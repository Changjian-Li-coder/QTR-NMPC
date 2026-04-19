#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ====================== 1. 基础库导入 ======================
import numpy as np
import rospy
from mavros_msgs.msg import State
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
from tf.transformations import euler_from_quaternion
import time
import os
import matplotlib.pyplot as plt

# 解决matplotlib在ROS中的绘图问题（非阻塞）
plt.rcParams.update({'font.size': 10})
plt.switch_backend('TkAgg')  # 或使用'Qt5Agg'

# ====================== 2. 无人机物理参数（完全复用原代码） ======================
class UAVParams:
    def __init__(self):
        self.m = 2
        self.L = 0.18
        self.Ixx = 0.01
        self.Iyy = 0.01
        self.Izz = 0.02
        self.I = np.diag([self.Ixx, self.Iyy, self.Izz])
        self.g = 9.81
        self.thrust_max = 2.223 * self.g * 4
        self.acceleration_xy_max = 2
        self.acceleration_z_max = 1
        self.roll_pitch_acceleration_max = 3.0
        self.yaw_acceleration_max = 1.0
        self.nu = 6  # u=[Fx,Fy,Fz,τx,τy,τz] 单位：推力：N；力矩：mN·m
        self.u_min = np.array([-10, -10, -self.thrust_max * 0.8, -60, -60, -30])
        self.u_max = np.array([10, 10, self.thrust_max * 0.8, 60, 60, 30])
        self.du_min = np.array([-1, -1, -1, -15, -15, -10])
        self.du_max = np.array([1, 1, 1, 15, 15, 10])
        self.x_min = np.array([-10, -10, -1, -2, -2, -0.2,
                               np.deg2rad(-90), np.deg2rad(-90), np.deg2rad(-180),
                               np.deg2rad(-60), np.deg2rad(-60), np.deg2rad(-20)])
        self.x_max = np.array([10, 10, 1.5, 2, 2, 0.2,
                               np.deg2rad(90), np.deg2rad(90), np.deg2rad(180),
                               np.deg2rad(60), np.deg2rad(60), np.deg2rad(60)])

# ====================== 3. PID参数配置（替换原NMPCParams） ======================
class PIDParams:
    def __init__(self):
        # 基础参数
        self.Ts = 0.01  # 控制周期（与原NMPC一致）

        # 位置环PID参数 (x,y,z)
        self.pos_kp = np.array([8.0, 8.0, 10.0])    # 比例增益
        self.pos_ki = np.array([0.1, 0.1, 0.2])     # 积分增益
        self.pos_kd = np.array([4.0, 4.0, 5.0])     # 微分增益
        self.pos_int_limit = np.array([1.0, 1.0, 1.5])  # 积分限幅

        # 速度环PID参数 (vx,vy,vz)
        self.vel_kp = np.array([4.0, 4.0, 5.0])
        self.vel_ki = np.array([0.05, 0.05, 0.1])
        self.vel_kd = np.array([1.0, 1.0, 1.5])
        self.vel_int_limit = np.array([0.5, 0.5, 0.8])

        # 姿态环PID参数 (roll,pitch,yaw)
        self.att_kp = np.array([60.0, 60.0, 30.0])
        self.att_ki = np.array([1.0, 1.0, 0.5])
        self.att_kd = np.array([10.0, 10.0, 5.0])
        self.att_int_limit = np.array([2.0, 2.0, 1.0])

        # 角速度环PID参数 (p,q,r)
        self.omega_kp = np.array([8.0, 8.0, 4.0])
        self.omega_ki = np.array([0.5, 0.5, 0.2])
        self.omega_kd = np.array([1.0, 1.0, 0.5])
        self.omega_int_limit = np.array([1.0, 1.0, 0.5])

        # 悬停配平（与原NMPC一致）
        hover_thrust = 2 * 9.81
        self.u_trim = np.array([0.0, 0.0, hover_thrust, 0.0, 0.0, 0.0])

# ====================== 4. PID控制器实现（替换原NMPCController） ======================
class PIDController:
    def __init__(self, uav_params, pid_params):
        self.uav = uav_params
        self.pid = pid_params

        # 初始化PID状态变量
        self.pos_int = np.zeros(3)    # 位置积分
        self.vel_int = np.zeros(3)    # 速度积分
        self.att_int = np.zeros(3)    # 姿态积分
        self.omega_int = np.zeros(3)   # 角速度积分

        # 历史状态记录（用于微分计算）
        self.last_pos_err = np.zeros(3)
        self.last_vel_err = np.zeros(3)
        self.last_att_err = np.zeros(3)
        self.last_omega_err = np.zeros(3)

        # 控制历史
        self.solve_time_history = []
        self.u_prev = self.pid.u_trim  # 上一帧控制量

    def normalize_angle_np(self, angle):
        """角度归一化（与原NMPC一致）"""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def pid_core(self, err, last_err, integral, kp, ki, kd, dt, int_limit):
        """通用PID计算核心"""
        # 比例项
        p_term = kp * err

        # 积分项
        integral = integral + ki * err * dt
        integral = np.clip(integral, -int_limit, int_limit)

        # 微分项
        d_term = kd * (err - last_err) / dt if dt > 1e-6 else 0.0

        # 总输出
        output = p_term + integral + d_term
        return output, integral, err

    def solve(self, x_current, x_ref):
        """
        PID控制计算（替代原NMPC的solve方法）
        :param x_current: 当前12维状态 [pos, vel, euler, omega]
        :param x_ref: 参考轨迹 (12, Np+1)，取第0列作为当前参考
        :return: 控制量u_opt, 成功标志, 计算耗时
        """
        # 计时（与原NMPC一致）
        solve_start = time.perf_counter()

        try:
            # 1. 提取当前状态和参考值
            pos = x_current[0:3]          # 当前位置
            vel = x_current[3:6]          # 当前速度
            att = x_current[6:9]          # 当前姿态（欧拉角）
            omega = x_current[9:12]       # 当前角速度

            ref_pos = x_ref[0:3, 0]       # 参考位置
            ref_vel = x_ref[3:6, 0]       # 参考速度
            ref_att = x_ref[6:9, 0]       # 参考姿态
            ref_omega = x_ref[9:12, 0]    # 参考角速度

            # 角度归一化
            att[2] = self.normalize_angle_np(att[2])
            ref_att[2] = self.normalize_angle_np(ref_att[2])

            # 2. 串级PID计算
            # 2.1 位置环 → 速度指令
            pos_err = ref_pos - pos
            vel_cmd, self.pos_int, self.last_pos_err = self.pid_core(
                pos_err, self.last_pos_err, self.pos_int,
                self.pid.pos_kp, self.pid.pos_ki, self.pid.pos_kd,
                self.pid.Ts, self.pid.pos_int_limit
            )

            # 2.2 速度环 → 加速度/推力指令（机体系）
            vel_err = (ref_vel + vel_cmd) - vel
            acc_cmd, self.vel_int, self.last_vel_err = self.pid_core(
                vel_err, self.last_vel_err, self.vel_int,
                self.pid.vel_kp, self.pid.vel_ki, self.pid.vel_kd,
                self.pid.Ts, self.pid.vel_int_limit
            )

            # 转换为机体系推力 (Fx,Fy)
            phi, theta, psi = att
            cos_phi = np.cos(phi)
            sin_phi = np.sin(phi)
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            cos_psi = np.cos(psi)
            sin_psi = np.sin(psi)
            Fx = self.uav.m * (acc_cmd[0] * cos_theta * cos_psi + acc_cmd[1] * sin_phi * sin_theta * cos_psi - acc_cmd[1] * cos_phi * sin_psi)
            Fy = self.uav.m * (acc_cmd[0] * cos_theta * sin_psi + acc_cmd[1] * sin_phi * sin_theta * sin_psi + acc_cmd[1] * cos_phi * cos_psi)
            Fz = self.uav.m * (acc_cmd[2] + self.uav.g)  # Z轴推力（含重力补偿）

            # 2.3 姿态环 → 角速度指令
            att_err = ref_att - att
            att_err[2] = self.normalize_angle_np(att_err[2])  # 偏航角误差归一化
            omega_cmd, self.att_int, self.last_att_err = self.pid_core(
                att_err, self.last_att_err, self.att_int,
                self.pid.att_kp, self.pid.att_ki, self.pid.att_kd,
                self.pid.Ts, self.pid.att_int_limit
            )

            # 2.4 角速度环 → 力矩指令
            omega_err = omega_cmd - omega
            torque, self.omega_int, self.last_omega_err = self.pid_core(
                omega_err, self.last_omega_err, self.omega_int,
                self.pid.omega_kp, self.pid.omega_ki, self.pid.omega_kd,
                self.pid.Ts, self.pid.omega_int_limit
            )

            # 组合控制量 [Fx,Fy,Fz,τx,τy,τz]
            u_opt = np.array([Fx, Fy, Fz, torque[0], torque[1], torque[2]])

            # 控制量约束（与原NMPC一致）
            u_opt = np.clip(u_opt, self.uav.u_min, self.uav.u_max)
            self.u_prev = u_opt

            # 计算耗时
            solve_time = time.perf_counter() - solve_start
            self.solve_time_history.append(solve_time)

            return u_opt, True, solve_time

        except Exception as e:
            solve_time = time.perf_counter() - solve_start
            self.solve_time_history.append(solve_time)
            rospy.logerr(f"⚠️ PID计算失败：{e}，使用悬停配平控制")
            return self.pid.u_trim, False, solve_time

# ====================== 5. 实物飞行主控制器（仅替换控制器类型） ======================
class UAVHostController:
    def __init__(self):
        self.uav_params = UAVParams()
        self.pid_params = PIDParams()  # 替换为PID参数
        self.controller = PIDController(self.uav_params, self.pid_params)  # 替换为PID控制器

        # 参考轨迹参数（与原NMPC一致）
        self.ref_radius = 1.0
        self.ref_h_max = 3.0
        self.ref_vz = 0.2
        self.ref_turns_to_hmax = 0.5

        # ROS状态变量（与原NMPC一致）
        self.x_current = None
        self.current_state = None
        self.state_ready = False
        self.t0 = None
        self.ref_pos_hover = None

        # 新增：数据记录相关（与原NMPC一致）
        self.is_armed = False
        self.is_recording = False
        self.recorded_data = {
            'time': [],
            'state': [],
            'control': [],
            'solve_time': [],
            'solve_success': []
        }
        # ROS通信（与原NMPC完全一致）
        self.pose_sub = rospy.Subscriber("/mavros/local_position/odom", Odometry, self.odom_callback, queue_size=5)
        self.state_sub = rospy.Subscriber("/mavros/state", State, self.state_callback, queue_size=5)
        self.control_pub = rospy.Publisher("/nmpc/control_cmd", Float64MultiArray, queue_size=5)

    def state_callback(self, msg):
        """监听mavros/state，检测解锁/上锁状态（与原NMPC一致）"""
        self.current_state = msg
        prev_armed = self.is_armed
        self.is_armed = msg.armed

        # 解锁：开始记录
        if self.is_armed and not prev_armed:
            rospy.loginfo("✅ 无人机解锁，开始记录数据！")
            self.is_recording = True
            self.recorded_data = {  # 重置记录数据
                'time': [],
                'state': [],
                'control': [],
                'solve_time': [],
                'solve_success': []
            }
            self.t0 = rospy.Time.now().to_sec()  # 重置时间戳

        # 上锁：停止记录并绘图
        if not self.is_armed and prev_armed:
            rospy.loginfo("🛑 无人机上锁，停止记录并绘图！")
            self.is_recording = False
            if len(self.recorded_data['time']) > 0:
                self.plot_recorded_data()  # 绘制数据
            else:
                rospy.logwarn("⚠️ 无记录数据，跳过绘图")

    def odom_callback(self, msg):
        """从/mavros/local_position/odom更新12维状态（与原NMPC完全一致）"""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        vz = msg.twist.twist.linear.z
        q = msg.pose.pose.orientation
        phi, theta, psi = euler_from_quaternion([q.x, q.y, q.z, q.w])

        # 关键：与原代码保持一致的坐标符号转换
        theta = -theta
        psi = -psi
        p_rate = msg.twist.twist.angular.x
        q_rate = -msg.twist.twist.angular.y
        r_rate = -msg.twist.twist.angular.z

        self.x_current = np.array([x, y, z, vx, vy, vz, phi, theta, psi, p_rate, q_rate, r_rate])
        if not self.state_ready:
            self.state_ready = True

        # 初始化时间戳
        if self.t0 is None:
            self.t0 = rospy.Time.now().to_sec()

    def generate_reference_trajectory(self, t_current):
        """
        动态生成参考轨迹（与原NMPC接口完全一致）
        :param t_current: 当前时刻（秒）
        :return: 参考轨迹矩阵, shape=(12, Np+1)
        """
        # 为了兼容PID的输入格式，保留Np+1维度（实际PID仅使用第0列）
        Np = 20  # 与原NMPC的预测时域一致（仅占位）
        Ts = self.pid_params.Ts
        x_ref = np.zeros((12, Np + 1))

        # 固定参考目标：悬停（与原NMPC一致）
        if self.ref_pos_hover is None:
            self.ref_pos_hover = self.x_current[0:3].copy()
            self.ref_pos_hover[2] = 0.65  # 固定高度
        ref_pos = self.ref_pos_hover.copy()
        ref_pos[2] = -0.55  # 与原NMPC一致的高度设置
        ref_vel = np.array([0.0, 0.0, 0.0])
        ref_euler = np.array([0.0, 0.0, self.x_current[8].copy()])
        ref_omega = np.array([0.0, 0.0, 0.0])

        ref_state = np.concatenate([ref_pos, ref_vel, ref_euler, ref_omega])
        for i in range(Np + 1):
            x_ref[:, i] = ref_state

        return x_ref

    def publish_control(self, u):
        """发布控制量到ROS话题（与原NMPC一致）"""
        msg = Float64MultiArray()
        msg.data = u.astype(float).tolist()
        self.control_pub.publish(msg)

    def plot_recorded_data(self):
        """绘制记录的无人机状态/控制量曲线（与原NMPC完全一致）"""
        # 提取数据
        time_arr = np.array(self.recorded_data['time'])
        state_arr = np.array(self.recorded_data['state'])  # (N,12)
        control_arr = np.array(self.recorded_data['control'])  # (N,6)
        solve_time_arr = np.array(self.recorded_data['solve_time'])
        solve_success_arr = np.array(self.recorded_data['solve_success'])

        # 创建子图
        fig, axes = plt.subplots(4, 2, figsize=(16, 12))
        fig.suptitle('UAV PID Flight Data', fontsize=16)

        # 1. 位置 (x,y,z)
        ax1 = axes[0,0]
        ax1.plot(time_arr, state_arr[:,2], label='z [m]', linewidth=1.5)
        ax1.set_title('Position')
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Position [m]')
        ax1.legend()
        ax1.grid(True)

        # 3. 姿态 (phi,theta,psi) → 转换为角度
        ax3 = axes[0,1]
        ax3.plot(time_arr, np.rad2deg(state_arr[:,6]), label='roll [deg]', linewidth=1.5)
        ax3.plot(time_arr, np.rad2deg(state_arr[:,7]), label='pitch [deg]', linewidth=1.5)
        ax3.set_title('Attitude (Euler Angles)')
        ax3.set_xlabel('Time [s]')
        ax3.set_ylabel('Angle [deg]')
        ax3.legend()
        ax3.grid(True)

        # 2. 速度 (vx,vy,vz)
        ax2 = axes[1,0]
        ax2.plot(time_arr, state_arr[:,3], label='vx [m/s]', linewidth=1.5)
        ax2.plot(time_arr, state_arr[:,4], label='vy [m/s]', linewidth=1.5)
        ax2.plot(time_arr, state_arr[:,5], label='vz [m/s]', linewidth=1.5)
        ax2.set_title('Velocity')
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Velocity [m/s]')
        ax2.legend()
        ax2.grid(True)

        # 4. 角速度 (p,q,r) → 转换为角度/秒
        ax4 = axes[1,1]
        ax4.plot(time_arr, np.rad2deg(state_arr[:,9]), label='p [deg/s]', linewidth=1.5)
        ax4.plot(time_arr, np.rad2deg(state_arr[:,10]), label='q [deg/s]', linewidth=1.5)
        ax4.plot(time_arr, np.rad2deg(state_arr[:,11]), label='r [deg/s]', linewidth=1.5)
        ax4.set_title('Angular Velocity')
        ax4.set_xlabel('Time [s]')
        ax4.set_ylabel('Angular Velocity [deg/s]')
        ax4.legend()
        ax4.grid(True)

        # 5. 推力 (Fx,Fy,Fz)
        ax5 = axes[2,0]
        ax5.plot(time_arr, control_arr[:,2], label='Fz [N]', linewidth=1.5)
        ax5.set_title('Thrust (Body Frame)')
        ax5.set_xlabel('Time [s]')
        ax5.set_ylabel('Force [N]')
        ax5.legend()
        ax5.grid(True)

        # 6. 力矩 (τx,τy,τz)
        ax6 = axes[2,1]
        ax6.plot(time_arr, control_arr[:,3], label='τx [N·m]', linewidth=1.5)
        ax6.plot(time_arr, control_arr[:,4], label='τy [N·m]', linewidth=1.5)
        ax6.plot(time_arr, control_arr[:,5], label='τz [N·m]', linewidth=1.5)
        ax6.set_title('Torque (Body Frame)')
        ax6.set_xlabel('Time [s]')
        ax6.set_ylabel('Torque [N·m]')
        ax6.legend()
        ax6.grid(True)

        # 7. 求解时间
        ax7 = axes[3,0]
        ax7.plot(time_arr, solve_time_arr * 1000, label='Compute Time [ms]', color='orange', linewidth=1.5)
        ax7.set_title('PID Compute Time')
        ax7.set_xlabel('Time [s]')
        ax7.set_ylabel('Time [ms]')
        ax7.legend()
        ax7.grid(True)

        # 8. 求解成功率
        ax8 = axes[3,1]
        ax8.plot(time_arr, solve_success_arr, label='Compute Success', color='green', linewidth=1.5, drawstyle='steps-post')
        ax8.set_title('PID Compute Success (1=Success, 0=Fail)')
        ax8.set_xlabel('Time [s]')
        ax8.set_ylabel('Success Flag')
        ax8.set_ylim(-0.1, 1.1)
        ax8.legend()
        ax8.grid(True)

        # 调整布局并保存/显示
        plt.tight_layout()
        plt.savefig(f"uav_pid_flight_data_{int(time.time())}.png", dpi=150)
        rospy.loginfo("📊 飞行数据图已保存！")
        plt.show(block=True)

    def run(self):
        """实物飞行主循环（与原NMPC一致）"""
        rospy.loginfo("PID控制器启动，等待/mavros/local_position/odom状态...")
        rate = rospy.Rate(120)  # 与原NMPC一致的控制频率

        while not rospy.is_shutdown():
            if not self.state_ready or self.t0 is None:
                rate.sleep()
                continue

            # 计算当前时间
            t_current = rospy.Time.now().to_sec() - self.t0

            # 1. 生成参考轨迹
            x_ref = self.generate_reference_trajectory(t_current)

            # 2. 求解PID（替代原NMPC求解）
            u_opt, success, solve_time = self.controller.solve(self.x_current, x_ref)

            # 控制量符号调整（与原NMPC一致）
            send_u = u_opt.copy()

            # 3. 发布控制指令
            self.publish_control(send_u)
            if success :#and self.is_recording:
                rospy.loginfo_throttle(0.2, f"求解:{'成功' if success else '失败'}"
                                            f"控制指令: {np.array2string(send_u, precision=4, floatmode='fixed', suppress_small=True, max_line_width=1000)}")

            # 4. 记录数据（仅在解锁时）
            if self.is_recording and self.x_current is not None:
                self.recorded_data['time'].append(t_current)
                self.recorded_data['state'].append(self.x_current.copy())
                self.recorded_data['control'].append(u_opt.copy())
                self.recorded_data['solve_time'].append(solve_time)
                self.recorded_data['solve_success'].append(1 if success else 0)

            rate.sleep()

# ====================== 7. 程序入口（与原NMPC一致） ======================
if __name__ == "__main__":
    try:
        rospy.init_node("pid_flight_controller", anonymous=False)
        host_controller = UAVHostController()
        host_controller.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("程序中断")
