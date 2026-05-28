#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import numpy as np
import time
from uav_config import UAVParams
from nmpc_config import NMPCParams
from dynamics_model import Dynamics_Model
from acados_model import AcadosModelBuilder
from reference_trajectory import ReferenceTrajectory, TrajectoryType
from generate_trajectory import GenerateTrajectory
from plot_record_data import PlotRecordData

from mavros_msgs.msg import State, RCIn, ActuatorControl
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion, quaternion_matrix
from geometry_msgs.msg import PoseStamped, TwistStamped

class FlightPhase:
    """飞行阶段枚举"""
    INIT = 0       # 初始等待：悬停在当前位置
    TAKEOFF = 1    # 起飞：爬升至目标高度
    HOVER = 2      # 悬停稳定：等待稳定后触发轨迹
    TRAJECTORY = 3 # 执行轨迹：沿预设航点飞行
    LAND = 4       # 降落：下降到地面


class NMPCController:
    def __init__(self):
        self.uav_params = UAVParams()
        self.nmpc_params = NMPCParams()
        self.dynamics_model = Dynamics_Model()
        self.nmpc_solver = AcadosModelBuilder()
        self.nmpc_solver.create_acados_model()
        self.reference_trajectory = ReferenceTrajectory()
        self.generate_trajectory = GenerateTrajectory()
        self.plot_recorder = PlotRecordData()

        # PI补偿参数（增强：抑制水平运动时掉高度）
        self.kp_z = 2.0          # 比例增益，快速响应高度偏差
        self.ki_z = 3.0          # 积分增益，消除静差
        self.z_int_limit = 5.0   # 积分限幅放宽，允许更大补偿
        self.z_error_integral = 0.0

        # ROS通信 - 替换为指定的订阅话题
        self.mavros_odom_sub = rospy.Subscriber("/mavros/local_position/odom", Odometry, self.mavros_odom_callback, queue_size=5)
        self.state_sub = rospy.Subscriber("/mavros/state", State, self.state_callback, queue_size=5)
        self.rc_in_sub = rospy.Subscriber("/mavros/rc/in", RCIn, self.rc_in_callback, queue_size=5)
        self.control_pub = rospy.Publisher("/mavros/actuator_control", ActuatorControl, queue_size=5)
        
        # ROS状态变量
        self.x_current = None  # 原12维状态
        self.x_prev = None     # 上一帧状态
        self.current_state = None
        self.state_ready = False
        self.t0 = None
        self.ref_pos_hover = None
        self.ref_yaw_hover = None
        self.send_prev = None

        # 新增：分状态存储变量
        self.vrpn_pose = None       # 存储VRPN的位置和姿态
        self.vrpn_twist = None      # 存储VRPN的线速度
        self.mavros_odom = None  # 存储MAVROS的角速度
        self.rc_in = None            # 存储遥控器输入（如果使用）

        # ================== 飞行阶段管理 ==================
        self.flight_phase = FlightPhase.INIT      # 当前飞行阶段
        self.hover_start_time = None              # 进入悬停的时刻
        self.hover_stable_duration = 1.0          # 悬停稳定等待时间 [s]
        self._takeoff_pos = None                  # 起飞位置 [x, y, z]

        # 俯仰角缓启动参数
        self._pitch_ramp_start_time = None   # 缓启动开始时刻（秒）
        self._pitch_ramp_duration = 2.0       # 缓启动持续时间（秒），从 0 均匀增加到目标角度

        # 数据记录
        self.is_armed = False
        self.is_recording = False
        self.recorded_data = {
            'time': [], 'state': [], 'control': [], 'solve_time': [], 'solve_success': []
        }

    def normalize_angle_np(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def state_callback(self, msg):
        self.current_state = msg
        self.is_armed = msg.armed

    def mavros_odom_callback(self, msg):
        self.mavros_odom = msg
        self.update_x_current()  # 每次收到新的里程计数据时更新状态

    def update_x_current(self):
        # 检查所有数据是否就绪
        if self.mavros_odom is None:
            return
        # 1. 惯性坐标系位置 (x,y,z) - 来自MAVROS Odom
        x = self.mavros_odom.pose.pose.position.x  # 使用MAVROS的x位置
        y = self.mavros_odom.pose.pose.position.y  # 使用MAVROS的y位置
        z = self.mavros_odom.pose.pose.position.z  # 使用MAVROS的z位置

        # 2. 机体坐标系姿态 (roll,pitch,yaw) - 来自MAVROS Odom的四元数转欧拉角
        q = self.mavros_odom.pose.pose.orientation  # 使用MAVROS的姿态四元数
        phi, theta, psi = euler_from_quaternion([q.x, q.y, q.z, q.w])
        phi = self.normalize_angle_np(phi)
        theta = self.normalize_angle_np(theta)
        psi = self.normalize_angle_np(psi)

        # 3. 机体坐标系线速度 (vx,vy,vz) - 来自MAVROS Odom
        vx_body = self.mavros_odom.twist.twist.linear.x
        vy_body = self.mavros_odom.twist.twist.linear.y
        vz_body = self.mavros_odom.twist.twist.linear.z
        vel_body = np.array([vx_body, vy_body, vz_body])

        # 将机体坐标系速度转换到惯性坐标系
        R = quaternion_matrix([q.x, q.y, q.z, q.w])[0:3, 0:3]  # 从四元数构造旋转矩阵
        vel_inertial = R @ vel_body  # 旋转到惯性坐标系
        vx, vy, vz = vel_inertial

        
        # 4. 机体坐标系角速度 (p,q,r) - 来自MAVROS Odom
        p_rate = self.mavros_odom.twist.twist.angular.x
        q_rate = self.mavros_odom.twist.twist.angular.y
        r_rate = self.mavros_odom.twist.twist.angular.z

        # 拼接12维原状态
        self.x_current = np.array([x, y, z, vx, vy, vz, phi, theta, psi, p_rate, q_rate, r_rate])

        if self.x_prev is not None:
            pos_abn = np.any(np.abs(self.x_current[0:3] - self.x_prev[0:3]) > 0.1)   # 位置数据异常
            vel_abn = np.any(np.abs(self.x_current[3:6] - self.x_prev[3:6]) > 1 )   # 速度数据异常
            # 欧拉角异常检测：先对角度差做归一化，避免±π跳变误判
            eul_diff = self.x_current[6:9] - self.x_prev[6:9]
            eul_diff_normalized = np.array([
                self.normalize_angle_np(eul_diff[0]),
                self.normalize_angle_np(eul_diff[1]),
                self.normalize_angle_np(eul_diff[2])
            ])
            eul_abn = np.any(np.abs(eul_diff_normalized) > 0.15)   # 欧拉角数据异常
            ome_abn = np.any(np.abs(self.x_current[9:12] - self.x_prev[9:12]) > 2.5) # 角速度数据异常
            if pos_abn or vel_abn or eul_abn or ome_abn:  # 状态跳变过大，可能是数据异常，保持上一帧状态
                rospy.logwarn_throttle(0.2, f"⚠️ 状态跳变过大，保持上一帧状态！当前状态：{np.array2string(self.x_current, precision=4, floatmode='fixed', suppress_small=True, max_line_width=1000)}")
                rospy.logwarn_throttle(0.2, f"⚠️ 上一帧状态：{np.array2string(self.x_prev, precision=4, floatmode='fixed', suppress_small=True, max_line_width=1000)}")
                self.x_current = 0.5 * self.x_prev.copy() + 0.5 * self.x_current.copy()

        self.x_prev = self.x_current.copy()

        if not self.state_ready:
            # 初始化增广NMPC的初始状态
            self.nmpc_solver.ocp.constraints.x0 = np.hstack([self.x_current, np.zeros(6)])
            self.nmpc_solver.acados_solver.set(0, 'lbx', self.nmpc_solver.ocp.constraints.x0)
            self.nmpc_solver.acados_solver.set(0, 'ubx', self.nmpc_solver.ocp.constraints.x0)
        self.state_ready = True

        if self.t0 is None:
            self.t0 = rospy.Time.now().to_sec()
    def rc_in_callback(self, msg):
        """处理遥控器输入（控制数据记录）"""
        self.rc_in = msg.channels
        prev_recording = self.is_recording
        # 通道6 > 1500：开始记录数据
        if self.rc_in[5] > 1500 and not prev_recording:
            rospy.loginfo("✅ 遥控器指令：开始记录数据！")
            self.is_recording = True
            self.recorded_data = {
                'time': [], 'state': [], 'control': [], 'solve_time': [], 'solve_success': []
            }
            self.t0 = rospy.Time.now().to_sec()
        # 通道6 < 1500：停止记录并绘图
        elif self.rc_in[5] < 1500 and prev_recording:
            rospy.loginfo("🛑 遥控器指令：停止记录并绘图！")
            self.is_recording = False
            if len(self.recorded_data['time']) > 0:
                self.plot_recorder.plot_recorded_data(self.recorded_data)

    def publish_control(self, send_data):
        msg = ActuatorControl()
        for i in range(0,6):
            msg.controls[i] = send_data[i]
        self.control_pub.publish(msg)

    def z_pi_compensate(self, u_opt, current_z, target_z):
        # 误差
        z_error = target_z - current_z

        # rospy.loginfo_throttle(0.1,f"z_error:{z_error:.4f}")

        # 积分
        if self.is_armed and self.x_current[2] > 0.1:
            self.z_error_integral += z_error * self.nmpc_params.Ts * 1.5

        # 积分限幅
        self.z_error_integral = np.clip(self.z_error_integral, -self.z_int_limit, self.z_int_limit)

        # rospy.loginfo_throttle(0.1,f"z_error_integral:{self.z_error_integral:.4f}")

        # PI输出
        T_z = self.kp_z * z_error + self.ki_z * self.z_error_integral

        # rospy.loginfo_throttle(0.1,f"z_integral:{self.ki_z * self.z_error_integral:.4f}")

        # 叠加到力矩
        u_comp = u_opt.copy()
        u_comp[2] += T_z

        # 防超限
        u_comp = np.clip(u_comp, self.nmpc_params.u_min, self.nmpc_params.u_max)
        # rospy.loginfo_throttle(0.1,f"u_opt:{np.array2string(u_opt[2], precision=3, floatmode='fixed', suppress_small=True, max_line_width=1000)}   u_comp:{np.array2string(u_comp[2], precision=3, floatmode='fixed', suppress_small=True, max_line_width=1000)}")
        return u_comp

    def run(self):
        rospy.loginfo("积分增广NMPC控制器启动，等待状态...")
        rate = rospy.Rate(100)
        x_target = np.zeros(12)
        set_angle = np.deg2rad(0)
        flag = 0
        init = 0
        num = 0
        while not rospy.is_shutdown():
            self.update_x_current()
            if not self.state_ready or self.t0 is None:
                rate.sleep()
                continue
            t_current = rospy.Time.now().to_sec() - self.t0

            # ========== 初始化：悬停在当前位置 ==========
            if init == 0:
                x_target = np.array([self.x_current[0].copy(), self.x_current[1].copy(), self.x_current[2].copy(),
                                    0.0, 0.0, 0.0,
                                    0.0, 0.0, self.x_current[8].copy(),
                                    0.0, 0.0, 0.0])
                self._takeoff_pos = self.x_current[0:3].copy()
                init = 1

            # ========== 遥控器指令（优先处理） ==========
            if self.rc_in is not None:
                if self.rc_in[6] < 1500 and flag == 0:  # 拨杆向上 → 起飞
                    x_target = np.array([self.x_current[0].copy(), self.x_current[1].copy(), 0.5,
                                        0.0, 0.0, 0.0,
                                        0.0, 0.0, self.x_current[8].copy(),
                                        0.0, 0.0, 0.0])
                    flag = 1
                    self.flight_phase = FlightPhase.TAKEOFF
                    self._takeoff_pos = self.x_current[0:3].copy()
                    self._pitch_ramp_start_time = None
                    rospy.loginfo("🛫 起飞指令")

                elif self.rc_in[6] > 1500 and flag == 1:  # 拨杆向下 → 降落（随时中断）
                    x_target = np.array([self.x_current[0].copy(), self.x_current[1].copy(), -1.0,
                                        0.0, 0.0, 0.0,
                                        0.0, 0.0, self.x_current[8].copy(),
                                        0.0, 0.0, 0.0])
                    flag = 0
                    self.flight_phase = FlightPhase.LAND
                    self._pitch_ramp_start_time = None
                    rospy.loginfo("🛬 遥控器降落指令")

            # ========== 高度计数（用于起飞检测） ==========
            if self.x_current[2] > 0.3 and self.flight_phase >= FlightPhase.TAKEOFF:
                num += 1

            # ========== 飞行阶段状态机 ==========
            #   INIT(0) → TAKEOFF(1) → HOVER(2) → TRAJECTORY(3) → LAND(4)
            #   任何阶段均可由 RC 降落指令中断至 LAND(4)
            # ====================================================

            if self.flight_phase == FlightPhase.TAKEOFF:  # 1-起飞爬升
                x_target[2] = 0.5
                x_target[7] = 0.0
                if self.x_current[2] > 0.45 and num > 50:
                    self.flight_phase = FlightPhase.HOVER
                    self.hover_start_time = rospy.Time.now().to_sec()
                    rospy.loginfo("✅ 到达目标高度，进入悬停稳定阶段")

            elif self.flight_phase == FlightPhase.HOVER:  # 2-悬停稳定
                x_target[0] = self._takeoff_pos[0]
                x_target[1] = self._takeoff_pos[1]
                x_target[2] = 0.5
                x_target[7] = 0.0
                hover_elapsed = rospy.Time.now().to_sec() - self.hover_start_time
                rospy.loginfo_throttle(1.0,
                    f"⏳ 悬停稳定中... {hover_elapsed:.1f}/{self.hover_stable_duration:.1f}s"
                    f"  num={num}")
                # 悬停足够时间且稳定 → 开始轨迹
                if hover_elapsed > self.hover_stable_duration and num > 200:
                    self.flight_phase = FlightPhase.TRAJECTORY
                    self._pitch_ramp_start_time = None
                    # 初始化直线轨迹：从起飞位置向前飞行
                    start_pos = self._takeoff_pos.copy()
                    start_pos[2] = 0.5
                    end_pos = np.array([start_pos[0] + 0.8, start_pos[1], 0.5])
                    self.reference_trajectory.init_line(
                        start_pos, end_pos, yaw=0.0, speed=1.0)
                    rospy.loginfo(
                        f"➡️ 开始执行直线轨迹: [{start_pos[0]:.1f},{start_pos[1]:.1f}]"
                        f" → [{end_pos[0]:.1f},{end_pos[1]:.1f}]")

            elif self.flight_phase == FlightPhase.TRAJECTORY:  # 3-执行轨迹
                # 检测轨迹完成（同时判断时间 + 实际位置到达终点）
                dist_to_end = np.linalg.norm(
                    self.x_current[0:2] - self.reference_trajectory.line_end[0:2])
                rospy.loginfo_throttle(0.5,
                    f"✈️ 轨迹飞行中  距终点: {dist_to_end:.2f}m")
                if self.reference_trajectory.is_trajectory_done(self.x_current[0:3].copy()):
                    self.flight_phase = FlightPhase.LAND
                    x_target = np.array([self.x_current[0].copy(), self.x_current[1].copy(), -1.0,
                                        0.0, 0.0, 0.0,
                                        0.0, 0.0, self.x_current[8].copy(),
                                        0.0, 0.0, 0.0])
                    self._pitch_ramp_start_time = None
                    rospy.loginfo(f"✅ 轨迹完成，开始降落  (距终点 {dist_to_end:.2f}m)")
                else:
                    x_target[2] = 0.5  # 保持高度

            elif self.flight_phase == FlightPhase.LAND:  # 4-降落
                x_target[2] = -1.0
                x_target[7] = 0.0
                self._pitch_ramp_start_time = None

            # ========== 俯仰角缓启动（仅轨迹阶段） ==========
            if self.flight_phase == FlightPhase.TRAJECTORY:
                if self._pitch_ramp_start_time is None:
                    self._pitch_ramp_start_time = rospy.Time.now().to_sec()
                    rospy.loginfo(
                        f"🔄 缓启动俯仰角: 0 → {np.rad2deg(set_angle):.1f}°")
                elapsed = rospy.Time.now().to_sec() - self._pitch_ramp_start_time
                ramp_ratio = np.clip(elapsed / self._pitch_ramp_duration, 0.0, 1.0)
                x_target[7] = set_angle * ramp_ratio
            else:
                x_target[7] = 0.0

            # ========== 生成参考轨迹 ==========
            if self.flight_phase == FlightPhase.TRAJECTORY:
                # 轨迹模式：使用 ReferenceTrajectory 生成完整参考
                x_ref = self.reference_trajectory.step(self.x_current.copy())
                # 覆盖俯仰参考（前飞需求），roll/yaw 由轨迹生成器处理
                x_ref[7, :] = x_target[7]
            else:
                # 非轨迹模式：使用 GenerateTrajectory 平滑插值到目标
                x_ref = self.generate_trajectory.generate_line_reference_trajectory(
                    self.x_current.copy(), x_target.copy())

            # rospy.loginfo_throttle(0.2,f"x_current : {np.array2string(self.x_current, precision=4, floatmode='fixed', suppress_small=True, max_line_width=1000)}")
            rospy.loginfo_throttle(0.2,f"x_target  : {np.array2string(x_target, precision=4, floatmode='fixed', suppress_small=True, max_line_width=1000)}")

            # 求解积分增广NMPC
            u_opt, success, solve_time = self.nmpc_solver.solve(self.x_current.copy(), x_ref)
            # rospy.loginfo_throttle(0.2, f"solve time:{solve_time}")
            # PI补偿
            send_u = self.z_pi_compensate(u_opt.copy(), self.x_current[2].copy(), x_target[2].copy())
            # send_u = u_opt.copy()
            send_u[4] += 0.1
            send_u[3] -= 0.05
            send_u[0] -= 0.5

            if self.send_prev is not None:
                thrust_xy_abn = np.any(np.abs(send_u[0:2] - self.send_prev[0:2]) > 0.05)
                thrust_z_abn = np.abs(send_u[2] - self.send_prev[2]) > 0.3
                torque_abn = np.any(np.abs(send_u[3:6] - self.send_prev[3:6]) > 0.05)
                # if thrust_xy_abn or thrust_z_abn or torque_abn:
                    # rospy.logwarn_throttle(0.2, f"⚠️ 控制量跳变过大，保持上一帧控制量！当前控制量：{np.array2string(send_u, precision=4, floatmode='fixed', suppress_small=True, max_line_width=1000)}")
                    # send_u = 0.5 * self.send_prev.copy() + 0.5 * send_u
            self.send_prev = send_u.copy()

            self.publish_control(send_u)
            # if success :  #and self.is_recording
            #     rospy.loginfo_throttle(0.2, f"控制指令: {np.array2string(send_u, precision=4, floatmode='fixed', suppress_small=True, max_line_width=1000)}")

            # 记录数据
            if self.is_recording and self.x_current is not None:
                self.recorded_data['time'].append(t_current)
                self.recorded_data['state'].append(self.x_current.copy())
                self.recorded_data['control'].append(u_opt.copy())
                self.recorded_data['solve_time'].append(solve_time)
                self.recorded_data['solve_success'].append(1 if success else 0)

            rate.sleep()

# ====================== 程序入口 ======================
if __name__ == "__main__":
    try:
        rospy.init_node("nmpc_flight_controller", anonymous=False)
        host_controller = NMPCController()
        host_controller.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("程序中断")