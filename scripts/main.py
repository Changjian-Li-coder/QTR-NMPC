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
from position_pi_controller import PositionPIController

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

        # 位置PI控制器（补偿NMPC稳态误差）
        self.pos_pi_ctrl = PositionPIController(dt=self.nmpc_params.Ts)
        # 配置输出映射：x→Fx(idx0), y→Fy(idx1), z→Fz(idx2)
        self.pos_pi_ctrl.set_output_mapping(0, 1, 2)
        # 设置控制量限幅
        self.pos_pi_ctrl.set_limits(self.nmpc_params.u_min, self.nmpc_params.u_max)

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

            elif self.flight_phase == FlightPhase.HOVER:  # 2-悬停稳定（悬停在x_target点，不做轨迹跟踪）
                x_target[0] = self._takeoff_pos[0]
                x_target[1] = self._takeoff_pos[1]
                x_target[2] = 0.5
                x_target[7] = 0.0
                hover_elapsed = rospy.Time.now().to_sec() - self.hover_start_time
                rospy.loginfo_throttle(1.0,
                    f"⏳ 悬停稳定中... {hover_elapsed:.1f}/{self.hover_stable_duration:.1f}s"
                    f"  num={num}")
                # 悬停足够时间且稳定 → 开始轨迹
                if hover_elapsed > self.hover_stable_duration and num > 500:
                    self.flight_phase = FlightPhase.LAND
                #     self.flight_phase = FlightPhase.TRAJECTORY
                #     self._pitch_ramp_start_time = None
                #     # 初始化圆形轨迹：以起飞位置为圆心
                #     center_pos = self._takeoff_pos.copy()
                #     center_pos[1] -= 0.8
                #     center_pos[2] = 0.5
                #     self.reference_trajectory.init_circle(
                #         center=center_pos, radius=0.8, omega=0.6,
                #         z=0.5, phase_init=0.0, revolutions=1.0)
                #     rospy.loginfo(
                #         f"⭕ 开始执行圆形轨迹: 圆心=[{center_pos[0]:.1f},{center_pos[1]:.1f}]"
                #         f" 半径=0.8m 角速度=0.6rad/s 圈数=1")

            elif self.flight_phase == FlightPhase.TRAJECTORY:  # 3-执行轨迹
                # 检测轨迹完成（圈数检测）
                phase_traversed = (
                    self.reference_trajectory.circle_phase
                    - self.reference_trajectory._circle_phase_start)
                completed_cycles = phase_traversed / (2.0 * np.pi)
                rospy.loginfo_throttle(0.5,
                    f"⭕ 圆形轨迹飞行中  已完成: {completed_cycles:.2f}/{self.reference_trajectory.circle_revolutions:.0f}圈")
                if self.reference_trajectory.is_trajectory_done(self.x_current[0:3].copy()):
                    self.flight_phase = FlightPhase.LAND
                    x_target = np.array([self.x_current[0].copy(), self.x_current[1].copy(), -1.0,
                                        0.0, 0.0, 0.0,
                                        0.0, 0.0, self.x_current[8].copy(),
                                        0.0, 0.0, 0.0])
                    self._pitch_ramp_start_time = None
                    rospy.loginfo(f"✅ 圆形轨迹完成，开始降落  (已完成 {completed_cycles:.2f}圈)")
                else:
                    x_target[2] = 0.5  # 保持高度

            elif self.flight_phase == FlightPhase.LAND:  # 4-降落37
                x_target[2] = -1.0
                x_target[7] = 0.0
                self._pitch_ramp_start_time = None


            # ========== 生成参考轨迹 ==========
            if self.flight_phase == FlightPhase.TRAJECTORY:
                # 轨迹模式：使用 ReferenceTrajectory 生成完整参考
                x_ref = self.reference_trajectory.step(self.x_current.copy())
            else:
                # 非轨迹模式：使用 GenerateTrajectory 平滑插值到目标
                x_ref = self.generate_trajectory.generate_line_reference_trajectory(
                    self.x_current.copy(), x_target.copy())

            # rospy.loginfo_throttle(0.2,f"x_current : {np.array2string(self.x_current, precision=4, floatmode='fixed', suppress_small=True, max_line_width=1000)}")
            rospy.loginfo_throttle(0.2,f"x_target  : {np.array2string(x_ref[:,-1], precision=4, floatmode='fixed', suppress_small=True, max_line_width=1000)}")

            # 求解积分增广NMPC
            u_opt, success, solve_time = self.nmpc_solver.solve(self.x_current.copy(), x_ref)
            # rospy.loginfo_throttle(0.2, f"solve time:{solve_time}")
            # PI补偿（x,y,z三通道PI + 前馈补偿）
            target_pi = np.array([x_ref[0, -1], x_ref[1, -1], x_ref[2, -1]])
            send_u = self.pos_pi_ctrl.compute(
                u_opt.copy(), self.x_current[0:3].copy(), target_pi, self.is_armed)

            if self.send_prev is not None:
                thrust_xy_abn = np.any(np.abs(send_u[0:2] - self.send_prev[0:2]) > 0.05)
                thrust_z_abn = np.abs(send_u[2] - self.send_prev[2]) > 0.3
                torque_abn = np.any(np.abs(send_u[3:6] - self.send_prev[3:6]) > 0.05)
                # if thrust_xy_abn or thrust_z_abn or torque_abn:
                    # rospy.logwarn_throttle(0.2, f"⚠️ 控制量跳变过大，保持上一帧控制量！当前控制量：{np.array2string(send_u, precision=4, floatmode='fixed', suppress_small=True, max_line_width=1000)}")
                    # send_u = 0.5 * self.send_prev.copy() + 0.5 * send_u
            self.send_prev = send_u.copy()

            self.publish_control(send_u)
            if success :  #and self.is_recording
                rospy.loginfo_throttle(0.2, f"控制指令: {np.array2string(send_u, precision=4, floatmode='fixed', suppress_small=True, max_line_width=1000)}")

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