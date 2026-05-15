#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import numpy as np
import time
from uav_config import UAVParams
from nmpc_config import NMPCParams
from dynamics_model import Dynamics_Model
from acados_model import AcadosModelBuilder
from generate_trajectory import GenerateTrajectory
from plot_record_data import PlotRecordData

from mavros_msgs.msg import State, RCIn, ActuatorControl
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import PoseStamped, TwistStamped


class NMPCController:
    def __init__(self):
        self.uav_params = UAVParams()
        self.nmpc_params = NMPCParams()
        self.dynamics_model = Dynamics_Model()
        self.nmpc_solver = AcadosModelBuilder()
        self.nmpc_solver.create_acados_model()
        self.generate_trajectory = GenerateTrajectory()
        self.plot_recorder = PlotRecordData()

        # ROS通信 - 替换为指定的订阅话题
        self.vrpn_pose_sub = rospy.Subscriber("/vrpn_client_node/QTR2/pose", PoseStamped, self.vrpn_pose_callback, queue_size=5)
        self.vrpn_twist_sub = rospy.Subscriber("/vrpn_client_node/QTR2/twist", TwistStamped, self.vrpn_twist_callback, queue_size=5)
        self.mavros_odom_sub = rospy.Subscriber("/mavros/local_position/odom", Odometry, self.mavros_odom_callback, queue_size=5)
        self.state_sub = rospy.Subscriber("/mavros/state", State, self.state_callback, queue_size=5)
        self.rc_in_sub = rospy.Subscriber("/mavros/rc/in", RCIn, self.rc_in_callback, queue_size=5)
        self.control_pub = rospy.Publisher("/mavros/actuator_control", ActuatorControl, queue_size=5)
        
        # ROS状态变量
        self.x_current = None  # 原12维状态
        self.current_state = None
        self.state_ready = False
        self.t0 = None
        self.ref_pos_hover = None
        self.ref_yaw_hover = None

        # 新增：分状态存储变量
        self.vrpn_pose = None       # 存储VRPN的位置和姿态
        self.vrpn_twist = None      # 存储VRPN的线速度
        self.mavros_angular = None  # 存储MAVROS的角速度
        self.rc_in = None            # 存储遥控器输入（如果使用）

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

    def vrpn_pose_callback(self, msg):
        self.vrpn_pose = msg
        self.update_x_current()

    def vrpn_twist_callback(self, msg):
        self.vrpn_twist = msg

    def mavros_odom_callback(self, msg):
        self.mavros_angular = msg.twist.twist.angular

    def update_x_current(self):
        # 检查所有数据是否就绪
        if self.vrpn_pose is None or self.vrpn_twist is None or self.mavros_angular is None:
            return
        # 1. 位置 (x,y,z) - 来自VRPN Pose
        x = self.vrpn_pose.pose.position.x
        y = self.vrpn_pose.pose.position.y
        z = self.vrpn_pose.pose.position.z

        # 2. 线速度 (vx,vy,vz) - 来自VRPN Twist
        vx = self.vrpn_twist.twist.linear.x
        vy = self.vrpn_twist.twist.linear.y
        vz = self.vrpn_twist.twist.linear.z
        
        # 3. 姿态 (roll,pitch,yaw) - 来自VRPN Pose的四元数转欧拉角
        q = self.vrpn_pose.pose.orientation
        phi, theta, psi = euler_from_quaternion([q.x, q.y, q.z, q.w])
        phi = self.normalize_angle_np(phi)
        theta = self.normalize_angle_np(theta)
        psi = self.normalize_angle_np(psi)
        
        # 4. 角速度 (p,q,r) - 来自MAVROS Odom
        p_rate = self.mavros_angular.x
        q_rate = self.mavros_angular.y
        r_rate = self.mavros_angular.z

        # 拼接12维原状态
        self.x_current = np.array([x, y, z, vx, vy, vz, phi, theta, psi, p_rate, q_rate, r_rate])
        
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
        set_angle = np.deg2rad(20)
        flag = 0
        init = 0
        num = 0
        while not rospy.is_shutdown():
            if not self.state_ready or self.t0 is None:
                rate.sleep()
                continue
            t_current = rospy.Time.now().to_sec() - self.t0
            if init == 0:
                x_target = np.array([self.x_current[0].copy(), self.x_current[1].copy(), self.x_current[2].copy(),
                                    0.0, 0.0, 0.0,
                                    0.0, 0.0, self.x_current[8].copy(),
                                    0.0, 0.0, 0.0])
                init = 1
            
            if self.rc_in is not None:
                if self.rc_in[6] < 1500 and flag == 0: #拨杆向上，起飞
                    x_target = np.array([self.x_current[0].copy(), self.x_current[1].copy(), 0.4,
                                        0.0, 0.0, 0.0,
                                        0.0, 0.0, self.x_current[8].copy(),
                                        0.0, 0.0, 0.0])
                    flag = 1

                elif self.rc_in[6] > 1500 and flag == 1:  #拨杆向下，降落
                    x_target = np.array([self.x_current[0].copy(), self.x_current[1].copy(), -2.0,
                                        0.0, 0.0, 0.0,
                                        0.0, 0.0, self.x_current[8].copy(),
                                        0.0, 0.0, 0.0])
                    flag = 0
            if self.x_current[2] > 0.4:
                num += 1
            rospy.loginfo_throttle(0.2,f"num:{num}")
            if num > 200 and self.rc_in[6] < 1500:
                x_target[7] = set_angle

            x_ref = self.generate_trajectory.generate_line_reference_trajectory(self.x_current.copy(), x_target)
            # rospy.loginfo(f"x_current : {np.array2string(self.x_current, precision=4, floatmode='fixed', suppress_small=True, max_line_width=1000)}")
            # rospy.loginfo_throttle(0.2,f"x_target  : {np.array2string(x_target, precision=4, floatmode='fixed', suppress_small=True, max_line_width=1000)}")
            # for i in range(self.nmpc_params.Np + 1):
            #     rospy.loginfo(f"x_ref{i}    : {np.array2string(x_ref[:, i], precision=4, floatmode='fixed', suppress_small=True, max_line_width=1000)}")


            # 求解积分增广NMPC
            u_opt, success, solve_time = self.nmpc_solver.solve(self.x_current.copy(), x_ref)
            # rospy.loginfo_throttle(0.2, f"solve time:{solve_time}")
            send_u = u_opt.copy()
            send_u[3] -= 0.105
            send_u[4] += 0.085

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