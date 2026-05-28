#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VRPN数据记录测试节点（独立版）
- 订阅 /vrpn_client_node/QTR2/pose 获取位置和姿态
- 订阅 /vrpn_client_node/QTR2/twist 获取线速度（忽略角速度）
- 订阅 /mavros/rc/in 通过通道6控制数据记录
- 停止记录后独立绘图（位置、角度、线速度），不保存文件
"""
import rospy
import numpy as np
import matplotlib.pyplot as plt
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import RCIn
from tf.transformations import euler_from_quaternion


class VRPNDataRecorder:
    def __init__(self):
        # 订阅
        self.pose_sub = rospy.Subscriber(
            "/vrpn_client_node/QTR2/pose", PoseStamped, self.pose_callback, queue_size=5
        )
        self.twist_sub = rospy.Subscriber(
            "/vrpn_client_node/QTR2/twist", TwistStamped, self.twist_callback, queue_size=5
        )
        self.rc_sub = rospy.Subscriber(
            "/mavros/rc/in", RCIn, self.rc_callback, queue_size=5
        )

        # 状态变量
        self.vrpn_pose = None
        self.vrpn_linear_vel = None
        self.rc_in = None

        # 记录控制
        self.is_recording = False
        self.t0 = None
        self.recorded_data = {
            'time': [],
            'pos': [],     # [x, y, z]
            'rpy': [],     # [roll, pitch, yaw]
            'vel': [],     # [vx, vy, vz]
        }

    def normalize_angle_np(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def pose_callback(self, msg):
        self.vrpn_pose = msg

    def twist_callback(self, msg):
        self.vrpn_linear_vel = np.array([
            msg.twist.linear.x,
            msg.twist.linear.y,
            msg.twist.linear.z
        ])

    def get_current_data(self):
        """获取当前帧的位置、角度、线速度"""
        if self.vrpn_pose is None or self.vrpn_linear_vel is None:
            return None

        # 位置
        pos = np.array([
            self.vrpn_pose.pose.position.x,
            self.vrpn_pose.pose.position.y,
            self.vrpn_pose.pose.position.z
        ])

        # 姿态 -> 欧拉角
        q = self.vrpn_pose.pose.orientation
        phi, theta, psi = euler_from_quaternion([q.x, q.y, q.z, q.w])
        rpy = np.array([
            self.normalize_angle_np(phi),
            self.normalize_angle_np(theta),
            self.normalize_angle_np(psi)
        ])

        # 线速度
        vel = self.vrpn_linear_vel.copy()

        return pos, rpy, vel

    def rc_callback(self, msg):
        self.rc_in = msg.channels
        prev_recording = self.is_recording

        # 通道6 > 1500：开始记录
        if self.rc_in[5] > 1500 and not prev_recording:
            rospy.loginfo("✅ [VRPN Record] 遥控器指令：开始记录数据！")
            self.is_recording = True
            self.recorded_data = {
                'time': [], 'pos': [], 'rpy': [], 'vel': []
            }
            self.t0 = rospy.Time.now().to_sec()

        # 通道6 < 1500：停止记录并绘图
        elif self.rc_in[5] < 1500 and prev_recording:
            rospy.loginfo("🛑 [VRPN Record] 遥控器指令：停止记录并绘图！")
            self.is_recording = False
            if len(self.recorded_data['time']) > 0:
                self.plot_data()
            else:
                rospy.logwarn("[VRPN Record] 无数据可绘图")

    def plot_data(self):
        """独立绘图：位置、角度、线速度 3个子图，不保存文件"""
        time_arr = np.array(self.recorded_data['time'])
        pos_arr = np.array(self.recorded_data['pos'])       # (N, 3)
        rpy_arr = np.array(self.recorded_data['rpy'])       # (N, 3)
        vel_arr = np.array(self.recorded_data['vel'])       # (N, 3)

        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        fig.suptitle('VRPN Data Recording', fontsize=14, fontweight='bold')

        # ====== 子图1：位置 ======
        ax1 = axes[0]
        ax1.plot(time_arr, pos_arr[:, 0], label='X [m]', linewidth=1.5)
        ax1.plot(time_arr, pos_arr[:, 1], label='Y [m]', linewidth=1.5)
        ax1.plot(time_arr, pos_arr[:, 2], label='Z [m]', linewidth=1.5)
        ax1.set_ylabel('Position [m]')
        ax1.set_title('Position (XYZ)')
        ax1.legend()
        ax1.grid(True)

        # ====== 子图2：角度 ======
        ax2 = axes[1]
        ax2.plot(time_arr, np.rad2deg(rpy_arr[:, 0]), label='Roll [deg]', linewidth=1.5)
        ax2.plot(time_arr, np.rad2deg(rpy_arr[:, 1]), label='Pitch [deg]', linewidth=1.5)
        ax2.plot(time_arr, np.rad2deg(rpy_arr[:, 2]), label='Yaw [deg]', linewidth=1.5)
        ax2.set_ylabel('Angle [deg]')
        ax2.set_title('Attitude (Roll / Pitch / Yaw)')
        ax2.legend()
        ax2.grid(True)

        # ====== 子图3：线速度 ======
        ax3 = axes[2]
        ax3.plot(time_arr, vel_arr[:, 0], label='Vx [m/s]', linewidth=1.5)
        ax3.plot(time_arr, vel_arr[:, 1], label='Vy [m/s]', linewidth=1.5)
        ax3.plot(time_arr, vel_arr[:, 2], label='Vz [m/s]', linewidth=1.5)
        ax3.set_xlabel('Time [s]')
        ax3.set_ylabel('Velocity [m/s]')
        ax3.set_title('Linear Velocity (Vx / Vy / Vz)')
        ax3.legend()
        ax3.grid(True)

        plt.tight_layout()
        plt.show(block=True)

    def run(self):
        rospy.loginfo("🚀 VRPN数据记录节点启动，等待数据...")
        rate = rospy.Rate(100)

        while not rospy.is_shutdown():
            data = self.get_current_data()

            if data is not None and self.is_recording:
                pos, rpy, vel = data
                t_current = rospy.Time.now().to_sec() - self.t0

                self.recorded_data['time'].append(t_current)
                self.recorded_data['pos'].append(pos.copy())
                self.recorded_data['rpy'].append(rpy.copy())
                self.recorded_data['vel'].append(vel.copy())

                # 每50帧打印一次状态摘要
                if len(self.recorded_data['time']) % 50 == 0:
                    rospy.loginfo_throttle(
                        1.0,
                        f"[VRPN Record] 已记录 {len(self.recorded_data['time'])} 帧 | "
                        f"pos: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) | "
                        f"yaw: {np.rad2deg(rpy[2]):.1f}°"
                    )

            rate.sleep()


if __name__ == "__main__":
    try:
        rospy.init_node("vrpn_data_recorder", anonymous=False)
        recorder = VRPNDataRecorder()
        recorder.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("VRPN数据记录节点已退出")
