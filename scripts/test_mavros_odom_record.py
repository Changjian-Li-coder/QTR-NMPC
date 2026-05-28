#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MAVROS Odom数据记录测试节点
- 订阅 /mavros/local_position/odom 获取位置、速度、角度、角速度
- 机体速度 → 惯性坐标系速度
- 订阅 /mavros/rc/in 通过通道6控制数据记录
- 停止记录后绘图（位置/速度/角度/角速度），不保存文件
"""
import rospy
import numpy as np
import matplotlib.pyplot as plt
from nav_msgs.msg import Odometry
from mavros_msgs.msg import RCIn
from tf.transformations import euler_from_quaternion, quaternion_matrix


class MavrosOdomRecorder:
    def __init__(self):
        # 订阅
        self.odom_sub = rospy.Subscriber(
            "/mavros/local_position/odom", Odometry, self.odom_callback, queue_size=5
        )
        self.rc_sub = rospy.Subscriber(
            "/mavros/rc/in", RCIn, self.rc_callback, queue_size=5
        )

        # 状态变量
        self.mavros_odom = None
        self.rc_in = None

        # 记录控制
        self.is_recording = False
        self.t0 = None
        self.recorded_data = {
            'time': [],
            'pos': [],      # [x, y, z]
            'vel': [],      # [vx, vy, vz]  惯性系
            'rpy': [],      # [roll, pitch, yaw]
            'omega': [],    # [p, q, r]     角速度
        }

    def normalize_angle_np(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def odom_callback(self, msg):
        self.mavros_odom = msg

    def get_current_data(self):
        """从Odom提取数据，并将机体速度转换到惯性系"""
        if self.mavros_odom is None:
            return None

        odom = self.mavros_odom

        # 1. 位置
        pos = np.array([
            odom.pose.pose.position.x,
            odom.pose.pose.position.y,
            odom.pose.pose.position.z
        ])

        # 2. 姿态 -> 欧拉角
        q = odom.pose.pose.orientation
        phi, theta, psi = euler_from_quaternion([q.x, q.y, q.z, q.w])
        rpy = np.array([
            self.normalize_angle_np(phi),
            self.normalize_angle_np(theta),
            self.normalize_angle_np(psi)
        ])

        # 3. 机体坐标系线速度 → 惯性坐标系线速度
        vel_body = np.array([
            odom.twist.twist.linear.x,
            odom.twist.twist.linear.y,
            odom.twist.twist.linear.z
        ])
        R = quaternion_matrix([q.x, q.y, q.z, q.w])[0:3, 0:3]
        vel_inertial = R @ vel_body

        # 4. 角速度 (机体坐标系)
        omega = np.array([
            odom.twist.twist.angular.x,
            odom.twist.twist.angular.y,
            odom.twist.twist.angular.z
        ])

        return pos, vel_inertial, rpy, omega

    def rc_callback(self, msg):
        self.rc_in = msg.channels
        prev_recording = self.is_recording

        # 通道6 > 1500：开始记录
        if self.rc_in[5] > 1500 and not prev_recording:
            rospy.loginfo("✅ [Odom Record] 遥控器指令：开始记录数据！")
            self.is_recording = True
            self.recorded_data = {
                'time': [], 'pos': [], 'vel': [], 'rpy': [], 'omega': []
            }
            self.t0 = rospy.Time.now().to_sec()

        # 通道6 < 1500：停止记录并绘图
        elif self.rc_in[5] < 1500 and prev_recording:
            rospy.loginfo("🛑 [Odom Record] 遥控器指令：停止记录并绘图！")
            self.is_recording = False
            if len(self.recorded_data['time']) > 0:
                self.plot_data()
            else:
                rospy.logwarn("[Odom Record] 无数据可绘图")

    def plot_data(self):
        """绘制4×1子图：位置、速度(惯性系)、角度、角速度，不保存文件"""
        time_arr = np.array(self.recorded_data['time'])
        pos_arr  = np.array(self.recorded_data['pos'])    # (N, 3)
        vel_arr  = np.array(self.recorded_data['vel'])    # (N, 3)
        rpy_arr  = np.array(self.recorded_data['rpy'])    # (N, 3)
        omega_arr = np.array(self.recorded_data['omega']) # (N, 3)

        fig, axes = plt.subplots(4, 1, figsize=(14, 14))
        fig.suptitle('MAVROS Local Odom Data Recording', fontsize=14, fontweight='bold')

        # ====== 子图1：位置 ======
        ax1 = axes[0]
        ax1.plot(time_arr, pos_arr[:, 0], label='X [m]', linewidth=1.5)
        ax1.plot(time_arr, pos_arr[:, 1], label='Y [m]', linewidth=1.5)
        ax1.plot(time_arr, pos_arr[:, 2], label='Z [m]', linewidth=1.5)
        ax1.set_ylabel('Position [m]')
        ax1.set_title('Position (XYZ)')
        ax1.legend()
        ax1.grid(True)

        # ====== 子图2：惯性系速度 ======
        ax2 = axes[1]
        ax2.plot(time_arr, vel_arr[:, 0], label='Vx_inertial [m/s]', linewidth=1.5)
        ax2.plot(time_arr, vel_arr[:, 1], label='Vy_inertial [m/s]', linewidth=1.5)
        ax2.plot(time_arr, vel_arr[:, 2], label='Vz_inertial [m/s]', linewidth=1.5)
        ax2.set_ylabel('Velocity [m/s]')
        ax2.set_title('Velocity (Inertial Frame)')
        ax2.legend()
        ax2.grid(True)

        # ====== 子图3：角度 ======
        ax3 = axes[2]
        ax3.plot(time_arr, np.rad2deg(rpy_arr[:, 0]), label='Roll [deg]', linewidth=1.5)
        ax3.plot(time_arr, np.rad2deg(rpy_arr[:, 1]), label='Pitch [deg]', linewidth=1.5)
        ax3.plot(time_arr, np.rad2deg(rpy_arr[:, 2]), label='Yaw [deg]', linewidth=1.5)
        ax3.set_ylabel('Angle [deg]')
        ax3.set_title('Attitude (Roll / Pitch / Yaw)')
        ax3.legend()
        ax3.grid(True)

        # ====== 子图4：角速度 ======
        ax4 = axes[3]
        ax4.plot(time_arr, np.rad2deg(omega_arr[:, 0]), label='p [deg/s]', linewidth=1.5)
        ax4.plot(time_arr, np.rad2deg(omega_arr[:, 1]), label='q [deg/s]', linewidth=1.5)
        ax4.plot(time_arr, np.rad2deg(omega_arr[:, 2]), label='r [deg/s]', linewidth=1.5)
        ax4.set_xlabel('Time [s]')
        ax4.set_ylabel('Angular Rate [deg/s]')
        ax4.set_title('Angular Velocity (Body Frame)')
        ax4.legend()
        ax4.grid(True)

        plt.tight_layout()
        plt.show(block=True)

    def run(self):
        rospy.loginfo("🚀 MAVROS Odom记录节点启动，等待数据...")
        rate = rospy.Rate(100)

        while not rospy.is_shutdown():
            data = self.get_current_data()

            if data is not None and self.is_recording:
                pos, vel, rpy, omega = data
                t_current = rospy.Time.now().to_sec() - self.t0

                self.recorded_data['time'].append(t_current)
                self.recorded_data['pos'].append(pos.copy())
                self.recorded_data['vel'].append(vel.copy())
                self.recorded_data['rpy'].append(rpy.copy())
                self.recorded_data['omega'].append(omega.copy())

                if len(self.recorded_data['time']) % 90 == 0:
                    rospy.loginfo_throttle(
                        1.0,
                        f"[Odom Record] 已记录 {len(self.recorded_data['time'])} 帧 | "
                        f"pos: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) | "
                        f"yaw: {np.rad2deg(rpy[2]):.1f}°"
                    )

            rate.sleep()


if __name__ == "__main__":
    try:
        rospy.init_node("mavros_odom_recorder", anonymous=False)
        recorder = MavrosOdomRecorder()
        recorder.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("MAVROS Odom记录节点已退出")
