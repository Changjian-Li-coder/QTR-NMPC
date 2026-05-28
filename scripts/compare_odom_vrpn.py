#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Odom vs VRPN 位置对比记录节点
- 订阅 /mavros/local_position/odom 获取飞控位置
- 订阅 /vrpn_client_node/QTR2/pose 获取视觉定位位置
- 订阅 /mavros/rc/in 通过通道6控制数据记录
- 停止记录后绘图：3×1子图分别对比 X / Y / Z
- 不保存图片
"""
import rospy
import numpy as np
import matplotlib.pyplot as plt
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import RCIn


class OdomVRPNComparer:
    def __init__(self):
        # 订阅
        self.odom_sub = rospy.Subscriber(
            "/mavros/local_position/odom", Odometry, self.odom_callback, queue_size=5
        )
        self.vrpn_sub = rospy.Subscriber(
            "/vrpn_client_node/QTR2/pose", PoseStamped, self.vrpn_callback, queue_size=5
        )
        self.rc_sub = rospy.Subscriber(
            "/mavros/rc/in", RCIn, self.rc_callback, queue_size=5
        )

        # 状态变量
        self.mavros_odom = None
        self.vrpn_pose = None
        self.rc_in = None

        # 记录控制
        self.is_recording = False
        self.t0 = None
        self.recorded_data = {
            'time': [],
            'odom_pos': [],   # [x, y, z]
            'vrpn_pos': [],   # [x, y, z]
        }

    def odom_callback(self, msg):
        self.mavros_odom = msg

    def vrpn_callback(self, msg):
        self.vrpn_pose = msg

    def get_current_data(self):
        """同时获取 Odom 和 VRPN 的位置数据"""
        if self.mavros_odom is None or self.vrpn_pose is None:
            return None

        # Odom 位置
        odom_pos = np.array([
            self.mavros_odom.pose.pose.position.x,
            self.mavros_odom.pose.pose.position.y,
            self.mavros_odom.pose.pose.position.z
        ])

        # VRPN 位置
        vrpn_pos = np.array([
            self.vrpn_pose.pose.position.x,
            self.vrpn_pose.pose.position.y,
            self.vrpn_pose.pose.position.z
        ])

        return odom_pos, vrpn_pos

    def rc_callback(self, msg):
        self.rc_in = msg.channels
        prev_recording = self.is_recording

        # 通道6 > 1500：开始记录
        if self.rc_in[5] > 1500 and not prev_recording:
            rospy.loginfo("✅ [Odom vs VRPN] 遥控器指令：开始记录数据！")
            self.is_recording = True
            self.recorded_data = {
                'time': [], 'odom_pos': [], 'vrpn_pos': []
            }
            self.t0 = rospy.Time.now().to_sec()

        # 通道6 < 1500：停止记录并绘图
        elif self.rc_in[5] < 1500 and prev_recording:
            rospy.loginfo("🛑 [Odom vs VRPN] 遥控器指令：停止记录并绘图！")
            self.is_recording = False
            if len(self.recorded_data['time']) > 0:
                self.plot_comparison()
            else:
                rospy.logwarn("[Odom vs VRPN] 无数据可绘图")

    def plot_comparison(self):
        """绘制3×1子图，分别对比 X / Y / Z 的 odom 与 vrpn 数据"""
        time_arr = np.array(self.recorded_data['time'])
        odom_arr = np.array(self.recorded_data['odom_pos'])  # (N, 3)
        vrpn_arr = np.array(self.recorded_data['vrpn_pos'])  # (N, 3)

        labels = ['X', 'Y', 'Z']
        colors_odom = ['#E74C3C', '#3498DB', '#2ECC71']
        colors_vrpn = ['#C0392B', '#2980B9', '#27AE60']
        linestyles = ['-', '--', '-.']

        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        fig.suptitle(
            'Odom vs VRPN Position Comparison',
            fontsize=15, fontweight='bold'
        )

        for i in range(3):
            ax = axes[i]
            # Odom (实线)
            ax.plot(
                time_arr, odom_arr[:, i],
                label=f'Odom-{labels[i]}',
                color=colors_odom[i], linewidth=1.8, linestyle='-'
            )
            # VRPN (虚线)
            ax.plot(
                time_arr, vrpn_arr[:, i],
                label=f'VRPN-{labels[i]}',
                color=colors_vrpn[i], linewidth=1.8, linestyle='--'
            )

            ax.set_ylabel(f'{labels[i]} [m]', fontsize=12)
            ax.set_title(f'{labels[i]} Position Comparison', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

            # 计算并显示偏差统计
            diff = odom_arr[:, i] - vrpn_arr[:, i]
            rmse = np.sqrt(np.mean(diff ** 2))
            mean_err = np.mean(diff)
            std_err = np.std(diff)
            ax.text(
                0.98, 0.05,
                f'RMSE: {rmse:.3f} m\nMean: {mean_err:.3f} m\nStd: {std_err:.3f} m',
                transform=ax.transAxes,
                fontsize=9, verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.7)
            )

        axes[2].set_xlabel('Time [s]', fontsize=12)

        plt.tight_layout()
        plt.show(block=True)

    def run(self):
        rospy.loginfo("🚀 Odom vs VRPN 对比记录节点启动，等待数据...")
        rate = rospy.Rate(100)

        while not rospy.is_shutdown():
            data = self.get_current_data()

            if data is not None and self.is_recording:
                odom_pos, vrpn_pos = data
                t_current = rospy.Time.now().to_sec() - self.t0

                self.recorded_data['time'].append(t_current)
                self.recorded_data['odom_pos'].append(odom_pos.copy())
                self.recorded_data['vrpn_pos'].append(vrpn_pos.copy())

                if len(self.recorded_data['time']) % 90 == 0:
                    diff = odom_pos - vrpn_pos
                    rospy.loginfo_throttle(
                        1.0,
                        f"[Odom vs VRPN] 已记录 {len(self.recorded_data['time'])} 帧 | "
                        f"odom: ({odom_pos[0]:.3f}, {odom_pos[1]:.3f}, {odom_pos[2]:.3f}) | "
                        f"vrpn: ({vrpn_pos[0]:.3f}, {vrpn_pos[1]:.3f}, {vrpn_pos[2]:.3f}) | "
                        f"diff: ({diff[0]:.3f}, {diff[1]:.3f}, {diff[2]:.3f})"
                    )

            rate.sleep()


if __name__ == "__main__":
    try:
        rospy.init_node("odom_vrpn_comparer", anonymous=False)
        recorder = OdomVRPNComparer()
        recorder.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Odom vs VRPN 对比记录节点已退出")
