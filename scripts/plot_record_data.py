#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import pandas as pd  # 新增：用于数据保存
from mpl_toolkits.mplot3d import Axes3D  # 新增：3D绘图
import rospy
class PlotRecordData:
    def __init__(self):
        self.time_history = []
        self.state_history = []
        self.control_history = []
        self.solve_time_history = []
    def plot_recorded_data(self, recorded_data):
        """绘图函数（修改后：3D位置图 + 3×2子图 + 数据保存）"""
        time_arr = np.array(recorded_data['time'])
        state_arr = np.array(recorded_data['state'])
        control_arr = np.array(recorded_data['control'])
        solve_time_arr = np.array(recorded_data['solve_time'])
        solve_success_arr = np.array(recorded_data['solve_success'])

        # ====================== 1. 保存所有数据到指定路径的CSV/XLSX ======================
        # 定义指定保存路径（可根据需要修改）
        data_save_path = "catkin_ws/src/nmpc/data/"
        if not os.path.exists(data_save_path):
            os.makedirs(data_save_path)
        # 整理数据为DataFrame
        data_dict = {
            'time_s': time_arr,
            # 状态数据
            'pos_x_m': state_arr[:, 0],
            'pos_y_m': state_arr[:, 1],
            'pos_z_m': state_arr[:, 2],
            'vel_vx_m/s': state_arr[:, 3],
            'vel_vy_m/s': state_arr[:, 4],
            'vel_vz_m/s': state_arr[:, 5],
            'att_roll_rad': state_arr[:, 6],
            'att_pitch_rad': state_arr[:, 7],
            'att_yaw_rad': state_arr[:, 8],
            'ang_vel_p_rad/s': state_arr[:, 9],
            'ang_vel_q_rad/s': state_arr[:, 10],
            'ang_vel_r_rad/s': state_arr[:, 11],
            # 控制量数据
            'control_Fx_N': control_arr[:, 0],
            'control_Fy_N': control_arr[:, 1],
            'control_Fz_N': control_arr[:, 2],
            'control_tau_x_Nm': control_arr[:, 3],
            'control_tau_y_Nm': control_arr[:, 4],
            'control_tau_z_Nm': control_arr[:, 5],
            # NMPC求解数据
            'solve_time_s': solve_time_arr,
            'solve_success': solve_success_arr
        }
        df = pd.DataFrame(data_dict)
        
        # 保存为CSV文件
        csv_filename = os.path.join(data_save_path, f"uav_nmpc_data_{int(time.time())}.csv")
        df.to_csv(csv_filename, index=False, encoding='utf-8')
        # 保存为XLSX文件（需要openpyxl库，可通过pip install openpyxl安装）
        xlsx_filename = os.path.join(data_save_path, f"uav_nmpc_data_{int(time.time())}.xlsx")
        df.to_excel(xlsx_filename, index=False, engine='openpyxl')
        
        rospy.loginfo(f"📄 数据已保存到CSV：{csv_filename}")
        rospy.loginfo(f"📄 数据已保存到XLSX：{xlsx_filename}")


        # ====================== 2. 单独绘制XYZ位置3D图 ======================
        picture_save_path = "catkin_ws/src/nmpc/picture/"
        if not os.path.exists(picture_save_path):
            os.makedirs(picture_save_path)
        fig_3d = plt.figure(figsize=(10, 8))
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        # 绘制3D位置轨迹
        ax_3d.plot(state_arr[:, 0], state_arr[:, 1], state_arr[:, 2], 
                   label='UAV Position Trajectory', linewidth=2, color='blue')
        # 标记起点和终点
        ax_3d.scatter(state_arr[0, 0], state_arr[0, 1], state_arr[0, 2], 
                      color='red', s=50, label='Start Point', zorder=5)
        ax_3d.scatter(state_arr[-1, 0], state_arr[-1, 1], state_arr[-1, 2], 
                      color='green', s=50, label='End Point', zorder=5)
        # 设置坐标轴标签
        ax_3d.set_xlabel('X [m]', fontsize=12)
        ax_3d.set_ylabel('Y [m]', fontsize=12)
        ax_3d.set_zlabel('Z [m]', fontsize=12)
        ax_3d.set_title('UAV 3D Position Trajectory (XYZ)', fontsize=14)
        ax_3d.legend()
        ax_3d.grid(True)
        # 保存3D位置图
        pos_3d_filename = f"uav_3d_position_{int(time.time())}.png"
        fig_3d.savefig(os.path.join(picture_save_path, pos_3d_filename), dpi=150, bbox_inches='tight')
        rospy.loginfo(f"📊 3D位置轨迹图已保存：{pos_3d_filename}")
        plt.show(block=False)  # 非阻塞显示，避免卡住

        # ====================== 3. 绘制3×2子图（删去求解成功率） ======================
        fig, axes = plt.subplots(3, 2, figsize=(16, 15))
        fig.suptitle('UAV Augmented NMPC (Integral Augmentation) Flight Data', fontsize=16)

        # 子图1：姿态（roll/pitch/yaw）
        ax1 = axes[0, 0]
        ax1.plot(time_arr, np.rad2deg(state_arr[:, 6]), label='roll [deg]', linewidth=1.5)
        ax1.plot(time_arr, np.rad2deg(state_arr[:, 7]), label='pitch [deg]', linewidth=1.5)
        ax1.plot(time_arr, np.rad2deg(state_arr[:, 8]), label='yaw [deg]', linewidth=1.5)
        ax1.set_title('Attitude (Euler Angles)')
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Angle [deg]')
        ax1.legend()
        ax1.grid(True)

        # 子图2：速度（vx/vy/vz）
        ax2 = axes[0, 1]
        ax2.plot(time_arr, state_arr[:, 3], label='vx [m/s]', linewidth=1.5)
        ax2.plot(time_arr, state_arr[:, 4], label='vy [m/s]', linewidth=1.5)
        ax2.plot(time_arr, state_arr[:, 5], label='vz [m/s]', linewidth=1.5)
        ax2.set_title('Velocity')
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Velocity [m/s]')
        ax2.legend()
        ax2.grid(True)

        # 子图3：角速度（p/q/r）
        ax3 = axes[1, 0]
        ax3.plot(time_arr, np.rad2deg(state_arr[:, 9]), label='p [deg/s]', linewidth=1.5)
        ax3.plot(time_arr, np.rad2deg(state_arr[:, 10]), label='q [deg/s]', linewidth=1.5)
        ax3.plot(time_arr, np.rad2deg(state_arr[:, 11]), label='r [deg/s]', linewidth=1.5)
        ax3.set_title('Angular Velocity')
        ax3.set_xlabel('Time [s]')
        ax3.set_ylabel('Angular Velocity [deg/s]')
        ax3.legend()
        ax3.grid(True)

        # 子图4：推力（Fz）
        ax4 = axes[1, 1]
        ax4.plot(time_arr, control_arr[:, 0], label='Fx [N]', linewidth=1.5)
        ax4.plot(time_arr, control_arr[:, 1], label='Fy [N]', linewidth=1.5)
        ax4.plot(time_arr, control_arr[:, 2], label='Fz [N]', linewidth=1.5)
        ax4.set_title('Thrust (Body Frame Fz)')
        ax4.set_xlabel('Time [s]')
        ax4.set_ylabel('Force [N]')
        ax4.legend()
        ax4.grid(True)

        # 子图5：力矩（τx/τy/τz）
        ax5 = axes[2, 0]
        ax5.plot(time_arr, control_arr[:, 3], label='τx [N·m]', linewidth=1.5)
        ax5.plot(time_arr, control_arr[:, 4], label='τy [N·m]', linewidth=1.5)
        ax5.plot(time_arr, control_arr[:, 5], label='τz [N·m]', linewidth=1.5)
        ax5.set_title('Torque (Body Frame)')
        ax5.set_xlabel('Time [s]')
        ax5.set_ylabel('Torque [N·m]')
        ax5.legend()
        ax5.grid(True)

        # 子图6：求解时间
        ax6 = axes[2, 1]
        ax6.plot(time_arr, solve_time_arr * 1000, label='Solve Time [ms]', color='red', linewidth=1.5)
        ax6.set_title('Augmented NMPC Solve Time')
        ax6.set_xlabel('Time [s]')
        ax6.set_ylabel('Time [ms]')
        ax6.legend()
        ax6.grid(True)

        plt.tight_layout()
        # 保存3×2子图
        flight_data_filename = f"uav_augmented_nmpc_flight_data_{int(time.time())}.png"
        fig.savefig(os.path.join(picture_save_path, flight_data_filename), dpi=150, bbox_inches='tight')
        rospy.loginfo(f"📊 积分增广NMPC飞行数据图已保存：{flight_data_filename}")
        plt.show(block=True)  # 非阻塞显示，避免卡住