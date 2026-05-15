#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from nmpc_config import NMPCParams
from scipy.interpolate import CubicSpline
import numpy as np
import rospy

class GenerateTrajectory:
    def __init__(self):
        self.nmpc_params = NMPCParams()
        self.Np = self.nmpc_params.Np
        self.dt = self.nmpc_params.Ts
        self.max_v = self.nmpc_params.x_max[3:6]
        self.min_v = self.nmpc_params.x_min[3:6]
        self.max_w = self.nmpc_params.x_max[9:12]
        self.min_w = self.nmpc_params.x_min[9:12]

        # =========================== 8字绕飞核心参数===============================
        self.R = 1.0                  # 8字轨迹半径
        self.z_fixed = 1.0            # 固定飞行高度（z轴）
        self.w_theta = 0.2            # 绕飞角速度（控制8字飞行速度，越小越慢）
        self.center = [0, 0]          # 8字交汇点坐标 (x0, y0)
        # ========================================================================
        
        self.theta_now = 0.0          # 全局轨迹相位（核心：记录当前在8字上的位置）  

    def normalize_angle_np(self, angle):
        """归一化角度到[-π, π]"""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def generate_reference_trajectory(self, x_current, x_target):
        """生成平滑的参考轨迹（线性插值）"""
        Np = self.nmpc_params.Np
        x_ref = np.zeros((12, Np + 1))
        for i in range(12):
            x_ref[i, :] = np.linspace(x_current[i], x_target[i], Np + 1)
        # 角度归一化
        x_ref[6, :] = self.normalize_angle_np(x_ref[6, :])
        x_ref[7, :] = self.normalize_angle_np(x_ref[7, :])
        x_ref[8, :] = self.normalize_angle_np(x_ref[8, :])
        return x_ref

    def generate_line_reference_trajectory(self, x_current, x_target):
        """
        生成平滑的参考轨迹（基于三次样条的运动学插值）
        维度说明（12维状态向量）：
            0-2: 位置 (x, y, z)
            3-5: 线速度 (vx, vy, vz)
            6-8: 欧拉角 (roll, pitch, yaw)
            9-11: 角速度 (wx, wy, wz)
        逻辑：
            1. 位置/欧拉角：三次样条插值，约束两端速度为0
            2. 线速度：位置样条的一阶导数
            3. 角速度：欧拉角样条的一阶导数
        """
        # 初始化参考轨迹数组
        x_ref = np.zeros((12, self.Np + 1))
        # 生成时间节点（0到Np*dt，共Np+1个点）
        t_total = self.Np * self.dt
        t = np.linspace(0, t_total, self.Np + 1)

        # -------------------------- 位置插值（0-2维）--------------------------
        t_waypoints = np.array([0, t_total])
        for dim in range(3):
            y_waypoints = [x_current[dim], x_target[dim]]
            # 三次样条：约束起点/终点一阶导数为0（两端线速度为0）
            cs_pos = CubicSpline(t_waypoints, y_waypoints, bc_type=((1, 0.0), (1, 0.0)))
            # 位置轨迹
            x_ref[dim, :] = cs_pos(t)
            # 线速度 = 位置的一阶导数
            x_ref[dim + 3, :] = cs_pos(t, 1)
            # 速度限幅
            x_ref[dim + 3, :] = np.clip(x_ref[dim + 3, :], self.min_v[dim], self.max_v[dim])

        # -------------------------- 欧拉角插值（6-8维）--------------------------
        for dim in range(6, 9):
            # 先归一化当前和目标欧拉角
            current_angle = self.normalize_angle_np(x_current[dim])
            target_angle = self.normalize_angle_np(x_target[dim])
            y_waypoints = [current_angle, target_angle]
            # 三次样条：约束起点/终点一阶导数为0（两端角速度为0）
            cs_angle = CubicSpline(t_waypoints, y_waypoints, bc_type=((1, 0.0), (1, 0.0)))
            # 欧拉角轨迹（再次归一化防止插值过程中越界）
            angle_traj = self.normalize_angle_np(cs_angle(t))
            x_ref[dim, :] = angle_traj
            # 角速度 = 欧拉角的一阶导数（注：严格场景需考虑欧拉角→角速度的转换矩阵，此处简化为直接求导）
            x_ref[dim + 3, :] = cs_angle(t, 1)
            # 角速度限幅
            x_ref[dim + 3, :] = np.clip(x_ref[dim + 3, :], self.min_w[dim - 6], self.max_w[dim - 6])

        return x_ref

    def init_8_trajectory(self, R, z_fixed, w_theta, center):
        """初始化8字轨迹参数"""
        self.R = R
        self.z_fixed = z_fixed
        self.w_theta = w_theta
        self.center = center

    def figure_8_trajectory(self, theta):
        """8字轨迹方程：输入相位theta，输出xy位置 + 速度 + 航向角"""

        # 8字基础坐标（相对交汇点）
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        denom = 1 + sin_t**2
        
        # 位置
        x = self.center[0] + self.R * cos_t / denom
        y = self.center[1] + self.R * sin_t * cos_t / denom
        z = self.z_fixed

        # 线速度（对相位求导，转换为时间导数）
        dx_dt = -self.R * sin_t * (1 + sin_t**2) - 2 * self.R * cos_t**2 * sin_t
        dx_dt = dx_dt / denom**2 * self.w_theta
        dy_dt = self.R * (cos_t**2 - sin_t**2 - sin_t**4) / denom**2 * self.w_theta
        dz_dt = 0.0

        # 航向角yaw（机头朝向飞行方向）
        yaw = np.arctan2(dy_dt, dx_dt)
        # 横滚/俯仰固定为0（平飞）
        roll = 0.0
        pitch = 0.0

        # 角速度（角度对时间求导）
        wy = np.gradient(yaw, self.dt) if len(yaw.shape) else 0.0
        wx = 0.0
        wz = 0.0

        return np.array([x, y, z, dx_dt, dy_dt, dz_dt, roll, pitch, yaw, wx, wy, wz])

    def generate_8_reference_trajectory(self, x_current, x_target=None):
        """
        生成NMPC预测窗口内的8字短轨迹
        兼容原接口：x_target无用，自动生成8字片段
        输出：12维参考轨迹 [12, Np+1]
        """
        x_ref = np.zeros((12, self.Np + 1))
        # 生成未来Np步的时间序列 + 相位序列
        t = np.linspace(0, self.Np * self.dt, self.Np + 1)
        theta_list = self.theta_now + self.w_theta * t

        # 逐点生成8字轨迹片段
        for i, theta in enumerate(theta_list):
            state = self.figure_8_trajectory(theta)
            x_ref[:, i] = state

        # ===================== 核心：更新相位，实现滚动绕飞 =====================
        self.theta_now = theta_list[-1]
        # 相位归一化（防止数值过大，0~2π循环）
        self.theta_now = self.theta_now % (2 * np.pi)

        return x_ref