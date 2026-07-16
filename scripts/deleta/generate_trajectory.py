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