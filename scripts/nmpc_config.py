#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from uav_config import UAVParams
import numpy as np
class NMPCParams:
    def __init__(self):
        self.Ts = 0.015 # 预测步长
        self.Np = 10   # 预测时域
        self.Nc = 5    # 控制时域

        self.nx = 18 # 状态维度：12维原状态 + 6维积分状态
        self.nx_original = 12 # 原状态维度
        self.nx_integral = 6 # 积分状态维度
        self.nu = 6  # 控制量维度：[Fx,Fy,Fz,τx,τy,τz]

        self.uav_params = UAVParams()

        # 悬停配平
        self.hover_thrust = self.uav_params.m * self.uav_params.g
        self.u_trim = np.array([0.0, 0.0, self.hover_thrust, 0.0, 0.0, 0.0])

        self.u_min = np.array([-30, -30, -self.hover_thrust, -6, -6, -6])
        self.u_max = np.array([30, 30, self.hover_thrust * 1.5, 6, 6, 6])
        self.du_min = np.array([-5, -5, -5, -5, -5, -5])
        self.du_max = np.array([5, 5, 5, 5, 5, 5])
        
        # 原12维状态约束 + 6维积分状态约束（积分项限幅）
        x_original_min = np.array([-10, -10, -1, -2, -2, -0.3,
                                   np.deg2rad(-90), np.deg2rad(-90), np.deg2rad(-180),
                                   np.deg2rad(-60), np.deg2rad(-60), np.deg2rad(-60)])
        x_integral_min = np.array([-5.0, -5.0, -2.0,  # 位置误差积分限幅
                                   np.deg2rad(-45), np.deg2rad(-45), np.deg2rad(-90)])  # 姿态误差积分限幅
        self.x_min = np.hstack([x_original_min, x_integral_min])  # 增广后18维状态下界
        
        x_original_max = np.array([1000, 1000, 1500, 200, 200, 500,
                                   np.deg2rad(90), np.deg2rad(90), np.deg2rad(180),
                                   np.deg2rad(60), np.deg2rad(60), np.deg2rad(60)])
        x_integral_max = np.array([5.0, 5.0, 2.0,
                                   np.deg2rad(45), np.deg2rad(45), np.deg2rad(90)])
        self.x_max = np.hstack([x_original_max, x_integral_max])  # 增广后18维状态上界

# ========== NMPC代价权重矩阵（固定一套权重） ==========
        self.Q = np.diag([
            150, 150, 300,          # 位置 (x,y,z)
            10, 10, 200,            # 速度 (vx,vy,vz)
            170.0, 170.0, 20.0,     # 姿态 (roll/pitch/yaw)
            10.0, 10.0, 5.0,        # 角速度 (p,q,r)
            20.0, 20.0, 2.0,        # 位置误差积分
            0.0, 0.0, 0.0           # 姿态误差积分
        ])
        self.P = self.Q * 1.5  # 终端权重
        self.R = np.diag([
            0, 0, 0,
            0, 0, 0
        ])
        self.S = np.diag([
            0, 0, 0,
            0, 0, 0
        ])