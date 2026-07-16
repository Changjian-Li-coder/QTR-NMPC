#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
class PIDParams:
    def __init__(self):
        # 基础参数
        self.dt = 0.015  # 控制周期（与原NMPC一致）

        # 位置环PID参数 (x,y,z)
        self.pos_kp = np.array([8.0, 8.0, 10.0])    # 比例增益
        self.pos_ki = np.array([0.1, 0.1, 0.2])     # 积分增益
        self.pos_kd = np.array([4.0, 4.0, 5.0])     # 微分增益
        self.pos_int_limit = np.array([1.0, 1.0, 1.5])  # 积分限幅

        # 速度环PID参数 (vx,vy,vz)
        self.vel_kp = np.array([4.0, 4.0, 5.0])
        self.vel_ki = np.array([0.05, 0.05, 0.1])
        self.vel_kd = np.array([1.0, 1.0, 1.5])
        self.vel_int_limit = np.array([0.5, 0.5, 0.8])

        # 姿态环PID参数 (roll,pitch,yaw)
        self.att_kp = np.array([60.0, 60.0, 30.0])
        self.att_ki = np.array([1.0, 1.0, 0.5])
        self.att_kd = np.array([10.0, 10.0, 5.0])
        self.att_int_limit = np.array([2.0, 2.0, 1.0])

        # 角速度环PID参数 (p,q,r)
        self.omega_kp = np.array([8.0, 8.0, 4.0])
        self.omega_ki = np.array([0.5, 0.5, 0.2])
        self.omega_kd = np.array([1.0, 1.0, 0.5])
        self.omega_int_limit = np.array([1.0, 1.0, 0.5])

        # 悬停配平（与原NMPC一致）
        hover_thrust = 2.645 * 9.81
        self.u_trim = np.array([0.0, 0.0, hover_thrust, 0.0, 0.0, 0.0])