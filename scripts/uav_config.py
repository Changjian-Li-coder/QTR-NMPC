#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
class UAVParams:
    def __init__(self):
        self.m = 2.645   # 无人机质量（kg）
        self.L = 0.18    # 无人机臂长（m）
        self.Ixx = 0.0390 # 无人机绕x轴转动惯量（kg·m²）
        self.Iyy = 0.0401 # 无人机绕y轴转动惯量（kg·m²）
        # self.Izz = 0.0690    # 无人机绕z轴转动惯量（kg·m²）
        # self.Ixx = 0.06 # 无人机绕x轴转动惯量（kg·m²）
        # self.Iyy = 0.06 # 无人机绕y轴转动惯量（kg·m²）
        self.Izz = 0.10  # 无人机绕z轴转动惯量（kg·m²）
        self.I = np.diag([self.Ixx, self.Iyy, self.Izz]) # 无人机转动惯量矩阵
        self.g = 9.81   # 重力加速度（m/s²）