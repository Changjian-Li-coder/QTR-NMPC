#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

# 全局变量定义
_lambda_thrust = 0.01
_lambda_servo = 0.1
_thrust_min = 0.1
_thrust_max = 0.9
_servo_angle_limit_rad = np.pi / 8.0
_max_iter = 30
_lr_thrust = 0.1
_lr_servo = 0.3

_L = 0.182  # 轴距365mm
_kf = 0.015
sqrt2_2 = 0.7071

# 初始化矩阵和向量
_matrix_A = np.array([
    [0.0, -sqrt2_2, 0.0,  sqrt2_2, 0.0, sqrt2_2, 0.0, -sqrt2_2],
    [0.0,  sqrt2_2, 0.0, -sqrt2_2, 0.0, sqrt2_2, 0.0, -sqrt2_2],
    [-1.0, 0.0, -1.0, 0.0, -1.0, 0.0, -1.0, 0.0],
    [-sqrt2_2 * _L,  sqrt2_2 * _kf,  sqrt2_2 * _L, -sqrt2_2 * _kf, sqrt2_2 * _L,  sqrt2_2 * _kf, -sqrt2_2 * _L, -sqrt2_2 * _kf],
    [ sqrt2_2 * _L,  sqrt2_2 * _kf, -sqrt2_2 * _L, -sqrt2_2 * _kf, sqrt2_2 * _L, -sqrt2_2 * _kf, -sqrt2_2 * _L,  sqrt2_2 * _kf],
    [_kf, _L, _kf, _L, -_kf, _L, -_kf, _L]
])

_W = np.array([1, 1, 1, 10, 10, 1])

def solve_allocation(U_target):
    x_out = np.zeros(8)  # 初始化输出向量

    # 热启动保护：如果全为0则初始化一个微小推力
    if np.linalg.norm(x_out) < 0.01:
        for i in range(4):
            x_out[2 * i] = 0.1  # 微小推力
            x_out[2 * i + 1] = 0.0
        # x_out[0] = 4.9
        # x_out[2] = 1.363
        # x_out[4] = 1.355
        # x_out[6] = 7.568
    # else:
    #     for i in range(4):
    #         x_out[2 * i + 1] = 0.0

    # 预先计算转置矩阵
    A_T = _matrix_A.T

    # 投影梯度下降迭代
    for _ in range(_max_iter):
        # 1. 计算当前的虚拟控制量 b
        b = np.zeros(8)
        for i in range(4):
            b[2 * i] = x_out[2 * i] * np.cos(x_out[2 * i + 1])
            b[2 * i + 1] = x_out[2 * i] * np.sin(x_out[2 * i + 1])

        # 2. 计算误差向量 e = A*b - U_target
        error = _matrix_A @ b - U_target

        # 应用加权矩阵 W，提升特定轴（如 x/y 轴力矩）的敏感度
        error *= _W

        # 3. 计算对 b 的梯度: g_b = 2 * A^T * error
        g_b = 2.0 * A_T @ error

        # 4. 使用链式法则计算对状态变量 x 的解析梯度
        grad = np.zeros(8)
        for i in range(4):
            F = x_out[2 * i]
            alpha = x_out[2 * i + 1]
            cos_a = np.cos(alpha)
            sin_a = np.sin(alpha)

            # 对推力 F_i 的梯度
            grad[2 * i] = g_b[2 * i] * cos_a + g_b[2 * i + 1] * sin_a + 2.0 * _lambda_thrust * F

            # 对角度 alpha_i 的梯度
            # grad[2 * i + 1] = 0

            grad[2 * i + 1] = -g_b[2 * i] * F * sin_a + g_b[2 * i + 1] * F * cos_a + 2.0 * _lambda_servo * alpha

        # 5. 更新状态
        for i in range(4):
            x_out[2 * i] -= grad[2 * i] * _lr_thrust  # 更新推力
            x_out[2 * i + 1] -= grad[2 * i + 1] * _lr_servo  # 更新舵机角度

        # 6. 边界约束 (投影)
        for i in range(4):
            # 推力约束
            x_out[2 * i] = np.clip(x_out[2 * i], _thrust_min, _thrust_max)

            # 舵机角度约束
            x_out[2 * i + 1] = np.clip(x_out[2 * i + 1], -_servo_angle_limit_rad, _servo_angle_limit_rad)

    # 输出最终结果
    for i in range(4):
        print(f"Thruster {i + 1}: F = {x_out[2 * i]:.3f}, alpha = {2 * x_out[2 * i + 1]:.3f}")

if __name__ == "__main__":
    # 定义期望的控制输入 (x/y/z 力和滚/俯/偏转矩)
    u = np.array([0.0, 0.0, -1.82, -0.23, 0.36, -0.01])
    print("control input:", u)

    # 调用分配求解器
    solve_allocation(u)
