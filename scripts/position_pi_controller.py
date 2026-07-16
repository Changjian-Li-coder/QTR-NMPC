#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
位置PI补偿控制器

对 x,y,z 三通道分别进行 PI 控制，补偿 NMPC 的稳态误差。
输出直接叠加到 NMPC 控制量的对应通道。

控制量维度：[Fx, Fy, Fz, τx, τy, τz]
  默认映射: x误差→Fx(idx0), y误差→Fy(idx1), z误差→Fz(idx2)

使用方式:
    pi_ctrl = PositionPIController(dt=0.015)
    u_comp = pi_ctrl.compute(u_opt, current_pos, target_pos, is_armed)
"""

import numpy as np


class PositionPIController:
    """位置PI补偿控制器"""

    def __init__(self, dt=0.015):
        # ===================== PI增益（保守参数） =====================
        # 说明：NMPC已做主控制，PI仅补偿稳态误差，增益不宜过大
        # x通道（默认→Fx）
        self.kp_x = 1.0            # 0.5m误差 → 0.25N
        self.ki_x = 0.5           # 缓慢消除静差
        self.err_int_x = 0.0
        self.int_limit_x = 1.5     # 积分限幅小，防windup

        # y通道（默认→Fy）
        self.kp_y = 1.0
        self.ki_y = 0.5
        self.err_int_y = 0.0
        self.int_limit_y = 1.5

        # z通道（默认→Fz）
        self.kp_z = 2.0            # 比原来略保守
        self.ki_z = 3.0            # 积分慢一点，防震荡
        self.err_int_z = 0.0
        self.int_limit_z = 5.0     # 限幅收窄

        # ===================== 采样时间 =====================
        self.dt = dt

        # ===================== 输出通道映射 =====================
        # out_idx[0] = x误差输出到的控制量索引
        # out_idx[1] = y误差输出到的控制量索引
        # out_idx[2] = z误差输出到的控制量索引
        self.out_idx = [0, 1, 2]   # 默认: x→Fx, y→Fy, z→Fz

        # ===================== 前馈补偿 =====================
        # 固定偏置，始终叠加到控制量
        self.feedforward = np.array([-0.5, 0.0, 0.0, -0.05, 0.1, 0.0])

        # ===================== 控制量限幅 =====================
        self.u_min = None
        self.u_max = None

    # ------------------------------------------------------------------
    #  公共接口
    # ------------------------------------------------------------------

    def set_output_mapping(self, x_idx, y_idx, z_idx):
        """
        设置位置误差到控制通道的映射

        Args:
            x_idx: x误差的输出通道索引
            y_idx: y误差的输出通道索引
            z_idx: z误差的输出通道索引
        """
        self.out_idx = [int(x_idx), int(y_idx), int(z_idx)]

    def set_limits(self, u_min, u_max):
        """设置控制量限幅"""
        self.u_min = np.asarray(u_min, dtype=float)
        self.u_max = np.asarray(u_max, dtype=float)

    def reset(self):
        """重置所有积分项"""
        self.err_int_x = 0.0
        self.err_int_y = 0.0
        self.err_int_z = 0.0

    def compute(self, u_opt, current_pos, target_pos, is_armed, dt=None):
        """
        计算PI补偿后的控制量

        Args:
            u_opt:       NMPC原始控制量, shape (6,)
            current_pos: 当前位置 [x, y, z]
            target_pos:  目标位置 [x, y, z]
            is_armed:    是否已解锁（解锁后才积分）
            dt:          采样时间（可选，覆盖__init__中的值）

        Returns:
            u_comp: PI补偿后的控制量, shape (6,)
        """
        if dt is not None:
            self.dt = dt

        # ---- 误差 ----
        err_x = target_pos[0] - current_pos[0]
        err_y = target_pos[1] - current_pos[1]
        err_z = target_pos[2] - current_pos[2]

        # ---- 积分（仅解锁且离地后积分，防止地面累积） ----
        if is_armed and current_pos[2] > 0.1:
            self.err_int_x += err_x * self.dt
            self.err_int_y += err_y * self.dt
            self.err_int_z += err_z * self.dt

        # ---- 积分限幅 ----
        self.err_int_x = np.clip(self.err_int_x, -self.int_limit_x, self.int_limit_x)
        self.err_int_y = np.clip(self.err_int_y, -self.int_limit_y, self.int_limit_y)
        self.err_int_z = np.clip(self.err_int_z, -self.int_limit_z, self.int_limit_z)

        # ---- PI输出 ----
        pi_x = self.kp_x * err_x + self.ki_x * self.err_int_x
        pi_y = self.kp_y * err_y + self.ki_y * self.err_int_y
        pi_z = self.kp_z * err_z + self.ki_z * self.err_int_z

        # ---- 叠加到控制量 ----
        u_comp = u_opt.copy()
        u_comp[self.out_idx[0]] += pi_x
        u_comp[self.out_idx[1]] += pi_y
        u_comp[self.out_idx[2]] += pi_z

        # ---- 前馈补偿 ----
        for i in range(min(len(self.feedforward), len(u_comp))):
            if self.feedforward[i] != 0.0:
                u_comp[i] += self.feedforward[i]

        # ---- 控制量限幅 ----
        if self.u_min is not None and self.u_max is not None:
            u_comp = np.clip(u_comp, self.u_min, self.u_max)

        return u_comp

    def get_integrals(self):
        """获取当前积分值（用于调试/监控）"""
        return np.array([self.err_int_x, self.err_int_y, self.err_int_z])

    def set_x_gains(self, kp, ki, int_limit=1.0):
        """设置x通道PI参数"""
        self.kp_x = kp
        self.ki_x = ki
        self.int_limit_x = int_limit

    def set_y_gains(self, kp, ki, int_limit=1.0):
        """设置y通道PI参数"""
        self.kp_y = kp
        self.ki_y = ki
        self.int_limit_y = int_limit

    def set_z_gains(self, kp, ki, int_limit=5.0):
        """设置z通道PI参数"""
        self.kp_z = kp
        self.ki_z = ki
        self.int_limit_z = int_limit


# ================================================================
#  简单自测
# ================================================================
if __name__ == "__main__":
    ctrl = PositionPIController(dt=0.015)
    print("=== 位置PI控制器测试 ===")
    print(f"默认输出映射: x→idx{ctrl.out_idx[0]}, y→idx{ctrl.out_idx[1]}, z→idx{ctrl.out_idx[2]}")
    print(f"前馈补偿: {ctrl.feedforward}")

    u_opt = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
    current_pos = np.array([0.0, 0.0, 0.5])
    target_pos  = np.array([0.5, 0.0, 0.5])

    for step in range(10):
        u_comp = ctrl.compute(u_opt, current_pos, target_pos, is_armed=True)
        if step < 3:
            print(f"  step {step}: u_comp={np.array2string(u_comp, precision=4, floatmode='fixed', suppress_small=True)}")

    print(f"  积分值: {ctrl.get_integrals()}")
    ctrl.reset()
    print(f"  重置后积分: {ctrl.get_integrals()}")
    print("测试完成 ✓")
