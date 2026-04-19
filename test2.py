#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from numpy.linalg import pinv

# ====================== 配置类：集中管理参数（对齐C++参数） ======================
class TiltQuadConfig:
    """倾转四旋翼参数配置类（统一管理，便于调参）"""
    def __init__(self):
        # 基础物理参数
        self.sqrt2_2 = np.sqrt(2) / 2  # √2/2 ≈ 0.7071
        self.L = 0.182                 # 力臂长度 (m)
        self.kf = 0.01                 # 反扭矩系数
        self.n_motors = 4              # 旋翼数量
        self.n_inputs = 8              # 控制变量维度（4旋翼×2分量）
        self.n_outputs = 6             # 虚拟控制维度（Fx,Fy,Fz,Tx,Ty,Tz）

        # 执行器约束（对齐C++的servo_angle_limit）
        self.F_min = 0.0               # 推力非负（对齐C++的0下限）
        self.F_max = 50.0              # 最大推力（电机上限）
        self.servo_angle_limit_deg = 30.0  # 舵机角度限幅（度），对应±π/6
        self.alpha_min = -np.pi/6      # 最小倾转角（-30°）
        self.alpha_max = np.pi/6       # 最大倾转角（30°）

        # 旋翼权重矩阵（8维，对齐C++的_W）
        self.W = np.diag([
            1.0, 1.0,  # 旋翼1 x/y分量权重
            1.0, 1.0,  # 旋翼2 x/y分量权重
            1.0, 1.0,  # 旋翼3 x/y分量权重
            1.0, 1.0   # 旋翼4 x/y分量权重
        ])

        # 6维输出权重矩阵（对齐C++的W_u: [1,1,1,5,5,1]）
        self.W_u = np.diag([1.0, 1.0, 1.0, 5.0, 5.0, 1.0])

        # 构建核心A矩阵
        self.matrix_A = self._build_A_matrix()

    def _build_A_matrix(self):
        """构建控制分配核心A矩阵（与C++/原逻辑完全一致）"""
        L, kf, s2 = self.L, self.kf, self.sqrt2_2
        return np.array([
            [0.0, -s2, 0.0,  s2, 0.0, s2, 0.0, -s2],
            [0.0, -s2, 0.0,  s2, 0.0,-s2, 0.0,  s2],
            [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            [-s2*L,  s2*kf,  s2*L, -s2*kf, s2*L,  s2*kf, -s2*L, -s2*kf],
            [-s2*L,  s2*kf,  s2*L, -s2*kf,-s2*L, -s2*kf,  s2*L,  s2*kf],
            [-kf, -L, -kf, -L, kf, -L, kf, -L]
        ])

# ====================== 核心控制分配类（对齐C++逻辑） ======================
class TiltQuadControlAllocator:
    def __init__(self, config: TiltQuadConfig):
        self.cfg = config
        self.matrix_A = config.matrix_A
        self.W = config.W
        self.W_u = config.W_u
        # 预计算加权伪逆（避免重复计算，提升效率）
        self.A_W_pinv = self._calc_weighted_pinv()
        # 初始化变量（对齐C++）
        self.b = np.zeros(8)          # 8维中间变量
        self.F = np.zeros(4)          # 初始推力
        self.alpha = np.zeros(4)      # 初始倾角（弧度）
        self.alpha_clamped_rad = np.zeros(4)  # 限幅后倾角（弧度）

    def _calc_weighted_pinv(self):
        """计算加权伪逆（鲁棒版：处理奇异矩阵，对齐C++逻辑）"""
        A = self.matrix_A
        W = self.W
        A_T = A.T

        # 论文公式：A_W† = W·A^T·(A·W·A^T)^{-1}
        try:
            A_W_T = W @ A_T
            A_W_A_T = A @ A_W_T
            # 用pinv替代inv，增强鲁棒性（对应C++的geninv）
            A_W_A_T_inv = pinv(A_W_A_T)
            A_W_pinv = A_W_T @ A_W_A_T_inv
        except np.linalg.LinAlgError:
            # 降级为普通伪逆（兜底方案，对齐C++的回退逻辑）
            A_W_pinv = pinv(A)
        return A_W_pinv

    def _clip_alpha_rad(self, alpha):
        """倾角限幅（对齐C++的手动限幅逻辑）"""
        alpha_clamped = np.zeros(4)
        for i in range(4):
            if alpha[i] > self.cfg.alpha_max:
                alpha_clamped[i] = self.cfg.alpha_max
            elif alpha[i] < self.cfg.alpha_min:
                alpha_clamped[i] = self.cfg.alpha_min
            else:
                alpha_clamped[i] = alpha[i]
        return alpha_clamped

    def _clip_F_non_negative(self, F):
        """推力非负限幅（对齐C++的推力非负约束）"""
        F_clamped = np.maximum(F, self.cfg.F_min)
        # 可选：保留最大推力约束（C++未实现，如需对齐可取消注释）
        # F_clamped = np.clip(F_clamped, self.cfg.F_min, self.cfg.F_max)
        return F_clamped

    def control_allocation(self, F_d: np.ndarray, Tau_d: np.ndarray):
        """
        对齐C++的控制分配逻辑：
        1. 加权伪逆求b → 初始F/alpha
        2. 倾角限幅
        3. 固定限幅倾角，重构矩阵求解最优推力
        4. 推力非负限幅
        :param F_d: 期望合力 [Fx, Fy, Fz] (3,)
        :param Tau_d: 期望力矩 [Tx, Ty, Tz] (3,)
        :return:
            F_clamped: 最终推力 [F1,F2,F3,F4] (4,)
            alpha_clamped_deg: 限幅后倾角（度） [α1,α2,α3,α4] (4,)
            x_out: 输出向量（前4位推力，后4位倾角(度)）(8,)
            U_error: 控制分配误差
        """
        # 1. 拼接6维虚拟控制量（对齐C++的U_target）
        U_target = np.concatenate([F_d, Tau_d])  # (6,)
        x_prev = np.zeros(8)  # 回退用的初始值（对齐C++的x_prev）

        # 2. 求解8维中间变量b（对齐C++第一步）
        try:
            self.b = self.A_W_pinv @ U_target
        except Exception:
            self.b = x_prev[:8]  # 求逆失败回退

        # 3. 极坐标变换解算初始F和alpha（对齐C++）
        for i in range(4):
            F_ix = self.b[2*i]
            F_iy = self.b[2*i+1]
            self.F[i] = np.hypot(F_ix, F_iy)
            self.alpha[i] = np.arctan2(F_iy, F_ix)

        # 4. 倾角限幅（对齐C++的手动限幅）
        self.alpha_clamped_rad = self._clip_alpha_rad(self.alpha)

        # 5. 构造8x4矩阵C（对齐C++）
        C = np.zeros((8, 4))
        for i in range(4):
            C[2*i, i] = np.cos(self.alpha_clamped_rad[i])
            C[2*i+1, i] = np.sin(self.alpha_clamped_rad[i])

        # 6. 构建M = A·C（6x4，对齐C++）
        M = self.matrix_A @ C
        M_T = M.T

        # 7. 加权最小二乘求解最优F（对齐C++：F = (M^T·W_u·M)^{-1}·M^T·W_u·U_target）
        try:
            M_T_Wu_M = M_T @ self.W_u @ M
            M_T_Wu_M_inv = pinv(M_T_Wu_M)  # 鲁棒求逆（对应C++的geninv）
            F_clamped = M_T_Wu_M_inv @ M_T @ self.W_u @ U_target
        except np.linalg.LinAlgError:
            # 求逆失败回退到初始F（对齐C++的回退逻辑）
            PX4_WARN = print  # 模拟PX4日志
            PX4_WARN("Constrained matrix inversion failed, use initial F")
            F_clamped = self.F

        # 8. 推力非负限幅（对齐C++）
        F_clamped = self._clip_F_non_negative(F_clamped)

        # 9. 构造最终输出向量x_out（对齐C++：前4位推力，后4位倾角(度)）
        alpha_clamped_deg = np.rad2deg(self.alpha_clamped_rad)
        x_out = np.concatenate([F_clamped, alpha_clamped_deg])

        # 10. 计算分配误差（验证精度）
        U_calc = self.matrix_A @ self.b
        U_error = np.linalg.norm(U_calc - U_target)

        return F_clamped, alpha_clamped_deg, x_out, U_error

# ====================== 测试用例（验证与C++对齐） ======================
def run_tests():
    # 1. 初始化配置和分配器
    cfg = TiltQuadConfig()
    allocator = TiltQuadControlAllocator(cfg)

    # 全局打印配置（抑制科学计数法，保留4位小数）
    np.set_printoptions(suppress=True, precision=4)

    # 测试2：前飞+偏航工况（与C++输入对齐）
    print("\n===== 测试2：前飞+偏航工况 =====")
    F_d_forward = np.array([0.0, 0.0, 20.0])  # X前飞+Y侧向力
    Tau_d_yaw = np.array([0.00, -1.00, 0.00])        # 偏航力矩
    F_forward, alpha_forward_deg, x_out_forward, err_forward = allocator.control_allocation(F_d_forward, Tau_d_yaw)
    print(f"旋翼推力 F [N]: {F_forward}")
    print(f"倾转角 α [deg]: {alpha_forward_deg}")
    print(f"最终输出x_out [F1-F4, α1-α4(deg)]: {x_out_forward}")
    print(f"分配误差: {err_forward:.6f}")
    print(f"倾角约束范围: [{np.rad2deg(cfg.alpha_min):.1f}°, {np.rad2deg(cfg.alpha_max):.1f}°]")

if __name__ == "__main__":
    run_tests()
