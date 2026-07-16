#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全局轨迹参考生成器 GlobalTrajectory

===== 架构 =====
"全局轨迹离线预生成 + 在线截取参考切片" 的标准架构。
起飞 → 悬停 → 8字轨迹 → 悬停 → 降落，段间 C² 连续。

===== 轨迹分段 =====
  Segment 0: 起飞段 (Takeoff)
     时间: [0, T_takeoff]
     z 轴: 五次多项式 0 → H，起止 v=0, a=0
     xy 轴: 保持初始位置 (x0, y0)

  Segment 1: 悬停段 1 (Hover1)
     时间: [T_takeoff, T_takeoff + T_hover1]
     位置: (x0, y0, H)，速度/加速度为零

  Segment 2: 8字进入过渡 (Figure8 Entry)
     时间: [T_hover1_end, T_hover1_end + T_trans]
     变系数缩放: s(t) 从 0→1（五次多项式）
     8字形幅值乘以 s(t)，确保入口处 C² 连续

  Segment 3: 全幅8字轨迹 (Figure8 Full)
     时间: [T_entry_end, T_entry_end + T_f8]
     全幅 Lemniscate of Bernoulli:
       x(t) = A·sin(ωt) + cx
       y(t) = B·sin(2ωt) + cy
       z    = H

  Segment 4: 8字退出过渡 (Figure8 Exit)
     时间: [T_f8_end, T_f8_end + T_trans]
     变系数缩放: s(t) 从 1→0，出口处 C² 连续

  Segment 5: 悬停段 2 (Hover2)
     时间: [T_exit_end, T_exit_end + T_hover2]
     位置: (cx, cy, H)，速度/加速度为零

  Segment 6: 降落段 (Landing)
     时间: [T_hover2_end, T_hover2_end + T_land]
     z 轴: 五次多项式 H → 0，起止 v=0, a=0
     xy 轴: 保持 (cx, cy)

===== NMPC 参考接口 =====
  get_reference(t)        → (pos, vel, acc)          9维
  get_reference_12d(t)    → 12维状态 [pos, vel, eul, omega]
  get_reference_sequence(x_current, Ts, Np) → [12, Np+1] 矩阵

===== 扩展到基于弧长的索引 =====
  当前实现基于时间索引。若要扩展到基于弧长 (arc-length) 索引，
  可在 get_reference() 中增加弧长参数化映射 t = f(s)，或预计算弧长
  查找表 (LUT) 实现 t(s) 插值。弧长参数化有利于等速跟踪。
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from nmpc_config import NMPCParams


# ================================================================
#  数据结构
# ================================================================

@dataclass
class TrajectoryState:
    """轨迹状态：位置 + 速度 + 加速度"""
    position: np.ndarray      # [x, y, z]
    velocity: np.ndarray      # [vx, vy, vz]
    acceleration: np.ndarray  # [ax, ay, az]

    def to_array_9d(self) -> np.ndarray:
        """展平为 9 维向量"""
        return np.concatenate([self.position, self.velocity, self.acceleration])

    @staticmethod
    def from_array_9d(arr: np.ndarray) -> 'TrajectoryState':
        """从 9 维向量恢复"""
        return TrajectoryState(
            position=arr[0:3].copy(),
            velocity=arr[3:6].copy(),
            acceleration=arr[6:9].copy()
        )

    def to_12d_state(self, yaw: float = 0.0) -> np.ndarray:
        """
        转换为标准12维状态向量

        返回: [x,y,z, vx,vy,vz, roll,pitch,yaw, wx,wy,wz]
        注: roll=pitch=0（平飞假设），角速度从加速度推算
        """
        state = np.zeros(12)
        state[0:3] = self.position
        state[3:6] = self.velocity
        state[6] = 0.0     # roll ≈ 0（平飞）
        state[7] = 0.0     # pitch ≈ 0（平飞）
        state[8] = yaw
        state[9:12] = 0.0  # 角速度
        return state


class SegmentType:
    """轨迹段类型枚举"""
    TAKEOFF = 0
    HOVER = 1
    FIGURE8_ENTRY = 2
    FIGURE8_FULL = 3
    FIGURE8_EXIT = 4
    LANDING = 5


# ================================================================
#  五次多项式工具函数
# ================================================================

def _quintic_coeffs(z0: float, zf: float, T: float) -> np.ndarray:
    """
    解算五次多项式系数

    边界条件:
        t=0:  z=z0,  v=0, a=0
        t=T:  z=zf,  v=0, a=0

    多项式: z(t) = a0 + a1·t + a2·t² + a3·t³ + a4·t⁴ + a5·t⁵

    Args:
        z0: 起始位置
        zf: 终止位置
        T:  总时间 [s]

    Returns:
        coeffs: [a0, a1, a2, a3, a4, a5]
    """
    if T < 1e-8:
        return np.array([zf, 0.0, 0.0, 0.0, 0.0, 0.0])

    dz = zf - z0
    T2 = T * T
    T3 = T2 * T
    T4 = T3 * T
    T5 = T4 * T

    return np.array([
        z0,              # a0
        0.0,             # a1
        0.0,             # a2
        10.0 * dz / T3,  # a3
        -15.0 * dz / T4, # a4
        6.0 * dz / T5    # a5
    ])


def _eval_quintic(t: float, coeffs: np.ndarray,
                  T: float) -> Tuple[float, float, float]:
    """
    计算五次多项式在时刻 t 的值、一阶导、二阶导

    Args:
        t:      当前时刻 [s]
        coeffs: [a0, ..., a5]
        T:      总时间（用于 clamp）

    Returns:
        (z, v, a): 位置、速度、加速度
    """
    t_clamped = np.clip(t, 0.0, T)
    a0, a1, a2, a3, a4, a5 = coeffs

    z = a0 + a1 * t_clamped + a2 * t_clamped ** 2 \
        + a3 * t_clamped ** 3 + a4 * t_clamped ** 4 \
        + a5 * t_clamped ** 5

    v = a1 + 2.0 * a2 * t_clamped \
        + 3.0 * a3 * t_clamped ** 2 \
        + 4.0 * a4 * t_clamped ** 3 \
        + 5.0 * a5 * t_clamped ** 4

    a = 2.0 * a2 + 6.0 * a3 * t_clamped \
        + 12.0 * a4 * t_clamped ** 2 \
        + 20.0 * a5 * t_clamped ** 3

    return z, v, a


def _quintic_ramp_coeffs(T: float) -> np.ndarray:
    """
    标准五次多项式 ramp: 0→1, 两端 v=a=0

    s(0)=0, s'(0)=0, s''(0)=0
    s(T)=1, s'(T)=0, s''(T)=0

    用于 8 字轨迹的平滑进入/退出过渡。
    """
    return _quintic_coeffs(0.0, 1.0, T)


def _eval_ramp(t: float, coeffs: np.ndarray,
               T: float) -> Tuple[float, float, float]:
    """计算 ramp 在时刻 t 的值 s, ds/dt, d²s/dt²"""
    return _eval_quintic(t, coeffs, T)


# ================================================================
#  8字轨迹 (Lemniscate of Bernoulli) 解析计算
# ================================================================

def _figure8_state(theta: float, A: float, B: float,
                   omega: float, cx: float, cy: float
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算 8 字轨迹在给定相位处的状态

    参数方程:
        x = cx + A·sin(θ)
        y = cy + B·sin(2θ)
        θ = ωt

    Args:
        theta: 当前相位 θ = ωt [rad]
        A, B:  xy 方向幅值 [m]
        omega: 角速度 [rad/s]
        cx, cy: 交汇点坐标 [m]

    Returns:
        (pos, vel, acc): 位置、速度、加速度向量 (均为 3 维)
    """
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    sin_2t = np.sin(2.0 * theta)
    cos_2t = np.cos(2.0 * theta)

    # 位置
    pos = np.array([
        cx + A * sin_t,
        cy + B * sin_2t,
        0.0  # z 在外部处理
    ])

    # 速度 (dx/dt = A·ω·cos(ωt), dy/dt = 2B·ω·cos(2ωt))
    vel = np.array([
        A * omega * cos_t,
        2.0 * B * omega * cos_2t,
        0.0
    ])

    # 加速度 (dvx/dt = -A·ω²·sin(ωt), dvy/dt = -4B·ω²·sin(2ωt))
    acc = np.array([
        -A * omega ** 2 * sin_t,
        -4.0 * B * omega ** 2 * sin_2t,
        0.0
    ])

    return pos, vel, acc


# ================================================================
#  主类
# ================================================================

class GlobalTrajectory:
    """
    全局轨迹参考生成器

    生成包含起飞→悬停→8字轨迹(可变系数过渡)→悬停→降落的完整轨迹，
    段间 C² 连续。提供滚动时域截取接口供 NMPC 使用。

    Usage:
        traj = GlobalTrajectory(
            takeoff_height=0.5,
            takeoff_time=3.0,
            hover_time_initial=2.0,
            figure8_A=1.5,
            figure8_B=0.8,
            figure8_omega=0.4,
            figure8_duration=16.0,
            transition_time=1.5,
            hover_time_final=1.0,
            landing_time=3.0,
            center_pos=np.array([0.0, 0.0])
        )
        traj.build_global_trajectory()

        # 查询单个时刻
        state = traj.get_reference(5.0)

        # 获取 NMPC 滚动时域参考
        x_ref = traj.get_reference_sequence(x_current, Ts=0.015, Np=10)
    """

    def __init__(self,
                 takeoff_height: float = 0.5,
                 takeoff_time: float = 3.0,
                 hover_time_initial: float = 2.0,
                 figure8_A: float = 1.5,
                 figure8_B: float = 0.8,
                 figure8_omega: float = 0.4,
                 figure8_duration: float = 16.0,
                 transition_time: float = 1.5,
                 hover_time_final: float = 1.0,
                 landing_time: float = 3.0,
                 center_pos: Optional[np.ndarray] = None):
        """
        Args:
            takeoff_height:     起飞目标高度 [m]（默认 0.5）
            takeoff_time:       起飞持续时间 [s]（默认 3.0）
            hover_time_initial: 起飞后悬停时间 [s]（默认 2.0）
            figure8_A:          8字 x 方向幅值 [m]（默认 1.5）
            figure8_B:          8字 y 方向幅值 [m]（默认 0.8）
            figure8_omega:      8字角速度 [rad/s]（默认 0.4）
            figure8_duration:   全幅8字飞行时间 [s]（默认 16.0）
            transition_time:    8字进入/退出过渡时间 [s]（默认 1.5）
            hover_time_final:   降落后悬停时间 [s]（默认 1.0）
            landing_time:       降落持续时间 [s]（默认 3.0）
            center_pos:         8字交汇点 / 悬停点坐标 [x, y]
                                默认取 (0, 0)
        """
        # ===== 用户参数 =====
        self.H = max(takeoff_height, 0.05)
        self.T_takeoff = max(takeoff_time, 0.5)
        self.T_hover1 = max(hover_time_initial, 0.0)
        self.f8_A = max(figure8_A, 0.1)
        self.f8_B = max(figure8_B, 0.1)
        self.f8_omega = max(abs(figure8_omega), 0.01)
        self.f8_omega = figure8_omega  # 保持符号，允许负数
        self.T_f8_desired = max(figure8_duration, 0.1)  # 用户期望值
        self.T_f8 = self.T_f8_desired                     # 实际值（在 build 中自动修正）
        self.T_trans = max(transition_time, 0.1)
        self.T_hover2 = max(hover_time_final, 0.0)
        self.T_land = max(landing_time, 0.5)

        if center_pos is not None:
            self.cx = float(center_pos[0])
            self.cy = float(center_pos[1])
        else:
            self.cx = 0.0
            self.cy = 0.0

        # ===== NMPC 参数（用于截断/悬停保持） =====
        self.nmpc_params = NMPCParams()
        self._Ts = self.nmpc_params.Ts
        self._Np = self.nmpc_params.Np

        # ===== 预计算系数 =====
        self._built = False

        # 起飞/降落五次多项式系数
        self._takeoff_coeffs = None   # z: 0 → H
        self._landing_coeffs = None   # z: H → 0

        # 8字进入/退出 ramp 系数
        self._ramp_in_coeffs = None   # 0 → 1
        self._ramp_out_coeffs = None  # 1 → 0

        # 段切换时间
        self.T0 = 0.0                 # 轨迹起点
        self.T1 = 0.0                 # 起飞结束
        self.T2 = 0.0                 # 悬停1结束
        self.T3 = 0.0                 # 8字进入过渡结束
        self.T4 = 0.0                 # 全幅8字结束
        self.T5 = 0.0                 # 8字退出过渡结束
        self.T6 = 0.0                 # 悬停2结束
        self.T7 = 0.0                 # 降落结束 = 总时长

        # ===== 轨迹后段前馈参考位置 =====
        # 降落时 xy 位置保持点（8字完成后可能不在中心→拉回中心点）
        self._landing_xy = np.zeros(2)

    # ---------------------------------------------------------------
    #  构建全局轨迹
    # ---------------------------------------------------------------

    def build_global_trajectory(self) -> 'GlobalTrajectory':
        """
        预计算所有段的多项式系数和时间节点。

        自动修正 T_f8，使全幅8字段结束时的总相位恰为 2π 的整数倍，
        确保退出过渡从中心点 (cx, cy) 开始，消除不合理轨迹重叠。

        调用此方法后才能使用 get_reference() 和 get_reference_sequence()。
        支持链式调用。

        Timeline 结构:
            0          T1        T2     T3          T4        T5      T6       T7
            |-- 起飞 --|- 悬停1 -|- 进入 -|- 全幅8字 --|- 退出 -|- 悬停2 -|- 降落 -|
        """
        # ---- 0. 自动修正 T_f8：使全幅段结束相位 = n·2π（退出过渡始于中心） ----
        # 进入过渡结束时相位 = ω·T_trans
        # 全幅段结束时相位 = ω·T_trans + ω·T_f8 = n·2π
        # → T_f8 = n·2π/ω - T_trans
        entry_phase = self.f8_omega * self.T_trans
        full_phase_desired = self.f8_omega * self.T_f8_desired
        total_phase_desired = entry_phase + full_phase_desired
        n_cycles = max(1, round(total_phase_desired / (2.0 * np.pi)))
        self.T_f8 = max((n_cycles * 2.0 * np.pi / self.f8_omega) - self.T_trans, 0.1)

        # ---- 1. 解算五次多项式系数 ----
        self._takeoff_coeffs = _quintic_coeffs(0.0, self.H, self.T_takeoff)
        self._landing_coeffs = _quintic_coeffs(self.H, 0.0, self.T_land)
        self._ramp_in_coeffs = _quintic_ramp_coeffs(self.T_trans)
        self._ramp_out_coeffs = _quintic_ramp_coeffs(self.T_trans)

        # ---- 2. 计算段切换时间 ----
        self.T1 = self.T_takeoff
        self.T2 = self.T1 + self.T_hover1
        self.T3 = self.T2 + self.T_trans
        self.T4 = self.T3 + self.T_f8
        self.T5 = self.T4 + self.T_trans
        self.T6 = self.T5 + self.T_hover2
        self.T7 = self.T6 + self.T_land

        self._total_duration = self.T7
        self._built = True

        # 打印修正信息（仅在 T_f8 被显著调整时）
        _diff = abs(self.T_f8 - self.T_f8_desired)
        if _diff > 0.1:
            _phase_end = self.f8_omega * (self.T_trans + self.T_f8)
            _n = _phase_end / (2.0 * np.pi)
            print(f"[GlobalTrajectory] T_f8 自动修正: {self.T_f8_desired:.2f}s → {self.T_f8:.2f}s "
                  f"(全幅段结束相位={_phase_end:.3f}rad = {_n:.2f}×2π)")

        return self

    @property
    def total_duration(self) -> float:
        """全局轨迹总时长 [s]"""
        if not self._built:
            raise RuntimeError("请先调用 build_global_trajectory()")
        return self._total_duration

    @property
    def segment_times(self) -> dict:
        """各段切换时间（用于调试/可视化）"""
        return {
            'takeoff_end': self.T1,
            'hover1_end': self.T2,
            'f8_entry_end': self.T3,
            'f8_full_end': self.T4,
            'f8_exit_end': self.T5,
            'hover2_end': self.T6,
            'landing_end': self.T7,
        }

    # ---------------------------------------------------------------
    #  核心查询接口
    # ---------------------------------------------------------------

    def get_reference(self, t: float) -> TrajectoryState:
        """
        获取全局轨迹在时刻 t 的参考状态

        Args:
            t: 当前时间 [s]（0 ~ total_duration）

        Returns:
            TrajectoryState: 包含 position, velocity, acceleration
        """
        if not self._built:
            raise RuntimeError("请先调用 build_global_trajectory()")

        # ---- 截断处理：超出总时长 → 保持地面悬停 ----
        if t >= self.T7:
            return TrajectoryState(
                position=np.array([self.cx, self.cy, 0.0]),
                velocity=np.zeros(3),
                acceleration=np.zeros(3)
            )

        # ---- 分段求值 ----
        if t < self.T1:
            # Segment 0: 起飞段
            return self._eval_takeoff(t)
        elif t < self.T2:
            # Segment 1: 悬停段 1
            return self._eval_hover(t, self.T1, self.T2,
                                    np.array([self.cx, self.cy, self.H]))
        elif t < self.T3:
            # Segment 2: 8字进入过渡
            return self._eval_figure8_transition(
                t, self.T2, self.T3, is_entry=True)
        elif t < self.T4:
            # Segment 3: 全幅8字
            return self._eval_figure8_full(t, self.T3, self.T4)
        elif t < self.T5:
            # Segment 4: 8字退出过渡
            return self._eval_figure8_transition(
                t, self.T4, self.T5, is_entry=False)
        elif t < self.T6:
            # Segment 5: 悬停段 2
            # 降落前将无人机拉回中心点 (cx, cy)
            return self._eval_hover(t, self.T5, self.T6,
                                    np.array([self.cx, self.cy, self.H]))
        else:
            # Segment 6: 降落段
            return self._eval_landing(t)

    def get_reference_12d(self, t: float, x_current: Optional[np.ndarray] = None
                          ) -> np.ndarray:
        """
        获取时刻 t 的 12 维参考状态

        返回: [x,y,z, vx,vy,vz, roll,pitch,yaw, wx,wy,wz]
        - roll/pitch 假设为 0（平飞）
        - yaw 从速度方向计算（指向航向）
        - 角速度从偏航变化率推算

        Args:
            t: 当前时间 [s]
            x_current: 当前实际状态（可选，用于偏航解缠绕）

        Returns:
            np.ndarray, shape (12,)
        """
        state = self.get_reference(t)

        # 计算偏航角（指向速度方向）
        v_xy = state.velocity[0:2]
        v_norm = np.linalg.norm(v_xy)
        if v_norm > 0.01:
            yaw = np.arctan2(v_xy[1], v_xy[0])
        else:
            yaw = 0.0

        # 偏航角速度：从加速度推算（简化）
        # yaw = atan2(vy, vx) → yaw_dot = (vx*ay - vy*ax) / (vx²+vy²)
        if v_norm > 0.01:
            vx, vy = v_xy
            ax, ay = state.acceleration[0:2]
            yaw_rate = (vx * ay - vy * ax) / (v_norm ** 2)
        else:
            yaw_rate = 0.0

        ref = np.zeros(12)
        ref[0:3] = state.position
        ref[3:6] = state.velocity
        ref[6] = 0.0        # roll
        ref[7] = 0.0        # pitch
        ref[8] = yaw
        ref[9:12] = 0.0     # p, q
        ref[11] = yaw_rate  # r

        return ref

    def get_reference_sequence(self, x_current: np.ndarray,
                               Ts: Optional[float] = None,
                               Np: Optional[int] = None) -> np.ndarray:
        """
        获取 NMPC 滚动时域参考序列（核心接口）

        从当前时间开始，以 Ts 为步长，生成未来 Np 步的参考轨迹片段。

        Args:
            x_current: 当前 12 维（或 18 维增广）状态向量
                       state[0:3] 用于确定当前时间位置以对齐轨迹
                       state[8]   用于偏航解缠绕
            Ts:        预测步长 [s]（默认使用 nmpc_config 中的值）
            Np:        预测步数（默认使用 nmpc_config 中的值）

        Returns:
            x_ref: np.ndarray, shape [12, Np+1]
                   每一列为一个时刻的 12 维参考状态:
                     0-2:  位置 (x, y, z)
                     3-5:  线速度 (vx, vy, vz)
                     6-8:  欧拉角 (roll, pitch, yaw)
                     9-11: 角速度 (wx, wy, wz)
        """
        if not self._built:
            raise RuntimeError("请先调用 build_global_trajectory()")

        _Ts = Ts if Ts is not None else self._Ts
        _Np = Np if Np is not None else self._Np

        # ---- 从当前位置推算全局时间 ----
        # 方法：找到轨迹上最接近当前位置的时刻
        # 先验知识：当前在大地盘坐标中，通过位置反推时间
        t_now = self._estimate_time_from_position(x_current[0:3])

        # ---- 生成参考序列 ----
        x_ref = np.zeros((12, _Np + 1))

        for i in range(_Np + 1):
            t_query = t_now + i * _Ts
            ref_12d = self.get_reference_12d(t_query, x_current)
            x_ref[:, i] = ref_12d

        # ---- 偏航角解缠绕 ----
        yaw_current = x_current[8]
        x_ref[8, :] = self._unwrap_yaw_series(x_ref[8, :], yaw_current)

        # ---- 重新计算角速度（与解缠绕偏航一致） ----
        for i in range(1, _Np + 1):
            dyaw = x_ref[8, i] - x_ref[8, i - 1]
            # 处理 ±π 跳变
            dyaw = (dyaw + np.pi) % (2.0 * np.pi) - np.pi
            x_ref[11, i] = dyaw / _Ts

        return x_ref

    # ---------------------------------------------------------------
    #  内部：各段求值
    # ---------------------------------------------------------------

    def _eval_takeoff(self, t: float) -> TrajectoryState:
        """起飞段求值"""
        z, vz, az = _eval_quintic(t, self._takeoff_coeffs, self.T_takeoff)

        return TrajectoryState(
            position=np.array([self.cx, self.cy, z]),
            velocity=np.array([0.0, 0.0, vz]),
            acceleration=np.array([0.0, 0.0, az])
        )

    def _eval_hover(self, t: float, t_start: float, t_end: float,
                    pos: np.ndarray) -> TrajectoryState:
        """悬停段求值（位置恒定，速度/加速度为零）"""
        return TrajectoryState(
            position=pos.copy(),
            velocity=np.zeros(3),
            acceleration=np.zeros(3)
        )

    def _eval_figure8_full(self, t: float,
                           t_start: float, t_end: float) -> TrajectoryState:
        """全幅8字轨迹段求值"""
        # theta_offset = 进入过渡段结束时的累积相位（保证段间连续）
        theta_offset = self.f8_omega * self.T_trans
        theta = theta_offset + self.f8_omega * (t - t_start)
        pos_rel, vel_rel, acc_rel = _figure8_state(
            theta, self.f8_A, self.f8_B,
            self.f8_omega, self.cx, self.cy)

        return TrajectoryState(
            position=np.array([pos_rel[0], pos_rel[1], self.H]),
            velocity=np.array([vel_rel[0], vel_rel[1], 0.0]),
            acceleration=np.array([acc_rel[0], acc_rel[1], 0.0])
        )

    def _eval_figure8_transition(self, t: float,
                                  t_start: float, t_end: float,
                                  is_entry: bool) -> TrajectoryState:
        """
        8字轨迹过渡段求值（进入或退出）

        使用五次多项式 ramp 缩放 8 字幅值，确保 C² 连续:
            p(t) = s(t) * f8(t)
            v(t) = s'(t) * f8(t) + s(t) * f8'(t)
            a(t) = s''(t)*f8(t) + 2*s'(t)*f8'(t) + s(t)*f8''(t)

        Args:
            t:        当前时间
            t_start:  过渡段起始时间
            t_end:    过渡段结束时间
            is_entry: True=进入(0→1), False=退出(1→0)
        """
        t_local = t - t_start
        T_trans = t_end - t_start

        coeffs = self._ramp_in_coeffs if is_entry else self._ramp_out_coeffs
        s, s_dot, s_ddot = _eval_ramp(t_local, coeffs, T_trans)

        # 退出时 ramp 从 1 → 0
        if not is_entry:
            s = 1.0 - s
            s_dot = -s_dot
            s_ddot = -s_ddot

        # 8 字在过渡段起始处的相位
        theta_offset = self.f8_omega * (t_start - self.T2)
        theta = theta_offset + self.f8_omega * t_local

        pos_rel, vel_rel, acc_rel = _figure8_state(
            theta, self.f8_A, self.f8_B,
            self.f8_omega, 0.0, 0.0)  # 相对位置（不加 cx, cy）

        # ---- 缩放后的状态 ----
        # 位置: p = s * p_f8 + [cx, cy, H]
        pos = np.array([
            s * pos_rel[0] + self.cx,
            s * pos_rel[1] + self.cy,
            self.H
        ])

        # 速度: v = s' * p_f8 + s * v_f8
        vel = np.array([
            s_dot * pos_rel[0] + s * vel_rel[0],
            s_dot * pos_rel[1] + s * vel_rel[1],
            0.0
        ])

        # 加速度: a = s''*p_f8 + 2*s'*v_f8 + s*a_f8
        acc = np.array([
            s_ddot * pos_rel[0] + 2.0 * s_dot * vel_rel[0] + s * acc_rel[0],
            s_ddot * pos_rel[1] + 2.0 * s_dot * vel_rel[1] + s * acc_rel[1],
            0.0
        ])

        return TrajectoryState(position=pos, velocity=vel, acceleration=acc)

    def _eval_landing(self, t: float) -> TrajectoryState:
        """降落段求值"""
        t_local = t - self.T6
        z, vz, az = _eval_quintic(t_local, self._landing_coeffs, self.T_land)

        return TrajectoryState(
            position=np.array([self.cx, self.cy, z]),
            velocity=np.array([0.0, 0.0, vz]),
            acceleration=np.array([0.0, 0.0, az])
        )

    # ---------------------------------------------------------------
    #  辅助方法
    # ---------------------------------------------------------------

    @staticmethod
    def normalize_angle_np(angle: float) -> float:
        """归一化角度到 [-π, π]"""
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    @staticmethod
    def _unwrap_yaw_series(yaw_seq: np.ndarray,
                           yaw_current: float) -> np.ndarray:
        """将偏航角序列解缠绕，使其相对 yaw_current 连续"""
        yaw_out = np.zeros_like(yaw_seq)
        yaw_out[0] = yaw_current + GlobalTrajectory.normalize_angle_np(
            yaw_seq[0] - yaw_current)
        for i in range(1, len(yaw_seq)):
            yaw_out[i] = yaw_out[i-1] + GlobalTrajectory.normalize_angle_np(
                yaw_seq[i] - yaw_out[i-1])
        return yaw_out

    def _estimate_time_from_position(self, pos: np.ndarray) -> float:
        """
        根据当前位置估算全局轨迹时间

        用于在 get_reference_sequence 中定位当前时间位置。
        采用暴力搜索方法，在离散时间点上找最近位置。
        若轨迹已完成，返回总时长。

        未来改进: 可替换为基于弧长的索引查找表。
        """
        if not self._built:
            return 0.0

        # 采样分辨率（与 NMPC 步长一致）
        dt_search = self._Ts
        t_max = self.T7
        n_samples = int(t_max / dt_search) + 1

        # 只需要搜索当前高度对应的段
        z = pos[2]

        # 确定搜索范围
        if z <= 0.02:
            # 接近地面 → 可能在起飞前或降落后
            return 0.0 if self.T7 > 10.0 else self.T7

        # 粗搜索：在可能的段内搜索
        # 先确定大致段范围
        search_start = 0.0
        search_end = t_max

        if z < self.H * 0.3:
            # 低高度 → 起飞段或降落段
            search_start = 0.0
            search_end = self.T1 + self._Ts * 10
            # 也可能在降落段
            if self.T6 > 0:
                search_end = max(search_end, self.T7)
        elif z > self.H * 0.7:
            # 接近目标高度 → 悬停或8字段
            search_start = max(0.0, self.T1 - self._Ts * 20)
            search_end = min(t_max, self.T6 + self._Ts * 20)

        # 暴力搜索最小误差
        min_dist = float('inf')
        best_t = 0.0

        n_search = max(1, int((search_end - search_start) / dt_search))
        for i in range(n_search):
            t_candidate = search_start + i * dt_search
            if t_candidate > t_max:
                break
            ref = self.get_reference(t_candidate)
            dist = np.linalg.norm(ref.position - pos)
            if dist < min_dist:
                min_dist = dist
                best_t = t_candidate

        return best_t


# ================================================================
#  自测与可视化
# ================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (启用 projection='3d')

    # ---- 中文字体配置（防止方格乱码） ----
    plt.rcParams['font.sans-serif'] = [
        'WenQuanYi Micro Hei', 'Noto Sans CJK SC',
        'Droid Sans Fallback', 'SimHei',
        'DejaVu Sans'
    ]
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

    print("===== 全局轨迹生成器自测 =====")

    # ---- 创建轨迹 ----
    traj = GlobalTrajectory(
        takeoff_height=0.5,
        takeoff_time=3.0,
        hover_time_initial=2.0,
        figure8_A=1.5,
        figure8_B=0.8,
        figure8_omega=0.4,
        figure8_duration=16.0,
        transition_time=1.5,
        hover_time_final=1.0,
        landing_time=3.0,
        center_pos=np.array([0.0, 0.0])
    )
    traj.build_global_trajectory()

    print(f"总时长: {traj.total_duration:.2f} s")
    print(f"段切换时间: {traj.segment_times}")

    # ---- 离散采样 ----
    dt_sample = 0.01
    t_samples = np.arange(0, traj.total_duration, dt_sample)
    n_samples = len(t_samples)

    pos_history = np.zeros((n_samples, 3))
    vel_history = np.zeros((n_samples, 3))
    acc_history = np.zeros((n_samples, 3))

    for i, t in enumerate(t_samples):
        state = traj.get_reference(t)
        pos_history[i] = state.position
        vel_history[i] = state.velocity
        acc_history[i] = state.acceleration

    # ---- 验证C²连续性检查 ----
    # 检查各段切换点处位置/速度/加速度是否连续
    switch_times = [traj.T1, traj.T2, traj.T3, traj.T4, traj.T5, traj.T6]
    print("\n----- C² 连续性检查 -----")
    for st in switch_times:
        eps = 1e-6
        state_m = traj.get_reference(st - eps)
        state_p = traj.get_reference(st + eps)

        pos_err = np.linalg.norm(state_m.position - state_p.position)
        vel_err = np.linalg.norm(state_m.velocity - state_p.velocity)
        acc_err = np.linalg.norm(state_m.acceleration - state_p.acceleration)

        print(f"t={st:.3f}s:  pos_err={pos_err:.2e}  "
              f"vel_err={vel_err:.2e}  acc_err={acc_err:.2e}")

    # ---- 打印轨迹统计 ----
    print(f"\n----- 轨迹统计 -----")
    print(f"Z 范围: [{pos_history[:, 2].min():.3f}, {pos_history[:, 2].max():.3f}] m")
    print(f"XY 范围: x=[{pos_history[:, 0].min():.3f}, {pos_history[:, 0].max():.3f}], "
          f"y=[{pos_history[:, 1].min():.3f}, {pos_history[:, 1].max():.3f}]")
    print(f"最大速度: {np.max(np.linalg.norm(vel_history, axis=1)):.3f} m/s")
    print(f"最大加速度: {np.max(np.linalg.norm(acc_history, axis=1)):.3f} m/s²")

    # ---- 可视化 ----
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle('全局轨迹参考生成器 — 自测结果', fontsize=14)

    # 3D 轨迹
    ax = axes[0, 0]
    ax.plot(pos_history[:, 0], pos_history[:, 1], 'b-', linewidth=1.0)
    ax.scatter(pos_history[0, 0], pos_history[0, 1],
               c='g', s=80, marker='o', label='起点', zorder=5)
    ax.scatter(pos_history[-1, 0], pos_history[-1, 1],
               c='r', s=80, marker='x', label='终点', zorder=5)
    ax.scatter(0, 0, c='orange', s=60, marker='+', label='交汇点', zorder=5)
    # 标记段切换点
    for st in switch_times:
        idx = int(st / dt_sample)
        if idx < n_samples:
            ax.scatter(pos_history[idx, 0], pos_history[idx, 1],
                       c='purple', s=30, marker='.', alpha=0.6)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('XY 平面轨迹')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.legend(fontsize=8)

    # Z 轴剖面
    ax = axes[0, 1]
    ax.plot(t_samples, pos_history[:, 2], 'b-', linewidth=1.5)
    ax.axhline(traj.H, color='gray', linestyle='--', alpha=0.5, label=f'H={traj.H}m')
    for st in switch_times:
        ax.axvline(st, color='purple', linestyle=':', alpha=0.4)
    ax.set_xlabel('t [s]')
    ax.set_ylabel('z [m]')
    ax.set_title('Z 轴剖面 (起飞→悬停→降落)')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Z 轴速度
    ax = axes[0, 2]
    ax.plot(t_samples, vel_history[:, 2], 'g-', linewidth=1.5)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
    for st in switch_times:
        ax.axvline(st, color='purple', linestyle=':', alpha=0.4)
    ax.set_xlabel('t [s]')
    ax.set_ylabel('vz [m/s]')
    ax.set_title('Z 轴速度')
    ax.grid(True, alpha=0.3)

    # XY 速度
    ax = axes[1, 0]
    ax.plot(t_samples, np.linalg.norm(vel_history[:, 0:2], axis=1),
            'orange', linewidth=1.5)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
    for st in switch_times:
        ax.axvline(st, color='purple', linestyle=':', alpha=0.4)
    ax.set_xlabel('t [s]')
    ax.set_ylabel('|v_xy| [m/s]')
    ax.set_title('水平面速度模值')
    ax.grid(True, alpha=0.3)

    # 加速度模值
    ax = axes[1, 1]
    ax.plot(t_samples, np.linalg.norm(acc_history, axis=1),
            'r-', linewidth=1.5)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
    for st in switch_times:
        ax.axvline(st, color='purple', linestyle=':', alpha=0.4)
    ax.set_xlabel('t [s]')
    ax.set_ylabel('|a| [m/s²]')
    ax.set_title('加速度模值')
    ax.grid(True, alpha=0.3)

    # 8字轨迹3D视图
    ax = axes[1, 2]
    ax = fig.add_subplot(2, 3, 6, projection='3d')
    ax.plot(pos_history[:, 0], pos_history[:, 1], pos_history[:, 2],
            'b-', linewidth=1.0, alpha=0.8)
    ax.scatter(pos_history[0, 0], pos_history[0, 1], pos_history[0, 2],
               c='g', s=60, marker='o', label='起点')
    ax.scatter(pos_history[-1, 0], pos_history[-1, 1], pos_history[-1, 2],
               c='r', s=60, marker='x', label='终点')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.set_title('3D 全局轨迹')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.show()

    print("\n===== 自测完成 =====")
