#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
无人机参考轨迹生成类 ReferenceTrajectory

支持四种轨迹类型，滑动窗口生成 [12, Np+1] 参考序列供 NMPC 使用：
    0-2:  位置 (x, y, z)           [m]
    3-5:  线速度 (vx, vy, vz)      [m/s]
    6-8:  欧拉角 (roll, pitch, yaw) [rad]
    9-11: 角速度 (wx, wy, wz)      [rad/s]

关键设计：
  - 每步调用 step(x_current) 生成未来 Np 步的参考轨迹片段
  - 内部维护相位/时间状态，实现滑动生成
  - 偏航角自动解缠绕，避免 ±π 跳变
  - 梯形速度曲线保证直线轨迹起停平滑
"""

from nmpc_config import NMPCParams
import numpy as np
from enum import Enum


class TrajectoryType(Enum):
    """轨迹类型枚举"""
    NONE = 0
    LINE = 1          # 直线轨迹
    LINE_ROUNDTRIP = 2  # 直线往返
    CIRCLE = 3        # 圆形轨迹
    FIGURE8 = 4       # 8字轨迹


class ReferenceTrajectory:
    def __init__(self):
        self.nmpc_params = NMPCParams()
        self.Np = self.nmpc_params.Np
        self.dt = self.nmpc_params.Ts

        # 当前轨迹类型
        self.traj_type = TrajectoryType.NONE

        # ================== 直线/直线往返 共享参数 ==================
        self.line_start = np.zeros(3)      # 起点
        self.line_end = np.zeros(3)        # 终点
        self.line_dir = np.zeros(3)        # 单位方向向量
        self.line_length = 0.0             # 总长度
        self.line_yaw = 0.0                # 期望偏航角 [rad]
        self.line_speed = 1.0              # 巡航速度 [m/s]
        self.line_total_time = 5.0         # 单程总时间 [s]

        # ================== 直线往返 专用参数 ==================
        self.rt_accel_time = 1.0           # 加减速段时间 [s]
        self.rt_dwell_time = 0.5           # 端点停留时间 [s]
        self.rt_oneway_time = 0.0          # 单程时间（含加减速）
        self.rt_half_cycle = 0.0           # 半周期（含停留）

        # ================== 圆形轨迹 参数 ==================
        self.circle_center = np.zeros(3)   # 圆心 [x, y, z]
        self.circle_radius = 1.0           # 半径 [m]
        self.circle_omega = 0.5            # 角速度 [rad/s]（正=逆时针）
        self.circle_phase = 0.0            # 当前相位 [rad]

        # ================== 8字轨迹 参数 ==================
        self.f8_center = np.zeros(2)       # 交汇点 [x, y]
        self.f8_R = 1.0                    # 半径 [m]
        self.f8_z = 1.0                    # 高度 [m]
        self.f8_omega = 0.3                # 角速度参数 [rad/s]
        self.f8_theta = 0.0                # 当前相位 [rad]

        # ================== 通用状态 ==================
        self.traj_time = 0.0               # 轨迹全局运行时间 [s]
        self.roll = 0.0                    # 默认横滚角（平飞）
        self.pitch = 0.0                   # 默认俯仰角（平飞）

        # ================== 任务完成与降落状态 ==================
        self._trajectory_done = True       # 当前轨迹段是否完成（直线达到终点）
        self.pos_arrival_threshold = 0.15   # 位置到达判断阈值 [m]
        self._arrival_time = None          # 首次到达终点附近的时刻（traj_time）
        self.hold_duration = 0.5           # 到达终点后悬停保持时间 [s]
        self._landing_active = False       # 是否正在降落阶段
        self._landing_complete = False     # 降落是否已完成
        self._landing_start_pos = np.zeros(3)  # 降落起始位置 [x, y, z]
        self._landing_speed = 0.3               # 降落下降速度 [m/s]
        self._landing_time = 0.0                # 降落计时器 [s]
        self._landing_duration = 0.0            # 预计降落时长 [s]

        # 限幅
        self.max_v = self.nmpc_params.x_max[3:6]
        self.max_w = self.nmpc_params.x_max[9:12]

    # ================================================================
    #  工具方法
    # ================================================================

    @staticmethod
    def normalize_angle_np(angle):
        """归一化角度到 [-π, π]"""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def _unwrap_yaw_series(yaw_seq, yaw_current):
        """将偏航角序列解缠绕，使其相对 yaw_current 连续无跳变"""
        yaw_out = np.zeros_like(yaw_seq)
        yaw_out[0] = yaw_current + ReferenceTrajectory.normalize_angle_np(yaw_seq[0] - yaw_current)
        for i in range(1, len(yaw_seq)):
            yaw_out[i] = yaw_out[i-1] + ReferenceTrajectory.normalize_angle_np(yaw_seq[i] - yaw_out[i-1])
        return yaw_out

    # ================================================================
    #  任务状态查询与降落控制
    # ================================================================

    def is_trajectory_done(self, current_pos=None):
        """
        当前轨迹段是否已完成

        对 LINE 类型：
          1. 实际位置到达终点附近（距离 < pos_arrival_threshold）
          2. 在终点处保持悬停 hold_duration 秒
        两条件均满足后才判完成。
        不传 current_pos 时降级为时间判断（向后兼容）。

        Args:
            current_pos: [x, y, z] 当前实际位置

        Returns: bool
        """
        # 非 LINE 类型（圆/8字/往返等）用原有 _trajectory_done 标记
        if self.traj_type != TrajectoryType.LINE:
            return self._trajectory_done
        # LINE 类型：距离 + 保持时间
        if current_pos is not None:
            dist = np.linalg.norm(np.asarray(current_pos)[:2] - self.line_end[:2])
            if dist <= self.pos_arrival_threshold:
                # 首次到达 → 记录到达时刻
                if self._arrival_time is None:
                    self._arrival_time = self.traj_time
                # 保持足够时间后才判完成
                if self.traj_time - self._arrival_time >= self.hold_duration:
                    if not self._trajectory_done:
                        self._trajectory_done = True
                    return True
                return False
            else:
                # 距离超出阈值时重置到达计时（防止抖动误触发）
                self._arrival_time = None
                return False
        # 无位置信息时降级为时间判断
        return self.traj_time >= self.line_total_time

    def is_landing_active(self):
        """是否正在降落阶段"""
        return self._landing_active

    def is_landing_complete(self):
        """降落是否已完成（z≈0 且保持足够时间）"""
        return self._landing_complete

    def start_landing(self, current_pos):
        """
        开始自动降落

        无人机从 current_pos 开始，平滑下降至地面 (z=0)。
        降落采用二次曲线速度剖面，触地柔和。

        Args:
            current_pos: [x, y, z] 当前无人机位置

        Returns: self
        """
        self.traj_type = TrajectoryType.NONE  # 降落期间无特定轨迹模式
        self._landing_active = True
        self._landing_complete = False
        self._landing_start_pos = np.asarray(current_pos, dtype=float).copy()
        self._landing_time = 0.0
        z0 = max(self._landing_start_pos[2], 0.05)
        self._landing_duration = z0 / self._landing_speed
        return self

    # ================================================================
    #  初始化接口
    # ================================================================

    def init_line(self, start_pos, end_pos, yaw=0.0, speed=1.0):
        """
        初始化直线轨迹

        Args:
            start_pos: [x, y, z] 起点位置
            end_pos:   [x, y, z] 终点位置
            yaw:       期望偏航角 [rad]，机头指向
            speed:     期望巡航速度 [m/s]

        Returns: self（链式调用）
        """
        self.traj_type = TrajectoryType.LINE
        self.line_start = np.asarray(start_pos, dtype=float).copy()
        self.line_end = np.asarray(end_pos, dtype=float).copy()
        self.line_yaw = float(yaw)
        self.line_speed = max(float(speed), 0.01)

        diff = self.line_end - self.line_start
        self.line_length = np.linalg.norm(diff)
        if self.line_length > 1e-6:
            self.line_dir = diff / self.line_length
        else:
            self.line_dir = np.zeros(3)

        self.line_total_time = self.line_length / self.line_speed
        # 重置任务状态
        self._trajectory_done = False
        self._arrival_time = None
        self._landing_active = False
        self._landing_complete = False
        self.traj_time = 0.0
        return self

    def init_line_roundtrip(self, start_pos, end_pos, yaw=0.0, speed=1.0,
                            accel_time=1.0, dwell_time=0.5):
        """
        初始化直线往返轨迹

        无人机在 start_pos 和 end_pos 之间反复匀速往返，
        两端加减速平滑过渡，端点可设置停留时间。

        Args:
            start_pos:   [x, y, z] 起点
            end_pos:     [x, y, z] 终点
            yaw:         期望偏航角 [rad]
            speed:       巡航速度 [m/s]
            accel_time:  加减速时间 [s]（默认 1.0）
            dwell_time:  端点停留时间 [s]（默认 0.5）

        Returns: self
        """
        self.traj_type = TrajectoryType.LINE_ROUNDTRIP
        self.line_start = np.asarray(start_pos, dtype=float).copy()
        self.line_end = np.asarray(end_pos, dtype=float).copy()
        self.line_yaw = float(yaw)
        self.line_speed = max(float(speed), 0.01)

        diff = self.line_end - self.line_start
        self.line_length = np.linalg.norm(diff)
        if self.line_length > 1e-6:
            self.line_dir = diff / self.line_length
        else:
            self.line_dir = np.zeros(3)

        self.line_total_time = self.line_length / self.line_speed
        self.rt_accel_time = max(float(accel_time), 0.1)
        self.rt_dwell_time = max(float(dwell_time), 0.0)

        # 限制加速段时间不超过单程时间的一半
        max_allowed_accel = self.line_total_time / 2.0
        if self.rt_accel_time > max_allowed_accel:
            self.rt_accel_time = max_allowed_accel

        self.rt_oneway_time = self.line_total_time
        self.rt_half_cycle = self.rt_oneway_time + self.rt_dwell_time
        # 重置任务状态
        self._trajectory_done = False
        self._landing_active = False
        self._landing_complete = False
        self.traj_time = 0.0
        return self

    def init_circle(self, center, radius, omega=0.5, z=1.0, phase_init=0.0):
        """
        初始化圆形轨迹

        Args:
            center:    [x, y] 或 [x, y, z] 圆心坐标
            radius:    半径 [m]
            omega:     角速度 [rad/s]（正=逆时针，负=顺时针）
            z:         飞行高度 [m]（center 只给2维时使用）
            phase_init: 初始相位 [rad]

        Returns: self
        """
        self.traj_type = TrajectoryType.CIRCLE
        center_arr = np.asarray(center, dtype=float)
        if len(center_arr) >= 3:
            self.circle_center = center_arr[:3].copy()
        else:
            self.circle_center = np.array([center_arr[0], center_arr[1], float(z)])
        self.circle_radius = max(float(radius), 0.1)
        self.circle_omega = float(omega)
        self.circle_phase = float(phase_init)
        # 重置任务状态
        self._trajectory_done = False
        self._landing_active = False
        self._landing_complete = False
        self.traj_time = 0.0
        return self

    def init_figure8(self, center, R, z=1.0, omega=0.3):
        """
        初始化8字轨迹（lemniscate of Gerono）

        参数方程（相对交汇点）:
            x = R * cos(θ) / (1 + sin²(θ))
            y = R * sin(θ) * cos(θ) / (1 + sin²(θ))

        Args:
            center: [x, y] 交汇点坐标
            R:      轨迹半径 [m]
            z:      飞行高度 [m]
            omega:  角速度参数 [rad/s]（控制8字飞行速度）

        Returns: self
        """
        self.traj_type = TrajectoryType.FIGURE8
        self.f8_center = np.asarray(center, dtype=float).copy()
        self.f8_R = float(R)
        self.f8_z = float(z)
        self.f8_omega = float(omega)
        self.f8_theta = 0.0
        # 重置任务状态
        self._trajectory_done = False
        self._landing_active = False
        self._landing_complete = False
        self.traj_time = 0.0
        return self

    # ================================================================
    #  核心：滑动窗口生成参考轨迹
    # ================================================================

    def step(self, x_current=None):
        """
        生成当前时刻预测时域内的参考轨迹片段（滑动窗口）

        每个控制周期调用一次，内部状态自动推进一个 dt。

        Args:
            x_current: 当前12维状态 [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]
                       用于偏航角解缠绕。为 None 时不执行解缠绕。

        Returns:
            x_ref: np.ndarray, shape [12, Np+1]
                   预测时域内各时刻的参考状态
        """
        # ========== 1. 降落阶段：生成降落参考轨迹 ==========
        if self._landing_active:
            x_ref = self._generate_landing(x_current)
            # 检查降落是否完成（到达地面后继续保持 0.5s）
            if self._landing_time >= self._landing_duration + 0.5:
                if not self._landing_complete:
                    self._landing_complete = True
            self.traj_time += self.dt
            return x_ref

        # ========== 2. 根据轨迹类型生成参考 ==========
        if self.traj_type == TrajectoryType.LINE:
            x_ref = self._generate_line()
            # 直线轨迹完成由 is_trajectory_done() 基于距离判断，此处不做任何标记
        elif self.traj_type == TrajectoryType.LINE_ROUNDTRIP:
            x_ref = self._generate_line_roundtrip()
        elif self.traj_type == TrajectoryType.CIRCLE:
            x_ref = self._generate_circle()
        elif self.traj_type == TrajectoryType.FIGURE8:
            x_ref = self._generate_figure8()
        else:
            # 默认：悬停（全零速度/角速度）
            x_ref = np.zeros((12, self.Np + 1))
            if x_current is not None:
                for d in range(3):
                    x_ref[d, :] = x_current[d]
                x_ref[8, :] = x_current[8]  # 保持当前偏航

        # ========== 3. 偏航角解缠绕 ==========
        if x_current is not None and self.traj_type != TrajectoryType.NONE:
            yaw_current = x_current[8]
            x_ref[8, :] = self._unwrap_yaw_series(x_ref[8, :], yaw_current)
            # 重新计算角速度（使角速度与解缠绕后的偏航一致）
            for i in range(1, self.Np + 1):
                dyaw = x_ref[8, i] - x_ref[8, i - 1]
                x_ref[11, i] = dyaw / self.dt

        self.traj_time += self.dt
        return x_ref

    # ================================================================
    #  降落参考轨迹生成器
    # ================================================================

    def _generate_landing(self, x_current=None):
        """
        生成降落参考轨迹

        从 _landing_start_pos 开始，平滑下降至地面 (z=0)。
        采用二次曲线速度剖面：z = z0 * (1-s)²，触地柔和。
        xy 位置保持不变，偏航保持当前航向。

        Args:
            x_current: 当前12维状态，用于获取偏航角

        Returns:
            x_ref: np.ndarray, shape [12, Np+1]
        """
        x_ref = np.zeros((12, self.Np + 1))
        t_grid = np.linspace(0, self.Np * self.dt, self.Np + 1)

        z0 = self._landing_start_pos[2]
        yaw_ref = x_current[8] if x_current is not None else 0.0

        for i, dt_i in enumerate(t_grid):
            t_land = self._landing_time + dt_i
            s = np.clip(t_land / self._landing_duration, 0.0, 1.0)  # 归一化进度 [0, 1]

            # ---- 位置：xy 保持降落起始点，z 二次曲线下降 ----
            x_ref[0, i] = self._landing_start_pos[0]
            x_ref[1, i] = self._landing_start_pos[1]
            x_ref[2, i] = z0 * (1.0 - s) ** 2

            # ---- 速度：xy 零速，z 缓慢下降 ----
            x_ref[3, i] = 0.0
            x_ref[4, i] = 0.0
            x_ref[5, i] = -2.0 * z0 * (1.0 - s) / self._landing_duration

            # ---- 姿态：水平，偏航保持 ----
            x_ref[6, i] = 0.0
            x_ref[7, i] = 0.0
            x_ref[8, i] = yaw_ref

            # ---- 角速度：零 ----
            x_ref[9:12, i] = 0.0

        # 推进降落计时
        self._landing_time += self.Np * self.dt

        return x_ref

    # ================================================================
    #  内部轨迹生成器
    # ================================================================

    def _trapezoidal_profile(self, t, total_time, accel_time):
        """
        梯形速度曲线：加速 → 匀速 → 减速

        Args:
            t:           绝对时间 [s]
            total_time:  总运动时间 [s]
            accel_time:  加减速时间 [s]

        Returns:
            s:     归一化位移 [0, 1]
            s_dot: 归一化速度 [1/s]
        """
        if total_time < 1e-8:
            return 0.0, 0.0

        decel_start = total_time - accel_time

        if t <= 0.0:
            return 0.0, 0.0
        elif t >= total_time:
            return 1.0, 0.0
        elif t <= accel_time:
            # 加速段：s = 0.5*(t/t_acc)², s_dot = t/(t_acc*t_total)
            s = 0.5 * (t / accel_time) ** 2
            s_dot = t / (accel_time * total_time)
        elif t <= decel_start:
            # 匀速段
            s_mid = 0.5 * accel_time / total_time
            s = s_mid + (t - accel_time) / total_time
            s_dot = 1.0 / total_time
        else:
            # 减速段
            t_dec = t - decel_start
            s = 1.0 - 0.5 * (1.0 - t_dec / accel_time) ** 2
            s_dot = (1.0 - t_dec / accel_time) / total_time

        return s, s_dot

    def _generate_line(self):
        """生成直线轨迹片段"""
        x_ref = np.zeros((12, self.Np + 1))
        t_grid = np.linspace(0, self.Np * self.dt, self.Np + 1)

        accel_time = min(self.line_total_time / 3.0, 1.0)

        for i, dt_i in enumerate(t_grid):
            t_abs = self.traj_time + dt_i
            s, s_dot = self._trapezoidal_profile(t_abs, self.line_total_time, accel_time)

            # 位置
            x_ref[0:3, i] = self.line_start + self.line_dir * s * self.line_length

            # 速度
            vel = self.line_dir * s_dot * self.line_length
            x_ref[3:6, i] = np.clip(vel, -self.max_v, self.max_v)

            # 欧拉角
            x_ref[6, i] = self.roll
            x_ref[7, i] = self.pitch
            x_ref[8, i] = self.line_yaw

            # 角速度（直线飞行偏航不变）
            x_ref[9:12, i] = 0.0

        return x_ref

    def _generate_line_roundtrip(self):
        """生成直线往返轨迹片段"""
        x_ref = np.zeros((12, self.Np + 1))
        t_grid = np.linspace(0, self.Np * self.dt, self.Np + 1)

        t_total = self.rt_half_cycle * 2.0  # 完整往返周期
        accel_time = self.rt_accel_time
        oneway_time = self.rt_oneway_time
        dwell_time = self.rt_dwell_time

        for i, dt_i in enumerate(t_grid):
            t_abs = self.traj_time + dt_i
            t_mod = t_abs % t_total

            # 判断半周期：正向还是反向
            if t_mod < self.rt_half_cycle:
                direction = 1
                t_local = t_mod
                pos_start = self.line_start
                pos_end = self.line_end
            else:
                direction = -1
                t_local = t_mod - self.rt_half_cycle
                pos_start = self.line_end
                pos_end = self.line_start

            diff = pos_end - pos_start
            length = np.linalg.norm(diff)
            if length > 1e-6:
                dir_vec = diff / length
            else:
                dir_vec = np.zeros(3)

            # 梯形速度曲线
            if t_local <= accel_time:
                # 加速
                s = 0.5 * (t_local / accel_time) ** 2
                s_dot = t_local / (accel_time * oneway_time)
            elif t_local <= oneway_time - accel_time:
                # 匀速
                s = 0.5 * accel_time / oneway_time + (t_local - accel_time) / oneway_time
                s_dot = 1.0 / oneway_time
            elif t_local <= oneway_time:
                # 减速
                t_dec = t_local - (oneway_time - accel_time)
                s = 1.0 - 0.5 * (1.0 - t_dec / accel_time) ** 2
                s_dot = (1.0 - t_dec / accel_time) / oneway_time
            else:
                # 停留
                s = 1.0
                s_dot = 0.0

            # 位置
            x_ref[0:3, i] = pos_start + dir_vec * s * length

            # 速度（方向控制正反向）
            vel = dir_vec * s_dot * length * direction
            x_ref[3:6, i] = np.clip(vel, -self.max_v, self.max_v)

            # 欧拉角
            x_ref[6, i] = self.roll
            x_ref[7, i] = self.pitch
            x_ref[8, i] = self.line_yaw

            # 角速度
            x_ref[9:12, i] = 0.0

        return x_ref

    def _generate_circle(self):
        """生成圆形轨迹片段"""
        x_ref = np.zeros((12, self.Np + 1))
        t_grid = np.linspace(0, self.Np * self.dt, self.Np + 1)

        for i, dt_i in enumerate(t_grid):
            theta = self.circle_phase + self.circle_omega * dt_i

            # ---- 位置 ----
            x = self.circle_center[0] + self.circle_radius * np.cos(theta)
            y = self.circle_center[1] + self.circle_radius * np.sin(theta)
            z = self.circle_center[2]
            x_ref[0:3, i] = [x, y, z]

            # ---- 速度（解析导数） ----
            vx = -self.circle_radius * self.circle_omega * np.sin(theta)
            vy = self.circle_radius * self.circle_omega * np.cos(theta)
            vz = 0.0
            x_ref[3:6, i] = np.clip([vx, vy, vz], -self.max_v, self.max_v)

            # ---- 欧拉角（偏航指向速度切线方向） ----
            x_ref[6, i] = self.roll
            x_ref[7, i] = self.pitch
            x_ref[8, i] = np.arctan2(vy, vx)

            # ---- 角速度 ----
            # 偏航角速度 = 圆周角速度（偏航跟随圆周运动）
            x_ref[9:12, i] = [0.0, 0.0, self.circle_omega]

        # 滑动：更新相位
        self.circle_phase += self.circle_omega * self.Np * self.dt
        self.circle_phase = self.normalize_angle_np(self.circle_phase)

        return x_ref

    def _generate_figure8(self):
        """生成8字轨迹片段（lemniscate of Gerono）"""
        x_ref = np.zeros((12, self.Np + 1))
        t_grid = np.linspace(0, self.Np * self.dt, self.Np + 1)

        # 先计算所有位置和速度，便于后面计算角速度
        pos_all = np.zeros((3, self.Np + 1))
        vel_all = np.zeros((3, self.Np + 1))
        yaw_all = np.zeros(self.Np + 1)

        for i, dt_i in enumerate(t_grid):
            theta = self.f8_theta + self.f8_omega * dt_i

            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            denom = 1.0 + sin_t ** 2

            # ---- 位置 ----
            x = self.f8_center[0] + self.f8_R * cos_t / denom
            y = self.f8_center[1] + self.f8_R * sin_t * cos_t / denom
            z = self.f8_z
            pos_all[:, i] = [x, y, z]

            # ---- 速度（解析导数） ----
            dx_dtheta = (-self.f8_R * sin_t * (1.0 + sin_t ** 2)
                         - 2.0 * self.f8_R * cos_t ** 2 * sin_t) / denom ** 2
            dy_dtheta = (self.f8_R * (cos_t ** 2 - sin_t ** 2 - sin_t ** 4)) / denom ** 2
            vx = dx_dtheta * self.f8_omega
            vy = dy_dtheta * self.f8_omega
            vz = 0.0
            vel_all[:, i] = [vx, vy, vz]

            # ---- 偏航角（指向速度方向） ----
            yaw_all[i] = np.arctan2(vy, vx)

        # 写入位置 + 速度
        x_ref[0:3, :] = pos_all
        x_ref[3:6, :] = np.clip(vel_all, -self.max_v[:, None], self.max_v[:, None])

        # ---- 欧拉角 ----
        x_ref[6, :] = self.roll
        x_ref[7, :] = self.pitch
        x_ref[8, :] = yaw_all

        # ---- 角速度（对偏航有限差分） ----
        x_ref[9:12, :] = 0.0
        for i in range(1, self.Np + 1):
            dyaw = self.normalize_angle_np(yaw_all[i] - yaw_all[i - 1])
            x_ref[11, i] = dyaw / self.dt

        # 滑动：更新相位
        self.f8_theta += self.f8_omega * self.Np * self.dt
        self.f8_theta = self.f8_theta % (2.0 * np.pi)

        return x_ref


# ================================================================
#  简单自测
# ================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    traj = ReferenceTrajectory()
    nmpc = NMPCParams()
    steps = 500

    # -------- 测试：直线轨迹 --------
    print("===== 测试直线轨迹 =====")
    traj.init_line([0, 0, 0.6], [3, 2, 0.6], yaw=np.deg2rad(30), speed=1.0)
    ref_history = []
    for _ in range(steps):
        x_ref = traj.step(np.array([0, 0, 0.6, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        ref_history.append(x_ref[:, 0].copy())
    ref_history = np.array(ref_history)
    print(f"直线轨迹 - 起点: {ref_history[0, 0:3]}, 终点: {ref_history[-1, 0:3]}")
    print(f"直线轨迹 - 最大速度: {np.max(np.linalg.norm(ref_history[:, 3:6], axis=1)):.3f} m/s")

    # -------- 测试：直线往返轨迹 --------
    print("\n===== 测试直线往返轨迹 =====")
    traj.init_line_roundtrip([0, 0, 0.6], [3, 0, 0.6], yaw=0, speed=1.0, accel_time=1.0, dwell_time=0.5)
    ref_history_rt = []
    for _ in range(steps):
        x_ref = traj.step(np.array([0, 0, 0.6, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        ref_history_rt.append(x_ref[:, 0].copy())
    ref_history_rt = np.array(ref_history_rt)
    print(f"往返轨迹 - x范围: [{ref_history_rt[:, 0].min():.2f}, {ref_history_rt[:, 0].max():.2f}]")
    print(f"往返轨迹 - 最大速度: {np.max(np.linalg.norm(ref_history_rt[:, 3:6], axis=1)):.3f} m/s")

    # -------- 测试：圆形轨迹 --------
    print("\n===== 测试圆形轨迹 =====")
    traj.init_circle([0, 0], 2.0, omega=0.5, z=1.0)
    ref_history_c = []
    for _ in range(steps):
        x_ref = traj.step(np.array([2, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        ref_history_c.append(x_ref[:, 0].copy())
    ref_history_c = np.array(ref_history_c)
    print(f"圆形轨迹 - xy半径: {np.mean(np.linalg.norm(ref_history_c[:, 0:2], axis=1)):.3f} m")
    print(f"圆形轨迹 - z均值: {ref_history_c[:, 2].mean():.3f} m")

    # -------- 测试：8字轨迹 --------
    print("\n===== 测试8字轨迹 =====")
    traj.init_figure8([0, 0], R=2.0, z=1.0, omega=0.3)
    ref_history_8 = []
    for _ in range(steps):
        x_ref = traj.step(np.array([0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        ref_history_8.append(x_ref[:, 0].copy())
    ref_history_8 = np.array(ref_history_8)

    # -------- 绘图 --------
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 直线
    ax = axes[0, 0]
    ax.plot(ref_history[:, 0], ref_history[:, 1], 'b-', linewidth=1.5)
    ax.scatter(ref_history[0, 0], ref_history[0, 1], c='g', s=60, marker='o', label='起点')
    ax.scatter(ref_history[-1, 0], ref_history[-1, 1], c='r', s=60, marker='x', label='终点')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('直线轨迹')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.legend()

    # 直线往返
    ax = axes[0, 1]
    ax.plot(ref_history_rt[:, 0], ref_history_rt[:, 1], 'b-', linewidth=1.5)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('直线往返轨迹')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # 圆形
    ax = axes[1, 0]
    ax.plot(ref_history_c[:, 0], ref_history_c[:, 1], 'b-', linewidth=1.5)
    ax.scatter(0, 0, c='r', s=40, marker='+', label='圆心')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('圆形轨迹')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.legend()

    # 8字
    ax = axes[1, 1]
    ax.plot(ref_history_8[:, 0], ref_history_8[:, 1], 'b-', linewidth=1.5)
    ax.scatter(0, 0, c='r', s=40, marker='+', label='交汇点')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('8字轨迹')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.legend()

    plt.suptitle('参考轨迹生成测试', fontsize=14)
    plt.tight_layout()
    plt.show()
