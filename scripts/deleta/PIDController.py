import numpy as np
import time
import rospy
from uav_config import UAVParams
from PID_config import PIDParams
from nmpc_config import NMPCParams

class PIDController:
    def __init__(self):
        self.uav = UAVParams()
        self.pid = PIDParams()
        self.nmpc = NMPCParams()

        # 初始化PID状态变量
        self.pos_int = np.zeros(3)    # 位置积分
        self.vel_int = np.zeros(3)    # 速度积分
        self.att_int = np.zeros(3)    # 姿态积分
        self.omega_int = np.zeros(3)   # 角速度积分

        # 历史状态记录（用于微分计算）
        self.last_pos_err = np.zeros(3)
        self.last_vel_err = np.zeros(3)
        self.last_att_err = np.zeros(3)
        self.last_omega_err = np.zeros(3)

        # 控制历史
        self.solve_time_history = []
        self.u_prev = self.pid.u_trim  # 上一帧控制量

    def normalize_angle_np(self, angle):
        """角度归一化（与原NMPC一致）"""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def pid_core(self, err, last_err, integral, kp, ki, kd, dt, int_limit):
        """通用PID计算核心"""
        # 比例项
        p_term = kp * err

        # 积分项
        integral = integral + err * dt
        i_term = ki * integral
        i_term = np.clip(i_term, -int_limit, int_limit)

        # 微分项
        d_term = kd * (err - last_err) / dt if dt > 1e-6 else 0.0

        # 总输出
        output = p_term + i_term + d_term
        return output, integral, err

    def solve(self, x_current, x_ref):
        """
        PID控制计算（替代原NMPC的solve方法）
        :param x_current: 当前12维状态 [pos, vel, euler, omega]
        :param x_ref: 参考轨迹 (12, Np+1)，取第0列作为当前参考
        :return: 控制量u_opt, 成功标志, 计算耗时
        """
        # 计时（与原NMPC一致）
        solve_start = time.perf_counter()

        try:
            # 1. 提取当前状态和参考值
            pos = x_current[0:3]          # 当前位置
            vel = x_current[3:6]          # 当前速度
            att = x_current[6:9]          # 当前姿态（欧拉角）
            omega = x_current[9:12]       # 当前角速度

            ref_pos = x_ref[0:3, -1]       # 参考位置
            ref_vel = x_ref[3:6, -1]       # 参考速度
            ref_att = x_ref[6:9, -1]       # 参考姿态
            ref_omega = x_ref[9:12, -1]    # 参考角速度

            # 角度归一化
            att[2] = self.normalize_angle_np(att[2])
            ref_att[2] = self.normalize_angle_np(ref_att[2])

            # 2. 串级PID计算
            # 2.1 位置环 → 速度指令
            pos_err = ref_pos - pos
            vel_cmd, self.pos_int, self.last_pos_err = self.pid_core(
                pos_err, self.last_pos_err, self.pos_int,
                self.pid.pos_kp, self.pid.pos_ki, self.pid.pos_kd,
                self.pid.dt, self.pid.pos_int_limit
            )
            rospy.loginfo_throttle(0.2, f"pos_err: {np.array2string(pos_err, precision=8, floatmode='fixed', suppress_small=True, max_line_width=1000)}; vel_cmd: {np.array2string(vel_cmd, precision=8, floatmode='fixed', suppress_small=True, max_line_width=1000)}")

            # 2.2 速度环 → 加速度/推力指令（机体系）
            vel_err = (ref_vel + vel_cmd) - vel
            acc_cmd, self.vel_int, self.last_vel_err = self.pid_core(
                vel_err, self.last_vel_err, self.vel_int,
                self.pid.vel_kp, self.pid.vel_ki, self.pid.vel_kd,
                self.pid.dt, self.pid.vel_int_limit
            )

            # 转换为机体系推力 (Fx,Fy)
            phi, theta, psi = att
            cos_phi = np.cos(phi)
            sin_phi = np.sin(phi)
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            cos_psi = np.cos(psi)
            sin_psi = np.sin(psi)
            Fx = self.uav.m * (acc_cmd[0] * cos_theta * cos_psi + acc_cmd[1] * sin_phi * sin_theta * cos_psi - acc_cmd[1] * cos_phi * sin_psi)
            Fy = self.uav.m * (acc_cmd[0] * cos_theta * sin_psi + acc_cmd[1] * sin_phi * sin_theta * sin_psi + acc_cmd[1] * cos_phi * cos_psi)
            Fz = self.uav.m * (acc_cmd[2] + self.uav.g)  # Z轴推力（含重力补偿）

            # 2.3 姿态环 → 角速度指令
            att_err = ref_att - att
            att_err[2] = self.normalize_angle_np(att_err[2])  # 偏航角误差归一化
            omega_cmd, self.att_int, self.last_att_err = self.pid_core(
                att_err, self.last_att_err, self.att_int,
                self.pid.att_kp, self.pid.att_ki, self.pid.att_kd,
                self.pid.dt, self.pid.att_int_limit
            )

            # 2.4 角速度环 → 力矩指令
            omega_err = omega_cmd - omega
            torque, self.omega_int, self.last_omega_err = self.pid_core(
                omega_err, self.last_omega_err, self.omega_int,
                self.pid.omega_kp, self.pid.omega_ki, self.pid.omega_kd,
                self.pid.dt, self.pid.omega_int_limit
            )

            # 组合控制量 [Fx,Fy,Fz,τx,τy,τz]
            u_opt = np.array([Fx, Fy, Fz, torque[0], torque[1], torque[2]])

            # 控制量约束（与原NMPC一致）
            u_opt = np.clip(u_opt, self.nmpc.u_min, self.nmpc.u_max)
            self.u_prev = u_opt

            # 计算耗时
            solve_time = time.perf_counter() - solve_start
            self.solve_time_history.append(solve_time)

            return u_opt, True, solve_time

        except Exception as e:
            solve_time = time.perf_counter() - solve_start
            self.solve_time_history.append(solve_time)
            rospy.logerr(f"⚠️ PID计算失败：{e}，使用悬停配平控制")
            return self.pid.u_trim, False, solve_time