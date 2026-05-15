#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import casadi as ca
from uav_config import UAVParams
from nmpc_config import NMPCParams

class Dynamics_Model:
    def __init__(self):
        self.uav_params = UAVParams()
        self.nmpc_params = NMPCParams()
        self.m = self.uav_params.m
        self.g = self.uav_params.g
        self.I = self.uav_params.I
        self.inv_I = ca.diagcat(1/self.uav_params.I[0,0],
                                1/self.uav_params.I[1,1],
                                1/self.uav_params.I[2,2])

    def get_R_IB(self, phi, theta, psi):
        # 机体系→惯性系旋转矩阵
        R_IB = ca.SX.zeros(3, 3)
        R_IB[0,0] = ca.cos(psi)*ca.cos(theta)
        R_IB[0,1] = ca.cos(psi)*ca.sin(theta)*ca.sin(phi) - ca.sin(psi)*ca.cos(phi)
        R_IB[0,2] = ca.cos(psi)*ca.sin(theta)*ca.cos(phi) + ca.sin(psi)*ca.sin(phi)
        R_IB[1,0] = ca.sin(psi)*ca.cos(theta)
        R_IB[1,1] = ca.sin(psi)*ca.sin(theta)*ca.sin(phi) + ca.cos(psi)*ca.cos(phi)
        R_IB[1,2] = ca.sin(psi)*ca.sin(theta)*ca.cos(phi) - ca.cos(psi)*ca.sin(phi)
        R_IB[2,0] = -ca.sin(theta)
        R_IB[2,1] = ca.cos(theta)*ca.sin(phi)
        R_IB[2,2] = ca.cos(theta)*ca.cos(phi)
        return R_IB
    def get_T(self, phi, theta):
        # 欧拉角速率矩阵
        T = ca.SX.zeros(3,3)
        T[0,0] = 1
        T[0,1] = ca.sin(phi)*ca.tan(theta)
        T[0,2] = ca.cos(phi)*ca.tan(theta)
        T[1,0] = 0
        T[1,1] = ca.cos(phi)
        T[1,2] = -ca.sin(phi)
        T[2,0] = 0
        T[2,1] = ca.sin(phi)/ca.cos(theta)
        T[2,2] = ca.cos(phi)/ca.cos(theta)
        return T
    def normalize_angle_ca(self, angle):
        """CasADi符号变量的角度归一化（替代numpy版本）"""
        return ca.atan2(ca.sin(angle), ca.cos(angle))
        # return ca.fmod(angle + ca.pi, 2 * ca.pi) - ca.pi
    def build_dynamics_model(self):
        """构建无人机积分增广动力学模型"""
        nx = self.nmpc_params.nx
        nu = self.nmpc_params.nu
        # 定义CasADi符号变量
        xa = ca.SX.sym('xa', nx)          # 增广状态 [x_original, x_integral] 18维：[位置、速度、姿态、角速度, 位置误差积分, 姿态误差积分]
        u = ca.SX.sym('u', nu)          # 控制量 [Fx,Fy,Fz,τx,τy,τz]
        xa_ref = ca.SX.sym('xa_ref', nx)  # 增广参考轨迹
        u_prev = ca.SX.sym('u_prev', nu)  # 前一时刻控制量（用于控制率惩罚）
        xa_dot = ca.SX.sym('xa_dot', nx)  # 增广状态导数

        pos = xa[0:3]    # x,y,z
        vel = xa[3:6]    # vx,vy,vz
        euler = xa[6:9]  # roll,pitch,yaw
        omega = xa[9:12] # p,q,r
        x_integral = xa[12:18] # 位置误差积分、姿态误差积分

        F_B = u[0:3]      # 机体系推力
        tau_B = u[3:6]    # 机体系力矩

        phi, theta, psi = euler[0], euler[1], euler[2]
        p, q, r = omega[0], omega[1], omega[2]

        pos_ref = xa_ref[0:3]    # 参考位置
        vel_ref = xa_ref[3:6]    # 参考速度
        euler_ref = xa_ref[6:9]  # 参考姿态
        omega_ref = xa_ref[9:12]  # 参考角速度

        R_IB = self.get_R_IB(phi, theta, psi)
        T = self.get_T(phi, theta)
        g_I = ca.SX([0, 0, -self.g])

        # 核心动力学
        dp_dt = vel       # 位置导数
        dv_dt = ca.mtimes(R_IB, F_B) / self.m + g_I  # 速度导数
        deuler_dt = ca.mtimes(T, omega)                    # 姿态导数
        cross_term = ca.cross(omega, ca.mtimes(self.I, omega))
        domega_dt = ca.mtimes(self.inv_I, -cross_term + tau_B)  # 角速度导数

        x_original_dot = ca.vertcat(dp_dt, dv_dt, deuler_dt, domega_dt)

        # 积分项动力学
        
        pos_err = pos - pos_ref  # 位置误差（真实值 - 参考值）
        vel_err = vel - vel_ref  # 速度误差（真实值 - 参考值）
        roll_err = self.normalize_angle_ca(euler[0] - euler_ref[0])
        pitch_err = self.normalize_angle_ca(euler[1] - euler_ref[1])
        yaw_err = self.normalize_angle_ca(euler[2] - euler_ref[2])
        euler_err = ca.vertcat(roll_err, pitch_err, yaw_err)  # 姿态误差
        omega_err = omega_ref - omega  # 角速度误差


        x_integral_dot = ca.vertcat(pos_err[0], pos_err[1], pos_err[2], euler_err[0], euler_err[1], euler_err[2])
        # --------------------- 4.3 增广状态总导数 ---------------------
        f_expl = ca.vertcat(x_original_dot, x_integral_dot) # 显式动力学表达式
        f_impl = xa_dot - f_expl  # 隐式动力学表达式（f_impl = xa_dot - f_expl）

        cost_expr = ca.vertcat(pos, vel, euler, omega, x_integral, u, u) # 代价表达式：位置误差、速度误差、姿态误差、角速度误差、积分状态、控制量偏离配平、控制率
        cost_expr_e = ca.vertcat(pos, vel, euler, omega, x_integral)  # 终端代价表达式

        return xa, xa_dot, u, xa_ref, u_prev, f_expl, f_impl, cost_expr, cost_expr_e