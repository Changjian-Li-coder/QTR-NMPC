# -*- coding: utf-8 -*-
# ====================== 1. 基础库导入 ======================
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import time
from pymavlink import mavutil
import serial
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ====================== 2. 无人机物理参数定义（严格匹配文档） ======================
class UAVParams:
    def __init__(self):
        # 基础质量与惯量
        self.m = 2.5  # 无人机质量，单位kg，可根据实际机型修改
        self.Ixx = 0.05  # x轴转动惯量，单位kg·m²
        self.Iyy = 0.05  # y轴转动惯量，单位kg·m²
        self.Izz = 0.08  # z轴转动惯量，单位kg·m²
        self.I = np.diag([self.Ixx, self.Iyy, self.Izz])  # 惯量矩阵
        self.g = 9.81  # 重力加速度，单位m/s²

        # 螺旋桨与舵机参数（文档乾丰6042三叶桨）
        self.k_f = 0.18  # 推力常数
        self.k_m = 0.0018  # 反扭矩常数
        self.k_torque = self.k_m / self.k_f  # 反扭矩-推力比例系数
        self.L = 0.182  # 机臂长度，单位m，可根据实际机型修改
        self.I_alpha = 0.001  # 旋翼组件转动惯量，单位kg·m²

        # 执行器约束（严格匹配文档）
        self.thrust_max = 2.223 * self.g   # 单电机最大推力，单位N
        # 控制输入u = [T1,T2,T3,T4,α1,α2,α3,α4]  单位：N和弧度
        self.u_min = np.array([0, 0, 0, 0, np.deg2rad(-20), np.deg2rad(-20), np.deg2rad(-20), np.deg2rad(-20)])
        self.u_max = np.array([0.9 * self.thrust_max, 0.9 * self.thrust_max, 0.9 * self.thrust_max, 0.9 * self.thrust_max, np.deg2rad(20), np.deg2rad(20), np.deg2rad(20), np.deg2rad(20)])
        # 控制增量约束
        self.du_min = np.array([-0.4 * self.thrust_max, -0.4 * self.thrust_max, -0.4 * self.thrust_max, -0.4 * self.thrust_max, np.deg2rad(-20), np.deg2rad(-20), np.deg2rad(-20), np.deg2rad(-20)])
        self.du_max = np.array([0.4 * self.thrust_max, 0.4 * self.thrust_max, 0.4 * self.thrust_max, 0.4 * self.thrust_max, np.deg2rad(20), np.deg2rad(20), np.deg2rad(20), np.deg2rad(20)])
        # 状态安全约束
        # 状态向量x = [x,y,z, vx,vy,vz, phi,theta,psi, p,q,r]
        self.x_min = np.array([-100, -100, -100, -20, -20, -20, np.deg2rad(-150), np.deg2rad(-150), np.deg2rad(-180), np.deg2rad(-1200), np.deg2rad(-1200), np.deg2rad(-1200)])
        self.x_max = np.array([100, 100, 100, 20, 20, 20, np.deg2rad(150), np.deg2rad(150), np.deg2rad(180), np.deg2rad(1200), np.deg2rad(1200), np.deg2rad(1200)])

        # 电机转向系数（文档定义：1、2号-1，3、4号1）
        self.sigma = np.array([-1, -1, 1, 1])
        # 电机位置单位矢量（文档定义）
        self.u_i = np.array([
            [1/np.sqrt(2), 1/np.sqrt(2), 0],
            [-1/np.sqrt(2), -1/np.sqrt(2), 0],
            [1/np.sqrt(2), -1/np.sqrt(2), 0],
            [-1/np.sqrt(2), 1/np.sqrt(2), 0]
        ])
        # 初始零倾转推力单位矢量（文档定义）
        self.n0 = np.array([0, 0, -1])

# ====================== 3. NMPC超参数定义 ======================
class NMPCParams:
    def __init__(self):
        self.Ts = 0.05  # 控制周期，单位s（20Hz控制频率，匹配飞控主流频率）
        self.Np = 10  # 预测时域步长
        self.Nc = 5  # 控制时域步长

        # 权重矩阵（调参核心：跟踪性能/控制平顺性平衡）
        # 状态权重Q: [x,y,z, vx,vy,vz, phi,theta,psi, p,q,r]
        self.Q = np.diag([150, 150, 180, 3, 3, 3, 5, 5, 3, 0.1, 0.1, 0.1])
        # 终端状态权重P
        self.P = self.Q * 1.5
        # 控制量权重R（惩罚偏离配平值）
        self.R = np.diag([0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 1.0])
        # 控制增量权重S（惩罚剧烈动作）
        self.S = np.diag([5, 5, 5, 5, 3, 3, 3, 3])

        # 悬停配平控制量（初始基准值，降低优化难度）
        hover_thrust = (2.5 * 9.81 / 4)   #单电机悬停推力
        self.u_trim = np.array([hover_thrust]*4 + [0, 0, 0, 0])
        print(f"悬停配平控制量 u_trim: {self.u_trim.round(4)}")
        print(self.u_trim)

# ====================== 4. 非线性动力学模型构建（匹配文档推导） ======================
def build_dynamics_model(uav_params):
    # 符号定义
    nx = 12  # 状态量维度（文档定义x∈R^12）
    nu = 8   # 控制量维度（文档定义u∈R^8）

    # 状态向量 x = [x,y,z, vx,vy,vz, phi,theta,psi, p,q,r]
    x = ca.SX.sym('x', nx)
    pos = x[0:3]       # 惯性系位置 p
    vel = x[3:6]       # 机体系线速度 v
    euler = x[6:9]     # 欧拉角 η=[phi,theta,psi]
    omega = x[9:12]    # 机体角速度 ω=[p,q,r]

    # 控制输入向量 u = [T1,T2,T3,T4, α1,α2,α3,α4]
    u = ca.SX.sym('u', nu)
    thrust = u[0:4]  # 电机推力 T_i
    alpha = u[4:8]        # 舵机倾角 α_i

    # 1. 机体系→惯性系旋转矩阵R_IB（文档定义）
    phi, theta, psi = euler[0], euler[1], euler[2]
    R_IB = ca.SX.zeros(3, 3)
    R_IB[0, 0] = ca.cos(psi) * ca.cos(theta)
    R_IB[0, 1] = ca.sin(psi) * ca.cos(phi) + ca.cos(psi) * ca.sin(theta) * ca.sin(phi)
    R_IB[0, 2] = ca.sin(psi) * ca.sin(phi) + ca.cos(psi) * ca.sin(theta) * ca.cos(phi)
    R_IB[1, 0] = ca.sin(psi) * ca.cos(theta)
    R_IB[1, 1] = ca.cos(psi) * ca.cos(phi) + ca.sin(psi) * ca.sin(theta) * ca.sin(phi)
    R_IB[1, 2] = -ca.cos(psi) * ca.sin(phi) + ca.sin(psi) * ca.sin(theta) * ca.cos(phi)
    R_IB[2, 0] = -ca.sin(theta)
    R_IB[2, 1] = ca.cos(theta) * ca.sin(phi)
    R_IB[2, 2] = ca.cos(theta) * ca.cos(phi)

    # 2. 欧拉角速率变换矩阵T（修正文档笔误，标准ZYX欧拉角变换）
    T = ca.SX.zeros(3, 3)
    T[0, 0] = 1
    T[0, 1] = ca.sin(phi) * ca.tan(theta)
    T[0, 2] = ca.cos(phi) * ca.tan(theta)
    T[1, 0] = 0
    T[1, 1] = ca.cos(phi)
    T[1, 2] = -ca.sin(phi)
    T[2, 0] = 0
    T[2, 1] = ca.sin(phi) / ca.cos(theta)
    T[2, 2] = ca.cos(phi) / ca.cos(theta)

    # 3. 单电机推力与倾转后推力矢量（文档罗德里格斯旋转公式）
    T_i = thrust   # 单电机推力大小
    n_i = ca.SX.zeros(3, 4)  # 每个电机的推力单位矢量
    for i in range(4):
        u_i = uav_params.u_i[i, :].T
        cross_term = ca.cross(u_i, uav_params.n0)
        n_i[:, i] = uav_params.n0 * ca.cos(alpha[i]) + cross_term * ca.sin(alpha[i])

    # 4. 机体总推力F（文档推导公式）
    F = ca.SX.zeros(3)
    for i in range(4):
        F += T_i[i] * n_i[:, i]

    # 5. 机体总扭矩τ = 推力力矩τ_F + 反扭矩τ_anti（文档推导公式）
    tau_F = ca.SX.zeros(3)
    for i in range(4):
        l_i = uav_params.L * uav_params.u_i[i, :].T
        tau_F += ca.cross(l_i, T_i[i] * n_i[:, i])

    # 旋翼反扭矩计算
    tau_anti = ca.SX.zeros(3)
    for i in range(4):
        anti_vec = -uav_params.sigma[i] * uav_params.k_torque * T_i[i] * n_i[:, i]
        tau_anti += anti_vec
    tau = tau_F + tau_anti

    # 6. 动力学微分方程（文档定义）
    dp_dt = R_IB @ vel  # 平动动力学：位置导数
    # 平动运动学：机体系速度导数（重力在机体系下的投影）
    g_I = ca.SX([0, 0, uav_params.g])  # 惯性系重力（+z向下）
    g_B = R_IB.T @ g_I
    dv_dt = (F + uav_params.m * g_B) / uav_params.m
    # 转动动力学：欧拉角导数
    deuler_dt = T @ omega
    # 转动运动学：机体角速度导数
    domega_dt = ca.inv(uav_params.I) @ (-ca.cross(omega, uav_params.I @ omega) + tau)

    # 状态导数dx/dt
    dx_dt = ca.vertcat(dp_dt, dv_dt, deuler_dt, domega_dt)
    # 构建CasADi函数
    dynamics_func = ca.Function('dynamics', [x, u], [dx_dt], ['x', 'u'], ['dx_dt'])

    return dynamics_func, nx, nu

# ====================== 5. NMPC优化问题构建 ======================
class NMPCController:
    def __init__(self, uav_params, nmpc_params):
        self.uav = uav_params
        self.nmpc = nmpc_params
        self.dynamics_func, self.nx, self.nu = build_dynamics_model(uav_params)

        # 构建优化问题
        self.opti = ca.Opti()
        self.Ts = nmpc_params.Ts
        self.Np = nmpc_params.Np
        self.Nc = nmpc_params.Nc

        # 优化变量：状态序列、控制序列
        self.X = self.opti.variable(self.nx, self.Np + 1)  # 预测时域状态序列
        self.U = self.opti.variable(self.nu, self.Np)      # 预测时域控制序列
        self.U_prev = self.opti.parameter(self.nu)          # 上一时刻控制量（增量约束）
        self.X0 = self.opti.parameter(self.nx)              # 初始状态
        self.X_ref = self.opti.parameter(self.nx, self.Np + 1)  # 参考轨迹

        # ========== 1. 代价函数构建（严格匹配文档） ==========
        cost = 0
        # 定义一个辅助函数：计算归一化的角度差（CasADi符号表达式）
        def angle_diff(psi, psi_ref):
            return ca.atan2(ca.sin(psi - psi_ref), ca.cos(psi - psi_ref))
        # --- 终端代价 ---
        x_term = self.X[:, -1]
        x_ref_term = self.X_ref[:, -1]

        # 1. 提取偏航角（状态向量索引8）并单独计算归一化误差
        psi_term = x_term[8]
        psi_ref_term = x_ref_term[8]
        delta_psi_term = angle_diff(psi_term, psi_ref_term)

        # 2. 构建终端误差向量：其他状态直接做差，偏航角用归一化误差
        x_err_term = x_term - x_ref_term
        x_err_term[8] = delta_psi_term  # 替换偏航角误差

        # 3. 计算终端代价
        cost += ca.mtimes([x_err_term.T, self.nmpc.P, x_err_term])
        # 状态跟踪代价
        for i in range(self.Np):
            cost += ca.mtimes([(self.X[:, i] - self.X_ref[:, i]).T, self.nmpc.Q, (self.X[:, i] - self.X_ref[:, i])])
        # 控制量与控制增量代价
        for i in range(self.Nc):
            # 控制量偏离配平值惩罚
            cost += ca.mtimes([(self.U[:, i] - self.nmpc.u_trim).T, self.nmpc.R, (self.U[:, i] - self.nmpc.u_trim)])
            # 控制增量惩罚
            if i == 0:
                du = self.U[:, i] - self.U_prev
            else:
                du = self.U[:, i] - self.U[:, i-1]
            cost += ca.mtimes([du.T, self.nmpc.S, du])

        self.opti.minimize(cost)

        # ========== 2. 约束条件构建（严格匹配文档） ==========
        # 初始状态约束
        self.opti.subject_to(self.X[:, 0] == self.X0)
        # 动力学约束（欧拉离散化）
        for i in range(self.Np):
            x_next = self.X[:, i] + self.Ts * self.dynamics_func(self.X[:, i], self.U[:, i])
            self.opti.subject_to(self.X[:, i+1] == x_next)
        # 控制输入约束
        for i in range(self.Np):
            self.opti.subject_to(self.U[:, i] >= self.uav.u_min)
            self.opti.subject_to(self.U[:, i] <= self.uav.u_max)
        # 控制增量约束
        for i in range(self.Nc):
            if i == 0:
                du = self.U[:, i] - self.U_prev
            else:
                du = self.U[:, i] - self.U[:, i-1]
            self.opti.subject_to(du >= self.uav.du_min)
            self.opti.subject_to(du <= self.uav.du_max)
        # 状态安全约束
        for i in range(self.Np + 1):
            self.opti.subject_to(self.X[:, i] >= self.uav.x_min)
            self.opti.subject_to(self.X[:, i] <= self.uav.x_max)
        # 控制时域后控制量保持
        for i in range(self.Nc, self.Np):
            self.opti.subject_to(self.U[:, i] == self.U[:, self.Nc-1])

        # ========== 3. 求解器设置 ==========
        solver_opts = {
            "ipopt": {
                "max_iter": 50, # 增加最大迭代次数，提升求解成功率
                "print_level": 0, # 0-3逐级增加求解日志输出，调试时可设置为1或2
                "tol": 1e-2, # 适当放宽求解精度要求，提升求解速度和成功率
                "acceptable_tol": 1e-1, # 允许的求解精度，启用后在达到acceptable_tol时提前终止，提升求解效率
                "acceptable_obj_change_tol": 1e-1, # 允许的目标函数变化率，启用后在目标函数变化率小于该值时提前终止，提升求解效率
                "constr_viol_tol": 1e-2, # 适当放宽约束违反容忍度，提升求解成功率
                "dual_inf_tol": 100.0, # 适当放宽对偶变量违反容忍度，提升求解成功率
                "compl_inf_tol": 1e-2, # 适当放宽互补违反容忍度，提升求解成功率
                "mu_strategy": "adaptive",     # 自适应barrier参数（加速收敛）
                "warm_start_init_point": "yes",# 开启热启动（关键！利用上一步的解作为初始值）
                "warm_start_bound_push": 1e-6,
                "warm_start_mult_bound_push": 1e-6,
                "hessian_approximation": "limited-memory", # 使用拟牛顿近似，提升求解效率
                "linear_solver": "mumps",      # 保持MUMPS（如果有HSL MA27/MA57，换成"ma27"或"ma57"，速度再快2~3倍）
                "mumps_pivtol": 1e-6,          # MUMPS枢轴容差（加速MUMPS求解）
            },
            "print_time": 0
        }

        self.opti.solver("ipopt", solver_opts)

        # 初始化控制量
        self.u_prev = self.nmpc.u_trim

    def solve(self, x0, x_ref):
        """
        NMPC求解核心函数
        :param x0: 当前状态,12维数组
        :param x_ref: 参考轨迹,shape=(12, Np+1)
        :return: 最优控制量u_opt, 求解状态
        """
        print("=== 初始值约束检查 ===")
        
        # 1. 检查初始状态 x0
        x_viol_min = x0 < self.uav.x_min
        x_viol_max = x0 > self.uav.x_max

        if np.any(x_viol_min) or np.any(x_viol_max):
            print(f"❌ 初始状态 x0 违反约束！")
            print(f"   x0 = {x0.round(4)}")
            print(f"   违反下限的位置: {np.where(x_viol_min)[0]}, 值: {x0[x_viol_min]}")
            print(f"   违反上限的位置: {np.where(x_viol_max)[0]}, 值: {x0[x_viol_max]}")
            # 安全模式：直接返回悬停配平控制量，停止剧烈动作
            u_opt = self.nmpc.u_trim.copy()
            self.u_prev = u_opt  # 重置上一时刻控制量
            solve_success = False
            print("⚠️  已切换到安全悬停控制量\n")
            return u_opt, solve_success
        else:
            print(f"✅ 初始状态 x0 满足约束")
        
        # 2. 检查上一时刻控制量 u_prev
        u_viol_min = self.u_prev < self.uav.u_min
        u_viol_max = self.u_prev > self.uav.u_max
        if np.any(u_viol_min) or np.any(u_viol_max):
            print(f"❌ 上一时刻控制量 u_prev 违反约束！")
            print(f"   u_prev = {self.u_prev.round(4)}")
            print(f"   违反下限的位置: {np.where(u_viol_min)[0]}, 值: {self.u_prev[u_viol_min]}")
            print(f"   违反上限的位置: {np.where(u_viol_max)[0]}, 值: {self.u_prev[u_viol_max]}")
        else:
            print(f"✅ 上一时刻控制量 u_prev 满足约束")
        print("========================\n")
        # 赋值参数
        self.opti.set_value(self.X0, x0)
        self.opti.set_value(self.X_ref, x_ref)
        self.opti.set_value(self.U_prev, self.u_prev)


        # 热启动（加速求解）
        try:
            self.opti.set_initial(self.X, np.tile(x0, (self.Np + 1, 1)).T)
            self.opti.set_initial(self.U, np.tile(self.u_prev, (self.Np, 1)).T)
        except:
            pass

        # 求解优化问题
        try:
            sol = self.opti.solve()
            u_opt = sol.value(self.U[:, 0])
            margin = 0.01  # 约束边界容忍度
            u_min_safe = self.uav.u_min * (1 - margin) + self.uav.u_max * margin
            u_max_safe = self.uav.u_min * margin + self.uav.u_max * (1 - margin)
            u_opt = np.clip(u_opt, u_min_safe, u_max_safe)

            self.u_prev = u_opt
            solve_success = True
        except Exception as e:
            print(f"NMPC求解失败: {e}")
            u_opt = self.u_prev
            solve_success = False

        return u_opt, solve_success

# ====================== 6. 上位机主程序 ======================
class UAVHostController:
    def __init__(self, mode='simulation'):
        """
        上位机主控制器
        :param mode: 模式选择，'simulation'=仿真验证，'hardware'=硬件飞行控制
        """
        self.mode = mode
        self.uav_params = UAVParams()
        self.nmpc_params = NMPCParams()
        self.controller = NMPCController(self.uav_params, self.nmpc_params)

        # 参考轨迹参数（螺旋上升）
        self.ref_radius = 1.0          # 螺旋半径（m）
        self.ref_h_max = 3.0          # 最大高度（m）
        self.ref_vz = 0.3             # 上升速度（m/s）
        self.ref_turns_to_hmax = 0.5   # 上升到最大高度前绕圈数

        # 仿真初始化
        if mode == 'simulation':
            self.x_current = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # 初始状态
            self.t_history = []
            self.x_history = []
            self.u_history = []

        # 硬件通信初始化（MAVLink协议，适配PX4/APM飞控）
        elif mode == 'hardware':
            self.serial_port = '/dev/ttyUSB0'  # 串口设备，Windows为'COM3'等
            self.baudrate = 57600
            self.mav_connection = mavutil.mavlink_connection(self.serial_port, baud=self.baudrate)
            print("等待飞控心跳...")
            self.mav_connection.wait_heartbeat()
            print(f"飞控心跳已接收,系统ID: {self.mav_connection.target_system}, 组件ID: {self.mav_connection.target_component}")

    def generate_reference_trajectory(self, t_current):
        """
        生成圆形上升参考轨迹
        :param t_current: 当前时刻（秒）
        :return: 参考轨迹矩阵,shape=(12, Np+1)
        """
        Np = self.nmpc_params.Np  # 预测时域步长
        Ts = self.nmpc_params.Ts  # 控制周期
        x_ref = np.zeros((12, Np + 1))
        
        # 螺旋上升轨迹参数
        radius = self.ref_radius
        h_max = self.ref_h_max
        v_z = self.ref_vz
        t_rise = h_max / max(v_z, 1e-6)  # 到达最大高度所需时间
        omega = 2 * np.pi * self.ref_turns_to_hmax / t_rise
        
        # 偏航角归一化函数
        def normalize_angle(angle):
            return (angle + np.pi) % (2 * np.pi) - np.pi

        # 遍历预测时域的每个步长，生成参考轨迹
        for i in range(Np + 1):
            # 计算当前预测步对应的绝对时间
            t_i = t_current + i * Ts

            # 1. 位置计算：螺旋上升（初始点在 (0, 0)）
            x = radius * (np.cos(omega * t_i)-1)  # 圆周x坐标，初始点调整到(0, 0)
            y = radius * np.sin(omega * t_i)  # 圆周y坐标
            z = min(v_z * t_i, h_max)      # z轴上升，最大h_max

            # 2. 速度计算：圆周运动速度 + 上升速度
            vx = -radius * omega * np.sin(omega * t_i)  # 圆周x方向速度
            vy = radius * omega * np.cos(omega * t_i)   # 圆周y方向速度
            vz = v_z if z < h_max else 0.0         # z轴速度（到达h_max后为0）

            # 3. 姿态与角速度：滚转/俯仰为0，偏航跟随圆周方向，角速度为0
            phi = 0.0
            theta = 0.0
            psi = normalize_angle(omega * t_i)  # 偏航角归一化

            p, q, r = 0.0, 0.0, omega

            # 组装参考状态（12维）
            ref_state = np.array([x, y, z, vx, vy, vz, phi, theta, psi, p, q, r])
            x_ref[:, i] = ref_state
        
        return x_ref

    def simulation_step(self, u, dt):
        """仿真单步状态更新(RK4离散化,提高精度)"""
        k1 = self.controller.dynamics_func(self.x_current, u).full().flatten()
        k2 = self.controller.dynamics_func(self.x_current + dt/2 * k1, u).full().flatten()
        k3 = self.controller.dynamics_func(self.x_current + dt/2 * k2, u).full().flatten()
        k4 = self.controller.dynamics_func(self.x_current + dt * k3, u).full().flatten()
        self.x_current += dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    def get_hardware_state(self):
        """从飞控获取当前状态(MAVLink协议)"""
        # 获取位置与速度
        msg = self.mav_connection.recv_match(type='LOCAL_POSITION_NED', blocking=True)
        x, y, z = msg.x, msg.y, -msg.z  # NED转前右下坐标系
        vx, vy, vz = msg.vx, msg.vy, -msg.vz

        # 获取姿态与角速度
        msg_att = self.mav_connection.recv_match(type='ATTITUDE', blocking=True)
        phi, theta, psi = msg_att.roll, msg_att.pitch, msg_att.yaw
        p, q, r = msg_att.rollspeed, msg_att.pitchspeed, msg_att.yawspeed

        # 组装12维状态
        x_current = np.array([x, y, z, vx, vy, vz, phi, theta, psi, p, q, r])
        return x_current

    def send_hardware_control(self, u):
        """向飞控发送控制量(MAVLink自定义通道输出)"""
        # 电机转速归一化转PWM（1000-2000us）
        pwm_motor = 1000 + u[0:4] * 1000
        # 舵机倾角转PWM（-20°~20° → 1000-2000us）
        alpha_norm = (u[4:8] - self.uav_params.u_min[4:8]) / (self.uav_params.u_max[4:8] - self.uav_params.u_min[4:8])
        pwm_servo = 1000 + alpha_norm * 1000
        # 合并PWM输出
        pwm_output = np.concatenate([pwm_motor, pwm_servo]).astype(np.uint16)

        # 发送MAVLink舵机输出指令
        self.mav_connection.mav.servo_output_raw_send(
            time_usec=int(time.time() * 1e6),
            port=0,
            servo1_raw=pwm_output[0],
            servo2_raw=pwm_output[1],
            servo3_raw=pwm_output[2],
            servo4_raw=pwm_output[3],
            servo5_raw=pwm_output[4],
            servo6_raw=pwm_output[5],
            servo7_raw=pwm_output[6],
            servo8_raw=pwm_output[7],
        )

    def run_simulation(self, simulation_time=40):
        """运行仿真验证"""
        dt = self.nmpc_params.Ts # 控制周期
        t_total = 0

        while t_total < simulation_time:
            t_start = time.time()
            # 生成参考轨迹
            x_ref = self.generate_reference_trajectory(t_total)
            # NMPC求解
            u_opt, success = self.controller.solve(self.x_current, x_ref)
            # 仿真状态更新
            self.simulation_step(u_opt, dt)

            # 记录数据
            self.t_history.append(t_total)
            self.x_history.append(self.x_current.copy())
            self.u_history.append(u_opt.copy())

            # 打印日志
            if int(t_total / dt) % 1 == 0:  # 每1个时间步打印一次日志
                print(f"时间: {t_total:.2f}s | 当前位置: {self.x_current[0:3].round(3)} | 求解状态: {'成功' if success else '失败'}")

            # 时间同步
            t_total += dt
            loop_time = time.time() - t_start
            if loop_time < dt:
                time.sleep(dt - loop_time)

        print("仿真结束，绘制结果...")
        self.plot_simulation_results()

    def run_hardware_control(self):
        """运行硬件飞行控制"""
        print("开始硬件控制,按Ctrl+C终止")
        try:
            while True:
                t_start = time.time()
                # 计算当前时间（相对启动时间）
                t_current = time.time() - self.start_time
                # 获取飞控状态
                x_current = self.get_hardware_state()
                # 生成参考轨迹
                x_ref = self.generate_reference_trajectory(t_current)
                # NMPC求解
                u_opt, success = self.controller.solve(x_current, x_ref)
                # 发送控制量到飞控
                self.send_hardware_control(u_opt)

                # 打印日志
                print(f"当前位置: {x_current[0:3].round(3)} | 求解状态: {'成功' if success else '失败'}")

                # 控制周期同步
                loop_time = time.time() - t_start
                if loop_time < self.nmpc_params.Ts:
                    time.sleep(self.nmpc_params.Ts - loop_time)
        except KeyboardInterrupt:
            print("控制终止")

    def plot_simulation_results(self):
        """仿真结果可视化"""
        t = np.array(self.t_history)
        x = np.array(self.x_history)
        u = np.array(self.u_history)
        x_ref_hist = np.array([self.generate_reference_trajectory(ti)[:, 0] for ti in t])

        # # 位置跟踪曲线
        # fig = plt.figure()
        # plt.plot(t, x[:, 0], label='x')
        # plt.plot(t, x[:, 1], label='y')
        # plt.plot(t, x[:, 2], label='z')
        # plt.xlabel('时间 (s)')
        # plt.ylabel('位置 (m)')
        # plt.title('位置跟踪曲线')
        # plt.legend()
        # plt.grid(True)

        # # 姿态角曲线
        # fig = plt.figure()
        # plt.plot(t, np.rad2deg(x[:, 6]), label='滚转 phi')
        # plt.plot(t, np.rad2deg(x[:, 7]), label='俯仰 theta')
        # plt.plot(t, np.rad2deg(x[:, 8]), label='偏航 psi')
        # plt.xlabel('时间 (s)')
        # plt.ylabel('姿态角 (°)')
        # plt.title('姿态角曲线')
        # plt.legend()
        # plt.grid(True)

        # # 电机转速曲线
        # fig = plt.figure()
        # plt.plot(t, u[:, 0], label='Ω1')
        # plt.plot(t, u[:, 1], label='Ω2')
        # plt.plot(t, u[:, 2], label='Ω3')
        # plt.plot(t, u[:, 3], label='Ω4')
        # plt.xlabel('时间 (s)')
        # plt.ylabel('电机归一化转速')
        # plt.title('电机转速控制量')
        # plt.legend()
        # plt.grid(True)

        # # 舵机倾角曲线
        # fig = plt.figure()
        # plt.plot(t, np.rad2deg(u[:, 4]), label='α1')
        # plt.plot(t, np.rad2deg(u[:, 5]), label='α2')
        # plt.plot(t, np.rad2deg(u[:, 6]), label='α3')
        # plt.plot(t, np.rad2deg(u[:, 7]), label='α4')
        # plt.xlabel('时间 (s)')
        # plt.ylabel('舵机倾角 (°)')
        # plt.title('舵机倾转角控制量')
        # plt.legend()
        # plt.grid(True)

        # 3D位置轨迹
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x[:, 0], x[:, 1], x[:, 2], label='飞行轨迹')
        ax.plot(x_ref_hist[:, 0], x_ref_hist[:, 1], x_ref_hist[:, 2], '--', linewidth=2, label='参考轨迹')
        ax.scatter(x[0, 0], x[0, 1], x[0, 2], c='green', label='起点')
        ax.scatter(x[-1, 0], x[-1, 1], x[-1, 2], c='red', label='终点')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('3D飞行轨迹')
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.show()

# ====================== 7. 程序入口 ======================
if __name__ == "__main__":
    # 模式选择：'simulation'=仿真，'hardware'=硬件飞行
    run_mode = 'simulation'

    # 初始化上位机控制器
    host_controller = UAVHostController(mode=run_mode)

    # 设定目标位置 [x, y, z] 单位m
    # target_position = [5, 3, 2]

    # 运行控制
    if run_mode == 'simulation':
        host_controller.run_simulation(simulation_time=4)
    elif run_mode == 'hardware':
        host_controller.run_hardware_control()
