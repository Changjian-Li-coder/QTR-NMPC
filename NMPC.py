#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ====================== 1. 基础库导入 ======================
import numpy as np
import casadi as ca
import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
from tf.transformations import euler_from_quaternion

# ====================== 2. 无人机物理参数定义（严格匹配文档） ======================
class UAVParams:
    def __init__(self):
        # 基础质量与惯量
        self.m = 1.66  # 无人机质量，单位kg，可根据实际机型修改
        self.Ixx = 0.01  # x轴转动惯量，单位kg·m²
        self.Iyy = 0.01  # y轴转动惯量，单位kg·m²
        self.Izz = 0.02  # z轴转动惯量，单位kg·m²
        self.I = np.diag([self.Ixx, self.Iyy, self.Izz])  # 惯量矩阵
        self.g = 9.81  # 重力加速度，单位m/s²

        # 螺旋桨与舵机参数（文档乾丰6042三叶桨）
        self.k_f = 0.18  # 推力常数
        self.k_m = 0.0018  # 反扭矩常数
        self.k_torque = self.k_m / self.k_f  # 反扭矩-推力比例系数
        self.L = 0.182  # 机臂长度，单位m，可根据实际机型修改
        self.I_alpha = 0.001  # 旋翼组件转动惯量，单位kg·m²

        # 执行器约束（严格匹配文档）
        self.thrust_max = 1 * self.g   # 单电机最大推力，单位N 2.223KG
        # 控制输入u = [T1,T2,T3,T4,α1,α2,α3,α4]  单位：N和弧度
        self.T_min = 0.1
        self.T_max = 0.9
        self.alpha_max = np.deg2rad(25)  # 最大舵机倾角，单位弧度
        self.u_min = np.array([self.T_min * self.thrust_max, self.T_min * self.thrust_max, self.T_min * self.thrust_max, self.T_min * self.thrust_max, -self.alpha_max, -self.alpha_max, -self.alpha_max, -self.alpha_max])
        self.u_max = np.array([self.T_max * self.thrust_max, self.T_max * self.thrust_max, self.T_max * self.thrust_max, self.T_max * self.thrust_max, self.alpha_max, self.alpha_max, self.alpha_max, self.alpha_max])
        # 控制增量约束
        self.dT_max = 0.2
        self.dalpha_max = np.deg2rad(5)
        self.du_min = np.array([-self.dT_max * self.thrust_max, -self.dalpha_max, -self.dalpha_max, -self.dalpha_max, -self.dalpha_max, -self.dalpha_max, -self.dalpha_max, -self.dalpha_max])
        self.du_max = np.array([self.dT_max * self.thrust_max, self.dalpha_max, self.dalpha_max, self.dalpha_max, self.dalpha_max, self.dalpha_max, self.dalpha_max, self.dalpha_max])
        # 状态安全约束
        # 状态向量x = [x,y,z, vx,vy,vz, phi,theta,psi, p,q,r]
        self.x_min = np.array([-100, -100, -100,  # x,y,z位置约束，单位m
                                -20, -20, -20,  # vx,vy,vz速度约束，单位m/s
                                np.deg2rad(-150), np.deg2rad(-150), np.deg2rad(-180),  # phi,theta,psi欧拉角约束，单位弧度
                                np.deg2rad(-1200), np.deg2rad(-1200), np.deg2rad(-1200)])  # p,q,r角速度约束，单位rad/s
        self.x_max = np.array([100, 100, 100,  # x,y,z位置约束，单位m
                               20, 20, 20,   # vx,vy,vz速度约束，单位m/s
                               np.deg2rad(150), np.deg2rad(150), np.deg2rad(180),  # phi,theta,psi欧拉角约束，单位弧度
                               np.deg2rad(1200), np.deg2rad(1200), np.deg2rad(1200)]) # p,q,r角速度约束，单位rad/s

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
        self.Ts = 0.02  # 控制周期，单位s（50Hz控制频率，匹配飞控主流频率）
        self.Np = 4  # 预测时域步长
        self.Nc = 2  # 控制时域步长

        # 权重矩阵（调参核心：跟踪性能/控制平顺性平衡）
        # 状态权重Q: [x,y,z, vx,vy,vz, phi,theta,psi, p,q,r]
        self.Q = np.diag([15, 15, 18, 3, 3, 3, 80, 80, 40, 10, 10, 10])
        # 终端状态权重P
        self.P = self.Q * 10
        # 控制量权重R（惩罚偏离配平值）
        self.R = np.diag([0.05, 0.05, 0.05, 0.05, 0.2, 0.2, 0.2, 0.2])
        # 控制增量权重S（惩罚剧烈动作）
        self.S = np.diag([2, 2, 2, 2, 5, 5, 5, 5])

        # 悬停配平控制量（初始基准值，降低优化难度）
        hover_thrust = -(1.66 * 9.81 / 4)  #单电机悬停推力
        self.u_trim = np.array([hover_thrust]*4 + [0, 0, 0, 0])

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
    T_i = u[0:4]  # 电机推力 T_i
    alpha = u[4:8]        # 舵机倾角 α_i

    # 1. 机体系→惯性系旋转矩阵R_IB（文档定义）
    phi, theta, psi = euler[0], euler[1], euler[2]
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
    n_i = ca.SX.zeros(3, 4)  # 每个电机的推力单位矢量
    for i in range(4):
        u_i = uav_params.u_i[i, :].T
        cross_term = ca.cross(u_i, uav_params.n0)
        # 原错误代码
        # n_i[:, i] = uav_params.n0 * ca.cos(alpha[i]) + cross_term * ca.sin(alpha[i])
        # 修复后完整罗德里格斯公式
        u_i_unit = uav_params.u_i[i, :].T  # 旋转轴单位向量
        dot_term = ca.dot(u_i_unit, uav_params.n0)  # 旋转轴与初始矢量的点积
        n_i[:, i] = (uav_params.n0 * ca.cos(alpha[i]) + 
             cross_term * ca.sin(alpha[i]) + 
             u_i_unit * dot_term * (1 - ca.cos(alpha[i])))

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
    dp_dt = vel  # 平动动力学：位置导数
    # 平动运动学：机体系速度导数（重力在机体系下的投影）
    g_I = ca.SX([0, 0, uav_params.g])  # 惯性系重力
    dv_dt = ca.mtimes(R_IB, F) / uav_params.m + g_I
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
                "print_level": 3, # 0-3逐级增加求解日志输出，调试时可设置为1或2
                "tol": 1e-1, # 适当放宽求解精度要求，提升求解速度和成功率
                "acceptable_tol": 5e-1, # 允许的求解精度，启用后在达到acceptable_tol时提前终止，提升求解效率
                "acceptable_obj_change_tol": 1e0, # 允许的目标函数变化率，启用后在目标函数变化率小于该值时提前终止，提升求解效率
                "constr_viol_tol": 1e-1, # 适当放宽约束违反容忍度，提升求解成功率
                "dual_inf_tol": 1000.0, # 适当放宽对偶变量违反容忍度，提升求解成功率
                "compl_inf_tol": 1e-2, # 适当放宽互补违反容忍度，提升求解成功率
                "mu_strategy": "adaptive",     # 自适应barrier参数（加速收敛）
                "warm_start_init_point": "yes",# 开启热启动（关键！利用上一步的解作为初始值）
                "warm_start_bound_push": 1e-4,
                "warm_start_mult_bound_push": 1e-4,
                "hessian_approximation": "limited-memory", # 使用拟牛顿近似，提升求解效率
                "linear_solver": "mumps",      # 保持MUMPS（如果有HSL MA27/MA57，换成"ma27"或"ma57"，速度再快2~3倍）
                "mumps_pivtol": 1e-6,          # MUMPS枢轴容差（加速MUMPS求解）
            },
            "print_time": 1
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
        # print("=== 初始值约束检查 ===")
        
        # 1. 检查初始状态 x0
        x_viol_min = x0 < self.uav.x_min
        x_viol_max = x0 > self.uav.x_max

        if np.any(x_viol_min) or np.any(x_viol_max):
            print(f"❌ 初始状态 x0 违反约束！")
            # print(f"   x0 = {x0.round(4)}")
            # print(f"   违反下限的位置: {np.where(x_viol_min)[0]}, 值: {x0[x_viol_min]}")
            # print(f"   违反上限的位置: {np.where(x_viol_max)[0]}, 值: {x0[x_viol_max]}")
            # 安全模式：直接返回悬停配平控制量，停止剧烈动作
            u_opt = self.nmpc.u_trim.copy()
            self.u_prev = u_opt  # 重置上一时刻控制量
            solve_success = False
            # print("⚠️  已切换到安全悬停控制量\n")
            return u_opt, solve_success
        else:
            # print(f"✅ 初始状态 x0 满足约束")
            pass

        # 2. 检查上一时刻控制量 u_prev
        u_viol_min = self.u_prev < self.uav.u_min
        u_viol_max = self.u_prev > self.uav.u_max
        if np.any(u_viol_min) or np.any(u_viol_max):
            print(f"❌ 上一时刻控制量 u_prev 违反约束！")
            # print(f"   u_prev = {self.u_prev.round(4)}")
            # print(f"   违反下限的位置: {np.where(u_viol_min)[0]}, 值: {self.u_prev[u_viol_min]}")
            # print(f"   违反上限的位置: {np.where(u_viol_max)[0]}, 值: {self.u_prev[u_viol_max]}")
        else:
            # print(f"✅ 上一时刻控制量 u_prev 满足约束")
            pass
        # print("========================\n")
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

# ====================== 6. 上位机主程序（仅实物飞行/ROS） ======================
class UAVHostController:
    def __init__(self):
        """上位机主控制器（仅ROS实物飞行链路）"""
        self.uav_params = UAVParams()
        self.nmpc_params = NMPCParams()
        self.controller = NMPCController(self.uav_params, self.nmpc_params)

        # 当前状态（由/mavros话题回调更新）
        self.x_current = None
        self.state_ready = False

        # ROS通信：订阅状态，发布控制量
        self.state_sub = rospy.Subscriber("/mavros/local_position/odom", Odometry, self.odom_callback, queue_size=10)
        self.control_pub = rospy.Publisher("/nmpc/control_cmd", Float64MultiArray, queue_size=10)

    def odom_callback(self, msg):
        """从/mavros/local_position/odom更新12维状态"""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z

        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        vz = msg.twist.twist.linear.z

        q = msg.pose.pose.orientation
        phi, theta, psi = euler_from_quaternion([q.x, q.y, q.z, q.w])
        theta = -theta
        psi = -psi

        p_rate = msg.twist.twist.angular.x
        q_rate = -msg.twist.twist.angular.y
        r_rate = -msg.twist.twist.angular.z

        self.x_current = np.array([x, y, z, vx, vy, vz, phi, theta, psi, p_rate, q_rate, r_rate])
        # rospy.loginfo_throttle(1,self.x_current[9:12])
        self.state_ready = True

    def generate_reference_trajectory(self, t_current):
        """
        动态生成参考轨迹：适配扰动回正，兼顾NMPC多步预测
        :param t_current: 当前时刻（秒）
        :return: 参考轨迹矩阵, shape=(12, Np+1)
        """
        Np = self.nmpc_params.Np  # 预测时域步长
        Ts = self.nmpc_params.Ts  # 控制周期
        x_ref = np.zeros((12, Np + 1))
        
        # 1. 提取当前状态（受扰后的实际值）
        current_pos = self.x_current[0:3]       # 当前位置
        current_euler = self.x_current[6:9]     # 当前欧拉角（phi/theta/psi）
        current_omega = self.x_current[9:12]    # 当前角速度（p/q/r）
        
        # 2. 设计回正参数（可调，类似PID的比例系数，但作用于参考轨迹）
        k_att = 5.0  # 姿态偏差→期望角速度的比例系数（越大回正越快）
        decay_rate = 0.7  # 期望角速度的衰减系数（避免震荡）
        converge_steps = min(4, Np)  # 姿态回正的预测步数（前5步完成回正）
        
        # 3. 遍历预测时域，生成每一步的期望状态
        for i in range(Np + 1):
            # --- 步骤1：期望位置/速度（保持悬停，无变化）---
            ref_pos = current_pos  # 位置保持当前值（悬停目标）
            # ref_pos = np.array([4.63,0.18,-0.49])
            ref_vel = np.array([0.0, 0.0, 0.0])  # 速度归零
            
            # --- 步骤2：期望欧拉角（逐步收敛到0）---
            # 前converge_steps步：从当前欧拉角线性收敛到0；之后保持0
            if i <= converge_steps:
                ref_euler = current_euler * (1 - i / converge_steps)
            else:
                ref_euler = np.array([0.0, 0.0, 0.0])
            # 偏航角特殊处理：保持当前值（悬停时无需纠偏）
            ref_euler[2] = current_euler[2]
            
            # --- 步骤3：期望角速度（由姿态偏差驱动，逐步衰减）---
            # 核心逻辑：期望角速度 = -k_att × 姿态偏差（负号：偏差方向与回正方向相反）
            att_error = ref_euler - current_euler  # 姿态偏差（期望-实际）
            ref_omega = k_att * att_error
            # 衰减：步数越多，期望角速度越接近0（避免回正过度）
            ref_omega = ref_omega * (decay_rate ** i)
            # 限制期望角速度范围（避免超过物理极限）
            ref_omega = np.clip(ref_omega, 
                            self.uav_params.x_min[9:12], 
                            self.uav_params.x_max[9:12])
            
            # --- 组装期望状态 ---
            ref_state = np.concatenate([
                ref_pos,    # 位置 [x,y,z]
                ref_vel,    # 速度 [vx,vy,vz]
                ref_euler,  # 欧拉角 [phi,theta,psi]
                ref_omega   # 角速度 [p,q,r]
            ])
            x_ref[:, i] = ref_state
        
        return x_ref

    def publish_control(self, u):
        """发布NMPC控制量到ROS话题，后续由你二次处理再下发飞控"""
        msg = Float64MultiArray()
        msg.data = u.astype(float).tolist()
        self.control_pub.publish(msg)

    def run(self):
        """运行控制主循环（状态来自MAVROS，控制量仅发布ROS话题）"""
        rospy.loginfo("NMPC控制启动，等待/mavros/local_position/odom状态...")
        rate = rospy.Rate(1.0 / self.nmpc_params.Ts)
        t0 = rospy.Time.now().to_sec()

        while not rospy.is_shutdown():
            if not self.state_ready:
                rate.sleep()
                continue

            t_current = rospy.Time.now().to_sec() - t0
            x_ref = self.generate_reference_trajectory(t_current)
            u_opt, success = self.controller.solve(self.x_current, x_ref)
            send_u = u_opt.copy()
            for i in range(0,4):
                send_u[i] = max(0.1, min(send_u[i] / self.uav_params.thrust_max, 0.9))
            if success:
                self.publish_control(send_u)

            rospy.loginfo_throttle(1.0, f"当前位置: {self.x_current[0:3].round(3)} | 求解状态: {'成功' if success else '失败'}")
            rate.sleep()

# ====================== 7. 程序入口 ======================
if __name__ == "__main__":
    rospy.init_node("nmpc_controller", anonymous=False)
    host_controller = UAVHostController()
    host_controller.run()
