#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ====================== 1. 基础库导入 ======================
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import time
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import os
os.environ["ACADOS_NO_TEMPLATES"] = "1"   # ✅ 强制关闭模板，解决 t_renderer 崩溃
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel



plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']  # 永久指定中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号乱码

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['figure.dpi'] = 100  # 统一分辨率
matplotlib.rcParams['savefig.dpi'] = 300  # 保存图片分辨率

# ====================== 新增：持续随机扰动函数 ======================
def add_continuous_attitude_disturbance(u, disturbance_range=(0.1, 0.2)):
    """
    为控制输入添加持续随机姿态扰动（力矩通道）
    :param u: 原始控制输入 [Fx,Fy,Fz,τx,τy,τz]
    :param disturbance_range: 扰动幅值范围 (min, max)，随机生成±范围内的数值
    :return: 叠加扰动后的控制输入
    """
    # 仅在力矩通道（τx,τy,τz）添加随机扰动
    disturbance = np.random.uniform(
        low=-disturbance_range[1], 
        high=disturbance_range[1], 
        size=3
    )
    # 限制扰动最小值（避免扰动过小无意义）
    disturbance = np.where(
        np.abs(disturbance) < disturbance_range[0], 
        np.sign(disturbance) * disturbance_range[0], 
        disturbance
    )
    
    u_disturbed = u.copy()
    u_disturbed[3:6] += disturbance  # 叠加到力矩通道
    return u_disturbed
    
# ====================== 2. 无人机物理参数（保持不变） ======================
class UAVParams:
    def __init__(self):
        self.m = 1.66
        self.Ixx = 0.05
        self.Iyy = 0.05
        self.Izz = 0.08
        self.I = np.diag([self.Ixx, self.Iyy, self.Izz])
        self.g = 9.81
        self.nu = 6  # u=[Fx,Fy,Fz,τx,τy,τz]
        self.u_min = np.array([-10, -10, -30, -5, -5, -5])  #z轴推力为负值表示向下的推力，所以u_min其实也是u_max
        self.u_max = np.array([10, 10, 30, 5, 5, 5])
        self.du_min = np.array([-2, -2, -2, -1, -1, -1])
        self.du_max = np.array([2, 2, 2, 1, 1, 1])
        self.x_min = np.array([-100, -100, -100, -10, -10, -10, 
                               np.deg2rad(-90), np.deg2rad(-90), np.deg2rad(-180), 
                               np.deg2rad(-600), np.deg2rad(-600), np.deg2rad(-600)])
        self.x_max = np.array([100, 100, 100, 10, 10, 10, 
                               np.deg2rad(90), np.deg2rad(90), np.deg2rad(180), 
                               np.deg2rad(600), np.deg2rad(600), np.deg2rad(600)])

# ====================== 3. NMPC超参数（保持不变） ======================
class NMPCParams:
    def __init__(self):
        self.Ts = 0.05
        self.Np = 8   # 预测时域
        self.Nc = 4   # 控制时域
        self.Q = np.diag([100, 100, 120, 1, 1, 1, 2, 2, 1, 0.05, 0.05, 0.05]) # 状态权重（位置更重，姿态适中，速度较轻）
        self.P = self.Q * 1.2  # 终端权重（更重，强化终端约束）
        self.R = np.diag([0.1, 0.1, 0.1, 0.2, 0.2, 0.2])  # 控制权重（推力较轻，力矩较重，鼓励使用推力调整姿态）
        self.S = np.diag([1, 1, 1, 0.5, 0.5, 0.5])  # 状态-控制交叉权重（鼓励状态误差通过推力调整来纠正）
        # 悬停配平：Fz=mg，其余为0
        hover_thrust = -1.66 * 9.81 # 注意：Fz为负值表示向下的推力
        self.u_trim = np.array([0.0, 0.0, hover_thrust, 0.0, 0.0, 0.0])

# ====================== 4. 动力学模型（适配acados） ======================
def build_acados_dynamics_model(uav_params):
    nx = 12
    nu = 6
    # 定义CasADi符号变量
    x = ca.SX.sym('x', nx)
    u = ca.SX.sym('u', nu)
    x_dot = ca.SX.sym('x_dot', nx)
    
    pos = x[0:3]   # x,y,z
    vel = x[3:6]   # vx,vy,vz (惯性系)
    euler = x[6:9] # phi,theta,psi
    omega = x[9:12]# p,q,r (机体系)
    phi, theta, psi = euler[0], euler[1], euler[2]
    p, q, r = omega[0], omega[1], omega[2]

    # 机体系 → 惯性系 旋转矩阵 R_IB
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

    # 欧拉角速率矩阵 T
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

    # 核心动力学公式
    F_B = u[0:3]      # 机体系期望推力 Fx,Fy,Fz
    tau_B = u[3:6]    # 机体系期望力矩 τx,τy,τz

    # 1. 位置导数（运动学）
    dp_dt = vel
    # 2. 速度导数（动力学）
    g_I = ca.SX([0, 0, uav_params.g])  # 惯性系重力 z轴向下为正
    dv_dt = ca.mtimes(R_IB, F_B) / uav_params.m + g_I
    # 3. 欧拉角导数
    deuler_dt = ca.mtimes(T, omega)
    # 4. 角速度导数（转动动力学）
    cross_term = ca.cross(omega, ca.mtimes(uav_params.I, omega))
    domega_dt = ca.mtimes(ca.inv(uav_params.I), -cross_term + tau_B)

    dx_dt = ca.vertcat(dp_dt, dv_dt, deuler_dt, domega_dt)
    
    # 构建acados模型
    acados_model = AcadosModel()
    acados_model.name = 'QTR_dynamics'
    acados_model.x = x
    acados_model.u = u
    acados_model.xdot = x_dot
    acados_model.f_expl_expr = dx_dt  # 显式动力学
    acados_model.f_impl_expr = x_dot - dx_dt  # 隐式动力学（acados要求）
    
    return acados_model, nx, nu

# ====================== 5. NMPC控制器（新增求解时间计时） ======================
class NMPCController:
    def __init__(self, uav_params, nmpc_params):
        self.uav = uav_params
        self.nmpc = nmpc_params
        self.acados_model, self.nx, self.nu = build_acados_dynamics_model(uav_params)
        self.ny = self.nx + self.nu  # 代价函数输出维度（状态+控制）
        self.ny_e = self.nx  # 终端代价维度（仅状态）
        
        # 初始化OCP问题
        self.ocp = AcadosOcp()
        self.ocp.model = self.acados_model
        self.ocp.dims.N = self.nmpc.Np  # 预测时域
        self.ocp.solver_options.tf = self.nmpc.Ts * self.nmpc.Np  # 总预测时间
        
        # 状态和控制变量初始化
        self.ocp.constraints.x0 = np.zeros(self.nx)  # 初始状态（后续每步更新）
        self.ocp.constraints.lbx = self.uav.x_min  # 状态下界
        self.ocp.constraints.ubx = self.uav.x_max  # 状态上界
        self.ocp.constraints.lbu = self.uav.u_min  # 控制下界
        self.ocp.constraints.ubu = self.uav.u_max  # 控制上界
        self.ocp.constraints.idxbu = np.arange(self.nu)  # 控制变量索引
        self.ocp.constraints.idxbx = np.arange(self.nx)  # 状态变量索引
        
        # ========== 修复核心：代价函数配置 ==========
        self.ocp.cost.cost_type = 'LINEAR_LS'
        self.ocp.cost.cost_type_e = 'LINEAR_LS'  # 终端代价类型
        
        # 1. 阶段代价权重：合并Q(12x12)和R(6x6)为18x18的对角矩阵
        W = np.block([
            [self.nmpc.Q, np.zeros((self.nx, self.nu))],
            [np.zeros((self.nu, self.nx)), self.nmpc.R]
        ])
        self.ocp.cost.W = W  # 阶段权重 (18x18)
        self.ocp.cost.W_e = self.nmpc.P  # 终端权重 (12x12)
        
        # 2. 修正Vx/Vu维度：Vx(18x12), Vu(18x6)
        self.ocp.cost.Vx = np.vstack([np.eye(self.nx), np.zeros((self.nu, self.nx))])  # 18x12
        self.ocp.cost.Vu = np.vstack([np.zeros((self.nx, self.nu)), np.eye(self.nu)])  # 18x6
        self.ocp.cost.Vx_e = np.eye(self.nx)  # 终端仅状态 (12x12)
        
        # 3. 参考值初始化（维度匹配）
        self.ocp.cost.yref = np.zeros(self.ny)  # 18维（12状态+6控制）
        self.ocp.cost.yref_e = np.zeros(self.ny_e)  # 12维（仅状态）
        
        # 求解器配置
        self.ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'  # HPIPM求解QP
        self.ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'  # 高斯牛顿近似
        self.ocp.solver_options.integrator_type = 'ERK'  # 显式龙格库塔积分
        self.ocp.solver_options.nlp_solver_type = 'SQP_RTI'  # SQP-RTI（实时迭代）
        self.ocp.solver_options.print_level = 0  # 关闭打印
        self.ocp.solver_options.nlp_solver_max_iter = 150  # 最大迭代次数
        
        # 初始化求解器
        self.acados_solver = AcadosOcpSolver(self.ocp, json_file='acados_ocp.json')
        
        # 控制历史（热启动用）
        self.u_prev = self.nmpc.u_trim
        self.X_ref_prev = np.tile(self.u_prev, (self.nmpc.Np+1, 1)).T
        # ========== 新增：初始化求解时间记录列表 ==========
        self.solve_time_history = []

    def angle_diff(self, psi, psi_ref):
        """角度差归一化"""
        return ca.atan2(ca.sin(psi - psi_ref), ca.cos(psi - psi_ref))

    def solve(self, x0, x_ref):
        # ========== 新增：开始计时（精确到微秒） ==========
        solve_start = time.perf_counter()
        
        # 安全约束检查
        x0 = np.clip(x0, self.uav.x_min, self.uav.x_max)
        
        
        self.acados_solver.set(0, 'lbx', x0)
        self.acados_solver.set(0, 'ubx', x0)
        
        # 更新参考轨迹和代价函数
        for i in range(self.nmpc.Np):
            # 角度归一化（偏航角）
            x_ref_i = x_ref[:, i].copy()
            x_ref_i[8] = self.angle_diff(x_ref_i[8], x_ref_i[8]).full()[0] if isinstance(x_ref_i[8], ca.SX) else x_ref_i[8]
            
            # 设置阶段参考：12状态 + 6控制（配平值）
            yref = np.concatenate([x_ref_i, self.nmpc.u_trim])
            self.acados_solver.set(i, 'yref', yref)
            
            # 控制率约束（前Nc步）
            if i < self.nmpc.Nc - 1:
                du = self.acados_solver.get(i, 'u') - self.acados_solver.get(i+1, 'u')
                du = np.clip(du, self.uav.du_min, self.uav.du_max)
                self.acados_solver.set(i+1, 'lbu', self.acados_solver.get(i, 'u') + self.uav.du_min)
                self.acados_solver.set(i+1, 'ubu', self.acados_solver.get(i, 'u') + self.uav.du_max)
        
        # 终端参考（仅状态）
        x_ref_e = x_ref[:, -1].copy()
        x_ref_e[8] = self.angle_diff(x_ref_e[8], x_ref_e[8]).full()[0] if isinstance(x_ref_e[8], ca.SX) else x_ref_e[8]
        self.acados_solver.set(self.nmpc.Np, 'yref', x_ref_e)
        
        # 热启动（使用上一次的解）
        for i in range(self.nmpc.Np):
            self.acados_solver.set(i, 'u', self.u_prev)
        for i in range(self.nmpc.Np + 1):
            self.acados_solver.set(i, 'x', x0)
        
        # 求解OCP问题
        try:
            status = self.acados_solver.solve()
            # ========== 新增：计算求解耗时（转换为秒） ==========
            solve_time = time.perf_counter() - solve_start
            self.solve_time_history.append(solve_time)
            
            if status != 0:
                raise RuntimeError(f"acados求解失败，状态码：{status}")
            
            # 获取最优控制（第一个控制量）
            u_opt = self.acados_solver.get(0, 'u')
            u_opt = np.clip(u_opt, self.uav.u_min, self.uav.u_max)
            self.u_prev = u_opt
            # ========== 新增：返回求解时间 ==========
            return u_opt, True, solve_time
        except Exception as e:
            # ========== 新增：失败时也记录时间 ==========
            solve_time = time.perf_counter() - solve_start
            self.solve_time_history.append(solve_time)
            print(f"⚠️ NMPC求解失败：{e}，使用悬停配平控制")
            # ========== 新增：返回求解时间 ==========
            return self.nmpc.u_trim, False, solve_time

# ====================== 6. 主程序（新增求解时间统计/打印/绘图） ======================
class UAVHostController:
    def __init__(self, mode='simulation'):
        self.mode = mode
        self.uav_params = UAVParams()
        self.nmpc_params = NMPCParams()
        self.controller = NMPCController(self.uav_params, self.nmpc_params)
        self.ref_radius = 1.0
        self.ref_h_max = 3.0
        self.ref_vz = 0.2
        self.ref_turns_to_hmax = 0.5
        if mode == 'simulation':
            self.x_current = np.zeros(12)
            self.x_current[6] = np.deg2rad(15)  # 初始滚转角15°
            self.x_current[7] = np.deg2rad(10)  # 初始俯仰角10°
            self.t_history = []
            self.x_history = []
            self.u_history = []
            # ========== 新增：存储每轮求解时间 ==========
            self.solve_time_history = []

            # 扰动参数
            self.disturbance_time = 3.0  # 扰动施加时间（仿真2秒时）
            self.disturbance_duration = 0.2  # 扰动持续时间
            self.disturbance_phi = np.deg2rad(2)  # 滚转扰动8°
            self.disturbance_theta = np.deg2rad(3)  # 俯仰扰动5°

    # def generate_reference_trajectory(self, t_current):
    #     Np = self.nmpc_params.Np
    #     Ts = self.nmpc_params.Ts
    #     x_ref = np.zeros((12, Np + 1))
    #     radius = self.ref_radius
    #     h_max = self.ref_h_max
    #     v_z = self.ref_vz
    #     omega = 2.5
    #     def normalize_angle(angle):
    #         return (angle + np.pi) % (2 * np.pi) - np.pi
    #     for i in range(Np + 1):
    #         t_i = t_current + i * Ts
    #         x = radius * (np.cos(omega * t_i)-1)
    #         y = radius * np.sin(omega * t_i)
    #         z = min(v_z * t_i, h_max)
    #         vx = -radius * omega * np.sin(omega * t_i)
    #         vy = radius * omega * np.cos(omega * t_i)
    #         vz = v_z if z < h_max else 0.0
    #         ref_state = np.array([x,y,z,vx,vy,vz,0,0,normalize_angle(omega*t_i),0,0,omega])
    #         x_ref[:,i] = ref_state
    #     return x_ref
    def generate_reference_trajectory(self, t_current):
        Np = self.nmpc_params.Np
        Ts = self.nmpc_params.Ts
        x_ref = np.zeros((12, Np + 1))

        for i in range(Np + 1):
            ref_state = np.array([0,0,0,0,0,0,0,0,0,0,0,0])
            x_ref[:,i] = ref_state
        return x_ref

    def add_attitude_disturbance(self, t_current):
        """在指定时间添加姿态扰动"""
        if (self.disturbance_time <= t_current < self.disturbance_time + self.disturbance_duration):
            self.x_current[6] += self.disturbance_phi  # 滚转扰动
            self.x_current[7] -= self.disturbance_theta  # 俯仰扰动
            print(f"⚠️ 施加姿态扰动：滚转+{np.rad2deg(self.disturbance_phi):.1f}°，俯仰+{np.rad2deg(self.disturbance_theta):.1f}°")

    def simulation_step(self, u, dt):
        # 1. 叠加持续随机扰动到控制输入
        u_disturbed = add_continuous_attitude_disturbance(u)
        
        # 2. 使用叠加扰动后的控制输入进行仿真
        k1 = ca.Function('f', [self.controller.acados_model.x, self.controller.acados_model.u], 
                         [self.controller.acados_model.f_expl_expr])(self.x_current, u_disturbed).full().flatten()
        k2 = ca.Function('f', [self.controller.acados_model.x, self.controller.acados_model.u], 
                         [self.controller.acados_model.f_expl_expr])(self.x_current + dt/2*k1, u_disturbed).full().flatten()
        k3 = ca.Function('f', [self.controller.acados_model.x, self.controller.acados_model.u], 
                         [self.controller.acados_model.f_expl_expr])(self.x_current + dt/2*k2, u_disturbed).full().flatten()
        k4 = ca.Function('f', [self.controller.acados_model.x, self.controller.acados_model.u], 
                         [self.controller.acados_model.f_expl_expr])(self.x_current + dt*k3, u_disturbed).full().flatten()
        self.x_current += dt/6*(k1+2*k2+2*k3+k4)
        

    def run_simulation(self, simulation_time=10):
        dt = self.nmpc_params.Ts
        t_total = 0
        while t_total < simulation_time:
            t_start = time.time()

            # 施加姿态扰动
            self.add_attitude_disturbance(t_total)

            # 生成参考轨迹
            x_ref = self.generate_reference_trajectory(t_total)

            # ========== 新增：接收求解时间 ==========
            u_opt, success, solve_time = self.controller.solve(self.x_current, x_ref)
            # ========== 新增：保存求解时间 ==========
            self.solve_time_history.append(solve_time)
            
            # 仿真步进
            self.simulation_step(u_opt, dt)

            # 记录数据
            self.t_history.append(t_total)
            self.x_history.append(self.x_current.copy())
            self.u_history.append(u_opt.copy())
            
            # ========== 新增：打印时显示求解时间（每10步） ==========
            # if int(t_total/dt) % 10 == 0:
            #     print(f"时间:{t_total:.2f}s | 位置:{self.x_current[0:3].round(3)} | 求解:{'成功' if success else '失败'} | 求解时间:{solve_time*1000:.2f}ms")
            
            t_total += dt
            loop_time = time.time() - t_start
            if loop_time < dt:
                time.sleep(dt-loop_time)
        
        # ========== 新增：仿真结束后输出求解时间统计 ==========
        solve_times = np.array(self.solve_time_history) * 1000  # 转换为毫秒
        print("\n===== NMPC求解时间统计 =====")
        print(f"总求解次数: {len(solve_times)}")
        print(f"平均求解时间: {np.mean(solve_times):.2f} ms")
        print(f"最大求解时间: {np.max(solve_times):.2f} ms")
        print(f"最小求解时间: {np.min(solve_times):.2f} ms")
        print(f"求解时间标准差: {np.std(solve_times):.2f} ms")
        print(f"95%分位数求解时间: {np.percentile(solve_times, 95):.2f} ms")
        print("============================")
        
        print("仿真完成！")
        self.plot_results()  # 启用绘图（包含求解时间）

    def plot_results(self):

        save_dir = "/home/li/catkin_ws/src/send_data/scripts/figure"
        t = np.array(self.t_history)
        x = np.array(self.x_history)
        u = np.array(self.u_history)

        # ========== 新增：滚转角/俯仰角绘图（核心需求） ==========
        plt.figure(figsize=(10, 5))
        # 滚转角
        plt.subplot(2,1,1)
        phi_deg = np.rad2deg(x[:, 6])  # 转换为角度
        plt.plot(t, phi_deg, color='#2E86AB', linewidth=2, label='Actual Roll Angle')
        plt.axhline(y=0, color='#E63946', linestyle='--', label='Reference Roll Angle (0°)', linewidth=1.5)
        plt.axvline(x=self.disturbance_time, color='gray', linestyle=':', label='Disturbance Application Time', linewidth=1.5)
        plt.title('UAV Roll Angle Response')
        plt.xlabel('Time (s)')
        plt.ylabel('Roll Angle (°)')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        # 俯仰角
        plt.subplot(2,1,2)
        theta_deg = np.rad2deg(x[:, 7])  # 转换为角度
        plt.plot(t, theta_deg, color='#F1A208', linewidth=2, label='Actual Pitch Angle')
        plt.axhline(y=0, color='#E63946', linestyle='--', label='Reference Pitch Angle (0°)', linewidth=1.5)
        plt.axvline(x=self.disturbance_time, color='gray', linestyle=':', label='Disturbance Application Time', linewidth=1.5)
        plt.title('UAV Pitch Angle Response')
        plt.xlabel('Time (s)')
        plt.ylabel('Pitch Angle (°)')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path = os.path.join(save_dir, 'attitude_response.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        # ========== 新增：求解时间数据（转换为毫秒） ==========
        # solve_times = np.array(self.solve_time_history) * 1000

        # ========== 新增：求解时间专用绘图 ==========
        # plt.figure(figsize=(10, 4))
        # plt.plot(t, solve_times, color='#2E86AB', label='Per-step Solve Time', linewidth=1.5)
        # plt.axhline(y=np.mean(solve_times), color='#E63946', linestyle='--', 
        #             label=f'Average: {np.mean(solve_times):.2f} ms', linewidth=2)
        # plt.fill_between(t, solve_times, alpha=0.3, color='#2E86AB')
        # plt.title('NMPC Solve Time_acados (Per Step)')
        # plt.xlabel('Simulation Time (s)')
        # plt.ylabel('Solve Time (ms)')
        # plt.legend(loc='upper right')
        # plt.grid(True, alpha=0.3)
        # plt.tight_layout()
        # save_path = os.path.join(save_dir, 'nmpc_solve_time.png')
        # plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # ========== 1. 推力/力矩绘图（修复中文显示） ==========
        plt.figure(figsize=(10,6))
        plt.subplot(2,1,1)
        plt.plot(t, u[:,0], label='Fx (X-Thrust)')
        plt.plot(t, u[:,1], label='Fy (Y-Thrust)')
        plt.plot(t, u[:,2], label='Fz (Z-Thrust)')
        plt.title('Desired Thrust_acados')
        plt.xlabel('Time (s)')
        plt.ylabel('Thrust (N)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.subplot(2,1,2)
        plt.plot(t, u[:,3], label='τx (X-Torque)')
        plt.plot(t, u[:,4], label='τy (Y-Torque)')
        plt.plot(t, u[:,5], label='τz (Z-Torque)')
        plt.title('Desired Torque_acados')
        plt.xlabel('Time (s)')
        plt.ylabel('Torque (N·m)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path = os.path.join(save_dir, 'thrust_torque_acados.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # # ========== 2. 3D轨迹绘图（核心修复：移除cls参数） ==========
        # x_ref_hist = np.array([self.generate_reference_trajectory(ti)[:, 0] for ti in t])
        # fig = plt.figure(figsize=(8,6))
        # # 正确创建3D子图：移除错误的cls=Axes3D参数
        # ax = fig.add_subplot(111, projection='3d')
        # # 绘制轨迹
        # ax.plot(x[:, 0], x[:, 1], x[:, 2], 'b-', linewidth=2, label='Actual Trajectory')
        # ax.plot(x_ref_hist[:, 0], x_ref_hist[:, 1], x_ref_hist[:, 2], 'r--', linewidth=1.5, label='Reference Trajectory')
        # # 起点/终点标记
        # ax.scatter(x[0, 0], x[0, 1], x[0, 2], c='green', s=50, label='Start')
        # ax.scatter(x[-1, 0], x[-1, 1], x[-1, 2], c='red', s=50, label='End')
        # # 坐标轴标签
        # ax.set_xlabel('X (m)')
        # ax.set_ylabel('Y (m)')
        # ax.set_zlabel('Z (m)')
        # ax.set_title('3D Flight Trajectory_acados')
        # ax.legend(loc='best')
        # # 优化3D视角
        # ax.view_init(elev=20, azim=45)
        # plt.tight_layout()
        # save_path = os.path.join(save_dir, '3d_trajectory_acados.png')
        # plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # 可选：显示图像（若闪退可注释，优先保存文件）
        try:
            plt.show()
        except Exception as e:
            print(f"⚠️ 图像显示失败（不影响文件保存）：{e}")

if __name__ == "__main__":
    host = UAVHostController(mode='simulation')
    host.run_simulation(simulation_time=5)