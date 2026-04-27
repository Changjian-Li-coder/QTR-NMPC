# -*- coding: utf-8 -*-
# ====================== 1. 基础库导入 ======================
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import time
import matplotlib
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

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
        self.u_min = np.array([-10, -10, -30, -5, -5, -5])
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
        self.Np = 7   # 预测时域
        self.Nc = 4   # 控制时域
        self.Q = np.diag([100, 100, 120, 1, 1, 1, 2, 2, 1, 0.05, 0.05, 0.05])
        self.P = self.Q * 1.2
        self.R = np.diag([0.1, 0.1, 0.1, 0.2, 0.2, 0.2])
        self.S = np.diag([1, 1, 1, 0.5, 0.5, 0.5])
        # 悬停配平：Fz=mg，其余为0
        hover_thrust = -1.66 * 9.81
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
    g_I = ca.SX([0, 0, uav_params.g])  # 惯性系重力
    dv_dt = ca.mtimes(R_IB, F_B) / uav_params.m + g_I
    # 3. 欧拉角导数
    deuler_dt = ca.mtimes(T, omega)
    # 4. 角速度导数（转动动力学）
    cross_term = ca.cross(omega, ca.mtimes(uav_params.I, omega))
    domega_dt = ca.mtimes(ca.inv(uav_params.I), -cross_term + tau_B)

    dx_dt = ca.vertcat(dp_dt, dv_dt, deuler_dt, domega_dt)
    
    # 构建acados模型
    acados_model = AcadosModel()
    acados_model.name = 'uav_dynamics'
    acados_model.x = x
    acados_model.u = u
    acados_model.xdot = x_dot
    acados_model.f_expl_expr = dx_dt  # 显式动力学
    acados_model.f_impl_expr = x_dot - dx_dt  # 隐式动力学（acados要求）
    
    return acados_model, nx, nu

# ====================== 5. NMPC控制器（替换为acados求解器） ======================
class NMPCController:
    def __init__(self, uav_params, nmpc_params):
        self.uav = uav_params
        self.nmpc = nmpc_params
        self.acados_model, self.nx, self.nu = build_acados_dynamics_model(uav_params)
        
        # 初始化OCP问题
        self.ocp = AcadosOcp()
        self.ocp.model = self.acados_model
        self.ocp.dims.N = self.nmpc.Np  # 预测时域
        self.ocp.solver_options.tf = self.nmpc.Ts * self.nmpc.Np  # 总预测时间
        
        # 状态和控制变量初始化
        self.ocp.constraints.x0 = np.zeros(self.nx)
        self.ocp.constraints.lbx = self.uav.x_min  # 状态下界
        self.ocp.constraints.ubx = self.uav.x_max  # 状态上界
        self.ocp.constraints.lbu = self.uav.u_min  # 控制下界
        self.ocp.constraints.ubu = self.uav.u_max  # 控制上界
        self.ocp.constraints.idxbu = np.arange(self.nu)  # 控制变量索引
        self.ocp.constraints.idxbx = np.arange(self.nx)  # 状态变量索引
        
        # 代价函数配置（LQR形式）
        self.ocp.cost.cost_type = 'LINEAR_LS'
        self.ocp.cost.cost_type_e = 'LINEAR_LS'  # 终端代价类型
        self.ocp.cost.W = self.nmpc.Q  # 阶段状态权重
        self.ocp.cost.W_e = self.nmpc.P  # 终端状态权重
        self.ocp.cost.W_u = self.nmpc.R  # 控制权重
        self.ocp.cost.W_uu = self.nmpc.S  # 控制率权重（通过约束近似）
        
        # 代价函数矩阵映射（单位矩阵，因为直接用W加权）
        self.ocp.cost.Vx = np.eye(self.nx)
        self.ocp.cost.Vu = np.eye(self.nu)
        self.ocp.cost.Vx_e = np.eye(self.nx)
        
        # 参考轨迹初始化
        self.ocp.cost.yref = np.zeros(self.nx + self.nu)
        self.ocp.cost.yref_e = np.zeros(self.nx)
        
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

    def angle_diff(self, psi, psi_ref):
        """角度差归一化"""
        return ca.atan2(ca.sin(psi - psi_ref), ca.cos(psi - psi_ref))

    def solve(self, x0, x_ref):
        # 安全约束检查
        x0 = np.clip(x0, self.uav.x_min, self.uav.x_max)
        
        # 更新初始状态
        self.acados_solver.set(0, 'lbx', x0)
        self.acados_solver.set(0, 'ubx', x0)
        
        # 更新参考轨迹和代价函数
        for i in range(self.nmpc.Np):
            # 角度归一化（偏航角）
            x_ref_i = x_ref[:, i].copy()
            x_ref_i[8] = self.angle_diff(x_ref_i[8], x_ref_i[8]).full()[0] if isinstance(x_ref_i[8], ca.SX) else x_ref_i[8]
            
            # 设置阶段参考
            yref = np.concatenate([x_ref_i, self.nmpc.u_trim])
            self.acados_solver.set(i, 'yref', yref)
            
            # 控制率约束（前Nc步）
            if i < self.nmpc.Nc - 1:
                du = self.acados_solver.get(i, 'u') - self.acados_solver.get(i+1, 'u')
                du = np.clip(du, self.uav.du_min, self.uav.du_max)
                self.acados_solver.set(i+1, 'lbu', self.acados_solver.get(i, 'u') + self.uav.du_min)
                self.acados_solver.set(i+1, 'ubu', self.acados_solver.get(i, 'u') + self.uav.du_max)
        
        # 终端参考
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
            if status != 0:
                raise RuntimeError(f"acados求解失败，状态码：{status}")
            
            # 获取最优控制（第一个控制量）
            u_opt = self.acados_solver.get(0, 'u')
            u_opt = np.clip(u_opt, self.uav.u_min, self.uav.u_max)
            self.u_prev = u_opt
            return u_opt, True
        except Exception as e:
            print(f"⚠️ NMPC求解失败：{e}，使用悬停配平控制")
            return self.nmpc.u_trim, False

# ====================== 6. 主程序（保持不变） ======================
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
            self.t_history = []
            self.x_history = []
            self.u_history = []

    def generate_reference_trajectory(self, t_current):
        Np = self.nmpc_params.Np
        Ts = self.nmpc_params.Ts
        x_ref = np.zeros((12, Np + 1))
        radius = self.ref_radius
        h_max = self.ref_h_max
        v_z = self.ref_vz
        omega = 0.5
        def normalize_angle(angle):
            return (angle + np.pi) % (2 * np.pi) - np.pi
        for i in range(Np + 1):
            t_i = t_current + i * Ts
            x = radius * (np.cos(omega * t_i)-1)
            y = radius * np.sin(omega * t_i)
            z = min(v_z * t_i, h_max)
            vx = -radius * omega * np.sin(omega * t_i)
            vy = radius * omega * np.cos(omega * t_i)
            vz = v_z if z < h_max else 0.0
            ref_state = np.array([x,y,z,vx,vy,vz,0,0,normalize_angle(omega*t_i),0,0,omega])
            x_ref[:,i] = ref_state
        return x_ref

    def simulation_step(self, u, dt):
        # 保留原有的RK4仿真步
        k1 = ca.Function('f', [self.controller.acados_model.x, self.controller.acados_model.u], 
                         [self.controller.acados_model.f_expl_expr])(self.x_current, u).full().flatten()
        k2 = ca.Function('f', [self.controller.acados_model.x, self.controller.acados_model.u], 
                         [self.controller.acados_model.f_expl_expr])(self.x_current + dt/2*k1, u).full().flatten()
        k3 = ca.Function('f', [self.controller.acados_model.x, self.controller.acados_model.u], 
                         [self.controller.acados_model.f_expl_expr])(self.x_current + dt/2*k2, u).full().flatten()
        k4 = ca.Function('f', [self.controller.acados_model.x, self.controller.acados_model.u], 
                         [self.controller.acados_model.f_expl_expr])(self.x_current + dt*k3, u).full().flatten()
        self.x_current += dt/6*(k1+2*k2+2*k3+k4)

    def run_simulation(self, simulation_time=10):
        dt = self.nmpc_params.Ts
        t_total = 0
        while t_total < simulation_time:
            t_start = time.time()
            x_ref = self.generate_reference_trajectory(t_total)
            u_opt, success = self.controller.solve(self.x_current, x_ref)
            self.simulation_step(u_opt, dt)
            self.t_history.append(t_total)
            self.x_history.append(self.x_current.copy())
            self.u_history.append(u_opt.copy())
            if int(t_total/dt) % 10 == 0:
                print(f"时间:{t_total:.2f}s | 位置:{self.x_current[0:3].round(3)} | 求解:{'成功' if success else '失败'}")
            t_total += dt
            loop_time = time.time() - t_start
            if loop_time < dt:
                time.sleep(dt-loop_time)
        print("仿真完成！")
        self.plot_results()

    def plot_results(self):
        t = np.array(self.t_history)
        x = np.array(self.x_history)
        u = np.array(self.u_history)
        plt.figure(figsize=(10,6))
        plt.subplot(2,1,1)
        plt.plot(t, u[:,0], label='Fx')
        plt.plot(t, u[:,1], label='Fy')
        plt.plot(t, u[:,2], label='Fz')
        plt.title('期望推力')
        plt.legend()
        plt.grid()
        plt.subplot(2,1,2)
        plt.plot(t, u[:,3], label='tx')
        plt.plot(t, u[:,4], label='ty')
        plt.plot(t, u[:,5], label='tz')
        plt.title('期望力矩')
        plt.legend()
        plt.grid()
        x_ref_hist = np.array([self.generate_reference_trajectory(ti)[:, 0] for ti in t])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x[:, 0], x[:, 1], x[:, 2], label='飞行轨迹')
        ax.plot(x_ref_hist[:, 0], x_ref_hist[:, 1], x_ref_hist[:, 2], '--', label='参考轨迹')
        ax.scatter(x[0, 0], x[0, 1], x[0, 2], c='green', label='起点')
        ax.scatter(x[-1, 0], x[-1, 1], x[-1, 2], c='red', label='终点')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('3D飞行轨迹')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    host = UAVHostController(mode='simulation')
    host.run_simulation(simulation_time=5)