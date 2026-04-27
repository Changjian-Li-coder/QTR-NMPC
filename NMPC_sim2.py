# -*- coding: utf-8 -*-
# ====================== 1. 基础库导入 ======================
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import time
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


# ====================== 2. 无人机物理参数（修复约束） ======================
class UAVParams:
    def __init__(self):
        self.m = 1.66
        self.Ixx = 0.05
        self.Iyy = 0.05
        self.Izz = 0.08
        self.I = np.diag([self.Ixx, self.Iyy, self.Izz])
        self.g = 9.81
        self.nu = 6  # u=[Fx,Fy,Fz,τx,τy,τz]
        # 【修复】缩小合理约束：水平推力不能太大，Fz≥0
        self.u_min = np.array([-10, -10, -30, -5, -5, -5])
        self.u_max = np.array([10, 10, 30, 5, 5, 5])
        self.du_min = np.array([-2, -2, -2, -1, -1, -1])
        self.du_max = np.array([2, 2, 2, 1, 1, 1])
        # 【修复】放宽状态约束，避免越界
        self.x_min = np.array([-100, -100, -100, -10, -10, -10, 
                               np.deg2rad(-90), np.deg2rad(-90), np.deg2rad(-180), 
                               np.deg2rad(-600), np.deg2rad(-600), np.deg2rad(-600)])
        self.x_max = np.array([100, 100, 100, 10, 10, 10, 
                               np.deg2rad(90), np.deg2rad(90), np.deg2rad(180), 
                               np.deg2rad(600), np.deg2rad(600), np.deg2rad(600)])

# ====================== 3. NMPC超参数（修复权重+配平） ======================
class NMPCParams:
    def __init__(self):
        self.Ts = 0.05
        self.Np = 7   # 【修复】缩短预测时域，降低计算量
        self.Nc = 4
        # 【修复】缩小权重，避免数值奇异
        self.Q = np.diag([100, 100, 120, 1, 1, 1, 2, 2, 1, 0.05, 0.05, 0.05])
        self.P = self.Q * 1.2
        self.R = np.diag([0.1, 0.1, 0.1, 0.2, 0.2, 0.2])
        self.S = np.diag([1, 1, 1, 0.5, 0.5, 0.5])
        # 悬停配平：Fz=mg，其余为0
        hover_thrust = -1.66 * 9.81
        self.u_trim = np.array([0.0, 0.0, hover_thrust, 0.0, 0.0, 0.0])
        # print(f"悬停配平推力: {hover_thrust:.2f}N")

# ====================== 4. 动力学模型【核心修复！！！】 ======================
def build_dynamics_model(uav_params):
    nx = 12
    nu = 6
    x = ca.SX.sym('x', nx)
    u = ca.SX.sym('u', nu)
    
    pos = x[0:3]   # x,y,z
    vel = x[3:6]   # vx,vy,vz (惯性系)
    euler = x[6:9]# phi,theta,psi
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


    # ====================== 【核心修复】正确动力学公式 ======================
    F_B = u[0:3]      # 机体系期望推力 Fx,Fy,Fz
    tau_B = u[3:6]    # 机体系期望力矩 τx,τy,τz

    # 1. 位置导数（运动学）：dp/dt = vel (惯性系速度)
    dp_dt = vel
    # 2. 速度导数（动力学）：m*dv/dt = R_IB*F_B + m*g_I → dv/dt = R_IB@F_B/m + [0,0,-g]
    g_I = ca.SX([0, 0, uav_params.g])  # 惯性系重力
    dv_dt = ca.mtimes(R_IB, F_B) / uav_params.m + g_I
    # 3. 欧拉角导数
    deuler_dt = ca.mtimes(T, omega)
    # 4. 角速度导数（转动动力学，正确）
    cross_term = ca.cross(omega, ca.mtimes(uav_params.I, omega))
    domega_dt = ca.mtimes(ca.inv(uav_params.I), -cross_term + tau_B)

    dx_dt = ca.vertcat(dp_dt, dv_dt, deuler_dt, domega_dt)
    dynamics_func = ca.Function('dynamics', [x, u], [dx_dt], ['x', 'u'], ['dx_dt'])
    return dynamics_func, nx, nu

# ====================== 5. NMPC控制器（修复求解器+热启动） ======================
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

        self.X = self.opti.variable(self.nx, self.Np + 1)
        self.U = self.opti.variable(self.nu, self.Np)
        self.U_prev = self.opti.parameter(self.nu)
        self.X0 = self.opti.parameter(self.nx)
        self.X_ref = self.opti.parameter(self.nx, self.Np + 1)

        # 代价函数
        cost = 0
        def angle_diff(psi, psi_ref):
            return ca.atan2(ca.sin(psi - psi_ref), ca.cos(psi - psi_ref))
        
        x_err_term = self.X[:, -1] - self.X_ref[:, -1]
        x_err_term[8] = angle_diff(self.X[8,-1], self.X_ref[8,-1])
        cost += ca.mtimes([x_err_term.T, self.nmpc.P, x_err_term])

        for i in range(self.Np):
            x_err = self.X[:, i] - self.X_ref[:, i]
            x_err[8] = angle_diff(self.X[8,i], self.X_ref[8,i])
            cost += ca.mtimes([x_err.T, self.nmpc.Q, x_err])

        for i in range(self.Nc):
            cost += ca.mtimes([(self.U[:,i]-self.nmpc.u_trim).T, self.nmpc.R, (self.U[:,i]-self.nmpc.u_trim)])
            du = self.U[:,i] - (self.U_prev if i==0 else self.U[:,i-1])
            cost += ca.mtimes([du.T, self.nmpc.S, du])
        self.opti.minimize(cost)

        # 约束
        self.opti.subject_to(self.X[:,0] == self.X0)
        for i in range(self.Np):
            x_next = self.X[:,i] + self.Ts * self.dynamics_func(self.X[:,i], self.U[:,i])
            self.opti.subject_to(self.X[:,i+1] == x_next)
        
        for i in range(self.Np):
            self.opti.subject_to(self.U[:,i] >= self.uav.u_min)
            self.opti.subject_to(self.U[:,i] <= self.uav.u_max)
        
        for i in range(self.Nc):
            du = self.U[:,i] - (self.U_prev if i==0 else self.U[:,i-1])
            self.opti.subject_to(du >= self.uav.du_min)
            self.opti.subject_to(du <= self.uav.du_max)

        for i in range(self.Np+1):
            self.opti.subject_to(self.X[:,i] >= self.uav.x_min)
            self.opti.subject_to(self.X[:,i] <= self.uav.x_max)

        for i in range(self.Nc, self.Np):
            self.opti.subject_to(self.U[:,i] == self.U[:, self.Nc-1])

        # 【修复】求解器参数：增加迭代+降低精度+加速收敛 spral,mumps
        solver_opts = {
            "ipopt": {
                "max_iter": 150,          # 从50→150，足够收敛
                "print_level": 3,         # 关闭冗余打印
                "tol": 1e-1,              # 放宽精度
                "acceptable_tol": 1e-1,
                "warm_start_init_point": "yes",
                "hessian_approximation": "limited-memory",
                "linear_solver": "spral",
            },
            "print_time": 1
        }
        self.opti.solver("ipopt", solver_opts)
        self.u_prev = self.nmpc.u_trim

    def solve(self, x0, x_ref):
        # 【修复】安全约束检查
        x0 = np.clip(x0, self.uav.x_min, self.uav.x_max)
        try:
            self.opti.set_value(self.X0, x0)
            self.opti.set_value(self.X_ref, x_ref)
            self.opti.set_value(self.U_prev, self.u_prev)
            
            # 【修复】稳定热启动
            self.opti.set_initial(self.X, np.tile(x0, (self.Np+1,1)).T)
            self.opti.set_initial(self.U, np.tile(self.u_prev, (self.Np,1)).T)
            
            sol = self.opti.solve()
            u_opt = sol.value(self.U[:,0])
            u_opt = np.clip(u_opt, self.uav.u_min, self.uav.u_max)
            self.u_prev = u_opt
            return u_opt, True
        except:
            # 【修复】求解失败：直接输出悬停配平，保证安全
            print("⚠️ NMPC求解失败，使用悬停配平控制")
            return self.nmpc.u_trim, False

# ====================== 6. 主程序（完全不变） ======================
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
        k1 = self.controller.dynamics_func(self.x_current, u).full().flatten()
        k2 = self.controller.dynamics_func(self.x_current + dt/2*k1, u).full().flatten()
        k3 = self.controller.dynamics_func(self.x_current + dt/2*k2, u).full().flatten()
        k4 = self.controller.dynamics_func(self.x_current + dt*k3, u).full().flatten()
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