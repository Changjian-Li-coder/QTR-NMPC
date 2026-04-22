#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ====================== 1. 基础库导入 ======================
import numpy as np
import casadi as ca
import rospy
from mavros_msgs.msg import State
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
from tf.transformations import euler_from_quaternion
import time
import os
import matplotlib.pyplot as plt  # 新增绘图库
os.environ["ACADOS_NO_TEMPLATES"] = "1"
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel

# 解决matplotlib在ROS中的绘图问题（非阻塞）
plt.rcParams.update({'font.size': 10})
plt.switch_backend('TkAgg')  # 或使用'Qt5Agg'

# ====================== 2. 无人机物理参数（完全复用sim_NMPC3.py） ======================
class UAVParams:
    def __init__(self):
        self.m = 2.4
        self.L = 0.18
        self.Ixx = 0.006
        self.Iyy = 0.007
        self.Izz = 0.015
        self.I = np.diag([self.Ixx, self.Iyy, self.Izz])
        self.g = 9.81
        self.thrust = self.m * self.g * 1.2
        self.acceleration_xy_max = 2
        self.acceleration_z_max = 1
        self.roll_pitch_acceleration_max = 3.0
        self.yaw_acceleration_max = 1.0
        self.nu = 6  # u=[Fx,Fy,Fz,τx,τy,τz] 单位：推力：N；力矩：mN·m
        self.u_min = np.array([-10, -10, -self.thrust, -6, -6, -6])
        self.u_max = np.array([10, 10, self.thrust, 6, 6, 6])
        self.du_min = np.array([-5, -5, -5, -3, -3, -3])
        self.du_max = np.array([5, 5, 5, 3, 3, 3])
        self.x_min = np.array([-2, -2, -1, -2, -2, -0.5,
                               np.deg2rad(-90), np.deg2rad(-90), np.deg2rad(-180),
                               np.deg2rad(-60), np.deg2rad(-60), np.deg2rad(-60)])
        self.x_max = np.array([2, 2, 1.5, 2, 2, 0.5,
                               np.deg2rad(90), np.deg2rad(90), np.deg2rad(180),
                               np.deg2rad(60), np.deg2rad(60), np.deg2rad(60)])

# ====================== 3. NMPC超参数（完全复用sim_NMPC3.py） ======================
class NMPCParams:
    def __init__(self):
        self.Ts = 0.01
        self.Np = 25   # 预测时域
        self.Nc = 15   # 控制时域
        self.Q = np.diag([45, 45, 18,
                          15, 15, 25,
                          5, 5, 2,
                          18, 18, 4])  # 状态权重（位置/速度/姿态/角速度）
        self.P = self.Q * 0.8  # 终端权重（更重视终端状态）
        self.R = np.diag([0.6, 0.6, 5.0, 30, 30, 25])  # 控制权重（推力/力矩）
        self.S = np.diag([0.35, 0.35, 1, 15, 15, 10])  # 控制率权重（仅前Nc步）
        # 悬停配平：Fz=mg，其余为0
        hover_thrust = 2.4 * 9.81
        self.u_trim = np.array([0.0, 0.0, hover_thrust, 0.0, 0.0, 0.0])

# ====================== 4. 动力学模型（完全复用sim_NMPC3.py） ======================
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
    # 机体系 → 惯性系 旋转矩阵 R^I_B
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
    F_B = u[0:3]      # 机体系期望推力 Fx,Fy,Fz（单位：N）
    tau_B = u[3:6]    # 机体系期望力矩 τx,τy,τz（单位：mN·m）
    # ========== 关键修改：力矩单位转换（mN·m → N·m） ==========
    # tau_B_SI = tau_B / 1#000.0  # 转换为国际单位制的N·m

    # 1. 位置导数（运动学）
    dp_dt = vel
    # 2. 速度导数（动力学）
    g_I = ca.SX([0, 0, -uav_params.g])  # 惯性系重力
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

# ====================== 5. NMPC控制器（修正维度问题） ======================
class NMPCController:
    def __init__(self, uav_params, nmpc_params):
        self.uav = uav_params
        self.nmpc = nmpc_params
        self.acados_model, self.nx, self.nu = build_acados_dynamics_model(uav_params)
        self.ny = self.nx + self.nu +self.nu  # 正确维度：12+6+6=24  12维状态，6维控制量，6维控制变化量
        self.ny_e = self.nx  # 终端代价维度：仅状态12维
        self.u_prev = self.nmpc.u_trim

        # 初始化OCP问题
        self.ocp = AcadosOcp()
        self.ocp.model = self.acados_model
        self.ocp.dims.N = self.nmpc.Np  # 预测时域
        self.ocp.solver_options.tf = self.nmpc.Ts * self.nmpc.Np  # 总预测时间

        # 状态和控制变量初始化
        self.ocp.constraints.x0 = np.zeros(self.nx)  # 保留默认值，首次odom会覆盖
        self.ocp.constraints.lbx = self.uav.x_min  # 状态下界
        self.ocp.constraints.ubx = self.uav.x_max  # 状态上界
        self.ocp.constraints.lbu = self.uav.u_min  # 控制下界
        self.ocp.constraints.ubu = self.uav.u_max  # 控制上界
        self.ocp.constraints.idxbu = np.arange(self.nu)  # 控制变量索引
        self.ocp.constraints.idxbx = np.arange(self.nx)  # 状态变量索引

        # ========== 修正：代价函数配置（核心修复维度问题） ==========
        self.ocp.cost.cost_type = 'LINEAR_LS'
        self.ocp.cost.cost_type_e = 'LINEAR_LS'

        # 1. 阶段代价权重W：维度必须是(ny, ny) = (24,24) = diag(Q, R, S)
        self.ocp.cost.W = np.block([
            [self.nmpc.Q, np.zeros((self.nx, self.nu)), np.zeros((self.nx, self.nu))],
            [np.zeros((self.nu, self.nx)), self.nmpc.R, np.zeros((self.nu, self.nu))],
            [np.zeros((self.nu, self.nx)), np.zeros((self.nu, self.nu)), self.nmpc.S]
        ])

        # 2. 终端代价权重：维度(ny_e, ny_e) = (12,12)
        self.ocp.cost.W_e = self.nmpc.P

        # 3. Vx/Vu矩阵：y = Vx·x + Vu·u（y维度18）
        # Vx: 24x12，对应y = [x; 0; 0]
        self.ocp.cost.Vx = np.vstack([
            np.eye(self.nx),
            np.zeros((self.nu, self.nx)),
            np.zeros((self.nu, self.nx))
        ])

        # Vu: 24x6，对应y = [0; u; u]
        self.ocp.cost.Vu = np.vstack([
            np.zeros((self.nx, self.nu)),
            np.eye(self.nu),
            np.eye(self.nu)
        ])

        # 4. 终端Vx矩阵：仅状态
        self.ocp.cost.Vx_e = np.eye(self.nx)  # 12x12
        self.ocp.cost.Vu_e = np.zeros((self.ny_e, self.nu))  # 终端无控制，补0

        # 5. 参考值初始化：维度匹配ny=24（12状态+6控制+6控制变化）
        self.ocp.cost.yref = np.zeros(self.ny)  # 24维
        self.ocp.cost.yref_e = np.zeros(self.ny_e)  # 12维

        # ========== 控制增量惩罚：换一种合规方式（约束+代价） ==========
        # 控制增量（Δu）惩罚不通过扩展W实现，而是通过：
        # 1. 控制率约束（原逻辑保留）；2. 对u的增量单独惩罚（可选：增加R的权重）

        # 求解器配置
        self.ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        self.ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        self.ocp.solver_options.nlp_solver_step_length = 0.3  # 降低步长，更平滑
        self.ocp.solver_options.min_step = 1e-4              # 放宽最小步长
        self.ocp.solver_options.nlp_solver_tol_eq = 1e-2     # 放宽收敛精度（无人机足够用）
        self.ocp.solver_options.nlp_solver_tol_ineq = 1e-2
        self.ocp.solver_options.integrator_type = 'ERK'
        self.ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        self.ocp.solver_options.print_level = 1
        self.ocp.solver_options.nlp_solver_max_iter = 50

        # 初始化求解器
        self.acados_solver = AcadosOcpSolver(self.ocp, json_file='acados_ocp.json')

        # 控制历史
        self.solve_time_history = []

    def normalize_angle_np(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def solve(self, x0, x_ref):
        # ========== 新增：开始计时（精确到微秒） ==========
        solve_start = time.perf_counter()

        # 安全约束检查
        x0 = np.clip(x0, self.uav.x_min, self.uav.x_max)

        # 更新初始状态
        self.acados_solver.set(0, 'lbx', x0)
        self.acados_solver.set(0, 'ubx', x0)

        # # ========== 修改4：先热启动（为yref提供u_{i-1}的值） ==========
        # for i in range(self.nmpc.Np):
        #     self.acados_solver.set(i, 'u', self.u_prev)
        # for i in range(self.nmpc.Np + 1):
        #     self.acados_solver.set(i, 'x', x0)

        # 更新参考轨迹和代价函数
        for i in range(self.nmpc.Np):
            x_ref_i = x_ref[:, i].copy()
            x_ref_i[8] = self.normalize_angle_np(x_ref_i[8])

            # 获取u_{i-1}：i=0时用上一次的u_prev，i>=1时用热启动的u_{i-1}
            if i == 0:
                u_prev_i = self.u_prev.copy()
            else:
                u_prev_i = self.acados_solver.get(i-1, 'u')

            # 设置阶段参考：12状态 + 6控制 + 6控制变化 → 18维（匹配ny）
            yref = np.concatenate([x_ref_i, self.nmpc.u_trim, u_prev_i])  # 18维
            self.acados_solver.set(i, 'yref', yref)

            # 控制率约束（前Nc步）：保留原逻辑，这是Δu的约束，而非代价
            if i < self.nmpc.Nc - 1:
                # 先获取上一步的控制量（避免索引越界）
                u_prev_step = self.acados_solver.get(i, 'u') if i >=0 else self.u_prev
                # 设置Δu约束：u_{i+1} ∈ [u_i + du_min, u_i + du_max]
                self.acados_solver.set(i+1, 'lbu', u_prev_step + self.uav.du_min)
                self.acados_solver.set(i+1, 'ubu', u_prev_step + self.uav.du_max)

        # 终端参考（仅状态）→ 12维（匹配ny_e）
        x_ref_e = x_ref[:, -1].copy()
        x_ref_e[8] = self.normalize_angle_np(x_ref_e[8])
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
            return self.nmpc.u_trim, False, solve_time
# ====================== 6. PI控制器 ======================
class PIController:
    def __init__(self, kp, ki, int_limit, speed):
        self.kp = kp
        self.ki = ki
        self.int_limit = int_limit
        self.speed = speed
        self.error_integral = 0.0

    def compensate(self, current, target, is_armed):
        # 误差
        error = target - current

        # 积分
        if is_armed:
            self.error_integral += error * 0.01 * self.speed  # 假设控制周期为10ms

        # 积分限幅
        self.error_integral = np.clip(self.error_integral, -self.int_limit, self.int_limit)

        # PI输出
        u_pi = self.kp * error + self.ki * self.error_integral

        return u_pi
# ====================== 7. 实物飞行主控制器（融合ROS逻辑） ======================
class UAVHostController:
    def __init__(self):
        self.uav_params = UAVParams()
        self.nmpc_params = NMPCParams()
        self.controller = NMPCController(self.uav_params, self.nmpc_params)
        self.x_pi_controller = PIController(kp=0.5, ki=2.0, int_limit=1.5, speed=self.nmpc_params.Ts * 1.5)
        self.y_pi_controller = PIController(kp=0.5, ki=2.0, int_limit=1.5, speed=self.nmpc_params.Ts * 1.5)
        self.z_pi_controller = PIController(kp=0.5, ki=2.0, int_limit=1.5, speed=self.nmpc_params.Ts * 1.5)
        self.roll_pi_controller = PIController(kp=0.5, ki=2.0, int_limit=np.deg2rad(30), speed=self.nmpc_params.Ts * 1.5)
        self.pitch_pi_controller = PIController(kp=0.5, ki=2.0, int_limit=np.deg2rad(30), speed=self.nmpc_params.Ts * 1.5)
        self.yaw_pi_controller = PIController(kp=0.5, ki=2.0, int_limit=np.deg2rad(30), speed=self.nmpc_params.Ts * 1.5)

        # ROS状态变量
        self.x_current = None
        self.current_state = None
        self.state_ready = False
        self.t0 = None
        self.ref_pos_hover = None
        self.ref_yaw_hover  = None
        self.reference_trajectory = None  # 存储订阅到的参考轨迹

        # 新增：数据记录相关
        self.is_armed = False          # 当前是否解锁
        self.is_recording = False       # 是否正在记录数据
        self.recorded_data = {
            'time': [],          # 时间戳
            'state': [],         # 12维状态
            'control': [],       # 6维控制量
            'solve_time': [],    # 求解时间
            'solve_success': []  # 求解是否成功
        }
        # ROS通信（参考NMPC3.py）
        self.pose_sub = rospy.Subscriber("/mavros/local_position/odom", Odometry, self.odom_callback, queue_size=5)
        self.state_sub = rospy.Subscriber("/mavros/state", State, self.state_callback, queue_size=5)
        self.trajectory_sub = rospy.Subscriber("/uav/reference_trajectory", Float64MultiArray, self.trajectory_callback, queue_size=5)
        self.control_pub = rospy.Publisher("/nmpc/control_cmd", Float64MultiArray, queue_size=5)

    def odom_callback(self, msg):
        """从/mavros/local_position/odom更新12维状态（坐标转换完全参考NMPC3.py）"""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        vz = msg.twist.twist.linear.z
        q = msg.pose.pose.orientation
        phi, theta, psi = euler_from_quaternion([q.x, q.y, q.z, q.w])

        # 关键：与NMPC3.py保持一致的坐标符号转换
        p_rate = msg.twist.twist.angular.x
        q_rate = msg.twist.twist.angular.y
        r_rate = msg.twist.twist.angular.z

        self.x_current = np.array([x, y, z, vx, vy, vz, phi, theta, psi, p_rate, q_rate, r_rate])
        if not self.state_ready:
            # 1. 赋值完整12维初始状态（而非仅位置）
            self.controller.ocp.constraints.x0 = self.x_current.copy()
            # 2. 同步更新acados_solver的初始状态约束（关键！）
            self.controller.acados_solver.set(0, 'lbx', self.x_current)
            self.controller.acados_solver.set(0, 'ubx', self.x_current)
            # rospy.loginfo(f"初始化NMPC初始状态x0：\n{np.array2string(self.x_current, precision=3, suppress_small=True)}")
        self.state_ready = True

        # 初始化时间戳
        if self.t0 is None:
            self.t0 = rospy.Time.now().to_sec()

    def trajectory_callback(self, msg):
        """订阅参考轨迹话题，重塑为12×(Np+1)数组"""
        try:
            traj_flat = np.array(msg.data)
            if traj_flat.size != self.controller.nx:
                rospy.logwarn(f"⚠️ 轨迹数据维度不匹配：收到{traj_flat.size}，期望{self.controller.nx}")
                return
            self.reference_trajectory = traj_flat
        except Exception as e:
            rospy.logwarn(f"⚠️ 解析轨迹数据失败：{e}")

    def state_callback(self, msg):
        """监听mavros/state，检测解锁/上锁状态，触发记录开始/结束"""
        self.current_state = msg
        prev_armed = self.is_armed
        self.is_armed = msg.armed

        # 解锁：开始记录
        if self.is_armed and not prev_armed:
            rospy.loginfo("✅ 无人机解锁，开始记录数据！")
            self.z_error_integral = 0.0
            self.is_recording = True
            self.recorded_data = {  # 重置记录数据
                'time': [],
                'state': [],
                'control': [],
                'solve_time': [],
                'solve_success': []
            }
            self.t0 = rospy.Time.now().to_sec()  # 重置时间戳

        # 上锁：停止记录并绘图
        if not self.is_armed and prev_armed:
            rospy.loginfo("🛑 无人机上锁，停止记录并绘图！")
            self.is_recording = False
            if len(self.recorded_data['time']) > 0:
                self.plot_recorded_data()  # 绘制数据
            else:
                rospy.logwarn("⚠️ 无记录数据，跳过绘图")

    def generate_reference_trajectory(self, t_current):
        """替换为订阅外部轨迹的逻辑"""
        Np = self.nmpc_params.Np
        # 兜底：未收到外部轨迹时，生成悬停轨迹（兼容原有逻辑）
        if self.reference_trajectory is None:
            rospy.logwarn_throttle(1, "⚠️ 未收到外部参考轨迹，使用本地悬停轨迹")
            x_ref = np.zeros((12, Np + 1))
            if self.ref_pos_hover is None:
                self.ref_pos_hover = self.x_current[0:3].copy()
                self.ref_pos_hover[2] = 0.55
            if self.ref_yaw_hover is None:
                self.ref_yaw_hover = self.x_current[8].copy()
            ref_pos = self.ref_pos_hover.copy()
            ref_vel = np.array([0.0, 0.0, 0.0])
            ref_euler = np.array([0.0, 0.0, self.ref_yaw_hover])
            ref_omega = np.array([0.0, 0.0, 0.0])
            ref_state = np.concatenate([ref_pos, ref_vel, ref_euler, ref_omega])
            for i in range(Np + 1):
                x_ref[:, i] = ref_state
            return x_ref
        # 直接返回订阅到的外部轨迹
        else:
            x_ref = np.zeros((12, Np + 1))
            start_state = self.x_current.copy()
            target_state = self.reference_trajectory[0:12].copy()

            for i in range(Np + 1):
                alpha = i / self.nmpc_params.Np
                # x_ref[:, i] = (1 - alpha) * start_state + alpha * target_state
                x_ref[:, i] = target_state  # 直接使用目标状态作为参考轨迹
                x_ref[8, i] = (x_ref[8, i] + np.pi) % (2 * np.pi) - np.pi  # 确保yaw角连续
            return x_ref

    def publish_control(self, u):
        """发布NMPC控制量到ROS话题"""
        msg = Float64MultiArray()
        msg.data = u.astype(float).tolist()
        self.control_pub.publish(msg)

    def plot_recorded_data(self):
        """绘制记录的无人机状态/控制量曲线"""
        # 提取数据
        time_arr = np.array(self.recorded_data['time'])
        state_arr = np.array(self.recorded_data['state'])  # (N,12)
        control_arr = np.array(self.recorded_data['control'])  # (N,6)
        solve_time_arr = np.array(self.recorded_data['solve_time'])
        solve_success_arr = np.array(self.recorded_data['solve_success'])

        # 创建子图
        fig, axes = plt.subplots(4, 2, figsize=(16, 12))
        fig.suptitle('UAV NMPC Flight Data', fontsize=16)

        # 1. 位置 (x,y,z)
        ax1 = axes[0,0]
        ax1.plot(time_arr, state_arr[:,0], label='x [m]', linewidth=1.5)
        ax1.plot(time_arr, state_arr[:,1], label='y [m]', linewidth=1.5)
        # ax1.plot(time_arr, state_arr[:,2], label='z [m]', linewidth=1.5)
        ax1.set_title('Position')
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Position [m]')
        ax1.legend()
        ax1.grid(True)

        # 3. 姿态 (phi,theta,psi) → 转换为角度
        ax3 = axes[0,1]
        ax3.plot(time_arr, np.rad2deg(state_arr[:,6]), label='roll [deg]', linewidth=1.5)
        ax3.plot(time_arr, np.rad2deg(state_arr[:,7]), label='pitch [deg]', linewidth=1.5)
        # ax3.plot(time_arr, np.rad2deg(state_arr[:,8]), label='yaw [deg]', linewidth=1.5)
        ax3.set_title('Attitude (Euler Angles)')
        ax3.set_xlabel('Time [s]')
        ax3.set_ylabel('Angle [deg]')
        ax3.legend()
        ax3.grid(True)

        # 2. 速度 (vx,vy,vz)
        ax2 = axes[1,0]
        ax2.plot(time_arr, state_arr[:,3], label='vx [m/s]', linewidth=1.5)
        ax2.plot(time_arr, state_arr[:,4], label='vy [m/s]', linewidth=1.5)
        ax2.plot(time_arr, state_arr[:,5], label='vz [m/s]', linewidth=1.5)
        ax2.set_title('Velocity')
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Velocity [m/s]')
        ax2.legend()
        ax2.grid(True)

        # 4. 角速度 (p,q,r) → 转换为角度/秒
        ax4 = axes[1,1]
        ax4.plot(time_arr, np.rad2deg(state_arr[:,9]), label='p [deg/s]', linewidth=1.5)
        ax4.plot(time_arr, np.rad2deg(state_arr[:,10]), label='q [deg/s]', linewidth=1.5)
        ax4.plot(time_arr, np.rad2deg(state_arr[:,11]), label='r [deg/s]', linewidth=1.5)
        ax4.set_title('Angular Velocity')
        ax4.set_xlabel('Time [s]')
        ax4.set_ylabel('Angular Velocity [deg/s]')
        ax4.legend()
        ax4.grid(True)

        # 5. 推力 (Fx,Fy,Fz)
        ax5 = axes[2,0]
        # ax5.plot(time_arr, control_arr[:,0], label='Fx [N]', linewidth=1.5)
        # ax5.plot(time_arr, control_arr[:,1], label='Fy [N]', linewidth=1.5)
        ax5.plot(time_arr, control_arr[:,2], label='Fz [N]', linewidth=1.5)
        ax5.set_title('Thrust (Body Frame)')
        ax5.set_xlabel('Time [s]')
        ax5.set_ylabel('Force [N]')
        ax5.legend()
        ax5.grid(True)

        # 6. 力矩 (τx,τy,τz)
        ax6 = axes[2,1]
        ax6.plot(time_arr, control_arr[:,3], label='τx [N·m]', linewidth=1.5)
        ax6.plot(time_arr, control_arr[:,4], label='τy [N·m]', linewidth=1.5)
        ax6.plot(time_arr, control_arr[:,5], label='τz [N·m]', linewidth=1.5)
        ax6.set_title('Torque (Body Frame)')
        ax6.set_xlabel('Time [s]')
        ax6.set_ylabel('Torque [N·m]')
        ax6.legend()
        ax6.grid(True)

        # 7. 求解时间
        ax7 = axes[3,0]
        ax7.plot(time_arr, solve_time_arr * 1000, label='Solve Time [ms]', color='orange', linewidth=1.5)
        ax7.set_title('NMPC Solve Time')
        ax7.set_xlabel('Time [s]')
        ax7.set_ylabel('Time [ms]')
        ax7.legend()
        ax7.grid(True)

        # 8. 求解成功率
        ax8 = axes[3,1]
        ax8.plot(time_arr, solve_success_arr, label='Solve Success', color='green', linewidth=1.5, drawstyle='steps-post')
        ax8.set_title('NMPC Solve Success (1=Success, 0=Fail)')
        ax8.set_xlabel('Time [s]')
        ax8.set_ylabel('Success Flag')
        ax8.set_ylim(-0.1, 1.1)
        ax8.legend()
        ax8.grid(True)

        # 调整布局并保存/显示
        plt.tight_layout()
        plt.savefig(f"uav_nmpc_flight_data_{int(time.time())}.png", dpi=150)
        rospy.loginfo("📊 飞行数据图已保存！")
        plt.show(block=True)  # 阻塞显示，关闭后继续程序


    def run(self):
        """实物飞行主循环"""
        rospy.loginfo("NMPC_acados实物飞行控制器启动，等待/mavros/local_position/odom状态...")
        rate = rospy.Rate(120)

        while not rospy.is_shutdown():
            if not self.state_ready or self.t0 is None:
                rate.sleep()
                continue

            # 计算当前时间
            t_current = rospy.Time.now().to_sec() - self.t0

            # 1. 生成参考轨迹
            x_ref = self.generate_reference_trajectory(t_current)
            rospy.loginfo_throttle(1, f"参考轨迹x_ref    :{np.array2string(x_ref[:, -1], precision=3, floatmode='fixed', suppress_small=True, max_line_width=1000)}")
            # rospy.loginfo_throttle(1, f"当前状态x_current:{np.array2string(self.x_current, precision=3, floatmode='fixed', suppress_small=True, max_line_width=1000)}")

            # 2. 求解NMPC
            u_opt, success, solve_time = self.controller.solve(self.x_current, x_ref)
            send_u = u_opt.copy()

            # 3. PI补偿
            x_compensated = self.x_pi_controller.compensate(self.x_current[0], x_ref[0, 0], self.is_armed)  # x位置补偿
            y_compensated = self.y_pi_controller.compensate(self.x_current[1], x_ref[1, 0], self.is_armed)  # y位置补偿
            z_compensated = self.z_pi_controller.compensate(self.x_current[2], x_ref[2, 0], self.is_armed)  # z位置补偿
            roll_compensated = self.roll_pi_controller.compensate(self.x_current[6], x_ref[6, 0], self.is_armed)  # roll角补偿
            pitch_compensated = self.pitch_pi_controller.compensate(self.x_current[7], x_ref[7, 0], self.is_armed)  # pitch角补偿
            yaw_compensated = self.yaw_pi_controller.compensate(self.x_current[8], x_ref[8, 0], self.is_armed)  # yaw角补偿

            # 4. 将PI补偿叠加到NMPC控制量上（仅在解锁时叠加，锁定时保持原NMPC输出）
            send_u[0] += x_compensated
            send_u[1] += y_compensated
            send_u[2] += z_compensated
            send_u[3] += roll_compensated
            send_u[4] += pitch_compensated
            send_u[5] += yaw_compensated

            send_u[3] = send_u[3] - 0.1  # 补静差

            # 5. 发布控制指令（仅在求解成功时发布，失败则保持上一帧或配平）
            self.publish_control(send_u)
            # if success :  #and self.is_recording
            #     rospy.loginfo_throttle(0.1, f"求解:{'成功' if success else '失败'}"
            #                                 f"控制指令: {np.array2string(send_u, precision=4, floatmode='fixed', suppress_small=True, max_line_width=1000)}")
            # 6. 记录数据（仅在解锁时）
            if self.is_recording and self.x_current is not None:
                self.recorded_data['time'].append(t_current)
                self.recorded_data['state'].append(self.x_current.copy())
                self.recorded_data['control'].append(u_opt.copy())
                self.recorded_data['solve_time'].append(solve_time)
                self.recorded_data['solve_success'].append(1 if success else 0)

            rate.sleep()

# ====================== 8. 程序入口 ======================
if __name__ == "__main__":
    try:
        rospy.init_node("nmpc_acados_flight_controller", anonymous=False)
        host_controller = UAVHostController()
        host_controller.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("程序中断")
