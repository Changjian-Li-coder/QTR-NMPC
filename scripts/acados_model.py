#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from acados_template import AcadosModel, AcadosOcpSolver, AcadosOcp
import numpy as np
import casadi as ca
import time
import rospy
from uav_config import UAVParams
from nmpc_config import NMPCParams
from dynamics_model import Dynamics_Model

class AcadosModelBuilder:
    def __init__(self):
        self.uav_params = UAVParams()
        self.nmpc_params = NMPCParams()
        self.dynamics_model = Dynamics_Model()
        self.nx = self.nmpc_params.nx
        self.nu = self.nmpc_params.nu
        self.x, self.x_dot, self.u, self.xa_ref, self.u_prev_sym, self.f_expl, self.f_impl, self.cost_expr, self.cost_expr_e = self.dynamics_model.build_dynamics_model()
        self.ny = self.cost_expr.size()[0]      # 18 (p_err, v_err, euler_err, u_err)
        self.ny_e = self.cost_expr_e.size()[0]  # 12 (p_err, v_err, euler_err)
        self.u_prev = self.nmpc_params.u_trim
        self.x_integral = np.zeros(self.nmpc_params.nx_integral)

    def create_acados_model(self):
        model = AcadosModel()
        model.name = 'uav_model'
        # ---- 定义模型变量 ----
        model.x = self.x  # 增广状态
        model.u = self.u  # 控制量
        model.xdot = self.x_dot  # 增广状态导数

        # 这里的参数只保留动力学里真正用到的参考轨迹 x_ref。
        # 控制率惩罚不放进 model.p，而是通过 cost_y_expr 的第二个 u 项配合 yref = u_prev 实现。
        model.p = self.xa_ref

        model.f_expl_expr = self.f_expl # 显式动力学表达式
        model.f_impl_expr = self.f_impl # 隐式动力学表达式（f_impl = x_dot - f_expl）

        # ----- 求解器参数 -----
        ocp = AcadosOcp()
        ocp.model = model
        ocp.solver_options.N_horizon = self.nmpc_params.Np
        ocp.solver_options.qp_solver_cond_N = self.nmpc_params.Np
        ocp.solver_options.tf = self.nmpc_params.Ts * self.nmpc_params.Np

        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.nlp_solver_step_length = 0.3
        ocp.solver_options.min_step = 1e-4
        ocp.solver_options.nlp_solver_tol_eq = 1e-4
        ocp.solver_options.nlp_solver_tol_ineq = 1e-4
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        ocp.solver_options.print_level = 0
        ocp.solver_options.nlp_solver_max_iter = 50

        # ----求解器代价设置----
        ocp.cost_type = 'LINEAR_LS'
        ocp.cost_type_e = 'LINEAR_LS'
        ocp.cost.W = np.diag(np.hstack([np.diag(self.nmpc_params.Q), np.diag(self.nmpc_params.R), np.diag(self.nmpc_params.S)])) # 过程权重 跟cost_expr中的项一一对应
        ocp.cost.W_e = self.nmpc_params.P  # 终端权重 与cost_y_expr_e中的项一一对应

        # Vx矩阵：30x18 → y = Vx·xa + Vu·u
        ocp.cost.Vx = np.vstack([
            np.eye(self.nx),                  # 增广状态（18维）
            np.zeros((self.nu, self.nx)),     # 控制量占位（6维）
            np.zeros((self.nu, self.nx))      # 控制率占位（6维）
        ])

        # Vu矩阵：30x6 → y = Vx·xa + Vu·u
        ocp.cost.Vu = np.vstack([
            np.zeros((self.nx, self.nu)),     # 增广状态占位（18维）
            np.eye(self.nu),                  # 控制量（6维）
            np.eye(self.nu)                   # 控制率（6维）
        ])

        # 终端Vx矩阵：18x18（仅增广状态）
        ocp.cost.Vx_e = np.eye(self.nx)
        ocp.cost.Vu_e = np.zeros((self.ny_e, self.nu))  # 终端无控制
        
        ocp.cost.yref = np.zeros(self.ny)
        ocp.cost.yref_e = np.zeros(self.ny_e)

        # ---- 求解器约束 ----
        ocp.constraints.x0 = np.zeros(self.nx)  # 初始增广状态
        ocp.constraints.lbx = self.nmpc_params.x_min  # 18维状态下界
        ocp.constraints.ubx = self.nmpc_params.x_max  # 18维状态上界
        ocp.constraints.lbu = self.nmpc_params.u_min  # 6维控制下界
        ocp.constraints.ubu = self.nmpc_params.u_max  # 6维控制上界
        ocp.constraints.idxbu = np.arange(self.nu)
        ocp.constraints.idxbx = np.arange(self.nx)

        # model.p 只包含参考轨迹 x_ref
        ocp.parameter_values = np.zeros(self.nmpc_params.nx)

        self.ocp = ocp
        self.acados_solver = AcadosOcpSolver(self.ocp, json_file='acados_ocp.json')
        self.solve_time_history = []
        
    def normalize_angle_np(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def update_external_integral(self, x, x_ref_0):
        """使用当前实测状态与当前参考值，更新外部积分状态。"""
        dt = self.nmpc_params.Ts

        pos_err = x[0:3] - x_ref_0[0:3]
        att_err = np.array([
            self.normalize_angle_np(x[6] - x_ref_0[6]),
            self.normalize_angle_np(x[7] - x_ref_0[7]),
            self.normalize_angle_np(x[8] - x_ref_0[8]),
        ])

        self.x_integral = self.x_integral + np.hstack([pos_err, att_err]) * dt
        self.x_integral = np.clip(
            self.x_integral,
            0.8*self.nmpc_params.x_min[self.nmpc_params.nx_original:],
            0.8*self.nmpc_params.x_max[self.nmpc_params.nx_original:],
        )
        
    def solve(self, x, x_ref):
        """
        求解积分增广NMPC
        :param x: 原12维状态（当前无人机状态）
        :param x_ref: 参考轨迹 (12, Np+1)
        :return: 最优控制量, 求解状态, 求解时间
        """
        solve_start = time.perf_counter()
        # 先用“当前实测状态 + 当前参考点”更新外部积分状态
        # self.update_external_integral(x, x_ref[:, -1])  # x, x_target

        # 构建增广初始状态：原状态 + 积分状态
        x0 = np.hstack([x, self.x_integral])  # 初始积分状态为0
        x0 = np.clip(x0, self.nmpc_params.x_min, self.nmpc_params.x_max)

        # 更新初始状态约束
        self.acados_solver.set(0, 'lbx', x0)
        self.acados_solver.set(0, 'ubx', x0)
        # 热启动
        for i in range(self.nmpc_params.Np):
            self.acados_solver.set(i, 'u', self.u_prev)
        for i in range(self.nmpc_params.Np + 1):
            self.acados_solver.set(i, 'x', x0)

        # --------------------- 更新参考轨迹和参数 ---------------------
        # 获取当前偏航，用于参考解缠绕（避免±π跳变导致代价函数误差虚大）
        yaw_current = x[8]

        for i in range(self.nmpc_params.Np):
            # 参考轨迹（原12维）
            x_ref_i = x_ref[:, i].copy()
            # 解缠绕参考偏航到当前偏航附近：保证参考yaw在 [yaw_current-π, yaw_current+π] 范围内
            x_ref_i[8] = yaw_current + self.normalize_angle_np(x_ref_i[8] - yaw_current)

            # 构建阶段参考值yref：18维增广状态参考 + 6维控制参考 + 6维控制率参考
            # 增广状态参考：原状态参考 + 积分状态参考（积分项期望为0，无静差）
            xa_ref_i = np.hstack([x_ref_i, np.zeros(self.nmpc_params.nx_integral)])  # 18维增广状态参考
            u_ref_i = self.nmpc_params.u_trim
            du_ref_i = self.u_prev  # 控制率参考
            yref = np.concatenate([xa_ref_i, u_ref_i, du_ref_i]) 
            self.acados_solver.set(i, 'yref', yref)
            self.acados_solver.set(i, 'p', xa_ref_i)  # 参数：仅参考轨迹

            # 控制率约束（前Nc步）
            if i < self.nmpc_params.Nc - 1:
                u_prev_step = self.acados_solver.get(i, 'u') if i >=0 else self.u_prev
                self.acados_solver.set(i+1, 'lbu', u_prev_step + self.nmpc_params.du_min)
                self.acados_solver.set(i+1, 'ubu', u_prev_step + self.nmpc_params.du_max)

        # 终端阶段
        x_ref_e = x_ref[:, -1].copy()
        # 终端参考偏航同样解缠绕到当前偏航附近
        x_ref_e[8] = yaw_current + self.normalize_angle_np(x_ref_e[8] - yaw_current)
        # 终端参考值：增广状态参考（原状态参考+积分0）
        xa_ref_e = np.hstack([x_ref_e, np.zeros(self.nmpc_params.nx_integral)])
        self.acados_solver.set(self.nmpc_params.Np, 'yref', xa_ref_e)
        self.acados_solver.set(self.nmpc_params.Np, 'p', xa_ref_e)  # 参数：仅终端参考轨迹

        # --------------------- 求解OCP ---------------------
        try:
            status = self.acados_solver.solve()
            solve_time = time.perf_counter() - solve_start
            self.solve_time_history.append(solve_time)

            if status != 0:
                self._diagnose_failure(status, x0)
                raise RuntimeError(f"acados求解失败，状态码：{status}")

            # 获取最优控制
            u_opt = self.acados_solver.get(1, 'u')
            u_opt = np.clip(u_opt, self.nmpc_params.u_min, self.nmpc_params.u_max)
            self.u_prev = u_opt

            # 输出acados求解器内部每一步的x状态
            # rospy.loginfo("===== ACADOS求解器各阶段状态x =====")
            # for i in range(self.nmpc_params.Np + 1):  # 包含终端状态，所以是Np+1
            #     x_i = self.acados_solver.get(i, 'x')
            #     rospy.loginfo(f"第{i}步状态x: {np.array2string(x_i, precision=4, floatmode='fixed', suppress_small=True, max_line_width=1000)}")
            #     # 如果需要更详细的分类输出（位置、速度、姿态等），可以取消下面的注释
            #     # pos = x_i[0:3]
            #     # vel = x_i[3:6]
            #     # euler = x_i[6:9]
            #     # angular_vel = x_i[9:12]
            #     # integral = x_i[12:]
            #     # rospy.loginfo(f"第{i}步 - 位置: {np.round(pos,4)}, 速度: {np.round(vel,4)}, 姿态: {np.round(euler,4)}, 角速度: {np.round(angular_vel,4)}, 积分项: {np.round(integral,4)}")
            # rospy.loginfo("====== ACADOS求解器各阶段状态u =======")
            # for i in range (self.nmpc_params.Nc):
            #     u_i = self.acados_solver.get(i, 'u')
            #     rospy.loginfo(f"第{i}步控制u: {np.array2string(u_i, precision=4, floatmode='fixed', suppress_small=True, max_line_width=1000)}")
            # rospy.loginfo("=====================================")
            return u_opt, True, solve_time
        except Exception as e:
            solve_time = time.perf_counter() - solve_start
            self.solve_time_history.append(solve_time)
            print(f"⚠️ 增广NMPC求解失败：{e}，使用悬停配平控制")
            return self.nmpc_params.u_trim, False, solve_time

    def _diagnose_failure(self, status, x0):
        """诊断acados求解失败的原因"""
        STATUS_MSG = {
            0: "ACADOS_SUCCESS",
            1: "ACADOS_FAILURE（通用失败）",
            2: "ACADOS_MAXITER（达到最大迭代次数）",
            3: "ACADOS_MINSTEP（步长过小，线搜索失败）",
            4: "ACADOS_QP_FAILURE（QP求解器失败，通常HPIPM问题）",
        }
        rospy.logwarn(f"\n{'='*60}")
        rospy.logwarn(f"🔍 NMPC求解失败诊断 (status={status})")
        rospy.logwarn(f"   状态码含义: {STATUS_MSG.get(status, '未知状态码')}")

        # 获取求解器统计信息
        try:
            stats = self.acados_solver.get_stats()
            rospy.logwarn(f"   求解迭代次数: {stats.get('nlp_iter', 'N/A')}")
            rospy.logwarn(f"   最后步长: {stats.get('step_length', 'N/A'):.6e}")
            rospy.logwarn(f"   QP求解次数: {stats.get('qp_iter', 'N/A')}")
        except:
            pass

        # ========== 检查初始状态x0是否在约束范围内 ==========
        lbx_0 = self.acados_solver.get(0, 'lbx')
        ubx_0 = self.acados_solver.get(0, 'ubx')
        vio_low = x0 < lbx_0 - 1e-6
        vio_high = x0 > ubx_0 + 1e-6
        if np.any(vio_low) or np.any(vio_high):
            rospy.logwarn(f"   ❌ 初始状态x0超出约束边界！")
            for idx in np.where(vio_low)[0]:
                rospy.logwarn(f"      x0[{idx}] = {x0[idx]:.4f} < lbx[{idx}] = {lbx_0[idx]:.4f}")
            for idx in np.where(vio_high)[0]:
                rospy.logwarn(f"      x0[{idx}] = {x0[idx]:.4f} > ubx[{idx}] = {ubx_0[idx]:.4f}")
        else:
            rospy.logwarn(f"   ✅ 初始状态x0在约束范围内")

        # ========== 遍历各阶段检查状态/控制是否碰到边界 ==========
        for i in range(self.nmpc_params.Np + 1):
            try:
                xi = self.acados_solver.get(i, 'x')
            except:
                continue

            lbx_i = self.acados_solver.get(i, 'lbx')
            ubx_i = self.acados_solver.get(i, 'ubx')

            # 状态触碰边界的索引
            near_low = np.abs(xi - lbx_i) < 1e-4
            near_high = np.abs(xi - ubx_i) < 1e-4
            violated_low = xi < lbx_i - 1e-4
            violated_high = xi > ubx_i + 1e-4

            has_active = np.any(near_low) or np.any(near_high)
            has_violated = np.any(violated_low) or np.any(violated_high)

            if has_active or has_violated:
                rospy.logwarn(f"   阶段[{i}] 状态约束:")
                for idx in np.where(near_low)[0]:
                    rospy.logwarn(f"      x[{idx}] = {xi[idx]:.4f} ≈ lbx[{idx}] = {lbx_i[idx]:.4f} [触碰下界]")
                for idx in np.where(near_high)[0]:
                    rospy.logwarn(f"      x[{idx}] = {xi[idx]:.4f} ≈ ubx[{idx}] = {ubx_i[idx]:.4f} [触碰上界]")
                for idx in np.where(violated_low)[0]:
                    rospy.logwarn(f"      x[{idx}] = {xi[idx]:.4f} < lbx[{idx}] = {lbx_i[idx]:.4f} [⚠️ 违反下界]")
                for idx in np.where(violated_high)[0]:
                    rospy.logwarn(f"      x[{idx}] = {xi[idx]:.4f} > ubx[{idx}] = {ubx_i[idx]:.4f} [⚠️ 违反上界]")

        # ========== 检查控制量约束 ==========
        for i in range(self.nmpc_params.Np):
            try:
                ui = self.acados_solver.get(i, 'u')
            except:
                continue
            lbu_i = self.acados_solver.get(i, 'lbu')
            ubu_i = self.acados_solver.get(i, 'ubu')

            near_low_u = np.abs(ui - lbu_i) < 1e-4
            near_high_u = np.abs(ui - ubu_i) < 1e-4
            violated_low_u = ui < lbu_i - 1e-4
            violated_high_u = ui > ubu_i + 1e-4

            has_active_u = np.any(near_low_u) or np.any(near_high_u)
            has_violated_u = np.any(violated_low_u) or np.any(violated_high_u)

            if has_active_u or has_violated_u:
                rospy.logwarn(f"   阶段[{i}] 控制约束:")
                for idx in np.where(near_low_u)[0]:
                    rospy.logwarn(f"      u[{idx}] = {ui[idx]:.4f} ≈ lbu[{idx}] = {lbu_i[idx]:.4f} [触碰下界]")
                for idx in np.where(near_high_u)[0]:
                    rospy.logwarn(f"      u[{idx}] = {ui[idx]:.4f} ≈ ubu[{idx}] = {ubu_i[idx]:.4f} [触碰上界]")
                for idx in np.where(violated_low_u)[0]:
                    rospy.logwarn(f"      u[{idx}] = {ui[idx]:.4f} < lbu[{idx}] = {lbu_i[idx]:.4f} [⚠️ 违反下界]")
                for idx in np.where(violated_high_u)[0]:
                    rospy.logwarn(f"      u[{idx}] = {ui[idx]:.4f} > ubu[{idx}] = {ubu_i[idx]:.4f} [⚠️ 违反上界]")

        # 如果状态码是 QP_FAILURE，额外提示
        if status == 4:
            rospy.logwarn(f"   💡 QP求解失败常见原因:")
            rospy.logwarn(f"      - 权重矩阵W不是正定矩阵（检查Q/R/S矩阵）")
            rospy.logwarn(f"      - 约束过于严格导致无可行解")
            rospy.logwarn(f"      - 动力学模型存在数值奇异性")
        elif status in [1, 2, 3]:
            rospy.logwarn(f"   💡 NLP求解失败常见原因:")
            rospy.logwarn(f"      - 参考轨迹突变过大，超出动力学可达范围")
            rospy.logwarn(f"      - 约束过于严格（放宽x_min/x_max或du_min/du_max试试）")
            rospy.logwarn(f"      - 初始状态x0离参考轨迹太远")
            rospy.logwarn(f"      - 控制率约束(du_min/du_max)过小导致控制量变化跟不上需求")

        rospy.logwarn(f"{'='*60}")