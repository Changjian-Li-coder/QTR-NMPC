#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import numpy as np
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray, Float32MultiArray
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import PoseStamped

class UAVTrajectoryGenerator:
    def __init__(self):
        # ========== 1. 核心配置参数（可动态调整） ==========
        self.Np = rospy.get_param("~Np", 25)    # NMPC预测时域
        self.Ts = rospy.get_param("~Ts", 0.01)  # 控制周期
        self.nx = 12  # 12维状态：位置(3)+速度(3)+欧拉角(3)+角速度(3)

        # 分步轨迹参数（原手动模式用）
        self.step_length = rospy.get_param("~step_length", 1)  # 平移步长（米）
        self.height_first = rospy.get_param("~height_first", 0.4)  # 先升高到该高度
        self.arrive_threshold = rospy.get_param("~arrive_threshold", 0.1)  # 到达目标阈值（米）

        # ========== 2. 轨迹模式选择（核心修改点1） ==========
        self.trajectory_mode = rospy.get_param("~trajectory_mode", "ab_round_trip")  # 可选："manual"（手动目标）、"ab_round_trip"（AB往返）、"circle"（飞圆圈）

        # ========== 3. 各模式配置参数 ==========
        # --- 手动模式参数（保留原有） ---
        self.manual_des_x = rospy.get_param("~manual_des_x", 1.75)
        self.manual_des_y = rospy.get_param("~manual_des_y", -0.1)
        self.manual_des_z = rospy.get_param("~manual_des_z", 0.4)
        self.manual_des_yaw = rospy.get_param("~manual_des_yaw", 0.0)
        self.manual_des_vel = rospy.get_param("~manual_des_vel", [0.0, 0.0, 0.0])

        # --- AB往返模式参数 ---
        self.ab_point_a = np.array(rospy.get_param("~ab_point_a", [1.6, -0.9]))  # A点xy坐标
        self.ab_point_b = np.array(rospy.get_param("~ab_point_b", [1.6, -0.2]))  # B点xy坐标
        self.ab_fixed_height = rospy.get_param("~ab_fixed_height", 0.4)  # 固定飞行高度
        self.ab_continuous = rospy.get_param("~ab_continuous", True)  # 是否持续往返（False则飞一次停止）
        self.ab_yaw = rospy.get_param("~ab_yaw", 0.0)  # 固定偏航角（弧度）
        # --- AB往返模式状态 ---
        self.ab_current_target_is_b = True  # 当前目标是否为B点（初始去B）
        self.ab_trip_count = 0  # 往返次数
        self.ab_sub_goals = []   # AB模式的插值子目标队列（新增）
        self.ab_current_sub_goal = None  # AB模式当前执行的子目标（新增）

        # --- 圆圈模式参数 ---
        self.circle_center = np.array(rospy.get_param("~circle_center", [0.0, 0.0]))  # 圆心xy坐标
        self.circle_radius = rospy.get_param("~circle_radius", 2.0)  # 圆圈半径（米）
        self.circle_height = rospy.get_param("~circle_height", 0.5)  # 固定飞行高度
        self.circle_angular_vel = rospy.get_param("~circle_angular_vel", 0.5)  # 角速度（rad/s），正=逆时针，负=顺时针
        self.circle_yaw_follow = rospy.get_param("~circle_yaw_follow", True)  # 偏航角是否跟随飞行方向（True=机头朝前进方向）
        self.circle_fixed_yaw = rospy.get_param("~circle_fixed_yaw", 0.0)  # 若不跟随，固定偏航角（弧度）

        # ========== 4. 状态变量初始化 ==========
        self.current_state = None  # 当前12维状态
        self.current_time = None   # 当前时间（用于圆圈模式计算角度）
        self.start_time = None     # 轨迹开始时间

        # --- 手动模式状态 ---
        self.total_desired_pose = None
        self.sub_goals = []
        self.current_sub_goal = None
        self.desired_vel = np.array(self.manual_des_vel)

        # --- AB往返模式状态 ---
        self.ab_current_target_is_b = True  # 当前目标是否为B点（初始去B）
        self.ab_trip_count = 0  # 往返次数

        # --- 圆圈模式状态 ---
        self.circle_initial_angle = 0.0  # 初始角度

        # ========== 5. ROS话题配置 ==========
        self.trajectory_pub = rospy.Publisher("/uav/reference_trajectory", Float64MultiArray, queue_size=5)
        # 仅保留当前状态订阅
        rospy.Subscriber("/mavros/local_position/odom", Odometry, self.odom_callback, queue_size=5)

        rospy.loginfo("✅ 多模式轨迹生成节点已启动！")
        rospy.loginfo(f"📌 当前模式：{self.trajectory_mode}")
        if self.trajectory_mode == "ab_round_trip":
            rospy.loginfo(f"📍 AB往返：A点{self.ab_point_a} → B点{self.ab_point_b}，高度{self.ab_fixed_height}m")
        elif self.trajectory_mode == "circle":
            rospy.loginfo(f"⭕ 飞圆圈：圆心{self.circle_center}，半径{self.circle_radius}m，高度{self.circle_height}m，角速度{self.circle_angular_vel}rad/s")

    def odom_callback(self, msg):
        """解析无人机当前12维状态（与原代码一致）"""
        # 位置
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        # 速度
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        vz = msg.twist.twist.linear.z
        # 欧拉角（四元数转欧拉角）
        q = msg.pose.pose.orientation
        phi, theta, psi = euler_from_quaternion([q.x, q.y, q.z, q.w])
        # 角速度
        p_rate = msg.twist.twist.angular.x
        q_rate = msg.twist.twist.angular.y
        r_rate = msg.twist.twist.angular.z

        # 组装12维状态
        self.current_state = np.array([
            x, y, z,
            vx, vy, vz,
            phi, theta, psi,
            p_rate, q_rate, r_rate
        ])

        # 首次获取状态时初始化轨迹
        if self.start_time is None:
            self.start_time = rospy.Time.now().to_sec()
            if self.trajectory_mode == "manual":
                self.init_manual_target()
            elif self.trajectory_mode == "circle":
                # 计算初始角度（无人机当前位置相对于圆心的角度）
                dx = self.current_state[0] - self.circle_center[0]
                dy = self.current_state[1] - self.circle_center[1]
                self.circle_initial_angle = np.arctan2(dy, dx)
                rospy.loginfo(f"⭕ 圆圈模式初始化，初始角度：{np.rad2deg(self.circle_initial_angle):.1f}°")

    def init_manual_target(self):
        """初始化手动目标（原代码逻辑）"""
        self.total_desired_pose = np.array([
            self.manual_des_x, self.manual_des_y, self.manual_des_z, self.current_state[8] if self.current_state is not None else self.manual_des_yaw
        ])
        self.sub_goals = []
        self.current_sub_goal = None
        self.generate_sub_goals()
        rospy.loginfo(f"✅ 手动目标初始化完成")

    def generate_sub_goals(self):
        """生成手动模式子目标队列（原代码逻辑）"""
        curr_x, curr_y, curr_z = self.current_state[0], self.current_state[1], self.current_state[2]
        final_x, final_y, final_z, final_yaw = self.total_desired_pose
        final_z = max(final_z, self.height_first)

        # Step 1: 升高到指定高度
        height_goal = [curr_x, curr_y, self.height_first, final_yaw]
        self.sub_goals.append(height_goal)
        rospy.loginfo(f"📝 子目标1（升高）：[{height_goal[0]:.2f}, {height_goal[1]:.2f}, {height_goal[2]:.2f}]")

        # Step 2: 等步长平移
        start_x, start_y = curr_x, curr_y
        end_x, end_y = final_x, final_y
        dx = end_x - start_x
        dy = end_y - start_y
        total_dist = np.hypot(dx, dy)

        if total_dist > self.step_length:
            step_num = int(np.ceil(total_dist / self.step_length))
            step_dx = dx / step_num
            step_dy = dy / step_num
            for i in range(1, step_num + 1):
                sub_x = start_x + step_dx * i
                sub_y = start_y + step_dy * i
                sub_goal = [sub_x, sub_y, self.height_first, final_yaw]
                self.sub_goals.append(sub_goal)
                rospy.loginfo(f"📝 子目标{i+1}（平移）：[{sub_x:.2f}, {sub_y:.2f}, {self.height_first:.2f}]")

        # Step 3: 最终目标
        final_sub_goal = [final_x, final_y, final_z, final_yaw]
        self.sub_goals.append(final_sub_goal)
        rospy.loginfo(f"📝 子目标{len(self.sub_goals)}（最终）：[{final_x:.2f}, {final_y:.2f}, {final_z:.2f}]")

        self.current_sub_goal = self.sub_goals[0]

    def check_arrive_sub_goal(self):
        """检查手动模式是否到达子目标（原代码逻辑）"""
        # rospy.loginfo_throttle(1, "🔍 检查是否到达子目标...")
        if self.current_state is None or self.current_sub_goal is None:
            return False
        curr_pos = self.current_state[0:3]
        sub_goal_pos = np.array(self.current_sub_goal[0:3])
        dist = np.linalg.norm(curr_pos - sub_goal_pos)
        # rospy.loginfo_throttle(1, f"📏 当前距离子目标的距离：{dist:.3f}m")
        if dist < self.arrive_threshold:
            rospy.loginfo(f"✅ 到达子目标：[{sub_goal_pos[0]:.2f}, {sub_goal_pos[1]:.2f}, {sub_goal_pos[2]:.2f}]（距离：{dist:.3f}m）")
            return True
        return False

    def switch_next_sub_goal(self):
        """手动模式切换子目标（原代码逻辑）"""
        if len(self.sub_goals) == 0:
            return
        self.sub_goals.pop(0)
        if len(self.sub_goals) > 0:
            self.current_sub_goal = self.sub_goals[0]
            rospy.loginfo(f"🔄 切换到下一个子目标：[{self.current_sub_goal[0]:.2f}, {self.current_sub_goal[1]:.2f}, {self.current_sub_goal[2]:.2f}]")
        else:
            self.current_sub_goal = None
            rospy.loginfo("🎉 所有子目标完成！")
    def generate_ab_sub_goals(self, start_pos, target_pos):
        """AB模式：生成从起点到目标点的插值子目标（分步移动）"""
        self.ab_sub_goals = []
        start_x, start_y, start_z = start_pos
        target_x, target_y, target_z = target_pos
        final_yaw = self.ab_yaw

        # Step 1: 先保持当前高度（或固定高度）
        height_goal = [start_x, start_y, self.ab_fixed_height, final_yaw]
        self.ab_sub_goals.append(height_goal)
        rospy.loginfo(f"📝 AB模式子目标1（定高）：[{height_goal[0]:.2f}, {height_goal[1]:.2f}, {height_goal[2]:.2f}]")

        # Step 2: 等步长平移到目标点xy坐标（插值核心）
        dx = target_x - start_x
        dy = target_y - start_y
        total_dist = np.hypot(dx, dy)

        if total_dist > self.step_length:
            step_num = int(np.ceil(total_dist / self.step_length))
            step_dx = dx / step_num
            step_dy = dy / step_num
            for i in range(1, step_num + 1):
                sub_x = start_x + step_dx * i
                sub_y = start_y + step_dy * i
                sub_goal = [sub_x, sub_y, self.ab_fixed_height, final_yaw]
                self.ab_sub_goals.append(sub_goal)
                rospy.loginfo(f"📝 AB模式子目标{i+1}（平移）：[{sub_x:.2f}, {sub_y:.2f}, {self.ab_fixed_height:.2f}]")

        # Step 3: 最终目标点（固定高度）
        final_sub_goal = [target_x, target_y, self.ab_fixed_height, final_yaw]
        self.ab_sub_goals.append(final_sub_goal)
        rospy.loginfo(f"📝 AB模式子目标{len(self.ab_sub_goals)}（最终）：[{target_x:.2f}, {target_y:.2f}, {self.ab_fixed_height:.2f}]")

        self.ab_current_sub_goal = self.ab_sub_goals[0]

    def check_ab_arrive_and_switch(self):
        """AB模式：检查子目标到达，切换子目标；子目标完成后切换总目标（A/B）"""
        if self.current_state is None:
            # 无状态时返回当前总目标点
            current_target_xy = self.ab_point_b if self.ab_current_target_is_b else self.ab_point_a
            return np.array([*current_target_xy, self.ab_fixed_height])

        # 1. 若未生成子目标，初始化从当前位置到总目标的插值子目标
        if not self.ab_sub_goals or self.ab_current_sub_goal is None:
            start_pos = self.current_state[0:3]
            current_target_xy = self.ab_point_b if self.ab_current_target_is_b else self.ab_point_a
            current_target = np.array([*current_target_xy, self.ab_fixed_height])
            self.generate_ab_sub_goals(start_pos, current_target)

        # 2. 检查是否到达当前子目标
        curr_pos = self.current_state[0:3]
        sub_goal_pos = np.array(self.ab_current_sub_goal[0:3])
        dist = np.linalg.norm(curr_pos - sub_goal_pos)

        if dist < self.arrive_threshold:
            rospy.loginfo(f"✅ AB模式到达子目标：[{sub_goal_pos[0]:.2f}, {sub_goal_pos[1]:.2f}, {sub_goal_pos[2]:.2f}]（距离：{dist:.3f}m）")
            # 弹出已完成的子目标，切换下一个
            self.ab_sub_goals.pop(0)
            if len(self.ab_sub_goals) > 0:
                self.ab_current_sub_goal = self.ab_sub_goals[0]
            else:
                # 3. 所有子目标完成 → 切换总目标（A↔B）
                self.ab_current_target_is_b = not self.ab_current_target_is_b
                self.ab_trip_count += 1
                next_target_name = "B" if self.ab_current_target_is_b else "A"
                next_target_xy = self.ab_point_b if self.ab_current_target_is_b else self.ab_point_a
                rospy.loginfo(f"🔄 AB模式完成第{self.ab_trip_count}次单程，切换目标到{next_target_name}点：{next_target_xy}")
                # 重置子目标（为下一次移动做准备）
                self.ab_current_sub_goal = None

        # 返回当前子目标（插值后的平滑目标）
        return np.array(self.ab_current_sub_goal[0:3]) if self.ab_current_sub_goal is not None else curr_pos

    def generate_circle_trajectory(self, time_since_start):
        """圆圈模式：生成当前时刻的参考轨迹"""
        # 计算当前角度
        current_angle = self.circle_initial_angle + self.circle_angular_vel * time_since_start

        # 计算当前目标位置（极坐标转直角坐标）
        target_x = self.circle_center[0] + self.circle_radius * np.cos(current_angle)
        target_y = self.circle_center[1] + self.circle_radius * np.sin(current_angle)
        target_z = self.circle_height

        # 计算期望速度（切线方向，v = ω × r）
        target_vx = -self.circle_angular_vel * self.circle_radius * np.sin(current_angle)
        target_vy = self.circle_angular_vel * self.circle_radius * np.cos(current_angle)
        target_vz = 0.0

        # 计算期望偏航角
        if self.circle_yaw_follow:
            # 偏航角跟随速度方向（机头朝前进方向）
            target_yaw = np.arctan2(target_vy, target_vx)
        else:
            target_yaw = self.circle_fixed_yaw

        # 组装目标状态
        target_state = np.zeros(self.nx)
        target_state[0:3] = [target_x, target_y, target_z]  # 位置
        target_state[3:6] = [target_vx, target_vy, target_vz]  # 速度（切线方向）
        target_state[6] = 0.0  # roll
        target_state[7] = 0.0  # pitch
        target_state[8] = target_yaw  # yaw
        target_state[9:12] = [0.0, 0.0, 0.0]  # 角速度

        return target_state

    def generate_smooth_trajectory(self):
        """根据模式生成参考轨迹（核心修改点2）"""
        ref_trajectory = np.zeros(self.nx)
        if self.current_state is None:
            rospy.logwarn_throttle(1, "⚠️ 未收到无人机状态，返回全零轨迹")
            return ref_trajectory

        # ========== 模式1：手动目标（原代码逻辑） ==========
        if self.trajectory_mode == "manual":
            if self.current_sub_goal is None:
                if self.total_desired_pose is not None:
                    self.current_sub_goal = self.total_desired_pose
                else:
                    hover_pos = self.current_state[0:3].copy()
                    hover_pos[2] = max(hover_pos[2], 0.5)
                    self.current_sub_goal = [hover_pos[0], hover_pos[1], hover_pos[2], self.current_state[8]]

            ref_trajectory[0:3] = self.current_sub_goal[0:3]
            ref_trajectory[3:6] = self.desired_vel
            ref_trajectory[6] = 0.0
            ref_trajectory[7] = 0.0
            ref_trajectory[8] = self.current_sub_goal[3]
            ref_trajectory[8] = (ref_trajectory[8] + np.pi) % (2 * np.pi) - np.pi
            ref_trajectory[9:12] = [0.0, 0.0, 0.0]

        # ========== 模式2：AB两点往返 ==========
        elif self.trajectory_mode == "ab_round_trip":
            # 获取当前插值子目标（平滑目标点）
            current_target = self.check_ab_arrive_and_switch()

            # 生成平滑轨迹（位置=子目标，速度小幅跟随，避免突变）
            ref_trajectory[0:3] = current_target
            # 速度：朝向目标点的小幅速度（替代直接归零，保证平滑移动）
            delta_pos = current_target - self.current_state[0:3]
            dist_to_target = np.linalg.norm(delta_pos)
            if dist_to_target > self.arrive_threshold:
                # 归一化方向 + 限速（避免速度过大）
                ref_vel = delta_pos / dist_to_target * 0.2  # 0.2m/s 移动速度，可配置化
            else:
                ref_vel = np.array([0.0, 0.0, 0.0])
            ref_trajectory[3:6] = ref_vel
            ref_trajectory[6] = 0.0  # roll
            ref_trajectory[7] = 0.0  # pitch
            ref_trajectory[8] = self.ab_yaw
            ref_trajectory[8] = (ref_trajectory[8] + np.pi) % (2 * np.pi) - np.pi
            ref_trajectory[9:12] = [0.0, 0.0, 0.0]  # 角速度归零

        # ========== 模式3：固定高度飞圆圈 ==========
        elif self.trajectory_mode == "circle":
            if self.start_time is None:
                return ref_trajectory

            # 计算时间
            time_since_start = rospy.Time.now().to_sec() - self.start_time

            # 生成预测时域内的轨迹（每个时间步对应一个圆圈上的点）
            for i in range(self.Np + 1):
                future_time = time_since_start + i * self.Ts
                target_state_i = self.generate_circle_trajectory(future_time)
                ref_trajectory[:, i] = target_state_i
                ref_trajectory[8, i] = (ref_trajectory[8, i] + np.pi) % (2 * np.pi) - np.pi

        return ref_trajectory

    def publish_trajectory(self):
        """发布参考轨迹（根据模式处理子目标切换）"""
        if self.current_state is None:
            return

        # 手动模式：检查并切换子目标
        if self.trajectory_mode == "manual":
            if self.check_arrive_sub_goal():
                self.switch_next_sub_goal()

        # 生成轨迹
        ref_traj = self.generate_smooth_trajectory()
        rospy.loginfo_throttle(1, f"发布参考轨迹:{ref_traj.round(3)}")  # 打印最后一个时间步的状态作为参考
        # 展平并发布
        traj_flat = ref_traj.flatten()
        msg = Float64MultiArray()
        msg.data = traj_flat.tolist()
        self.trajectory_pub.publish(msg)

    def run(self):
        """主循环"""
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            self.publish_trajectory()
            rate.sleep()

if __name__ == "__main__":
    try:
        rospy.init_node("uav_trajectory_generator", anonymous=False)
        generator = UAVTrajectoryGenerator()
        generator.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("🛑 多模式轨迹生成节点已中断")
