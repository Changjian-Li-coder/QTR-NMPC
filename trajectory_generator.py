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
        
        # 分步轨迹参数
        self.step_length = rospy.get_param("~step_length", 0.5)  # 平移步长（米）
        self.height_first = rospy.get_param("~height_first", 0.5)  # 先升高到该高度
        self.arrive_threshold = rospy.get_param("~arrive_threshold", 0.1)  # 到达目标阈值（米）
        
        # ========== 2. 状态变量初始化 ==========
        self.current_state = None  # 当前12维状态
        self.total_desired_pose = None  # 最终总目标 [x,y,z,yaw]
        self.sub_goals = []  # 子目标队列（分步到达总目标）
        self.current_sub_goal = None  # 当前执行的子目标
        self.desired_vel = np.array([0.0, 0.0, 0.0])  # 期望末端速度
        
        # ========== 3. ROS话题配置 ==========
        self.trajectory_pub = rospy.Publisher(
            "/uav/reference_trajectory", Float64MultiArray, queue_size=5
        )
        # 订阅当前状态
        rospy.Subscriber("/mavros/local_position/odom", Odometry, self.odom_callback, queue_size=5)
        # 订阅最终期望目标
        rospy.Subscriber("/uav/desired_pose", PoseStamped, self.desired_pose_callback, queue_size=5)
        # 可选：订阅期望速度
        rospy.Subscriber("/uav/desired_vel", Float32MultiArray, self.desired_vel_callback, queue_size=5)

        rospy.loginfo("✅ 分步轨迹生成节点已启动！")
        rospy.loginfo(f"📌 配置：步长={self.step_length}m | 先升高到={self.height_first}m | 到达阈值={self.arrive_threshold}m")

    def odom_callback(self, msg):
        """解析无人机当前12维状态（与NMPC3.py完全对齐）"""
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

    def desired_pose_callback(self, msg):
        """解析最终期望目标，生成子目标队列"""
        # 解析最终目标位置和偏航角
        des_x = msg.pose.position.x
        des_y = msg.pose.position.y
        des_z = msg.pose.position.z
        q = msg.pose.orientation
        _, _, des_yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.total_desired_pose = np.array([des_x, des_y, des_z, des_yaw])
        
        # 重置子目标队列（目标更新时重新生成）
        self.sub_goals = []
        self.current_sub_goal = None
        rospy.loginfo(f"🎯 收到最终目标：[{des_x:.2f}, {des_y:.2f}, {des_z:.2f}]，偏航角：{np.rad2deg(des_yaw):.1f}°")
        
        # 仅当获取到当前状态时，生成子目标队列
        if self.current_state is not None:
            self.generate_sub_goals()

    def desired_vel_callback(self, msg):
        """解析期望末端速度（可选）"""
        if len(msg.data) >= 3:
            self.desired_vel = np.array(msg.data[:3])

    def generate_sub_goals(self):
        """生成子目标队列：先升高度 → 等步长平移到最终目标"""
        # 当前位置
        curr_x, curr_y, curr_z = self.current_state[0], self.current_state[1], self.current_state[2]
        # 最终目标位置
        final_x, final_y, final_z, final_yaw = self.total_desired_pose
        # 修正最终目标高度（优先使用配置的height_first）
        final_z = max(final_z, self.height_first)
        
        # Step 1: 第一个子目标 → 升高到指定高度（保持xy不变）
        height_goal = [curr_x, curr_y, self.height_first, final_yaw]
        self.sub_goals.append(height_goal)
        rospy.loginfo(f"📝 子目标1（升高）：[{height_goal[0]:.2f}, {height_goal[1]:.2f}, {height_goal[2]:.2f}]")
        
        # Step 2: 计算从升高位置到最终目标的等步长子目标
        start_x, start_y = curr_x, curr_y  # 升高后的起始xy
        end_x, end_y = final_x, final_y    # 最终目标xy
        # 计算总位移向量
        dx = end_x - start_x
        dy = end_y - start_y
        total_dist = np.hypot(dx, dy)
        
        # 生成等步长子目标（如果总位移大于步长）
        if total_dist > self.step_length:
            # 计算需要的步数
            step_num = int(np.ceil(total_dist / self.step_length))
            # 计算每步的位移增量
            step_dx = dx / step_num
            step_dy = dy / step_num
            
            # 生成中间子目标
            for i in range(1, step_num + 1):
                sub_x = start_x + step_dx * i
                sub_y = start_y + step_dy * i
                sub_goal = [sub_x, sub_y, self.height_first, final_yaw]
                self.sub_goals.append(sub_goal)
                rospy.loginfo(f"📝 子目标{i+1}（平移）：[{sub_x:.2f}, {sub_y:.2f}, {self.height_first:.2f}]")
        
        # Step 3: 最后一个子目标 → 最终目标（修正z高度）
        final_sub_goal = [final_x, final_y, final_z, final_yaw]
        self.sub_goals.append(final_sub_goal)
        rospy.loginfo(f"📝 子目标{len(self.sub_goals)}（最终）：[{final_x:.2f}, {final_y:.2f}, {final_z:.2f}]")
        
        # 初始化当前子目标为第一个
        self.current_sub_goal = self.sub_goals[0]

    def check_arrive_sub_goal(self):
        """检查是否到达当前子目标（距离判断）"""
        if self.current_state is None or self.current_sub_goal is None:
            return False
        
        # 当前位置 vs 子目标位置
        curr_pos = self.current_state[0:3]
        sub_goal_pos = np.array(self.current_sub_goal[0:3])
        # 计算欧氏距离
        dist = np.linalg.norm(curr_pos - sub_goal_pos)
        
        if dist < self.arrive_threshold:
            rospy.loginfo(f"✅ 到达子目标：[{sub_goal_pos[0]:.2f}, {sub_goal_pos[1]:.2f}, {sub_goal_pos[2]:.2f}]（距离：{dist:.3f}m < {self.arrive_threshold}m）")
            return True
        return False

    def switch_next_sub_goal(self):
        """切换到下一个子目标"""
        if len(self.sub_goals) == 0:
            return
        
        # 移除已完成的当前子目标
        self.sub_goals.pop(0)
        # 更新当前子目标
        if len(self.sub_goals) > 0:
            self.current_sub_goal = self.sub_goals[0]
            rospy.loginfo(f"🔄 切换到下一个子目标：[{self.current_sub_goal[0]:.2f}, {self.current_sub_goal[1]:.2f}, {self.current_sub_goal[2]:.2f}]")
        else:
            self.current_sub_goal = None
            rospy.loginfo("🎉 所有子目标完成！")

    def generate_smooth_trajectory(self):
        """生成当前子目标的平滑参考轨迹（匹配NMPC预测时域）"""
        # 初始化轨迹数组（12维 × (Np+1)步）
        ref_trajectory = np.zeros((self.nx, self.Np + 1))

        # 兜底：无当前状态/子目标时返回全零
        if self.current_state is None:
            rospy.logwarn_throttle(1, "⚠️ 未收到无人机状态，返回全零轨迹")
            return ref_trajectory
        if self.current_sub_goal is None:
            # 所有子目标完成 → 悬停在最终目标
            if self.total_desired_pose is not None:
                self.current_sub_goal = self.total_desired_pose
            else:
                rospy.logwarn_throttle(1, "⚠️ 无可用子目标，生成悬停轨迹")
                hover_pos = self.current_state[0:3].copy()
                hover_pos[2] = max(hover_pos[2], 0.5)
                self.current_sub_goal = [hover_pos[0], hover_pos[1], hover_pos[2], self.current_state[8]]

        # ========== 轨迹生成核心逻辑 ==========
        # 1. 起始状态（当前状态）
        start_state = self.current_state.copy()
        # 2. 目标状态（当前子目标）
        target_state = np.zeros(self.nx)
        # 目标位置：当前子目标位置
        target_state[0:3] = self.current_sub_goal[0:3]
        # 目标速度：自定义期望速度（默认归零）
        target_state[3:6] = self.desired_vel
        # 目标姿态：roll/pitch归零，yaw为子目标偏航角
        target_state[6] = 0.0  # roll
        target_state[7] = 0.0  # pitch
        target_state[8] = self.current_sub_goal[3]  # yaw
        # 目标角速度：全归零
        target_state[9:12] = [0.0, 0.0, 0.0]

        # 3. 线性插值生成平滑轨迹（预测时域内逐步收敛到子目标）
        for i in range(self.Np + 1):
            alpha = i / self.Np  # 插值系数（0→当前，1→子目标）
            ref_trajectory[:, i] = (1 - alpha) * start_state + alpha * target_state
            # 偏航角归一化（防止角度突变）
            ref_trajectory[8, i] = (ref_trajectory[8, i] + np.pi) % (2 * np.pi) - np.pi

        return ref_trajectory

    def publish_trajectory(self):
        """发布参考轨迹（展平为一维数组）"""
        if self.current_state is None:
            return
        
        # 检查是否到达当前子目标，到达则切换
        if self.check_arrive_sub_goal():
            self.switch_next_sub_goal()
        
        # 生成轨迹
        ref_traj = self.generate_smooth_trajectory()
        # 展平并发布
        traj_flat = ref_traj.flatten()
        msg = Float64MultiArray()
        msg.data = traj_flat.tolist()
        self.trajectory_pub.publish(msg)

    def run(self):
        """主循环：按NMPC频率生成并发布轨迹"""
        rate = rospy.Rate(120)  # 与NMPC3.py频率一致
        while not rospy.is_shutdown():
            self.publish_trajectory()
            rate.sleep()

if __name__ == "__main__":
    try:
        rospy.init_node("uav_trajectory_generator", anonymous=False)
        generator = UAVTrajectoryGenerator()
        generator.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("🛑 分步轨迹生成节点已中断")