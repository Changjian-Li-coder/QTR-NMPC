#!/usr/bin/env python3
import rospy
import math
import numpy as np
import tf.transformations as tf_trans
from mavros_msgs.msg import State
from std_srvs.srv import SetBool
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import ActuatorControl
from mavros_msgs.msg import PositionTarget

# ===================== 全局变量 =====================
current_state = State()
correct_debug_msg = ActuatorControl()
vicon_pose = PoseStamped()
# 初始化发布者
nmpc_pub = None
vicon_pub = None
setpoint_pub = None

# ===================== 校准相关变量 =====================
CALIB_FRAMES = 10          # 校准采集帧数
calib_buffer = []          # 校准数据缓存
calib_offset = None        # 校准偏移量 (x, y, z)
calib_yaw_offset = 0.0     # 校准偏航角偏移 (弧度)
calibration_done = False   # 校准完成标志
calib_printed = False      # 防止重复打印校准信息

# ===================== 回调函数 =====================
def state_callback(msg):
    global current_state
    current_state = msg

def vicon_pose_callback(msg):
    global vicon_pose
    vicon_pose = msg

# ===================== 四元数工具函数 =====================
def quaternion_average(quaternions):
    """对多个四元数求平均（使用 Slerp 方法）"""
    q_array = np.array(quaternions)  # shape (N, 4)
    # 计算累加四元数并归一化
    q_sum = np.sum(q_array, axis=0)
    q_avg = q_sum / np.linalg.norm(q_sum)
    # 确保 w 为正（四元数符号歧义）
    if q_avg[3] < 0:
        q_avg = -q_avg
    return q_avg

# ===================== 执行校准 =====================
def perform_calibration():
    global calib_offset, calib_yaw_offset, calibration_done, calib_buffer

    positions = np.array([[p[0], p[1], p[2]] for p in calib_buffer])

    # 计算位置均值
    pos_mean = np.mean(positions, axis=0)
    calib_offset = pos_mean

    # 计算姿态均值，只提取偏航角
    quats = [q[3:] for q in calib_buffer]  # (x, y, z, w)
    q_avg = quaternion_average(quats)
    euler = tf_trans.euler_from_quaternion(q_avg)
    calib_yaw_offset = euler[2]  # 只保留偏航角偏移

    calibration_done = True

    rospy.loginfo("=" * 50)
    rospy.loginfo("Vicon 坐标系校准完成！")
    rospy.loginfo(f"  采集帧数: {CALIB_FRAMES}")
    rospy.loginfo(f"  位置偏移: x={calib_offset[0]:.6f}, y={calib_offset[1]:.6f}, z={calib_offset[2]:.6f}")
    rospy.loginfo(f"  偏航角偏移: {calib_yaw_offset:.6f} rad ({math.degrees(calib_yaw_offset):.2f}°)")
    rospy.loginfo("=" * 50)

# ===================== Vicon坐标系转换（带校准补偿） =====================
def transform_vicon_pose(raw_pose):
    global calib_offset, calib_yaw_offset, calibration_done

    transformed_pose = PoseStamped()
    transformed_pose.header = raw_pose.header
    transformed_pose.header.frame_id = "map"

    if calibration_done and calib_offset is not None:
        # 位置补偿：减去校准偏移量（将原点移动到无人机上电时的位置）
        transformed_pose.pose.position.x = raw_pose.pose.position.x - calib_offset[0]
        transformed_pose.pose.position.y = raw_pose.pose.position.y - calib_offset[1]
        transformed_pose.pose.position.z = raw_pose.pose.position.z - calib_offset[2]

        # 姿态补偿：只补偿偏航角，保持俯仰和滚转角不变
        q_raw = np.array([raw_pose.pose.orientation.x,
                          raw_pose.pose.orientation.y,
                          raw_pose.pose.orientation.z,
                          raw_pose.pose.orientation.w])
        euler = tf_trans.euler_from_quaternion(q_raw)
        q_compensated = tf_trans.quaternion_from_euler(
            euler[0],                              # roll — 保持不变
            euler[1],                              # pitch — 保持不变
            euler[2] - calib_yaw_offset            # yaw — 减去校准偏航偏移
        )

        transformed_pose.pose.orientation.x = q_compensated[0]
        transformed_pose.pose.orientation.y = q_compensated[1]
        transformed_pose.pose.orientation.z = q_compensated[2]
        transformed_pose.pose.orientation.w = q_compensated[3]
    else:
        # 校准完成前直接透传
        transformed_pose.pose.position.x = raw_pose.pose.position.x
        transformed_pose.pose.position.y = raw_pose.pose.position.y
        transformed_pose.pose.position.z = raw_pose.pose.position.z
        transformed_pose.pose.orientation.x = raw_pose.pose.orientation.x
        transformed_pose.pose.orientation.y = raw_pose.pose.orientation.y
        transformed_pose.pose.orientation.z = raw_pose.pose.orientation.z
        transformed_pose.pose.orientation.w = raw_pose.pose.orientation.w

    return transformed_pose


# ===================== 主函数 =====================
def main():
    global vicon_pub, setpoint_pub, calib_buffer, calibration_done, calib_printed
    rospy.init_node("debug_array_sender")

    # ------------ 发布者定义 ------------
    vicon_pub = rospy.Publisher("/mavros/vision_pose/pose", PoseStamped, queue_size=5)

    # ------------ 订阅者定义 ------------
    rospy.Subscriber("/vrpn_client_node/QTR2/pose", PoseStamped, vicon_pose_callback, queue_size=5)

    rate = rospy.Rate(100)

    rospy.loginfo("等待 Vicon 数据以进行坐标系校准...")
    rospy.loginfo(f"请确保无人机保持静止，将采集 {CALIB_FRAMES} 帧数据进行校准。")

    while not rospy.is_shutdown():
        # ========== 校准阶段：采集前 N 帧数据 ==========
        if not calibration_done:
            # 等待有效数据
            if vicon_pose.header.seq > 0 or vicon_pose.header.stamp.to_sec() > 0.0:
                pos = (vicon_pose.pose.position.x,
                       vicon_pose.pose.position.y,
                       vicon_pose.pose.position.z)
                orient = (vicon_pose.pose.orientation.x,
                          vicon_pose.pose.orientation.y,
                          vicon_pose.pose.orientation.z,
                          vicon_pose.pose.orientation.w)
                entry = pos + orient  # (x, y, z, qx, qy, qz, qw)

                # 去重：避免同一帧反复添加
                if len(calib_buffer) == 0 or not np.allclose(calib_buffer[-1], entry, atol=1e-9):
                    calib_buffer.append(entry)
                    rospy.loginfo(f"校准采集 [{len(calib_buffer)}/{CALIB_FRAMES}]: "
                                  f"pos=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")

                if len(calib_buffer) >= CALIB_FRAMES:
                    perform_calibration()

        # ========== 正常运行阶段：转换并发布 ==========
        transformed_pose = transform_vicon_pose(vicon_pose)

        # 校准完成后打印一次补偿后的初始位置
        if calibration_done and not calib_printed:
            rospy.loginfo(f"补偿后初始位置: "
                          f"({transformed_pose.pose.position.x:.4f}, "
                          f"{transformed_pose.pose.position.y:.4f}, "
                          f"{transformed_pose.pose.position.z:.4f}) — 应接近 (0, 0, 0)")
            calib_printed = True

        vicon_pub.publish(transformed_pose)  # 补偿后的数据发布
        # vicon_pub.publish(vicon_pose)      # 原始数据发布（用于对比）
        rate.sleep()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"错误: {str(e)}")
        raise
