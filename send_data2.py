#!/usr/bin/env python3
import rospy
import math
import tf.transformations as tf_trans
from mavros_msgs.msg import State
from std_srvs.srv import SetBool
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import ActuatorControl
# 新增：PX4 Offboard 必需的 位置设定值消息
from mavros_msgs.msg import PositionTarget

# ===================== 全局变量 =====================
current_state = State()
correct_debug_msg = ActuatorControl()
vicon_pose = PoseStamped()
# 初始化发布者
nmpc_pub = None
vicon_pub = None
setpoint_pub = None  # Offboard核心：位置设定值发布者

# ===================== 回调函数 =====================
def state_callback(msg):
    global current_state
    current_state = msg

def vicon_pose_callback(msg):
    global vicon_pose
    vicon_pose = msg

def data_callback(msg):
    global correct_debug_msg
    # 仅接收8维控制指令，可根据你的需求修改
    if len(msg.data) != 6:
        rospy.logwarn_throttle(2, f"控制指令长度错误: {len(msg.data)}")
        return
    for i in range(0,6):
        correct_debug_msg.controls[i] = msg.data[i]

# ===================== Vicon坐标系转换（已修复） =====================
def transform_vicon_pose(raw_pose):
    transformed_pose = PoseStamped()
    transformed_pose.header = raw_pose.header
    transformed_pose.header.frame_id = "map"

    # Vicon → MAVROS ENU 标准转换（你当前可用的正确转换）
    transformed_pose.pose.position.x = raw_pose.pose.position.y
    transformed_pose.pose.position.y = -raw_pose.pose.position.x
    transformed_pose.pose.position.z = raw_pose.pose.position.z


    transformed_pose.pose.orientation.x = raw_pose.pose.orientation.x
    transformed_pose.pose.orientation.y = raw_pose.pose.orientation.y
    transformed_pose.pose.orientation.z = raw_pose.pose.orientation.z
    transformed_pose.pose.orientation.w = raw_pose.pose.orientation.w

    return transformed_pose


# ===================== 主函数 =====================
def main():
    global nmpc_pub, vicon_pub, setpoint_pub
    rospy.init_node("debug_array_sender")

    # ------------ 发布者定义 ------------
    nmpc_pub = rospy.Publisher("/mavros/actuator_control", ActuatorControl, queue_size=5)
    vicon_pub = rospy.Publisher("/mavros/vision_pose/pose", PoseStamped, queue_size=5)

    # ------------ 订阅者定义 ------------
    rospy.Subscriber("/mavros/state", State, state_callback, queue_size=5)
    rospy.Subscriber("/nmpc/control_cmd", Float64MultiArray, data_callback, queue_size=5)
    rospy.Subscriber("/vrpn_client_node/LCJ_QTR_0409/pose", PoseStamped, vicon_pose_callback, queue_size=5)

    rate = rospy.Rate(200)  # 20Hz高频发送（PX4要求≥2Hz）

    while not rospy.is_shutdown():
        # 1. 转换并发布动捕定位数据（已正常工作）
        transformed_pose = transform_vicon_pose(vicon_pose)
        # vicon_pub.publish(vicon_pose)
        vicon_pub.publish(transformed_pose)

        # 3. 原有逻辑：安全指令 / 自定义控制指令
        if current_state is None:
            rospy.logwarn_throttle(5, "等待飞控状态...")
            rate.sleep()
            continue

        # # 未解锁 / 非Offboard模式 → 发送安全指令
        # if not current_state.armed :
        #     nmpc_pub.publish(safe_debug_msg)
        #     rospy.loginfo_once("等待解锁...")
        # # 已解锁 + Offboard模式 → 发送你的自定义控制指令
        # else:
        # for i in range(0,6):
        #     correct_debug_msg.controls[i] = 0
        # correct_debug_msg.controls[0] = -0.1885
        # correct_debug_msg.controls[1] = 0.2684
        # correct_debug_msg.controls[2] = -19.5998
        # correct_debug_msg.controls[3] = 1.408
        # correct_debug_msg.controls[4] = 1.7914
        # correct_debug_msg.controls[5] = 0.0000
        nmpc_pub.publish(correct_debug_msg)
        rospy.loginfo_throttle(2, "---输出控制量---")
        rospy.loginfo_throttle(2, "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f",correct_debug_msg.controls[0],correct_debug_msg.controls[1],correct_debug_msg.controls[2],correct_debug_msg.controls[3],correct_debug_msg.controls[4],correct_debug_msg.controls[5])
        # rospy.loginfo_once("✅ armed模式：已发送自定义控制指令")

        rate.sleep()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"错误: {str(e)}")
        raise
