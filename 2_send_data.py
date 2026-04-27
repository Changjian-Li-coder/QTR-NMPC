#!/usr/bin/env python3
import rospy

from mavros_msgs.msg import ActuatorControl

rospy.init_node("debug_array_sender")
pub = rospy.Publisher("/mavros/actuator_control", ActuatorControl, queue_size=10)

rate = rospy.Rate(10)
while not rospy.is_shutdown():
    msg = ActuatorControl()
    msg.controls = [1.1, 7.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8]  # 你的float数组
    pub.publish(msg)
    rate.sleep()
