#!/usr/bin/python3
# encoding:utf-8
import rospy
from std_msgs.msg import String

pid_str = "100:165:200"
robot_topic="pid"

def pid():
    global robot_param_str
    pub = rospy.Publisher(robot_topic, String, queue_size=10)
    rospy.init_node('pid', anonymous=True)
    pub.publish(pid_str)
    rospy.loginfo("完毕")

if __name__ == '__main__':
    try:
        pid()
    except rospy.ROSInterruptException:
        pass