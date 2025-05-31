#!/usr/bin/python3
# encoding:utf-8
import rospy
import sys,select,termios,tty
from std_msgs.msg import String

robot_param_str = "0.065:0.165:1320:5"  #轮子直径：小车间距：轮子转动一圈编码器计数:电机死区(电机最小驱动数值范围1-100，可不用设置)

def robot_param(id):
    global robot_param_str
    if "null" in id:
        pub = rospy.Publisher('robot_param', String, queue_size=10)
    else:
        pub = rospy.Publisher(id +"/robot_param", String, queue_size=10)
    rospy.init_node('robot_param', anonymous=True)
    pub.publish(robot_param_str)
    rospy.loginfo("设置完毕")

if __name__ == '__main__':
    id= sys.argv[1]
    settings = termios.tcgetattr(sys.stdin)
    robot_param(id)