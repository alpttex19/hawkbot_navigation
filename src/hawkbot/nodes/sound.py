#!/usr/bin/python3
# encoding:utf-8
import rospy
import sys,select,termios,tty
from std_msgs.msg import String

sound_str = "2000:1000,800:1000,600:1000,1500:2000"


def sound(id):
    global sound_str
    
    if "null" in id:
        pub = rospy.Publisher('sound', String, queue_size=10)
    else:
        pub = rospy.Publisher(id +"/sound", String, queue_size=10)
    
    rospy.init_node('sound', anonymous=True)
    pub.publish(sound_str)
    rospy.loginfo("播放完毕")


if __name__ == '__main__':
    
    id= sys.argv[1]
    settings = termios.tcgetattr(sys.stdin)
    sound(id)

    
