#!/usr/bin/python3
# encoding:utf-8
import time

import rospy
import sys,select,termios,tty
from std_msgs.msg import String
def servo(id):

    if "null" in id:
        pub = rospy.Publisher('servo', String, queue_size=10)
    else:
        pub = rospy.Publisher(id +"/servo", String, queue_size=10)
    
    rospy.init_node('servo', anonymous=True)
    pub.publish("1:90:1000:1")   #舵机ID:角度：时延：是否转到角度后释放舵机(0不释放，1释放)。
    # while True:
    #     time.sleep(1)
    #     pub.publish("2:20:1000:0")
    #     time.sleep(1)
    #     pub.publish("2:120:1000:0")
    rospy.loginfo("设置完毕")


if __name__ == '__main__':
    id= sys.argv[1]
    # id="null"
    settings = termios.tcgetattr(sys.stdin)
    servo(id)

    
