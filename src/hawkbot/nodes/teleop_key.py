#!/usr/bin/python3
# encoding:utf-8
from __future__ import print_function
import rospy
from geometry_msgs.msg import Twist
import sys,select,termios,tty

msg = """
======================
   u    i    o  
   j    k    l  
   m    ,    .  
======================
前: i       后: ,
左: j       右: l
左转弯：u   右转弯：o
左后退：m   右后退：.
增加/减少线速度: w/x
增加/减少角速度: e/c
退出: CTRL-C
"""
moveBindings = {
    'i':(1,0,0,0),
    'o':(1,0,0,-1),
    'j':(0,0,0,1),
    'l':(0,0,0,-1),
    'u':(1,0,0,1),
    ',':(-1,0,0,0),
    '.':(-1,0,0,1),
    'm':(-1,0,0,-1),
    'O':(1,-1,0,0),
    'I':(1,0,0,0),
    'J':(0,1,0,0),
    'L':(0,-1,0,0),
    'U':(1,1,0,0),
    '<':(-1,0,0,0),
    '>':(-1,-1,0,0),
    'M':(-1,1,0,0),
    't':(0,0,1,0),
    'b':(0,0,-1,0),
}

speedBindings = {
    'q':(1.1,1.1),
    'z':(.9,.9),
    'w':(1.1,1),
    'x':(.9,1),
    'e':(1,1.1),
    'c':(1,.9),
}


def getKey():
    tty.setraw(sys.stdin.fileno())
    select.select([sys.stdin],[],[],0)
    key = sys.stdin.read(1)
    termios.tcsetattr(sys.stdin,termios.TCSADRAIN,settings)
    return key


def vels(speed,turn):
    return "当前速度:\tspeed %s\tturn %s " % (speed,turn)


timeBegin = 0
timeEnd = 0

if __name__ == "__main__":
    id= sys.argv[1]
    settings = termios.tcgetattr(sys.stdin)
    if "null" in id:
        pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
    else:
        pub = rospy.Publisher("/" + id + "/cmd_vel", Twist, queue_size=1)

    rospy.init_node('teleop_twist_keyboard')
    speed = rospy.get_param("~speed",0.15)
    turn = rospy.get_param("~turn",1.0)
    x = 0
    y = 0
    z = 0
    th = 0
    status = 0
    print(msg)
    print(vels(speed,turn))

    try:

        while (1):
            key = getKey()
            if key in moveBindings.keys():
                x = moveBindings[key][0]
                y = moveBindings[key][1]
                z = moveBindings[key][2]
                th = moveBindings[key][3]
            elif key in speedBindings.keys():
                speed = speed * speedBindings[key][0]
                # if (speed > 0.3):
                #     speed = 0.3
                turn = turn * speedBindings[key][1]

                print(vels(speed,turn))
                if (status == 14):
                    print(msg)
                status = (status + 1) % 15
            else:
                x = 0
                y = 0
                z = 0
                th = 0
                if (key == '\x03'):
                    break

            twist = Twist()
            twist.linear.x = x * speed
            twist.linear.y = y * speed
            twist.linear.z = z * speed
            twist.angular.x = 0
            twist.angular.y = 0
            twist.angular.z = th * turn
            pub.publish(twist)

    except Exception as e:
        print(e)

    finally:
        twist = Twist()
        twist.linear.x = 0
        twist.linear.y = 0
        twist.linear.z = 0
        twist.angular.x = 0
        twist.angular.y = 0
        twist.angular.z = 0
        pub.publish(twist)

        termios.tcsetattr(sys.stdin,termios.TCSADRAIN,settings)
