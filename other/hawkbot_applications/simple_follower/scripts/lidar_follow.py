#!/usr/bin/python3
# encoding:utf-8

import rospy
import threading
import time
import numpy as np
from sensor_msgs.msg import Joy, LaserScan
from geometry_msgs.msg import Twist, Vector3
from simple_follower.msg import position as PositionMsg
from std_msgs.msg import String as StringMsg
from dynamic_reconfigure.server import Server
from simple_follower.cfg import laser_paramsConfig
from std_msgs.msg import Int8

angle = [0.0] * 3
distan = [0.0] * 3


class Follower:
    def __init__(self):

        # 当我们停止接收来自ps3控制器的Joy消息时，我们将停止所有移动:
        # self.controllerLossTimer = threading.Timer(1, self.controllerLoss) # 当失去连接时，将速度置零
        # self.controllerLossTimer.start() # 启动线程
        self.switchMode = rospy.get_param('~switchMode')  # 若该值设置为False，则必须一直按O键才能移动
        self.max_speed = rospy.get_param('~maxSpeed')  # 获取速度上限
        self.controllButtonIndex = rospy.get_param('~controllButtonIndex')

        self.buttonCallbackBusy = False
        self.active = False
        self.i = 0
        # 发布速度话题							话题名     消息类型    队列长度
        self.cmdVelPublisher = rospy.Publisher('/cmd_vel', Twist, queue_size=3)
        # tracker话题，给予我们正在追踪的目标的当前位置信息

        # 订阅目标位置话题											话题名					消息类型				回调函数
        self.positionSubscriber = rospy.Subscriber('/object_tracker/current_position', PositionMsg,
                                                   self.positionUpdateCallback)

        # 来自tracker的字符串信息。例如，告诉我们是否丢失了对象
        self.trackerInfoSubscriber = rospy.Subscriber('/object_tracker/info', StringMsg,
                                                      self.trackerInfoCallback)  # 订阅消息话题

        # 当按Ctrl+C终止运行时，调用回调函数controllerLoss使小车速度置零并发布消息
        rospy.on_shutdown(self.controllerLoss)

    def publish_flag(self):

        laser_follow_flag = Int8()
        laser_follow_flag.data = 1
        laserfwflagPublisher.publish(laser_follow_flag)
        rospy.loginfo('a=%d', laser_follow_flag.data)


    def trackerInfoCallback(self, info):
        # we do not handle any info from the object tracker specifically at the moment. just ignore that we lost the object for example
        # 我们不处理此时目标跟踪回传的任何信息，就当做此时没有跟踪的目标物
        rospy.logwarn(info.data)

    def positionUpdateCallback(self, position):

        angleX = position.angleX  # 与目标之间的角度
        distance = position.distance  # 与目标之间的距离

        angularSpeed = angleX
        if distance > 0.4:
            distance = 0.4

        linearSpeed = distance - 0.2

        # create the Twist message to send to the cmd_vel topic
        # 创建Twist消息发送到cmd_vel话题
        velocity = Twist()
        velocity.linear = Vector3(linearSpeed, 0, 0.)  # 线速度
        velocity.angular = Vector3(0., 0., angularSpeed)  # 角速度
        self.cmdVelPublisher.publish(velocity)
        if self.i < 10:
            self.i = self.i + 1
        elif self.i == 10:  # 语音识别标志
            self.publish_flag()
            self.i = 11

    # 将速度置零
    def stopMoving(self):
        velocity = Twist()
        velocity.linear = Vector3(0., 0., 0.)
        velocity.angular = Vector3(0., 0., 0.)
        self.cmdVelPublisher.publish(velocity)

    def controllerLoss(self):
        # we lost connection so we will stop moving and become inactive
        # 连接丢失，停止移动，并变成不活跃状态
        self.stopMoving()
        self.active = False
        rospy.loginfo('lost connection')


if __name__ == '__main__':

    print('starting')
    laserfwflagPublisher = rospy.Publisher('/laser_follow_flag', Int8, queue_size=1)
    rospy.init_node('follower')
    follower = Follower()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        print('exception')


