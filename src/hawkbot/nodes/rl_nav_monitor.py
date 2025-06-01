#!/usr/bin/env python3
# encoding: utf-8

import rospy
import numpy as np
import math
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray, String
from visualization_msgs.msg import Marker, MarkerArray
import tf2_ros
from tf.transformations import euler_from_quaternion

class RLNavMonitor:
    """RL导航监控节点 - 用于调试和可视化"""
    
    def __init__(self):
        rospy.init_node('rl_nav_monitor', anonymous=True)
        
        # 状态变量
        self.current_pose = None
        self.current_goal = None
        self.current_cmd_vel = None
        self.laser_data = None
        self.obstacles = []
        
        # 发布器
        self.debug_pub = rospy.Publisher('/rl_nav/debug_info', String, queue_size=1)
        self.obstacle_marker_pub = rospy.Publisher('/rl_nav/obstacles', MarkerArray, queue_size=1)
        self.state_pub = rospy.Publisher('/rl_nav/state', Float64MultiArray, queue_size=1)
        
        # 订阅器
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.pose_callback)
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback)
        rospy.Subscriber('/cmd_vel', Twist, self.cmd_vel_callback)
        rospy.Subscriber('/scan', LaserScan, self.laser_callback)
        rospy.Subscriber('/odom', Odometry, self.odom_callback)
        
        # 定时器
        self.monitor_timer = rospy.Timer(rospy.Duration(0.5), self.monitor_callback)
        
        rospy.loginfo("RL Navigation Monitor initialized")
    
    def pose_callback(self, msg):
        """接收位姿信息"""
        self.current_pose = msg.pose.pose
    
    def goal_callback(self, msg):
        """接收目标信息"""
        self.current_goal = msg
        rospy.loginfo(f"New goal: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})")
    
    def cmd_vel_callback(self, msg):
        """接收速度命令"""
        self.current_cmd_vel = msg
    
    def laser_callback(self, msg):
        """接收激光数据并提取障碍物"""
        self.laser_data = msg
        self.extract_obstacles(msg)
        self.publish_obstacle_markers()
    
    def odom_callback(self, msg):
        """接收里程计信息"""
        pass
    
    def extract_obstacles(self, laser_msg):
        """从激光数据提取障碍物"""
        if not self.current_pose:
            return
        
        robot_x = self.current_pose.position.x
        robot_y = self.current_pose.position.y
        orientation_q = self.current_pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (_, _, robot_yaw) = euler_from_quaternion(orientation_list)
        
        obstacles = []
        angle = laser_msg.angle_min
        
        for i, distance in enumerate(laser_msg.ranges):
            if laser_msg.range_min < distance < min(laser_msg.range_max, 5.0):  # 限制最大距离
                obstacle_x = robot_x + distance * math.cos(robot_yaw + angle)
                obstacle_y = robot_y + distance * math.sin(robot_yaw + angle)
                obstacles.append([obstacle_x, obstacle_y, distance])
            angle += laser_msg.angle_increment
        
        # 简单聚类
        self.obstacles = self.cluster_obstacles(obstacles)
    
    def cluster_obstacles(self, obstacles, cluster_distance=0.5):
        """障碍物聚类"""
        if not obstacles:
            return []
        
        clustered = []
        used = [False] * len(obstacles)
        
        for i, obs in enumerate(obstacles):
            if used[i]:
                continue
            
            cluster = [obs]
            used[i] = True
            
            for j, other_obs in enumerate(obstacles):
                if not used[j]:
                    dist = math.sqrt((obs[0] - other_obs[0])**2 + (obs[1] - other_obs[1])**2)
                    if dist < cluster_distance:
                        cluster.append(other_obs)
                        used[j] = True
            
            # 计算聚类中心和最小距离
            center_x = sum(pt[0] for pt in cluster) / len(cluster)
            center_y = sum(pt[1] for pt in cluster) / len(cluster)
            min_dist = min(pt[2] for pt in cluster)
            clustered.append([center_x, center_y, min_dist])
        
        # 按距离排序，取最近的20个
        clustered.sort(key=lambda x: x[2])
        return clustered[:20]
    
    def publish_obstacle_markers(self):
        """发布障碍物可视化标记"""
        marker_array = MarkerArray()
        
        for i, obs in enumerate(self.obstacles):
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = rospy.Time.now()
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            marker.pose.position.x = obs[0]
            marker.pose.position.y = obs[1]
            marker.pose.position.z = 0.1
            marker.pose.orientation.w = 1.0
            
            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.3
            
            # 根据距离设置颜色
            if obs[2] < 1.0:  # 很近的障碍物 - 红色
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
            elif obs[2] < 2.0:  # 中等距离 - 黄色
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 0.0
            else:  # 远距离 - 绿色
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
            
            marker.color.a = 0.7
            marker_array.markers.append(marker)
        
        # 清除多余的标记
        for i in range(len(self.obstacles), 50):
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.id = i
            marker.action = Marker.DELETE
            marker_array.markers.append(marker)
        
        self.obstacle_marker_pub.publish(marker_array)
    
    def get_current_state(self):
        """获取当前状态向量"""
        if not self.current_pose or not self.current_goal:
            return None
        
        # 机器人位置和方向
        x = self.current_pose.position.x
        y = self.current_pose.position.y
        orientation_q = self.current_pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (_, _, theta) = euler_from_quaternion(orientation_list)
        
        # 速度
        v = 0.0
        w = 0.0
        if self.current_cmd_vel:
            v = self.current_cmd_vel.linear.x
            w = self.current_cmd_vel.angular.z
        
        # 目标位置
        goal_x = self.current_goal.pose.position.x
        goal_y = self.current_goal.pose.position.y
        goal_dist = math.sqrt((x - goal_x)**2 + (y - goal_y)**2)
        
        # 最近障碍物距离
        min_obs_dist = float('inf')
        if self.obstacles:
            min_obs_dist = min(obs[2] for obs in self.obstacles)
        
        return [x, y, theta, v, w, goal_dist, min_obs_dist, len(self.obstacles)]
    
    def monitor_callback(self, event):
        """监控回调函数"""
        state = self.get_current_state()
        
        if state is not None:
            # 发布状态信息
            state_msg = Float64MultiArray()
            state_msg.data = state
            self.state_pub.publish(state_msg)
            
            # 创建调试信息
            debug_info = f"""
RL Navigation Status:
Position: ({state[0]:.2f}, {state[1]:.2f})
Orientation: {math.degrees(state[2]):.1f}°
Linear Vel: {state[3]:.2f} m/s
Angular Vel: {state[4]:.2f} rad/s
Goal Distance: {state[5]:.2f} m
Min Obstacle Distance: {state[6]:.2f} m
Obstacles Count: {int(state[7])}
"""
            
            # 发布调试信息
            debug_msg = String()
            debug_msg.data = debug_info
            self.debug_pub.publish(debug_msg)
            
            # 打印到控制台（可选）
            if rospy.get_param('~verbose', False):
                rospy.loginfo(debug_info)
    
    def run(self):
        """运行监控节点"""
        rospy.loginfo("RL Navigation Monitor is running...")
        rospy.spin()

if __name__ == '__main__':
    try:
        monitor = RLNavMonitor()
        monitor.run()
    except rospy.ROSInterruptException:
        pass
