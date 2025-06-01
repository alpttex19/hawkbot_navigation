#!/usr/bin/env python3
# encoding: utf-8

import rospy
import numpy as np
import math
import tf2_ros
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped, Point
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry
from std_msgs.msg import Header
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from stable_baselines import SAC, TD3
import threading
import time
from loguru import logger

# 导入你的环境类
from my_navigation.new_car_env_dy_obs import CarEnv
from my_navigation.simulation_car_curve_obs_new import Simulation


class RLNavigationNode:
    def __init__(self):
        rospy.init_node("rl_navigation_node", anonymous=True)

        # 参数设置
        self.goal_threshold = 0.02  # 到达目标的阈值
        self.max_linear_vel = 5.0  # 最大线速度
        self.max_angular_vel = math.pi / 6  # 最大角速度
        self.min_linear_vel = 0.2  # 最小线速度

        # 状态变量
        self.current_pose = None
        self.current_goal = None
        self.laser_data = None
        self.map_data = None
        self.goal_reached = False
        self.is_navigating = False

        # TF监听器
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # 强化学习模型
        self.model = None
        self.load_model()

        # 环境仿真器（用于状态处理）
        self.sim = Simulation(dt=0.01, goal_pos=np.array([0, 0]))

        # 障碍物信息
        self.obstacles = np.zeros((2, 20))  # 动态障碍物位置
        self.static_obstacles = []  # 静态障碍物从地图中提取

        # ROS Publishers
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.goal_status_pub = rospy.Publisher(
            "/rl_nav/goal_status", PoseStamped, queue_size=1
        )

        # ROS Subscribers
        rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.goal_callback)
        rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.pose_callback)
        rospy.Subscriber("/scan", LaserScan, self.laser_callback)
        rospy.Subscriber("/map", OccupancyGrid, self.map_callback)
        rospy.Subscriber("/odom", Odometry, self.odom_callback)

        # 控制循环
        self.control_rate = rospy.Rate(50)  # 50Hz控制频率
        self.nav_thread = threading.Thread(target=self.navigation_loop)
        self.nav_thread.daemon = True
        self.nav_thread.start()

        rospy.loginfo("RL Navigation Node initialized")

    def load_model(self):
        """加载训练好的强化学习模型"""
        try:
            # 根据你的模型路径修改
            model_path = rospy.get_param(
                "~model_path",
                "./nodes/my_navigation/train_result/model14/best_model.zip",
            )
            self.model = SAC.load(model_path)
            rospy.loginfo(f"Loaded RL model from: {model_path}")
        except Exception as e:
            rospy.logerr(f"Failed to load RL model: {e}")
            self.model = None

    def goal_callback(self, msg):
        """接收导航目标"""
        try:
            # 转换目标到map坐标系
            if msg.header.frame_id != "map":
                transform = self.tf_buffer.lookup_transform(
                    "map", msg.header.frame_id, rospy.Time()
                )
                # 手动进行坐标变换
                goal_transformed = PoseStamped()
                goal_transformed.header = msg.header
                goal_transformed.header.frame_id = "map"

                # 获取变换矩阵
                translation = transform.transform.translation
                rotation = transform.transform.rotation

                # 转换位置
                goal_transformed.pose.position.x = msg.pose.position.x + translation.x
                goal_transformed.pose.position.y = msg.pose.position.y + translation.y
                goal_transformed.pose.position.z = msg.pose.position.z + translation.z

                # 转换方向
                goal_transformed.pose.orientation = rotation
            else:
                goal_transformed = msg

            self.current_goal = goal_transformed
            self.goal_reached = False
            self.is_navigating = True

            # 更新仿真器的目标
            goal_pos = np.array(
                [goal_transformed.pose.position.x, goal_transformed.pose.position.y]
            )
            self.sim.goal_pos = goal_pos

            rospy.loginfo(f"New goal received: ({goal_pos[0]:.2f}, {goal_pos[1]:.2f})")

        except Exception as e:
            rospy.logerr(f"Error processing goal: {e}")

    def pose_callback(self, msg):
        """接收机器人位姿（来自AMCL）"""
        self.current_pose = msg.pose.pose

        # 更新仿真器状态
        if self.current_pose:
            x = self.current_pose.position.x
            y = self.current_pose.position.y

            # 从四元数获取yaw角
            orientation_q = self.current_pose.orientation
            orientation_list = [
                orientation_q.x,
                orientation_q.y,
                orientation_q.z,
                orientation_q.w,
            ]
            (_, _, yaw) = euler_from_quaternion(orientation_list)

            # 更新仿真器位置（注意：这里只更新位置，速度等从里程计获取）
            self.sim._state[2] = x  # x position
            self.sim._state[3] = y  # y position
            self.sim._state[4] = yaw  # theta

    def odom_callback(self, msg):
        """接收里程计信息（用于获取速度）"""
        if hasattr(self.sim, "_state"):
            # 更新线速度和角速度
            linear_vel = math.sqrt(
                msg.twist.twist.linear.x**2 + msg.twist.twist.linear.y**2
            )
            angular_vel = msg.twist.twist.angular.z

            self.sim._state[5] = linear_vel  # v
            self.sim._state[1] = angular_vel  # w

    def laser_callback(self, msg):
        """接收激光雷达数据"""
        self.laser_data = msg
        self.extract_obstacles_from_laser(msg)

    def map_callback(self, msg):
        """接收静态地图数据"""
        self.map_data = msg
        self.extract_static_obstacles(msg)

    def extract_obstacles_from_laser(self, laser_msg):
        """从激光雷达数据提取动态障碍物"""
        if not self.current_pose:
            return

        # 获取机器人当前位置和方向
        robot_x = self.current_pose.position.x
        robot_y = self.current_pose.position.y
        orientation_q = self.current_pose.orientation
        orientation_list = [
            orientation_q.x,
            orientation_q.y,
            orientation_q.z,
            orientation_q.w,
        ]
        (_, _, robot_yaw) = euler_from_quaternion(orientation_list)

        # 从激光数据提取障碍物位置
        obstacles = []
        angle = laser_msg.angle_min

        for i, distance in enumerate(laser_msg.ranges):
            if laser_msg.range_min < distance < laser_msg.range_max:
                # 计算障碍物在世界坐标系中的位置
                obstacle_x = robot_x + distance * math.cos(robot_yaw + angle)
                obstacle_y = robot_y + distance * math.sin(robot_yaw + angle)
                obstacles.append([obstacle_x, obstacle_y])

            angle += laser_msg.angle_increment
        # 聚类处理，提取主要障碍物
        if obstacles:
            clustered_obstacles = self.cluster_obstacles(obstacles)
            # 限制障碍物数量，取最近的20个
            if len(clustered_obstacles) > 20:
                # 按距离排序
                distances = [
                    math.sqrt((obs[0] - robot_x) ** 2 + (obs[1] - robot_y) ** 2)
                    for obs in clustered_obstacles
                ]
                sorted_indices = np.argsort(distances)[:20]
                clustered_obstacles = [clustered_obstacles[i] for i in sorted_indices]

            # 更新障碍物数组
            self.obstacles = np.zeros((2, 20))
            for i, obs in enumerate(clustered_obstacles[:20]):
                self.obstacles[0, i] = obs[0]
                self.obstacles[1, i] = obs[1]

    def cluster_obstacles(self, obstacles, cluster_distance=0.5):
        """简单的障碍物聚类"""
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
                if (
                    not used[j]
                    and math.sqrt(
                        (obs[0] - other_obs[0]) ** 2 + (obs[1] - other_obs[1]) ** 2
                    )
                    < cluster_distance
                ):
                    cluster.append(other_obs)
                    used[j] = True

            # 计算聚类中心
            center_x = sum(pt[0] for pt in cluster) / len(cluster)
            center_y = sum(pt[1] for pt in cluster) / len(cluster)
            clustered.append([center_x, center_y])

        return clustered

    def extract_static_obstacles(self, map_msg):
        """从静态地图提取障碍物"""
        # 这里可以实现从occupancy grid提取静态障碍物的逻辑
        # 简化处理，主要依赖激光雷达数据
        pass

    def get_observation(self):
        """获取强化学习模型的观测状态"""
        if not self.current_pose or not self.current_goal:
            return None

        # 当前位置
        x = self.current_pose.position.x
        y = self.current_pose.position.y

        # 当前方向
        orientation_q = self.current_pose.orientation
        orientation_list = [
            orientation_q.x,
            orientation_q.y,
            orientation_q.z,
            orientation_q.w,
        ]
        (_, _, theta) = euler_from_quaternion(orientation_list)

        # 速度信息（从仿真器获取）
        v = self.sim._state[5] if hasattr(self.sim, "_state") else 0.0
        phi = self.sim._state[6] if hasattr(self.sim, "_state") else 0.0

        # 找到最近的障碍物
        min_obs_dist = float("inf")
        closest_obs_idx = 0

        for i in range(20):
            obs_x = self.obstacles[0, i]
            obs_y = self.obstacles[1, i]
            if obs_x != 0 or obs_y != 0:  # 非零表示有效障碍物
                dist = math.sqrt((x - obs_x) ** 2 + (y - obs_y) ** 2)
                if dist < min_obs_dist:
                    min_obs_dist = dist
                    closest_obs_idx = i

        # 到目标的距离
        goal_x = self.current_goal.pose.position.x
        goal_y = self.current_goal.pose.position.y
        goal_dist = math.sqrt((x - goal_x) ** 2 + (y - goal_y) ** 2)

        # 构建观测向量 [x, y, theta, v, phi, obs_dist, goal_dist]
        observation = np.array(
            [x, y, theta, v, phi, min_obs_dist, goal_dist], dtype=np.float32
        )

        return observation

    def check_goal_reached(self):
        """检查是否到达目标"""
        if not self.current_pose or not self.current_goal:
            return False

        goal_x = self.current_goal.pose.position.x
        goal_y = self.current_goal.pose.position.y
        robot_x = self.current_pose.position.x
        robot_y = self.current_pose.position.y
        logger.info(f"robot_xy: {robot_x}, {robot_y}, goal_xy: {goal_x}, {goal_y}")

        distance = math.sqrt((goal_x - robot_x) ** 2 + (goal_y - robot_y) ** 2)
        return distance < self.goal_threshold

    def publish_cmd_vel(self, action):
        """发布速度命令"""
        if action is None:
            return

        u, w = action

        # 归一化动作到实际速度范围
        u = (np.tanh(u) + 1) / 2 * (
            self.max_linear_vel - self.min_linear_vel
        ) + self.min_linear_vel
        w = np.tanh(w) * self.max_angular_vel

        # 创建Twist消息
        cmd_vel = Twist()
        cmd_vel.linear.x = u
        cmd_vel.angular.z = w

        self.cmd_pub.publish(cmd_vel)

    def stop_robot(self):
        """停止机器人"""
        cmd_vel = Twist()
        self.cmd_pub.publish(cmd_vel)

    def navigation_loop(self):
        """主导航循环"""
        while not rospy.is_shutdown():
            if self.is_navigating and self.model is not None:
                # 检查是否到达目标
                if self.check_goal_reached():
                    self.goal_reached = True
                    self.is_navigating = False
                    self.stop_robot()

                    # 发布目标达到状态
                    if self.current_goal:
                        self.goal_status_pub.publish(self.current_goal)

                    rospy.loginfo("Goal reached!")
                    continue

                # 获取观测
                observation = self.get_observation()

                if observation is not None:
                    try:
                        # 使用强化学习模型预测动作
                        action, _ = self.model.predict(observation)

                        # 发布速度命令
                        self.publish_cmd_vel(action)

                    except Exception as e:
                        rospy.logerr(f"Error in RL prediction: {e}")
                        self.stop_robot()
                else:
                    # 如果没有有效观测，停止机器人
                    self.stop_robot()
            else:
                # 不在导航状态，确保机器人停止
                if not self.is_navigating:
                    self.stop_robot()

            self.control_rate.sleep()

    def run(self):
        """运行节点"""
        rospy.loginfo("RL Navigation Node is running...")
        rospy.spin()


if __name__ == "__main__":
    try:
        nav_node = RLNavigationNode()
        nav_node.run()
    except rospy.ROSInterruptException:
        pass
