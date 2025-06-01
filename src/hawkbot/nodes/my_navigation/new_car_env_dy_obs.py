import math
import gym
import numpy as np
from gym import spaces
from numpy import random
from .simulation_car_curve_obs_new import Simulation
from .viewer14 import Viewer

# from viewer import Viewer
from stable_baselines import SAC, TD3
from .gmm.gmm_utils import gmr_lyapunov
from .trajectory import load_V
import tensorflow as tf

REWARD_LIMIT = 500


class CarEnv(gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array"],
        "video.frames_per_second": 30,  # 视频每秒播放的帧数
    }

    def __init__(self):
        """
        Action spaces
            1. forward a
            2. change in rotational w

        Observation Space:
            1. x
            2. y
            3. ob0~ob3
            4. d1~d4

        Simulation: a,w,x,y,θ,v,φ

        """
        # Time delta per step
        self.dt = 0.01  # 0.01
        ## Environment parameters
        ### obstacle_central position
        # self.obstacle = np.array([[-7, 10, -17, 17, 15, -22, -8],
        #                   [-10, 6, 0, -13, 0, -6, 13]])    #简单地图
        # self.obstacle = np.array([[7, 10, 17, 17, 27, 22, 27],
        #                           [13, 6, 0, 13, 0, 6, 13]])  #原来位置
        # self.obstacle = np.array([[0, 10, -10, 5, 18, 20, 23],
        #                           [10, 0, 22, 20, 24, 5, -10]]) #复杂地图1
        # self.obstacle = np.array([[7, 15, 17, 19, 27, 22, 27],
        #                           [20, 16, 10, 26, 20, 6, 13]])  #第一象限地图
        # self.obstacle = np.array([[0, 0, 0, 0, 0, 0, 0],
        #                           [30, 30, 30, 30, 30, 30, 30]])  #地图边界外，当作地图里无障碍。
        ##尝试随机生成障碍(自己加的)
        self.obs_num = 20
        self.obstacle = np.zeros((2, self.obs_num))  # (2,7)
        for i in range(self.obs_num):
            #     # obs_x = random.choice(list(range(-5, -20)) + list(range(5, 20)))
            #     # obs_y = random.choice(list(range(-5, -20)) + list(range(5, 20)))
            obs_x = random.choice(list(range(-3, 0)) + list(range(0, 25)))  # (-3,25)
            obs_y = random.choice(list(range(-3, 0)) + list(range(0, 25)))
            self.obstacle[0, i] = obs_x
            self.obstacle[1, i] = obs_y

        # self.obstacle = np.zeros((2, 4))
        # for i in range(4):
        #     obs_x = random.choice(range(-25, 0))
        #     obs_y = random.choice(range(-25, 0))
        #     self.obstacle[0, i] = obs_x
        #     self.obstacle[1, i] = obs_y

        self.goal_pos = np.array([0, 0])

        self.sim = Simulation(self.dt, self.goal_pos)
        self.get_virtual_position = lambda: self.sim.position  # 获取虚拟位置
        self.get_virtual_theta = lambda: self.sim.theta  # 这里修改了的，和海森的不一样
        # self.seed(2)
        self.testItr = 200

        # Boundaries of the action
        self.MIN_ACC = 0.2
        self.MAX_ACC = 1.5
        self.MAX_omiga = np.pi / 6
        self.is_discrete_action = False

        # how close to goal = reach goal  离目标还有多远
        # self.dist_threshold = 0.1
        self.dist_threshold = 0.6
        # self.dist_threshold = 1.0

        # how close to obstacle = crash obstacle 离障碍物多近属于碰到障碍物
        # self.obs_threshold = 1.0  #原本
        self.obs_threshold = 0.1  # 第一象限时，0.7
        self.obs_index = 0

        # Action and observation spaces
        self.FIELD_SIZE_x_low = -5.0
        self.FIELD_SIZE_x_up = 30.0
        self.FIELD_SIZE_y_low = -5.0  # -15,-5
        self.FIELD_SIZE_y_up = 30.0  # 25,30，5
        self.theta_low = -np.pi
        self.theta_high = np.pi
        self.v_low = 0
        self.v_high = 5  # 5
        self.phi_low = -np.pi
        self.phi_high = np.pi  # pi/4
        self.obs_dist_low = -5  # -5
        self.obs_dist_high = 30  # 30
        self.goal_dist_low = -5  # -5
        self.goal_dist_high = 30  # 30
        self.state_low = np.array(
            [
                self.FIELD_SIZE_x_low,
                self.FIELD_SIZE_y_low,
                self.theta_low,
                self.v_low,
                self.phi_low,
                self.obs_dist_low,
                self.goal_dist_low,
            ],
            dtype=np.float32,
        )
        self.state_high = np.array(
            [
                self.FIELD_SIZE_x_up,
                self.FIELD_SIZE_y_up,
                self.theta_high,
                self.v_high,
                self.phi_high,
                self.obs_dist_high,
                self.goal_dist_high,
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=np.array([self.MIN_ACC, -self.MAX_omiga], dtype=np.float32),
            high=np.array([self.MAX_ACC, self.MAX_omiga], dtype=np.float32),
            dtype=np.float32,
        )  # 这里的action_space与原来海森的环境里的一样
        self.observation_space = spaces.Box(
            low=self.state_low,
            high=self.state_high,
            # low=np.concatenate([self.state_low, [-10] * 7]),
            # high=np.concatenate([self.state_high, [10] * 7]),
            dtype=np.float32,
        )

        # reward of V
        self.gparameter, self.Vxf = load_V()

        self.viewer = None
        # self.reset()  #这里是否注释掉不影响shape(14,) into shape(7,)的报错

    def sample_action(self):
        if self.is_discrete_action:  # 第83行，is_discrete 是设为 false
            # a = np.random.choice(list(range(3)))       #原本
            action = np.random.choice(list(range(3)))
        else:
            action = np.random.uniform(self.action_space, size=2)
        return action

    def _is_goal_reached(self):
        """
        Check if goal is reached.
        """
        return self._get_goal_dist() < self.dist_threshold

    def _is_done(self):
        return (
            self.sim.is_invalid()
            or self._is_goal_reached()
            or self.crash()
            or self.sim.time > 50
        )

    def _get_dist(self, p1: np.ndarray, p2: np.ndarray):
        return np.linalg.norm(p1 - p2)

    def _get_theta(self):
        return np.abs(self.sim.theta) < 0.1

    def _get_goal_dist(self):
        return self._get_dist(self.get_virtual_position(), self.goal_pos)

    def _get_obstacle_dist(self):
        return self._get_dist(self.get_virtual_position(), self.obstacle)

    def crash(self):
        """
        Check if goal is Carshed.
        """
        # return self._get_obstacle_dist() < self.obs_threshold \
        #        or self._get_dist(self.get_virtual_position(), self.obstacle1) < self.obs_threshold \
        #        or self._get_dist(self.get_virtual_position(),self.obstacle2)<self.obs_threshold\
        #        or self._get_dist(self.get_virtual_position(),self.obstacle3)<self.obs_threshold
        return self.barrier(self.get_virtual_position(), self.obstacle) > 1e-7

    def euclidean_distance(self, x, x_obs, Sum=True):
        # 计算向量的欧氏距离
        diff = x - x_obs
        squared_diff = np.square(diff)
        if Sum is True:
            summed = np.sum(squared_diff, axis=0)
        else:
            summed = squared_diff
        distance = np.sqrt(summed)
        return distance  # 1xN

    def barrier(self, x, x_so, k=1.1):  # 判断跟所有障碍物有没有发生至少一处的碰撞
        g = [np.array(())]
        G = []
        G_new = []
        G_obs = []
        r = 0.5  # 假设障碍物半径一致 #猜测是障碍物实际撞击范围 1.5
        R = 0.35  # 1.35
        gain = 1.4
        xi = np.array(0.00000000001)
        obs_R = np.array((r + R) * k)
        nbData = np.shape(x)
        nbobs = len(x_so.T)
        if nbobs <= 2:
            x_obs = x_so
            a = self.euclidean_distance(x, x_obs)
            theta = a - obs_R
            c = np.sqrt(np.square(theta) + 4 * xi)
            g = 0.5 * (c - theta)
            G.append(g * gain)
        else:
            num_obs = len(x_so[1, :])
            for i in range(num_obs):
                x_obs = x_so[:, i]
                a = self.euclidean_distance(x, x_obs)  # 欧式距离   [:, np.newaxis]
                # a = np.reshape(a, [1, nbData])
                theta = a - obs_R
                c = np.sqrt(np.square(theta) + 4 * xi)
                g = 0.5 * (c - theta)
                obs_indx = g > 1e-6
                G.append(g * gain)
        G_obs = np.sum(G, axis=0)
        # print(self.G_obs )
        # G_obs_hist.append(self.G_obs)
        return G_obs

    def obs_check(self, x, x_so, k=1.1):  # 找出是哪一个障碍物
        r = 0.5  # 假设障碍物半径一致 原是1.5
        R = 0.35  # 原1.35
        G = []
        xi = np.array(0.00000000001)
        obs_R = np.array((r + R) * k)
        for i in range(len(x_so[1, :])):
            x_obs = x_so[:, i]
            # a_ = self.euclidean_distance(x, x_obs)  # 欧式距离  [:, np.newaxis]
            a = self._get_dist(x, x_obs)
            # a = np.reshape(a, [1, np.shape(x)[1]])
            theta = a - obs_R
            c = np.sqrt(np.square(theta) + 4 * xi)
            g = 0.5 * (c - theta)
            G.append(g.squeeze())
            # print(a)   #加的，想看看a存的是什么数据
        self.obs_index = G.index(max(G))
        return self.obs_index

    def _get_real_reward(self):
        reach_reward = 0
        reward_directional = 0
        # punish_directional = 0 #
        reach_weight = 50  # （加的）
        target_weight = 0.001  # 0.01  #（加的）0.001
        direct_weight = 100  # 朝向权重  #sac 10, td3 200

        if self._is_goal_reached():  # （加的）
            reach_reward += (
                100 / self.sim.time
            )  # 100/sim.time暂时不知道具体如何从训练中获取，测试结果是固定为49.9

        # 目标奖励（加的）
        last_goal_dis = np.linalg.norm(self._last_pos - self.goal_pos)
        goal_distance = np.linalg.norm(self.sim.position - self.goal_pos)
        tar_velocity = 0.3  # 0.1  #0.03  ##状态反馈频率的时间步长，越小说明越频繁，0.3
        target_reward = np.max(
            [-tar_velocity, np.min([last_goal_dis - goal_distance, tar_velocity])]
        )

        # 新·朝向奖励（自制）
        delta_x = self.goal_pos[0] - self.sim.position[0]
        delta_y = self.goal_pos[1] - self.sim.position[1]
        pre_distance = np.sqrt(delta_x**2 + delta_y**2)
        agv_theta = self.sim.theta
        agv_goal_theta = np.arctan2(delta_y, delta_x)
        # if np.pi * 5/6 <= abs(agv_goal_theta + agv_theta) <= np.pi * 7/6:
        #     reward_directional = 0.02  #朝向奖励，若朝向目标点，则奖励0.02
        # if np.pi * 7/6 <= abs(agv_goal_theta + agv_theta) or abs(agv_goal_theta + agv_theta) <= np.pi * 2/3:
        #     reward_directional = -0.01 #（目前最优） 5/4,3/4
        # 新朝向函数2.0
        # if abs(agv_goal_theta - agv_theta) > np.pi * 1 / 4:  # 2024.7.14,尝试
        #     reward_directional = -0.01

        if agv_theta >= 0:
            if abs(agv_goal_theta - agv_theta) > np.pi * 1 / 4:  # 1/2，1/4
                reward_directional = -0.01
        else:
            if (
                abs(agv_goal_theta - agv_theta) > np.pi * 1 / 4
                or abs(agv_goal_theta - agv_theta) < np.pi * 7 / 4
            ):  # 3/2，7/4,5/3
                reward_directional = -0.01

        # #新朝向函数2.1，，试验失败，有待改进
        # if agv_theta >= 0:
        #     if agv_goal_theta <= np.pi and abs(agv_goal_theta - agv_theta) > np.pi * 1/4:  #1/2
        #         reward_directional = -0.01
        #     elif agv_goal_theta > np.pi and abs(agv_goal_theta + agv_theta) > agv_goal_theta + np.pi * 1/4:
        #         reward_directional = -0.01
        # if agv_theta < 0:
        #     if agv_goal_theta <= np.pi:
        #         if abs(agv_goal_theta - agv_theta) > np.pi * 1/4 or abs(agv_goal_theta - agv_theta) < np.pi * 7/4: #3/2
        #             reward_directional = -0.01
        #     elif agv_goal_theta > np.pi:
        #         if abs(agv_goal_theta - agv_theta) > np.pi*(2 + 1/4) or abs(agv_goal_theta - agv_theta) < np.pi*(1 + 3/4):
        #             reward_directional = -0.01

        # （加的）
        goal_reward = (
            reach_weight * reach_reward
            + target_weight * target_reward
            + direct_weight * reward_directional
        )
        if self.crash() or self.sim.is_invalid():
            goal_reward = -1000  # 如果撞到障碍或者出界，惩罚
            # 效果：1w和1k都偏大了，导致小车一直不敢靠近，会离开障碍物，感觉上会延长收敛时间）
        # if self.sim.is_invalid():
        #     goal_reward = -1000
        # if self.crash():
        #     goal_reward = -800
        # if abs(self.goal_pos[0] - self.sim.position[0])>5 and abs(self.goal_pos[1] - self.sim.position[1])>5:
        #     goal_reward -= 10
        L, dv = gmr_lyapunov(
            self.get_virtual_position(),
            self.obstacle[:, self.obs_index],
            self.Vxf["Priors"],
            self.Vxf["Mu"],
            self.Vxf["P"],
        )
        V = -L / 1e5 + goal_reward
        # V = -L / 1e5 + goal_reward + direct_weight * reward_directional - invalid_weight * self.sim.is_invalid()
        # V = -L / 1e7 + goal_reward
        self._print_info(V)
        return V

    #             #Computing the distance between obstacles
    #     # far count
    #     far_value = 0.2
    #     far = np.linalg.norm(self.sim.position - self.obstacle) > (self.obs_threshold + far_value) and \
    #           np.linalg.norm(self.sim.position - self.obstacle1) > (self.obs_threshold + far_value)and \
    #           np.linalg.norm(self.sim.position - self.obstacle2) > (self.obs_threshold + far_value) and \
    #           np.linalg.norm(self.sim.position - self.obstacle3) > (self.obs_threshold + far_value)
    #
    #     # near count
    #     near_value = 0.1
    #     near = self._get_dist(self.sim.position, self.obstacle) <= (self.obs_threshold + near_value) \
    #            or self._get_dist(self.sim.position, self.obstacle1) <= (self.obs_threshold + near_value)\
    #            or self._get_dist(self.sim.position, self.obstacle2) <= (self.obs_threshold + near_value)\
    #            or self._get_dist(self.sim.position, self.obstacle3) <= (self.obs_threshold + near_value)\
    #
    #     # obstacle Reward
    #     obstacle_cost = -100 if near else 0.1
    #
    #     # Directional Reward
    #     reward_directional = (np.pi - np.abs(self.sim.theta) * 5) * 0.1
    #     if reward_directional < 0:
    #         reward_directional *= 4
    #         if reward_directional < -np.pi * 2:
    #             reward_directional = -np.pi * 2
    #     # Distance reward ---sparse
    #     last_goal_dis = np.linalg.norm(self._last_pos - self.goal_pos)
    #     goal_distance = np.linalg.norm(self.sim.position - self.goal_pos)
    #     tar_velocity = 0.03
    #     target_reward = np.max([-tar_velocity, np.min([last_goal_dis - goal_distance, tar_velocity])])
    #     self._last_pos = self.sim.position
    #     # weight
    #     reach_weight = 1000 #+ 100/self.sim.time
    #     target_weight = 100
    #     obs_weight = 1
    #     far_weight = 0.2
    #     direction_weight = 0
    #     is_invalid_weight = 0.1
    #     # sum(reward)
    #     goal_reward = target_reward * target_weight + reach_weight * self._is_goal_reached() + 0.1 / self.sim.time
    #     final_reward = goal_reward +  \
    #                     obs_weight  * obstacle_cost +\
    #                     far * far_weight + \
    #                     reward_directional * direction_weight -\
    #                     is_invalid_weight * self.sim.is_invalid()
    #     return final_reward
    def _get_observation(self):

        x, y = self.get_virtual_position()
        theta = self.sim.theta
        v = self.sim.v
        phi = self.sim.phi
        # goal_relative = np.array([
        #     [np.cos(-theta), -np.sin(-theta)],
        #     [np.sin(-theta), np.cos(-theta)]
        # ]).dot(self.goal_pos - np.array([x, y]))

        # def get_rel(pos):
        #     return np.array([
        #         [np.cos(-theta), -np.sin(-theta)],
        #         [np.sin(-theta), np.cos(-theta)]
        #     ]).dot(pos - np.array([x, y]))

        # Test_observation
        # pos_info = np.hstack([goal_relative,
        #                       self.sim.position,
        #                       self.goal_pos,
        #                       self.goal_pos - self.sim.position,
        #                       self.obstacle,
        #                       self.obstacle1,
        #                       get_rel(self.obstacle),
        #                       get_rel(self.obstacle1),
        #                       np.linalg.norm(self.sim.position - self.obstacle),
        #                       np.linalg.norm(self.sim.position - self.obstacle1)
        #                       ])

        # pos_info = np.hstack([goal_relative,
        #                       get_rel(self.obstacle),
        #                       # get_rel(self.obstacle1),
        #                       # get_rel(self.obstacle2),
        #                       # get_rel(self.obstacle3),
        #                       np.linalg.norm(self.sim.position - self.obstacle),
        #                       np.linalg.norm(self.sim.position - self.obstacle1),
        #                       np.linalg.norm(self.sim.position - self.obstacle2),
        #                       np.linalg.norm(self.sim.position - self.obstacle3)
        #                       ])
        index = self.obs_check(self.get_virtual_position(), self.obstacle)

        pos_info = np.hstack(
            [
                x,
                y,
                theta,
                v,
                phi,
                # get_rel(self.obstacle1),
                # self.barrier(self.get_virtual_position(), self.obstacle[:, index], k=1.1),
                self._get_dist(self.sim.position, self.obstacle[:, index]),
                self._get_dist(self.sim.position, self.goal_pos),
                # 0, 0,
            ]
        )
        # print(x,y)
        return pos_info

    def step(self, action):  # 记录每一单步的信息
        """
        Args: np.ndarray
            - action (tuple): a and w .
        Returns:
            - observation (object):
            - reward (float):
            - done (boolean):
            - info (dict):
        """

        u, w = action
        # normalize the range of action is 0.2 to 1.5
        u = (np.tanh(u) + 1) / 2 * (self.MAX_ACC - self.MIN_ACC) + self.MIN_ACC
        # normalize
        w = np.tanh(w) * self.MAX_omiga

        # update the car position
        self.sim.step(np.array([u, w]))

        # compute distance rate
        dis_rate = self._get_goal_dist() / self.orig_dis * 10

        ##  动态障碍物，每一步应该怎么更新
        # while True:
        #     for i in range(self.obs_num):
        #         # # 假设上下垂直方向的速度是1，这里可以根据实际情况修改
        #         if i in range(self.obs_num - 2):
        #             self.obstacle[1, i] += 0.01
        #         else:
        #             self.obstacle[1, i] -= 0.01
        #         # 碰到边界时自动反弹
        #         if self.obstacle[1, i] > 5:  #0和-30是地图边界，可修改
        #             self.obstacle[1, i] = -30
        #         elif self.obstacle[1, i] < -30:
        #             self.obstacle[1, i] = 5
        # 假设左右水平方向的速度是1
        # if i in range(self.obs_num - 3):
        #     self.obstacle[0, i] += 0.01
        # else:
        #     self.obstacle[0, i] -= 0.01
        # # 碰到边界时自动反弹
        # if self.obstacle[0, i] > 5:  # 0和-30是地图边界，可修改
        #     self.obstacle[0, i] = -30
        # elif self.obstacle[0, i] < -30:
        #     self.obstacle[0, i] = 5
        # break
        return (
            (self._get_observation()),
            (self._get_real_reward()),
            self._is_done(),
            {"d": dis_rate},
        )

    def _print_info(self, reward):
        frequency = 50
        if self._is_done() or self.sim.ticks % np.round(1 / self.dt / frequency) == 0:
            u, w = self.sim.speed
            x, y = self.sim.position
            # print(f"T {self.sim.time}: Pos ({x:.4f}, {y:.4f}), action ({u:.4f}, {w:.4f}), reward {reward}")

    def reset(self):
        ## try to generate obstacle centeral point，试出来了，随机障碍物位置刷新点
        self.obstacle = np.zeros((2, self.obs_num))
        for i in range(self.obs_num):
            obs_x = np.random.choice(range(-3, 25))  # (-3,25)
            obs_y = np.random.choice(range(-3, 25))
            self.obstacle[0, i] = obs_x
            self.obstacle[1, i] = obs_y

        # genenrate goal pointg
        # N = self.obs_num
        # while True:
        #     rand_x = np.random.uniform(self.FIELD_SIZE_x_low, self.FIELD_SIZE_x_up)
        #     rand_y = np.random.uniform(self.FIELD_SIZE_y_low, self.FIELD_SIZE_y_up)
        #     curr = np.array(rand_x, rand_y)
        #     for i in range(0, self.obs_num):
        #         if np.linalg.norm(curr-self.obstacle[:, i]) > self.obs_threshold:
        #             N-=1
        #     if N == 0:
        #         break

        # reset simulation
        self.goal_pos = np.array(
            [0, 0]
        )  # 目标点位置的真正设置地方，原定位置坐标（0，0）
        self.sim.reset_mode(self.testItr, self.goal_pos)
        self._last_pos = self.sim.position
        self.sim.reset()
        self.orig_dis = self._get_dist(self.sim.goal_pos, self.sim.start_point)
        return self._get_observation()

    def render(self, mode="human"):
        if self.viewer is None:
            self.viewer = Viewer(self)
        return self.viewer.render(mode)

    def close(self):
        if self.viewer:
            self.viewer.viewer.close()
            self.viewer = None


if __name__ == "__main__":
    model = SAC.load(r"./train_result/model14/best_model.zip")
    # model = TD3.load(r'S:\yuanE\pycharm\train_result\TD3\model-sac22\best_model.zip')
    # g, vx = load_V()

    np.random.seed(1)
    env = CarEnv()
    for ep in range(100):
        sum_r = 0
        s = env.reset()
        # print("Training_state:", s)
        # print("orig_position：", env.sim._state)
        # print("goal",env.goal_pos)
        while True:
            env.render()
            # nb_actions = env.action_space.shape[0]
            # print("action:",nb_actions)
            # action = env.sample_action()
            # print("action:",action)
            action = model.predict(s)[0]

            s, r, done, _ = env.step(action)
            # print("eposid：", t)
            # print("update_state:", s)
            print("reward:", r)
            print("更新后小车的距离比率：", _)
            sum_r += r
            print("累积奖励", sum_r)
            # print("--------------------------------------------------------")
            if done:
                env._print_info(sum_r)
                break
        # exit(0)
