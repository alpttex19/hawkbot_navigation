import math
import numpy as np

class Simulation:

    def __init__(self, dt, goal_pos):

        self.FIELD_SIZE = 30.  #地图边界值
        # [v,w,x,y,θ,φ]×   #[a, w, x, y, theta, v, φ]√
        self._state = np.array([0., 0., 0., 0., 0., 0., 0.])    #2，3就是x，y
        # time
        self.ticks = 0
        self.dt = dt
        # self.dθ = math.pi / 30
        self.L = 0.001  #小车轴距，原0.001，后来改成1.4  ##2024.7.12，试着改成0.5
        self.goal_pos = goal_pos
    @property
    def time(self):
        return round(self.ticks * self.dt, 4)

    def is_invalid(self):
        _, _, x, y, _, _, _ = self._state
        # return x < -self.FIELD_SIZE or x > self.FIELD_SIZE or y < -self.FIELD_SIZE or y > self.FIELD_SIZE
        return x <= -5 or x > self.FIELD_SIZE or y <= -5 or y > self.FIELD_SIZE  #限制在第一象限，我测试的
        # return x <= -30 or x > 5 or y <= -30 or y > 5   #第三象限

    @property
    def speed(self):
        return self._state[[5, 1]]   #v，w
    @property
    def position(self):
        return self._state[[2, 3]]   #即小车的x，y位置坐标

    @property
    def theta(self):
        return self._state[4]  #theta

    @property
    def a(self):
        return self._state[0]  #a 加速度

    @property
    def v(self):
        return self._state[5]  #v 线速度

    @property
    def phi(self):
        return self._state[6]  #phi


    def step(self, action):
        next_a, next_w = action
        if next_w > 0:
            if 1-next_w < 0.1:
                next_w = next_w/2
        else:
            if 1 + next_w < 0.1:
                next_w = next_w/2

        a, w, x, y, theta, v, φ = self._state

        # Update state
        new_v = v + next_a * self.dt
        if new_v > 5:   #限速有效，原先限制为5
            new_v = 5
        # new_φ = φ + next_w * self.dt  #2024.7.10试用这个  #ps：好像不行
        new_φ =  next_w * self.dt   #之前SAC都是用的这个
        # if new_φ > np.pi/4:
        #     new_φ = np.pi/4
        # else:
        #     if new_φ < -np.pi/4:
        #         new_φ = np.pi/4
        new_theta = theta + (new_v * np.tan(new_φ) * self.dt / self.L)
        while new_theta > np.pi:
            new_theta -= np.pi * 2
        while new_theta < -np.pi:
            new_theta += np.pi * 2
        new_x = x + new_v * np.cos(theta) * self.dt
        new_y = y + new_v * np.sin(theta) * self.dt
    
        self.ticks += 1
        
        self._state = np.array([next_a, next_w, new_x, new_y, new_theta, new_v, new_φ])

    def reset(self):
        rand_x = np.random.uniform(5, 25)
        rand_y = np.random.uniform(5, 25)
        rand_theta = np.random.uniform(-np.pi, np.pi)
        rand_φ = np.random.uniform(-np.pi, np.pi)  #pi/3
        self._state = np.array([
            0,
            0,
            26,  #16
            # rand_x, #训练时的小车真正的初始位置x
            25,  #25
            # rand_y, #训练时的小车位置y
            rand_theta,
            # 0,
            0,
            rand_φ
            # 0
        ])
        self.start_state = self._state.copy()

    def _get_dist(self, p1: np.ndarray, p2: np.ndarray):
        return np.linalg.norm(p1 - p2)
    def set_orig(self,x,y):
        self._state[2] = x
        self._state[3] = y
    def reset_mode(self, itr, goal_pos, mode='gradual', ):
        self.ticks = 0
        self.goal_pos = goal_pos
        rand_x = np.random.uniform(5, 25)   #(5,25)
        rand_y = np.random.uniform(5, 25)
        # self.start_point = np.array([rand_x, rand_y])    #测试时的小车初始位置
        self.start_point = np.array([26, 25])
        if self.goal_pos[0] - self.start_point[0] > 0.2:
            if self.goal_pos[1] >= self.start_point[1]:
                rand_theta = 0 + np.random.uniform(0, np.pi / 10)
            else:
                rand_theta = 0 + np.random.uniform(0, -np.pi / 10)
        elif self.goal_pos[0] + 0.2 < self.start_point[0]:
            if self.goal_pos[1] >= self.start_point[1]:
                rand_theta = -np.pi + np.random.uniform(-np.pi / 10, 0)
            else:
                rand_theta = -np.pi + np.random.uniform(np.pi / 10, 0)
        else:
            if self.goal_pos[1] - self.start_point[1] >= 0.2:
                rand_theta = np.pi / 2 + np.random.uniform(-np.pi / 10, np.pi / 10)
            else:
                rand_theta = -np.pi / 2 + np.random.uniform(-np.pi / 10, np.pi / 10)

        rand_v = np.random.uniform(0.2, 5)  #0.2和1.5
        rand_φ = np.random.uniform(-np.pi, np.pi)  #-pi/3和pi/3

        if itr < 30:
            self._state = np.array([
                0,
                0,
                self.start_point[0],
                self.start_point[1],
                rand_theta,
                0,
                0
            ])
        elif itr < 100:
            self._state = np.array([
                0,
                0,
                rand_x,
                rand_y,
                rand_theta,
                rand_v,
                rand_φ
            ])
        elif itr < 150:
            self._state = np.array([
                0,
                0,
                rand_x * 2,
                rand_y * 2,
                rand_theta,
                rand_v * 2,
                rand_φ * 2
            ])
        else:
            self._state = np.array([
                0,
                0,
                self.start_point[0],
                self.start_point[1],
                rand_theta,
                rand_v * 2,
                rand_φ * 2
                # 0,0,0
            ])
        self.start_state = self._state.copy()
