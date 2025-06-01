import numpy as np
from os import path
from gym.envs.classic_control import rendering


class Viewer:
    def __init__(self, env):
        self.env = env
        self.sim = env.sim

        self.pathTrace = 30
        self.pathTraceSpace = 2
        self.pathTraceSpaceCounter = 0
        self.path = np.zeros([self.pathTrace, 2])
        self.pathPtr = 0

        # Set the display window size and range
        self.viewer = rendering.Viewer(500, 500)
        # self.viewer.set_bounds(-self.sim.FIELD_SIZE, self.sim.FIELD_SIZE, -self.sim.FIELD_SIZE,
        #                        self.sim.FIELD_SIZE)  # Scale (X,X,Y,Y) 这是原地图的大小
        # self.viewer.set_bounds(-5, self.sim.FIELD_SIZE, -5,
        #                        self.sim.FIELD_SIZE)  # Scale (X,X,Y,Y)  这是我测试的大小，设为第一象限
        # self.viewer.set_bounds(self.sim.FIELD_SIZE,5,
        # self.sim.FIELD_SIZE,5)  # Scale (X,X,Y,Y) 设为第3象限
        # self.viewer.set_bounds(-30, 5, -30, 5)
        self.viewer.set_bounds(-5, 30, -5, 30)  # 第一象限
        # Create the robot
        fname = path.join(
            path.dirname(__file__),
            "./assets/robot.png",
        )
        self.robotobj = rendering.Image(fname, 1.0, 1.0)  # 小车的大小尺寸原本1.5
        self.robot_t = rendering.Transform()
        self.robotobj.add_attr(self.robot_t)

        size = np.random.rand(self.env.obs_num) * 2
        # 圆形障碍半径，原[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        # Create obstacles using a loop
        self.obstacleobjs = []
        self.obstacle_ts = []
        for i in range(self.env.obs_num):
            obstacle = rendering.make_circle(radius=size[i], filled=False)
            obstacle.set_color(0, 0, 0)
            obstacle_t = rendering.Transform()
            obstacle.add_attr(obstacle_t)
            self.obstacleobjs.append(obstacle)
            self.obstacle_ts.append(obstacle_t)

        # Create the goal location
        self.goalobj = rendering.make_circle(1.0)  # 目标图像圆直径
        self.goalobj.set_color(0, 255, 0)
        self.goal_t = rendering.Transform()
        self.goalobj.add_attr(self.goal_t)
        self.viewer.add_geom(self.goalobj)
        self.goal_t.set_translation(*self.env.goal_pos)

        # Create trace path
        self.traceobj = []
        self.traceobj_t = []
        for i in range(self.pathTrace):
            self.traceobj.append(
                rendering.make_circle(0.02 + 0.03 * i / self.pathTrace)
            )
            # print(.5 * i / self.pathTrace, 1. - .5 * i / self.pathTrace, i / self.pathTrace)
            self.traceobj[i].set_color(
                1.5 - 1.5 * i / self.pathTrace,
                1.0 - 1.5 * i / self.pathTrace,
                i / self.pathTrace,
            )  # Setting the color gradiant for path
            self.traceobj_t.append(rendering.Transform())
            self.traceobj[i].add_attr(self.traceobj_t[i])
            self.traceobj_t[i].set_translation(-2 + i * 0.05, 0)
            self.viewer.add_geom(self.traceobj[i])

        self.goalPathobj = []
        self.goalPathobj_t = []

    def render(self, mode="human"):
        # Draw the robot
        self.viewer.add_onetime(self.robotobj)
        self.robot_t.set_translation(self.sim.position[0], self.sim.position[1])
        self.robot_t.set_rotation(self.sim.theta - np.pi / 2)

        self.viewer.add_onetime(self.goalobj)
        self.goal_t.set_translation(*self.env.sim.goal_pos)

        # Render obstacles using a loop
        for i in range(self.env.obs_num):
            self.viewer.add_onetime(self.obstacleobjs[i])
            self.obstacle_ts[i].set_translation(
                self.env.obstacle[0][i], self.env.obstacle[1][i]
            )

        # Update trace
        self.pathTraceSpaceCounter = (
            self.pathTraceSpaceCounter + 1
        ) % self.pathTraceSpace
        if self.pathTraceSpaceCounter == 0:
            self.path[self.pathPtr][0], self.path[self.pathPtr][1] = (
                self.sim.position[0],
                self.sim.position[1],
            )
            self.pathPtr = (self.pathPtr + 1) % self.pathTrace
            for i in range(self.pathTrace):
                counter = (i + self.pathPtr) % self.pathTrace
                self.traceobj_t[i].set_translation(
                    self.path[counter][0], self.path[counter][1]
                )

        self.viewer.geoms = self.viewer.geoms[: self.pathTrace]
        output = self.viewer.render(return_rgb_array=mode == "rgb_array")

        return output
