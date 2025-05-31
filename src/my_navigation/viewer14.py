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

        size = [1.0] * 7  # 圆形障碍半径，原[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        # Create the obstacle，创建障碍物
        # self.obstacleobj = rendering.make_circle(radius=1.2,filled = False)
        self.obstacleobj = rendering.make_circle(
            radius=size[0], filled=False
        )  # radius是半径，false表示不填充
        self.obstacleobj.set_color(0, 0, 0)
        self.obstacle_t = rendering.Transform()
        self.obstacleobj.add_attr(self.obstacle_t)

        # Create the obstacle
        # self.obstacleobj1 = rendering.make_circle(radius=0.8,filled = False)
        self.obstacleobj1 = rendering.make_circle(radius=size[1], filled=False)
        self.obstacleobj1.set_color(0, 0, 0)
        self.obstacle_t1 = rendering.Transform()
        self.obstacleobj1.add_attr(self.obstacle_t1)

        # Create the obstacle
        # self.obstacleobj2 = rendering.make_circle(radius=1,filled = False)
        self.obstacleobj2 = rendering.make_circle(radius=size[2], filled=False)
        self.obstacleobj2.set_color(0, 0, 0)
        self.obstacle_t2 = rendering.Transform()
        self.obstacleobj2.add_attr(self.obstacle_t2)

        self.obstacleobj3 = rendering.make_circle(radius=size[3], filled=False)
        self.obstacleobj3.set_color(0, 0, 0)
        self.obstacle_t3 = rendering.Transform()
        self.obstacleobj3.add_attr(self.obstacle_t3)

        # #当障碍物只有4个的时候，后面的3个障碍要注释掉
        self.obstacleobj4 = rendering.make_circle(radius=size[4], filled=False)
        self.obstacleobj4.set_color(0, 0, 0)
        self.obstacle_t4 = rendering.Transform()
        self.obstacleobj4.add_attr(self.obstacle_t4)

        self.obstacleobj5 = rendering.make_circle(radius=size[5], filled=False)
        self.obstacleobj5.set_color(0, 0, 0)
        self.obstacle_t5 = rendering.Transform()
        self.obstacleobj5.add_attr(self.obstacle_t5)

        self.obstacleobj6 = rendering.make_circle(radius=size[6], filled=False)
        self.obstacleobj6.set_color(0, 0, 0)
        self.obstacle_t6 = rendering.Transform()
        self.obstacleobj6.add_attr(self.obstacle_t6)

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

        self.viewer.add_onetime(self.obstacleobj)
        self.obstacle_t.set_translation(
            self.env.obstacle[0][0], self.env.obstacle[1][0]
        )

        self.viewer.add_onetime(self.obstacleobj1)
        self.obstacle_t1.set_translation(
            self.env.obstacle[0][1], self.env.obstacle[1][1]
        )

        self.viewer.add_onetime(self.obstacleobj2)
        self.obstacle_t2.set_translation(
            self.env.obstacle[0][2], self.env.obstacle[1][2]
        )

        self.viewer.add_onetime(self.obstacleobj3)
        self.obstacle_t3.set_translation(
            self.env.obstacle[0][3], self.env.obstacle[1][3]
        )

        self.viewer.add_onetime(self.obstacleobj4)  ##4个障碍物时，这里也要注释掉
        self.obstacle_t4.set_translation(
            self.env.obstacle[0][4], self.env.obstacle[1][4]
        )

        self.viewer.add_onetime(self.obstacleobj5)
        self.obstacle_t5.set_translation(
            self.env.obstacle[0][5], self.env.obstacle[1][5]
        )

        self.viewer.add_onetime(self.obstacleobj6)
        self.obstacle_t6.set_translation(
            self.env.obstacle[0][6], self.env.obstacle[1][6]
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
