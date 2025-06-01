import seaborn as sns
import numpy as np
from matplotlib import colors, pyplot as plt
from stable_baselines import SAC
from env14 import CarEnv

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
plt.rcParams["axes.unicode_minus"] = False
MAX_DIST = 100
MAX_VEL = 15


def plot_obstacle(obstacle):
    # 0.5=750,n*1500
    plt.scatter(
        obstacle[0][0], obstacle[1][0], color="#8D96A3", edgecolors="#3473C1", s=500
    )
    plt.scatter(
        obstacle[0][1], obstacle[1][1], color="#8D96A3", edgecolors="#3473C1", s=500
    )
    plt.scatter(
        obstacle[0][2], obstacle[1][2], color="#8D96A3", edgecolors="#3473C1", s=500
    )
    plt.scatter(
        obstacle[0][3], obstacle[1][3], color="#8D96A3", edgecolors="#3473C1", s=500
    )
    plt.scatter(
        obstacle[0][4], obstacle[1][4], color="#8D96A3", edgecolors="#3473C1", s=500
    )
    plt.scatter(
        obstacle[0][5], obstacle[1][5], color="#8D96A3", edgecolors="#3473C1", s=500
    )
    plt.scatter(
        obstacle[0][6], obstacle[1][6], color="#8D96A3", edgecolors="#3473C1", s=500
    )
    # plt.scatter(obstacle[0][0], obstacle[1][0], color='#8D96A3', edgecolors='#3473C1', s=1500, marker="o")
    # plt.scatter(obstacle[0][1], obstacle[1][1], color='#8D96A3', edgecolors='#3473C1', s=1500, marker="o")
    # plt.scatter(obstacle[0][2], obstacle[1][2], color='#8D96A3', edgecolors='#3473C1', s=1500, marker="o")
    # plt.scatter(obstacle[0][3], obstacle[1][3], color='#8D96A3', edgecolors='#3473C1', s=1500, marker="o")
    # plt.scatter(obstacle[0][4], obstacle[1][4], color='#8D96A3', edgecolors='#3473C1', s=1500, marker="o")
    # plt.scatter(obstacle[0][5], obstacle[1][5], color='#8D96A3', edgecolors='#3473C1', s=1500, marker="o")
    # plt.scatter(obstacle[0][6], obstacle[1][6], color='#8D96A3', edgecolors='#3473C1', s=1500, marker="o")


def plot(pos_x, pos_y, theta_list, ax):
    L = 0.5
    b = 0.32
    A_x = pos_x + L / 2 * np.cos(theta_list) - b / 2 * np.sin(theta_list)
    B_x = pos_x + L / 2 * np.cos(theta_list) + b / 2 * np.sin(theta_list)
    C_x = pos_x - L / 2 * np.cos(theta_list) + b / 2 * np.sin(theta_list)
    D_x = pos_x - L / 2 * np.cos(theta_list) - b / 2 * np.sin(theta_list)

    A_y = pos_y + L / 2 * np.sin(theta_list) + b / 2 * np.cos(theta_list)
    B_y = pos_y + L / 2 * np.sin(theta_list) - b / 2 * np.cos(theta_list)
    C_y = pos_y - L / 2 * np.sin(theta_list) - b / 2 * np.cos(theta_list)
    D_y = pos_y - L / 2 * np.sin(theta_list) + b / 2 * np.cos(theta_list)

    # plt.figure()
    # fig, ax = plt.subplots()
    for i in range(0, len(A_x)):
        ax.add_patch(
            plt.Polygon(
                xy=[
                    [A_x[i], A_y[i]],
                    [B_x[i], B_y[i]],
                    [C_x[i], C_y[i]],
                    [D_x[i], D_y[i]],
                ],
                color=(0.27, 0.50, 0.71),
                alpha=0.1,
            )
        )
        ax.add_patch(
            plt.Polygon(
                xy=[
                    [A_x[-1], A_y[-1]],
                    [B_x[-1], B_y[-1]],
                    [C_x[-1], C_y[-1]],
                    [D_x[-1], D_y[-1]],
                ],
                color=(0.27, 0.45, 0.5),
                alpha=0.3,
            )
        )
    # color=(0.27 0.45 0.5)
    font1 = {"family": "Times New Roman", "size": 12}
    font2 = {"family": "Times New Roman", "size": 16}

    plt.xlim([-5, 30])
    plt.ylim([-5, 30])  ##[-5, 30]
    plt.xlabel("Position x [m]", fontdict=font1)
    plt.ylabel("Position y [m]", fontdict=font1)
    print("============================")


def render_trajectory(test_env, i):
    try:
        # model = SAC.load(r'E:\pycharm\AGV\AGV_car_haisen_original\SAC\best_model.zip')    #用已经训练好的模型
        model = SAC.load(r"./train_result/model14/best_model.zip")

        # show the orig_image
        My_font = {
            "family": "Times New Roman",
            "weight": "normal",
            "size": 23,
        }
        sns.set(style="dark")
        fig = plt.figure(figsize=[6, 6])
        plt.title("Trajectory of the AGV")
        label_font = {"family": "Times New Roman"}
        plt.gcf().set_facecolor(np.ones(3) * 1)  # 生成画布的大小

        plt.xticks(np.arange(-5, 30, 5))  #
        plt.yticks(np.arange(-5, 30, 5))
        plt.xticks(fontsize=10, fontproperties=label_font)
        plt.yticks(fontsize=10, fontproperties=label_font)

        ax1 = plt.subplot(111)  # 绘制第一个图  （2行2列第1个）
        ax1.grid(linestyle="--")
        plt.xlim(-5, 30)  # [-5, 30]
        plt.ylim(-5, 30)
        plt.xlabel("(a)", fontsize=12)
        test_env = CarEnv()
        s = test_env.reset()
        # test_env.sim.set_orig(0,-3)  #小车起始点
        ax1.scatter(
            test_env.sim.position[0],
            test_env.sim.position[1],
            color="g",
            label="origin",
            s=200,
        )  # 小车大小
        ax1.scatter(
            test_env.sim.goal_pos[0],
            test_env.sim.goal_pos[1],
            color="r",
            label="target",
            s=200,
        )  # 小车大小
        t = 1
        # plot_obstacle(test_env.obstacle,test_env.obstacle1,test_env.obstacle2,test_env.obstacle3)
        for ep in range(1):
            pos_x = []
            pos_y = []
            theta_list = []
            while True:
                # test_env.render()
                action = model.predict(s)[0]
                print("action:", action)
                s, r, done, _ = test_env.step(action)
                print("eposid：", t)
                print("update_state:", s)
                print("reward:", r)
                print("update_Sim_state:", test_env.sim._state)
                print("theta", test_env.sim.theta)
                print("--------------------------------------------------------")
                plt.scatter(
                    test_env.obstacle[0][0],
                    test_env.obstacle[1][0],
                    color="#8D96A3",
                    edgecolors="#3473C1",
                    s=500,
                    marker="o",
                )
                plt.scatter(
                    test_env.obstacle[0][1],
                    test_env.obstacle[1][1],
                    color="#8D96A3",
                    edgecolors="#3473C1",
                    s=500,
                    marker="o",
                )
                plt.scatter(
                    test_env.obstacle[0][2],
                    test_env.obstacle[1][2],
                    color="#8D96A3",
                    edgecolors="#3473C1",
                    s=500,
                    marker="o",
                )
                plt.scatter(
                    test_env.obstacle[0][3],
                    test_env.obstacle[1][3],
                    color="#8D96A3",
                    edgecolors="#3473C1",
                    s=500,
                    marker="o",
                )
                plt.scatter(
                    test_env.obstacle[0][4],
                    test_env.obstacle[1][4],
                    color="#8D96A3",
                    edgecolors="#3473C1",
                    s=500,
                    marker="o",
                )
                plt.scatter(
                    test_env.obstacle[0][5],
                    test_env.obstacle[1][5],
                    color="#8D96A3",
                    edgecolors="#3473C1",
                    s=500,
                    marker="o",
                )
                plt.scatter(
                    test_env.obstacle[0][6],
                    test_env.obstacle[1][6],
                    color="#8D96A3",
                    edgecolors="#3473C1",
                    s=500,
                    marker="o",
                )

                if t == 0:
                    pos_x.append(test_env.sim.position[0])
                    pos_y.append(test_env.sim.position[1])
                    theta_list.append(test_env.sim.theta)
                if t % 8 == 0:
                    pos_x.append(test_env.sim.position[0])
                    pos_y.append(test_env.sim.position[1])
                    theta_list.append(test_env.sim.theta)
                t += 1
                if done:
                    pos_x = np.array(pos_x)
                    pos_y = np.array(pos_y)
                    theta_list = np.array(theta_list)
                    print("pos_x.shape：", pos_x.shape)
                    print("pos_y.shape：", pos_y.shape)
                    print("theta_lis1t.shape：", theta_list.shape)
                    plot(pos_x, pos_y, theta_list, ax1)
                    plt.legend(markerscale=0.75)
                    plt.show()
                    break

        # ax2 = plt.subplot(1,2,2)  #绘制图二
        # ax2.grid(linestyle='--')
        # plt.xlim(-5, 30)
        # plt.ylim(-5, 30)
        # plt.xlabel("(a)",fontsize=12)
        # test_env = CarEnv()
        # s = test_env.reset()
        # test_env.sim.set_orig(1,-3)
        # ax2.scatter(test_env.sim.position[0], test_env.sim.position[1],color='g', label='origin',s=200)
        # ax2.scatter(test_env.sim.goal_pos[0], test_env.sim.goal_pos[1], color='r', label='target',s=200)
        # t = 1
        # # plot_obstacle(test_env.obstacle,test_env.obstacle1,test_env.obstacle2,test_env.obstacle3)
        # for ep in range(1):
        #     pos_x = []
        #     pos_y = []
        #     theta_list = []
        #     while True:
        #         # test_env.render()
        #         action = model.predict(s)[0]
        #         print("action:", action)
        #         s, r, done, _ = test_env.step(action)
        #         print("eposid：", t)
        #         print("update_state:", s)
        #         print("reward:", r)
        #         print("update_Sim_state:", test_env.sim._state)
        #         print("theta",test_env.sim.theta)
        #         print("--------------------------------------------------------")
        #         #我删掉了一些东西，在注释的这个位置，具体对比原py文件。
        #         plt.scatter(test_env.obstacle[0][0], test_env.obstacle[1][0], color='#8D96A3', edgecolors='#3473C1',
        #                     s=150, marker="s")
        #         plt.scatter(test_env.obstacle[0][1], test_env.obstacle[1][1], color='#8D96A3', edgecolors='#3473C1',
        #                     s=150, marker="s")
        #         plt.scatter(test_env.obstacle[0][2], test_env.obstacle[1][2], color='#8D96A3', edgecolors='#3473C1',
        #                     s=150, marker="s")
        #         plt.scatter(test_env.obstacle[0][3], test_env.obstacle[1][3], color='#8D96A3', edgecolors='#3473C1',
        #                     s=150, marker="s")
        #         plt.scatter(test_env.obstacle[0][4], test_env.obstacle[1][4], color='#8D96A3', edgecolors='#3473C1',
        #                     s=150, marker="s")
        #         plt.scatter(test_env.obstacle[0][5], test_env.obstacle[1][5], color='#8D96A3', edgecolors='#3473C1',
        #                     s=150, marker="s")
        #         plt.scatter(test_env.obstacle[0][6], test_env.obstacle[1][6], color='#8D96A3', edgecolors='#3473C1',
        #                     s=150, marker="s")
        #         if t == 0:
        #             pos_x.append(test_env.sim.position[0])
        #             pos_y.append(test_env.sim.position[1])
        #             theta_list.append(test_env.sim.theta)
        #         if t % 8 == 0:
        #             pos_x.append(test_env.sim.position[0])
        #             pos_y.append(test_env.sim.position[1])
        #             theta_list.append(test_env.sim.theta)
        #         t += 1
        #         if done:
        #             pos_x = np.array(pos_x)
        #             pos_y = np.array(pos_y)
        #             theta_list = np.array(theta_list)
        #             print("pos_x.shape：",pos_x.shape)
        #             print("pos_y.shape：",pos_y.shape)
        #             print("theta_lis1t.shape：",theta_list.shape)
        #             plot(pos_x,pos_y,theta_list,ax2)
        #             plt.legend(markerscale=0.75)
        #             plt.show()
        #             break

        # ax3 = plt.subplot(2,2,3)   #第三个图
        # ax3.grid(linestyle='--')
        # plt.xlim(-5, 30)
        # plt.ylim(-5, 30)
        # plt.xlabel("(c)",fontsize=12)
        # test_env = CarEnv()
        # s = test_env.reset()
        # test_env.sim.set_orig(4.5,-3)
        # ax3.scatter(test_env.sim.position[0], test_env.sim.position[1],color='g', label='origin',s=200)
        #
        #
        # print("Training_state:", s)
        # print("orig_Sim_State：", test_env.sim._state)
        # t = 1
        # # plot_obstacle(test_env.obstacle,test_env.obstacle1,test_env.obstacle2,test_env.obstacle3)
        # for ep in range(1):
        #     pos_x = []
        #     pos_y = []
        #     theta_list = []
        #     plt.scatter(test_env.obstacle[0][0], test_env.obstacle[1][0], color='#8D96A3', edgecolors='#3473C1',
        #                 s=200)
        #     plt.scatter(test_env.obstacle[0][1], test_env.obstacle[1][1], color='#8D96A3', edgecolors='#3473C1',
        #                 s=200)
        #     plt.scatter(test_env.obstacle[0][2], test_env.obstacle[1][2], color='#8D96A3', edgecolors='#3473C1',
        #                 s=200)
        #     plt.scatter(test_env.obstacle[0][3], test_env.obstacle[1][3], color='#8D96A3', edgecolors='#3473C1',
        #                 s=200)
        #     plt.scatter(test_env.obstacle[0][4], test_env.obstacle[1][4], color='#8D96A3', edgecolors='#3473C1',
        #                 s=200, )
        #     plt.scatter(test_env.obstacle[0][5], test_env.obstacle[1][5], color='#8D96A3', edgecolors='#3473C1',
        #                 s=200)
        #     plt.scatter(test_env.obstacle[0][6], test_env.obstacle[1][6], color='#8D96A3', edgecolors='#3473C1',
        #                 s=200)
        #     # plt.scatter(-3,-2,color='#8D96A3',edgecolors='#3473C1',s=750*2*2)
        #     ax3.scatter(test_env.sim.goal_pos[0], test_env.sim.goal_pos[1], color='r', label='target',s=300)
        #     while True:
        #         # test_env.render()
        #         action = model.predict(s)[0]
        #         print("action:", action)
        #         s, r, done, _ = test_env.step(action)
        #         print("eposid：", t)
        #         print("update_state:", s)
        #         print("reward:", r)
        #         print("update_Sim_state:", test_env.sim._state)
        #         print("--------------------------------------------------------")
        #
        #         if t == 0:
        #             pos_x.append(test_env.sim.position[0])
        #             pos_y.append(test_env.sim.position[1])
        #             theta_list.append(test_env.sim.theta)
        #             # ax3.scatter(test_env.sim.position[0], test_env.sim.position[1],edgecolors='#3473C1',color='w',label="AGV")
        #         if t % 8 == 0:
        #             pos_x.append(test_env.sim.position[0])
        #             pos_y.append(test_env.sim.position[1])
        #             theta_list.append(test_env.sim.theta)
        #             # ax3.scatter(test_env.sim.position[0], test_env.sim.position[1],edgecolors='#3473C1',color='w')
        #         t += 1
        #         if done:
        #             pos_x = np.array(pos_x)
        #             pos_y = np.array(pos_y)
        #             theta_list = np.array(theta_list)
        #             print("pos_x.shape：",pos_x.shape)
        #             print("pos_y.shape：",pos_y.shape)
        #             print("theta_lis1t.shape：",theta_list.shape)
        #             plot(pos_x,pos_y,theta_list,ax3)
        #             plt.legend(markerscale=0.75)
        #             plt.show()
        #             break
        #
        # ax4 = plt.subplot(2,2,4)  #绘制第四个图
        # ax4.grid(linestyle='--')
        # plt.xlim(-5, 30)
        # plt.ylim(-5, 30)
        # plt.xlabel("(d)",fontsize=12)
        # test_env = CarEnv()
        # s = test_env.reset()
        # test_env.sim.set_orig(4,4)
        # ax4.scatter(test_env.sim.position[0], test_env.sim.position[1],color='g', label='origin',s=200)
        # ax4.scatter(test_env.sim.goal_pos[0], test_env.sim.goal_pos[1], color='r', label='target',s=300)
        # plt.scatter(test_env.obstacle[0][0], test_env.obstacle[1][0], color='#8D96A3', edgecolors='#3473C1',
        #             s=200, marker="s")
        # plt.scatter(test_env.obstacle[0][1], test_env.obstacle[1][1], color='#8D96A3', edgecolors='#3473C1',
        #             s=200, marker="p")
        # plt.scatter(test_env.obstacle[0][2], test_env.obstacle[1][2], color='#8D96A3', edgecolors='#3473C1',
        #             s=200, marker="s")
        # plt.scatter(test_env.obstacle[0][3], test_env.obstacle[1][3], color='#8D96A3', edgecolors='#3473C1',
        #             s=200, marker="s")
        # plt.scatter(test_env.obstacle[0][4], test_env.obstacle[1][4], color='#8D96A3', edgecolors='#3473C1',
        #             s=200, marker="s")
        # plt.scatter(test_env.obstacle[0][5], test_env.obstacle[1][5], color='#8D96A3', edgecolors='#3473C1',
        #             s=200, marker="s")
        # plt.scatter(test_env.obstacle[0][6], test_env.obstacle[1][6], color='#8D96A3', edgecolors='#3473C1',
        #             s=200, marker="s")
        # print("Training_state:", s)
        # print("orig_Sim_State：", test_env.sim._state)
        # t = 1
        # plot_obstacle(test_env.obstacle)
        # for ep in range(1):
        #     pos_x = []
        #     pos_y = []
        #     theta_list = []
        #     while True:
        #         # test_env.render()
        #         action = model.predict(s)[0]
        #         print("action:", action)
        #         s, r, done, _ = test_env.step(action)
        #         print("eposid：", t)
        #         print("update_state:", s)
        #         print("reward:", r)
        #         print("update_Sim_state:", test_env.sim._state)
        #         print("--------------------------------------------------------")
        #         if t == 0:
        #             pos_x.append(test_env.sim.position[0])
        #             pos_y.append(test_env.sim.position[1])
        #             theta_list.append(test_env.sim.theta)
        #             # ax4.scatter(test_env.sim.position[0], test_env.sim.position[1],edgecolors='#3473C1',color='w',label="AGV")
        #         if t % 8 == 0:
        #             pos_x.append(test_env.sim.position[0])
        #             pos_y.append(test_env.sim.position[1])
        #             theta_list.append(test_env.sim.theta)
        #             # ax4.scatter(test_env.sim.position[0], test_env.sim.position[1],edgecolors='#3473C1',color='w')
        #         t += 1
        #         if done:
        #             pos_x = np.array(pos_x)
        #             pos_y = np.array(pos_y)
        #             theta_list = np.array(theta_list)
        #             print("pos_x.shape：",pos_x.shape)
        #             print("pos_y.shape：",pos_y.shape)
        #             print("theta_lis1t.shape：",theta_list.shape)
        #             plot(pos_x,pos_y,theta_list,ax4)
        #             plt.legend(markerscale=0.75)
        #             plt.show()
        #             # name = "buguize" + i+"png"
        #             # plt.savefig('buguize.png')
        #             break

    except (RuntimeError, TypeError, NameError):
        pass


for i in range(50):
    render_trajectory(CarEnv(), i)
