import matplotlib
import time
from stable_baselines import SAC
from matplotlib import pyplot as plt
from stable_baselines.bench import Monitor
from stable_baselines.sac.policies import MlpPolicy as sacmlp
from stable_baselines.common.vec_env import DummyVecEnv
from utils import SaveOnBestTrainingRewardCallback  #显示运行后的timestep、最好奖励和目前奖励是多少的模块
from scipy.signal import savgol_filter
from env14 import CarEnv
from simulation14 import Simulation as sim

matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签

train_num = 1200000  # 60 0000
# run_mode = 'train'
run_mode = 'test'

if run_mode == 'train':

    car_env = CarEnv()

    SAC_log_dir = r'S:\yuanE\pycharm\train_result\SAC\model14-3'
    optimize_env = Monitor(car_env, SAC_log_dir, info_keywords=('d',))
    SAC_env = DummyVecEnv([lambda: optimize_env])

    print('RL Train Begin...')
    Begin_time = time.time()

    SAC_model = SAC(policy=sacmlp, env=SAC_env, verbose=0, gamma=0.99, learning_rate=1e-3, buffer_size=200000,
                    learning_starts=100, train_freq=1, batch_size=64, tau=0.005, ent_coef='auto',
                    target_update_interval=1, random_exploration=0.15, action_noise=None,   #随机探索率 和 白噪声
                    # tensorboard_log="./MyLogs/{}/".format("SAC"))   #每次训练都会存入Mylogs，导致磁盘占用越拉越大
                    ##所以为了避免本地磁盘爆掉，修改MyLogs的日志文件路径
                    tensorboard_log = "S:/yuanE/pycharm/train_result/MyLogs/{}/".format("SAC"))
    SAC_callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=SAC_log_dir)
    SAC_model.learn(total_timesteps=int(train_num), callback=SAC_callback)

    print('RL Train-Finish...')
    Finish_time = time.time()
    print("the total Training time is :", (Finish_time - Begin_time) / 60)
    # print('sim.time:', sim.time)


elif run_mode == 'test':
    model = SAC.load(r'S:\yuanE\pycharm\train_result\SAC\model14-1\best_model.zip')
    test_env = CarEnv()

    for ep in range(100):
        Begin_time = time.time()
        sum_r = 0
        state_list = []
        a_list = []
        xita_list = []
        ep_r = []
        s = test_env.reset()
        car_path = [(s[0],s[1])] #将小车的起始坐标信息放入路径中（自己加的）
        print("Training_state:", s)
        # exit(0)
        # state_list.append[s]
        print("orig_Sim_State：", test_env.sim._state)
        t = 1
        while True:
            test_env.render()
            # nb_actions = env.action_space.shape[0]
            # print("action:",nb_actions)
            action = model.predict(s)[0]
            a_list.append(action[0])
            xita_list.append(action[1])
            print("action:", action)
            s, r, done, _ = test_env.step(action)
            # state_list.append[s]
            ep_r.append(r)

            car_path.append(s[:2]) #将update的新state位置信息添加到移动路径中(自己加的)
            x = [point[0] for point in car_path]#加的
            y = [point[1] for point in car_path]#加的
            plt.plot(x, y, marker='o')  # 提取x和y坐标，并绘制路径(自己加的)
            print("eposid：", t)
            print("update_state:", s)
            print("reward:", r)
            print("update_Sim_state:", test_env.sim._state)
            sum_r += r
            Finish_time = time.time()
            print("the total Training time is :", (Finish_time - Begin_time))
            # print('sim.time:', sim.time)
            print("--------------------------------------------------------")
            t += 1
            if done:
                a_list = savgol_filter(a_list, 21, 1)
                xita_list = savgol_filter(xita_list, 41, 1)
                ep_r[0:len(ep_r) - 50] = savgol_filter(ep_r[0:len(ep_r) - 50], 21, 1)

                # plt.subplot(3, 1, 1) #最后显示的曲线图中3个子图里的第1个
                # plt.grid()  #在当前图形中添加水平和垂直的网格线
                # plt.plot(a_list[:len(ep_r)])
                # plt.ylabel("α")
                # plt.ylim(0.2, 2)
                #
                # plt.subplot(3, 1, 3) #3个子图里的第3个
                # plt.grid()  #在当前图形中添加水平和垂直的网格线
                # plt.plot(ep_r[2:-2][::-1])
                # plt.ylabel("r")
                # plt.xlabel("time step")
                #
                # plt.subplot(3, 1, 2) #3个子图里的第2个
                # plt.grid()  #在当前图形中添加水平和垂直的网格线
                # plt.plot(xita_list[:len(ep_r)])
                # plt.ylim([-1, 1])
                # plt.ylabel("ω")
                #
                # plt.show()

                state_list = []
                ep_r = []
                a_list = []
                xita_list = []
                test_env.reset()
                test_env._print_info(sum_r)
                break






















































































