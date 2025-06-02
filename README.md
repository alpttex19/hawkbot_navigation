## 每次代码更改都需要重新编译

## 进入工作空间
cd ~/catkin_ws
# 编译工作空间
catkin build


## 启动roscore
roscore

## 启动gazebo仿真环境
roslaunch hawkbot_gazebo hawkbot_autorace.launch 

## 一般同一个环境只需要建图一次，保存下来即可
## 启动gmapping建图
roslaunch hawkbot gmapping_slam.launch

## 启动遥控节点
roslaunch hawkbot teleop_key_sim.launch

## 遥控机器人探索环境，完成建图后保存地图
rosrun map_server map_saver -f hawkbot_map

#保存后，将地图文件复制到src/hawkbot/maps/hawkbot_map.yaml

#关闭建图和遥控节点，启动导航节点
roslaunch hawkbot navigation_sim.launch


## 启动gazebo仿真环境
roslaunch hawkbot_gazebo hawkbot_autorace.launch 
## 强化学习导航算法
roslaunch hawkbot rl_navigation.launch




