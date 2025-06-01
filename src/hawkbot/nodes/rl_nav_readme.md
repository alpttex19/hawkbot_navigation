# RL Navigation System

这个系统将你的强化学习避障算法集成到ROS导航框架中，替换传统的move_base路径规划器。

## 文件说明

### 核心节点
- **rl_navigation_node.py**: 主导航节点，使用强化学习模型进行路径规划和避障
- **rl_nav_monitor.py**: 监控和调试节点，用于可视化障碍物和状态信息
- **rl_send_mark.py**: 修改版的目标点设置节点，兼容RL导航系统

### Launch文件
- **rl_navigation.launch**: 基本的RL导航启动文件
- **complete_rl_navigation.launch**: 包含监控功能的完整启动文件

## 安装和配置

### 1. 依赖项
确保已安装以下Python包：
```bash
pip install stable-baselines3  # 或stable-baselines
pip install numpy
pip install tensorflow  # 如果使用TensorFlow模型
```

### 2. 文件部署
将以下文件放置到你的hawkbot包中：
```
hawkbot/
├── scripts/
│   ├── rl_navigation_node.py
│   ├── rl_nav_monitor.py
│   └── rl_send_mark.py
├── launch/
│   ├── rl_navigation.launch
│   └── complete_rl_navigation.launch
└── train_result/
    └── model14/
        └── best_model.zip  # 你的训练模型
```

### 3. 权限设置
```bash
chmod +x hawkbot/scripts/rl_navigation_node.py
chmod +x hawkbot/scripts/rl_nav_monitor.py
chmod +x hawkbot/scripts/rl_send_mark.py
```

## 使用方法

### 启动系统
```bash
# 基本启动（无监控）
roslaunch hawkbot rl_navigation.launch

# 完整启动（包含监控和调试）
roslaunch hawkbot complete_rl_navigation.launch

# 自定义模型路径
roslaunch hawkbot complete_rl_navigation.launch model_path:=/path/to/your/model.zip
```

### 设置目标点
1. 在RViz中使用"2D Nav Goal"工具点击设置目标
2. 或者在RViz中使用"Publish Point"工具点击设置目标
3. 支持多点巡航：依次点击多个目标点

### 监控和调试
启动完整系统后，可以通过以下topics监控状态：
```bash
# 查看调试信息
rostopic echo /rl_nav/debug_info

# 查看状态向量
rostopic echo /rl_nav/state

# 查看障碍物标记（在RViz中可视化）
rostopic echo /rl_nav/obstacles
```

## 系统架构

### 数据流
```
Laser Scan → RL Navigation Node → cmd_vel
     ↓              ↑
AMCL Pose ← → Goal Processing
     ↓              ↓
Monitor Node → Visualization
```

### 核心功能
1. **感知**: 从激光雷达和AMCL获取环境和位置信息
2. **决策**: 使用训练好的RL模型进行路径规划
3. **控制**: 发布速度命令到机器人
4. **监控**: 实时监控和可视化系统状态

## 参数配置

### RL Navigation Node参数
```xml
<param name="model_path" value="path/to/model.zip"/>      <!-- 模型路径 -->
<param name="goal_threshold" value="0.6"/>               <!-- 到达目标阈值(m) -->
<param name="max_linear_vel" value="5.0"/>               <!-- 最大线速度(m/s) -->
<param name="max_angular_vel" value="0.524"/>            <!-- 最大角速度(rad/s) -->
<param name="control_frequency" value="50"/>             <!-- 控制频率(Hz) -->
```

### Monitor Node参数
```xml
<param name="verbose" value="false"/>                    <!-- 详细日志输出 -->
```

## 故障排除

### 常见问题

1. **模型加载失败**
   - 检查模型路径是否正确
   - 确认stable-baselines版本兼容性
   - 查看节点日志：`rosnode info rl_navigation_node`

2. **机器人不移动**
   - 检查cmd_vel topic是否正确发布：`rostopic echo /cmd_vel`
   - 确认AMCL定位正常：`rostopic echo /amcl_pose`
   - 验证目标点是否设置：`rostopic echo /move_base_simple/goal`

3. **障碍物检测异常**
   - 检查激光雷达数据：`rostopic echo /scan`
   - 在RViz中查看激光点云
   - 调整聚类参数

4. **导航精度问题**
   - 调整goal_threshold参数
   - 检查AMCL参数配置
   - 重新训练模型（如果必要）

### 调试命令
```bash
# 查看节点状态
rosnode list | grep rl_nav

# 查看topic连接
rostopic list | grep rl_nav

# 实时监控关键topics
rostopic echo /cmd_vel
rostopic echo /amcl_pose
rostopic echo /rl_nav/debug_info
```

## 性能优化

1. **控制频率**: 根据硬件性能调整control_frequency
2. **障碍物数量**: 限制最大检测障碍物数量以提高性能
3. **模型优化**: 使用量化或压缩的模型以加速推理

## 扩展功能

### 添加新的观测信息
在`get_observation()`函数中添加新的状态变量：
```python
# 例如：添加IMU信息
imu_data = self.get_imu_data()
observation = np.hstack([observation, imu_data])
```

### 多机器人支持
修改节点名称和topic命名空间：
```xml
<group ns="robot1">
    <node pkg="hawkbot" type="rl_navigation_node.py" name="rl_navigation_node"/>
</group>
```

### 动态参数调整
使用dynamic_reconfigure包实现运行时参数调整。

## 版本信息
- ROS版本: Melodic/Noetic
- Python版本: 3.6+
- 依赖包: stable-baselines, numpy, tensorflow

如有问题，请检查日志文件或联系开发团队。
