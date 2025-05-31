FROM osrf/ros:noetic-desktop-full

# Set up apt sources
RUN apt-get update && apt-get install -y \
    apt-transport-https \
    ca-certificates \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Install additional development tools and dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-catkin-tools \
    python3-rosdep \
    gazebo11 \
    python3-wstool \
    python3-vcstool \
    git \
    libgflags-dev \
    libgoogle-glog-dev \
    python3-sphinx \
    ros-noetic-gmapping\
    ros-noetic-gmapping \
    ros-noetic-map-server \
    ros-noetic-amcl \
    ros-noetic-move-base \
    ros-noetic-navigation \
    && rm -rf /var/lib/apt/lists/*

# Update rosdep
RUN rosdep update

# Set up the workspace
WORKDIR /catkin_ws

# Source the workspace in the bashrc
RUN echo "alias sos='source /opt/ros/noetic/setup.bash'" >> ~/.bashrc
RUN echo "alias sis='source /catkin_ws/devel/setup.bash'" >> ~/.bashrc
RUN echo "alias sgs='source /usr/share/gazebo/setup.sh'" >> ~/.bashrc

# Set the entrypoint to keep container running
ENTRYPOINT ["/bin/bash", "-c", "while true; do sleep 1; done"] 