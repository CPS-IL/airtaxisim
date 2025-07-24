FROM ros:noetic

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    tar \
    wget \
    git \
    freeglut3-dev \
    libqglviewer-dev-qt5 \
    qtbase5-dev \
    python3-pip \
    libopencv-dev

# Install Eigen 3.3.4
RUN apt-get update && apt-get install -y --no-install-recommends \
    libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Boost 1.71.0
RUN apt-get update && apt-get install -y --no-install-recommends \
    libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

# Install ROS vision_msgs package
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-pcl-ros \
    ros-noetic-vision-msgs

RUN ln -s /usr/include/opencv4/opencv2 /usr/include/opencv2
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
RUN mkdir -p /catkin_ws/src
