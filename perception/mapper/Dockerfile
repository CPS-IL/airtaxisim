FROM ros:noetic

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update --fix-missing
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip

RUN apt-get -y install libc-dev libx11-dev libgl1-mesa-glx libxrender1
RUN apt-get -y install build-essential
RUN pip3 install -U pip

# Copy the files
COPY . /catkin_ws/src/mapper

# Source ROS
RUN echo "source /opt/ros/noetic/setup.bash" > ~/.bashrc

# Install required packages for octomap server
RUN apt-get update && apt-get install -y ros-noetic-octomap \
    ros-noetic-octomap-ros \
    ros-noetic-octomap-msgs \
    ros-noetic-octomap-server

# Build 
WORKDIR /catkin_ws
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && cd /catkin_ws && catkin_make"

# Source the built package
RUN echo "source /catkin_ws/devel/setup.bash" > ~/.bashrc