FROM nvidia/cudagl:11.4.2-devel-ubuntu20.04

ARG ROS_VERSION=noetic
# ARG GID=1000
# ARG UID=1000
# ARG USER=sim

ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/local/bin:${PATH}"

# ENV USERNAME ${USER}
# ENV HOME /home/$USERNAME
# RUN groupadd -f -g ${GID} ${USERNAME} && \
#     useradd -m $USERNAME -u ${UID} -g ${GID} && \
#     echo "$USERNAME:$USERNAME" | chpasswd && \
#     usermod --shell /bin/bash $USERNAME && \
#     usermod -aG sudo $USERNAME && \
#     mkdir -p /etc/sudoers.d && \
#     echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USERNAME && \
#     chmod 0440 /etc/sudoers.d/$USERNAME

RUN apt-get update && apt-get install -y --no-install-recommends curl software-properties-common

RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test &&\
    apt install -y g++-11

RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \
    apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 && \
    apt-get update && apt-get install -y --no-install-recommends \
        bash-completion \
        build-essential \
        cmake \
        command-not-found \
        freeglut3-dev \
        g++ \
        git \
        less \
        libboost-all-dev \
        libeigen3-dev \
        libomp-dev \
        libopencv-dev \
        libqglviewer-dev-qt5 \
        qt5-default qt5-qmake qtbase5-dev \
        qtbase5-dev-tools libqt5x11extras5-dev \
        libtbb-dev \
        lsb-release \
        make \
        pcl-tools \
        protobuf-compiler \
        python3-tk \
        qtbase5-dev \
        ros-${ROS_VERSION}-desktop \
        ros-${ROS_VERSION}-octomap \
        ros-${ROS_VERSION}-octomap-ros \
        ros-${ROS_VERSION}-octomap-msgs \
        ros-${ROS_VERSION}-octomap-server \
        ros-${ROS_VERSION}-rviz \
        ros-${ROS_VERSION}-vision-msgs \
        sudo \
        tar \
        tmux \
        vim \
        wget \
        xdg-user-dirs \
        xsel

# RUN echo "source /opt/ros/${ROS_VERSION}/setup.bash" >> $HOME/.bashrc && \
    # echo "source $HOME/simulator/devel/setup.bash" >> $HOME/.bashrc

RUN apt install -y python3-pip &&\
    \
    pip install \
        extensisq \
        opencv-python \
        pandas \
        pygame \
        rosdep \
        rosinstall \
        rosinstall-generator \
        scikit-spatial \
        scipy \
        setuptools \
        typing_extensions \
        torch \
        torchaudio \
        torchvision \
        wstool

COPY carla-0.9.15-cp38-cp38-linux_x86_64.whl /carla_python_api/carla-0.9.15-cp38-cp38-linux_x86_64.whl
WORKDIR /carla_python_api
RUN pip install carla-0.9.15-cp38-cp38-linux_x86_64.whl
# RUN python3 -c 'import carla'

# COPY yolov5 /yolov5
# WORKDIR /yolov5
# RUN pip install -r requirements.txt

RUN mkdir -p /catkin_ws/src

RUN pip install loguru ipdb

RUN apt-get update && apt-get install -y xvfb python3-pygame x11-xserver-utils
