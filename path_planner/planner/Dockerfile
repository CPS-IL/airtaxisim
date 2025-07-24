FROM ros:noetic

RUN apt-get update && apt-get install -y --no-install-recommends curl software-properties-common

RUN apt install -y python3-pip

# Install loguru
RUN pip install loguru ipdb tqdm matplotlib

# Set up ROS environment
ENV ROS_DISTRO=noetic
ENV ROS_ROOT=/opt/ros/noetic
ENV ROS_PACKAGE_PATH=$ROS_ROOT/share
ENV PATH=$ROS_ROOT/bin:$PATH
ENV PYTHONPATH=$ROS_ROOT/lib/python3/dist-packages:$PYTHONPATH
ENV LD_LIBRARY_PATH=$ROS_ROOT/lib:$LD_LIBRARY_PATH
