FROM ros:noetic

RUN apt-get update && apt-get install -y --no-install-recommends curl software-properties-common
RUN apt install -y python3-pip

# Copy the RRT Planner code
COPY ./rrt_planner /catkin_ws/src/rrt_planner

# Source ROS
RUN echo "source /opt/ros/noetic/setup.bash" > ~/.bashrc

# Install required packages
RUN pip3 install loguru matplotlib scipy

# Catkin make
WORKDIR /catkin_ws
RUN bash -c "source /opt/ros/noetic/setup.bash && cd /catkin_ws && catkin_make"

# Source the built environment
RUN echo "source /catkin_ws/devel/setup.bash" > ~/.bashrc