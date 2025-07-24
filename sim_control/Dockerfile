FROM ros:noetic

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip

RUN pip install loguru

RUN mkdir -p /catkin_ws/src