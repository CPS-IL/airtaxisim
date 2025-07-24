FROM ros:noetic

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update --fix-missing
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip

RUN apt-get -y install libc-dev libx11-dev libgl1-mesa-glx libxrender1
RUN apt-get -y install build-essential
RUN pip install -U pip

RUN pip install loguru 
RUN pip uninstall numpy -y
RUN pip install numpy>=1.21 tqdm pyvista
RUN mkdir -p /catkin_ws/src