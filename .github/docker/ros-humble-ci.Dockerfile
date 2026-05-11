FROM ros:humble-ros-base-jammy

LABEL org.opencontainers.image.source="https://github.com/ssu-robotics-abc/team_abc_ws"
LABEL org.opencontainers.image.description="ROS 2 Humble CI image for team_abc_ws"

ENV ROS_DISTRO=humble
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    python3-pip \
    python3-setuptools \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    python3-numpy \
    python3-scipy \
    python3-requests \
    python3-opencv \
    python3-pymodbus \
    ros-humble-ament-cmake \
    ros-humble-ament-cmake-python \
    ros-humble-rosidl-default-generators \
    ros-humble-rosidl-default-runtime \
    ros-humble-std-msgs \
    ros-humble-geometry-msgs \
    ros-humble-sensor-msgs \
    ros-humble-action-msgs \
    ros-humble-builtin-interfaces \
    ros-humble-cv-bridge \
    ros-humble-vision-msgs \
  && rm -rf /var/lib/apt/lists/*