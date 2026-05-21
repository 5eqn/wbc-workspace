FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl wget git sudo lsb-release software-properties-common tzdata build-essential cmake ninja-build pkg-config python3 python3-pip python3-venv python3-dev libyaml-cpp-dev libspdlog-dev libboost-all-dev libglfw3-dev libxinerama-dev libxcursor-dev libxi-dev libxrandr-dev libx11-dev libegl1 libgl1 libosmesa6 ffmpeg && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --no-cache-dir numpy scipy matplotlib pandas pyyaml mujoco imageio imageio-ffmpeg

WORKDIR /workspace
COPY thirdparties/unitree_mujoco /workspace/unitree_mujoco
