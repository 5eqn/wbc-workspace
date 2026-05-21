ARG CUDA_VERSION=12.1.1
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/opt/conda/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl wget git sudo lsb-release software-properties-common tzdata build-essential cmake ninja-build python3 python3-pip python3-venv python3-dev libyaml-cpp-dev libspdlog-dev libboost-all-dev libglfw3-dev libzmq3-dev ffmpeg && rm -rf /var/lib/apt/lists/*
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && bash /tmp/miniconda.sh -b -p /opt/conda && rm /tmp/miniconda.sh

WORKDIR /workspace
COPY thirdparties/HoloMotion /workspace/HoloMotion
COPY thirdparties/unitree_mujoco /workspace/unitree_mujoco
