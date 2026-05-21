ARG CUDA_VERSION=12.4.1
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV CUDA_HOME=/usr/local/cuda
ENV CUDAToolkit_ROOT=/usr/local/cuda
ENV TensorRT_ROOT=/opt/TensorRT
ENV onnxruntime_DIR=/opt/onnxruntime/lib/cmake/onnxruntime
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/TensorRT/lib:/opt/onnxruntime/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl wget git sudo lsb-release software-properties-common tzdata build-essential cmake ninja-build clang pkg-config python3 python3-pip python3-venv python3-dev libyaml-cpp-dev libspdlog-dev libboost-all-dev libglfw3-dev libzmq3-dev libmsgpack-dev ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY thirdparties/GR00T-WholeBodyControl /workspace/GR00T-WholeBodyControl
COPY thirdparties/unitree_mujoco /workspace/unitree_mujoco
