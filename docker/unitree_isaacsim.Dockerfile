FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
ENV CONDA_DIR=/opt/conda
ENV PATH=/opt/conda/bin:$PATH
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV PIP_EXTRA_INDEX_URL=https://pypi.nvidia.com
ENV PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

ARG ISAAC_SIM_VERSION=5.1.0
ARG PYTORCH_VERSION=2.7.0
ARG TORCHVISION_VERSION=0.22.0
ARG TORCHAUDIO_VERSION=2.7.0

RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|http://mirrors.aliyun.com/ubuntu/|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu/|http://mirrors.aliyun.com/ubuntu/|g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates cmake curl git git-lfs libglu1-mesa-dev libvulkan1 \
      ninja-build pkg-config unzip vulkan-tools wget && \
    rm -rf /var/lib/apt/lists/*

RUN wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p "$CONDA_DIR" && \
    rm /tmp/miniconda.sh && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda config --remove channels defaults && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge && \
    conda config --set show_channel_urls yes && \
    conda create -n unitree_isaacsim python=3.11 -y && \
    conda clean -afy

SHELL ["conda", "run", "-n", "unitree_isaacsim", "/bin/bash", "-c"]

RUN python -m pip install --upgrade pip && \
    python -m pip install \
      torch==${PYTORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} \
      --index-url https://download.pytorch.org/whl/cu126 && \
    python -m pip install "isaacsim[all,extscache]==${ISAAC_SIM_VERSION}"

WORKDIR /home/code
COPY thirdparties/IsaacLab /home/code/IsaacLab
COPY thirdparties/cyclonedds /home/code/cyclonedds
COPY thirdparties/GR00T-WholeBodyControl/external_dependencies/unitree_sdk2_python /home/code/unitree_sdk2_python

ENV ACCEPT_EULA=Y
ENV OMNI_KIT_ACCEPT_EULA=Y
ENV OMNI_KIT_ALLOW_ROOT=1
ENV OMNI_KIT_DISABLE_STARTUP=1

RUN touch /.dockerenv && \
    cd /home/code/IsaacLab && \
    ./isaaclab.sh --install none && \
    python -m pip install torchaudio==${TORCHAUDIO_VERSION} --index-url https://download.pytorch.org/whl/cu128 && \
    python -m pip install click==8.1.7 psutil==5.9.8

RUN cd /home/code/cyclonedds && mkdir -p build install && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/cyclonedds && \
    cmake --build . --target install -j"$(nproc)"

ENV CYCLONEDDS_HOME=/opt/cyclonedds
ENV CMAKE_PREFIX_PATH=/opt/cyclonedds

RUN python -m pip install -e /home/code/unitree_sdk2_python

RUN mkdir -p /home/code/unitree_sim_isaaclab
COPY thirdparties/unitree_sim_isaaclab/requirements.txt /home/code/unitree_sim_isaaclab/requirements.txt
COPY thirdparties/unitree_sim_isaaclab/action_provider /home/code/unitree_sim_isaaclab/action_provider
COPY thirdparties/unitree_sim_isaaclab/dds /home/code/unitree_sim_isaaclab/dds
COPY thirdparties/unitree_sim_isaaclab/layeredcontrol /home/code/unitree_sim_isaaclab/layeredcontrol
COPY thirdparties/unitree_sim_isaaclab/robots /home/code/unitree_sim_isaaclab/robots
COPY thirdparties/unitree_sim_isaaclab/tasks /home/code/unitree_sim_isaaclab/tasks
COPY thirdparties/unitree_sim_isaaclab/teleimager /home/code/unitree_sim_isaaclab/teleimager
COPY thirdparties/unitree_sim_isaaclab/tools /home/code/unitree_sim_isaaclab/tools
COPY thirdparties/unitree_sim_isaaclab/*.py /home/code/unitree_sim_isaaclab/

RUN python -m pip install -e /home/code/unitree_sim_isaaclab/teleimager && \
    python -m pip install -r /home/code/unitree_sim_isaaclab/requirements.txt && \
    python -m pip install \
      numpy==1.26.0 cryptography==44.0.0 pyopenssl==25.0.0 click==8.1.7 psutil==5.9.8

COPY thirdparties/unitree_sim_isaaclab/assets /home/code/unitree_sim_isaaclab/assets

FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
ENV CONDA_DIR=/opt/conda
ENV PATH=/opt/conda/bin:$PATH
ENV OMNI_KIT_ALLOW_ROOT=1
ENV OMNI_KIT_DISABLE_STARTUP=1
ENV ACCEPT_EULA=Y
ENV OMNI_KIT_ACCEPT_EULA=Y
ENV CYCLONEDDS_HOME=/opt/cyclonedds
ENV CMAKE_PREFIX_PATH=/opt/cyclonedds
ENV PROJECT_ROOT=/home/code/unitree_sim_isaaclab
ENV PYTHONPATH=/home/code/unitree_sim_isaaclab:/home/code/IsaacLab/source

RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|http://mirrors.aliyun.com/ubuntu/|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu/|http://mirrors.aliyun.com/ubuntu/|g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates ffmpeg git git-lfs libglu1-mesa libvulkan1 mesa-utils unzip vulkan-tools && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
      libxt6 libsm6 libice6 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/conda /opt/conda
COPY --from=builder /opt/cyclonedds /opt/cyclonedds
COPY --from=builder /home/code/IsaacLab /home/code/IsaacLab
COPY --from=builder /home/code/unitree_sdk2_python /home/code/unitree_sdk2_python
COPY --from=builder /home/code/unitree_sim_isaaclab /home/code/unitree_sim_isaaclab

WORKDIR /home/code

CMD ["conda", "run", "-n", "unitree_isaacsim", "/bin/bash"]
