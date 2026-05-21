ARG CUDA_VERSION=12.1.1
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/opt/conda/bin:/usr/local/cuda/bin:${PATH}
ENV CYCLONEDDS_HOME=/opt/cyclonedds/install
ENV CMAKE_PREFIX_PATH=/opt/cyclonedds/install:${CMAKE_PREFIX_PATH}
ENV LD_LIBRARY_PATH=/opt/cyclonedds/install/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl wget git sudo lsb-release software-properties-common tzdata build-essential cmake ninja-build python3 python3-pip python3-venv python3-dev libyaml-cpp-dev libspdlog-dev libboost-all-dev libglfw3-dev libzmq3-dev ffmpeg locales gnupg && rm -rf /var/lib/apt/lists/*
RUN locale-gen en_US en_US.UTF-8 && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
RUN add-apt-repository universe -y && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu jammy main" > /etc/apt/sources.list.d/ros2.list && apt-get update && apt-get install -y --no-install-recommends ros-humble-ros-base ros-humble-rmw-cyclonedds-cpp ros-humble-rosidl-generator-dds-idl ros-humble-rosidl-default-generators ros-humble-geometry-msgs python3-colcon-common-extensions python3-rosdep python3-argcomplete && rm -rf /var/lib/apt/lists/*
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && bash /tmp/miniconda.sh -b -p /opt/conda && rm /tmp/miniconda.sh
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

WORKDIR /workspace
COPY thirdparties/HoloMotion /workspace/HoloMotion
COPY thirdparties/unitree_mujoco /workspace/unitree_mujoco
RUN mkdir -p /workspace/HoloMotion/environments/environments && cp /workspace/HoloMotion/environments/requirements_deploy.txt /workspace/HoloMotion/environments/environments/requirements_deploy.txt
RUN cd /workspace/HoloMotion && conda env create -f environments/environment_deploy.yaml
RUN conda run -n holomotion_deploy python -m pip install --no-cache-dir pyzmq
RUN cp -a /workspace/HoloMotion/thirdparties/unitree_ros2 /opt/unitree_ros2
RUN cd /opt/unitree_ros2/cyclonedds_ws && PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/conda/bin bash -lc "source /opt/ros/humble/setup.bash && colcon build"
RUN cmake -S /workspace/HoloMotion/thirdparties/cyclonedds -B /tmp/cyclonedds-build -DCMAKE_INSTALL_PREFIX=/opt/cyclonedds/install -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF && cmake --build /tmp/cyclonedds-build --target install && rm -rf /tmp/cyclonedds-build
RUN apt-get update && apt-get install -y --no-install-recommends python3-yaml && rm -rf /var/lib/apt/lists/*
RUN cd /workspace/HoloMotion/deployment/unitree_g1_ros2_29dof && PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/conda/bin bash -lc "source /opt/ros/humble/setup.bash && source /opt/unitree_ros2/cyclonedds_ws/install/setup.bash && colcon build"
COPY thirdparties/GR00T-WholeBodyControl/external_dependencies/unitree_sdk2_python /workspace/unitree_sdk2_python
RUN /usr/bin/python3 -m pip install --no-cache-dir -e /workspace/unitree_sdk2_python
ENV PROFILE_PYTHON=/usr/bin/python3
