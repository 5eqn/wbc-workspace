## Task

- [ ] Run [HoloMotion](https://github.com/HorizonRobotics/HoloMotion) deployment for v1.3.0 & Unitree G1 29DOF motion tracking in a Docker container, give a Dockerfile and a run script. Use the ROS2 path, avoid changing or adding code if possible
- [ ] Run [GEAR-SONIC](https://github.com/NVlabs/GR00T-WholeBodyControl) deployment for latest commit & Unitree G1 29DOF motion tracking in a Docker container, give a Dockerfile and a run script
- [ ] Run [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco) of latest commit & Unitree G1 29DOF in a Docker container.
- [ ] A script that launches automated test for HoloMotion / GEAR-SONIC deployment + unitree_mujoco simulation. Leave logs that prove phase time (nominal motion time) = wall time = sim time, artifact enough to calculate motion RMSE & delay, and generate post-rendered side-by-side comparing video.

