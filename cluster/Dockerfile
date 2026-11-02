# The simulation + training environment for mscluster, as a container.
#
# Built on a machine with Docker (Docker Desktop on the Windows side here), saved to a
# tar, and converted to an Apptainer image on the cluster -- see cluster/README.md. The
# cluster gives no root and no Docker, so this is the only way to get ROS 2 Humble and
# Ignition Fortress onto its compute nodes.
#
# Deliberately NOT in the image: this repository and its colcon workspace. The code
# changes far more often than the dependencies, and a rebuilt-and-reuploaded multi-GB
# image per code change is the wrong trade. The repo is rsynced to the cluster and the
# workspace is built there, inside this image (cluster/build_ws.slurm).
#
# Package choices mirror the WSL install the project was developed and verified on:
# ros_gz / gz_ros2_control / ros2_controllers from apt (their vendored source checkouts in
# ros2_ws/src carry COLCON_IGNORE), and the Python versions pinned to the .venv's.

FROM ros:humble-ros-base-jammy

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-humble-xacro \
        ros-humble-robot-state-publisher \
        ros-humble-tf2-ros \
        ros-humble-ros-gz \
        ros-humble-ros-ign \
        ros-humble-ign-ros2-control \
        ros-humble-ros2-control \
        ros-humble-ros2-controllers \
        ros-humble-ros2controlcli \
        ros-humble-controller-manager \
        ros-humble-joint-state-publisher \
        ros-humble-cv-bridge \
        ros-humble-moveit-msgs \
        ignition-tools \
        ignition-transport11-cli \
        ruby \
        # Software OpenGL (llvmpipe) and a virtual X display for the depth camera. The
        # compute nodes have no display, and on WSL the GL context came from WSLg.
        libgl1-mesa-dri libegl1 libglu1-mesa mesa-utils xvfb xauth \
        python3-pip python3-yaml python3-matplotlib python3-pil \
        procps psmisc git rsync \
    && rm -rf /var/lib/apt/lists/*

# CPU torch: rendering is software and the compute nodes have no GPU to give it, and the
# CUDA wheels would add several GB for nothing.
RUN pip3 install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip3 install --no-cache-dir \
        numpy==1.26.4 \
        scipy==1.10.1 \
        gymnasium==0.29.1 \
        stable-baselines3==2.9.0 \
        tensorboard==2.21.0 \
        trimesh==5.0.0 \
        shapely==2.1.2 \
        rtree==1.4.1 \
        networkx==3.4.2 \
 # torch drags in a current setuptools, which breaks colcon's ament_python builds against
 # Ubuntu's older `packaging` (canonicalize_version(strip_trailing_zero=...)). Removing the
 # pip copy falls back to Ubuntu's setuptools 59.6 -- what the WSL install builds with.
 && pip3 uninstall -y setuptools

# Fails the build, rather than the first cluster job, if a pin did not resolve.
# cv_bridge and rclpy live under /opt/ros, so the ROS environment has to be sourced first.
RUN bash -c "source /opt/ros/humble/setup.bash \
 && python3 -c \"import torch, stable_baselines3, gymnasium, numpy, trimesh, cv_bridge, rclpy; \
print('torch', torch.__version__, 'sb3', stable_baselines3.__version__, 'numpy', numpy.__version__)\" \
 && ign gazebo --versions"
