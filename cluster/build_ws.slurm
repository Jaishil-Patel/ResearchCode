#!/bin/bash
# Build the colcon workspace inside the container, on a compute node (not the login node).
#
#   sbatch cluster/build_ws.slurm
#
# Re-run after `cluster/sync.sh up` whenever anything under ros2_ws/src changed. The
# vendored ros_ign / gz_ros2_control / ros2_controllers checkouts carry COLCON_IGNORE and
# come from apt in the image instead, as on WSL.
#SBATCH --job-name=build_ws
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --time=01:00:00
#SBATCH --output=cluster/logs/%x-%j.out
#SBATCH --error=cluster/logs/%x-%j.out

set -eo pipefail
SIF="${PEPPER_SIF:-$HOME/containers/pepper-sim.sif}"
cd "$SLURM_SUBMIT_DIR/ros2_ws"
apptainer exec --cleanenv "$SIF" bash -c '
    source /opt/ros/humble/setup.bash
    colcon build --merge-install --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
    source install/setup.bash
    ros2 pkg list | grep -E "^pepper_" '
