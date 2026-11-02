#!/bin/bash
# Runs INSIDE the Apptainer image: set up ROS + a display, then exec the given command.
#
#   cluster/in_container.sh python3 rl/scripts/run.py --room final_scan --branch standard
#
# Called by cluster/job.slurm; not normally run by hand.
set -eo pipefail
cd "$(dirname "$0")/.."

source /opt/ros/humble/setup.bash
if [ -f ros2_ws/install/setup.bash ]; then
    source ros2_ws/install/setup.bash
fi

# Keep this job's ROS 2 and Ignition traffic on this node. Both discover peers by
# multicast, so two jobs on different nodes (the two branches, say) would otherwise see
# each other's /pepper_robot topics across the cluster network, and train on a mixture.
export ROS_LOCALHOST_ONLY=1
export ROS_DOMAIN_ID=$(( ${SLURM_JOB_ID:-0} % 100 + 1 ))
export IGN_PARTITION="job${SLURM_JOB_ID:-local}"
export IGN_IP=127.0.0.1

# The depth camera renders through ogre2, which needs a GL context even with `-s`. A
# virtual X server gives it one, rasterised by Mesa llvmpipe exactly as on WSL
# (rl/sim_process.py sets LIBGL_ALWAYS_SOFTWARE=1 for the sim processes).
if [ -z "${DISPLAY:-}" ]; then
    export DISPLAY=":$(( ${SLURM_JOB_ID:-0} % 400 + 100 ))"
    Xvfb "$DISPLAY" -screen 0 1280x1024x24 -nolisten tcp >/dev/null 2>&1 &
    XVFB_PID=$!
    trap 'kill $XVFB_PID 2>/dev/null || true' EXIT
    sleep 2
fi

echo "host=$(hostname) cpus=$(nproc) display=$DISPLAY ros_domain=$ROS_DOMAIN_ID ign_partition=$IGN_PARTITION"
echo "repo at $(git rev-parse --short HEAD 2>/dev/null || echo '?') ($(git status --short 2>/dev/null | wc -l) uncommitted paths)"
"$@"
