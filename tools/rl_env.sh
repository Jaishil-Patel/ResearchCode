#!/usr/bin/env bash
# Source this to get ROS 2 and the RL stack in one shell. ORDER MATTERS.
#
#     source tools/rl_env.sh
#
# ROS first, venv second. The venv is created with --system-site-packages, so activating
# it keeps whatever is already on the path -- but sourcing ROS *after* activating puts
# /opt/ros/humble's python entries ahead of the venv's, and `import torch` then resolves
# against the system interpreter, which has no torch. The failure is a bare
# ModuleNotFoundError with a perfectly healthy-looking venv sitting right there.
#
# This is the same class of trap as tools/run_shim.sh: the py2 NAOqi SDK ships a `qi`
# package that breaks every Python 3 import if it reaches a global PYTHONPATH. Scope the
# environment deliberately rather than accumulating it.

_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source /opt/ros/humble/setup.bash
if [ -f "${_repo}/ros2_ws/install/local_setup.bash" ]; then
  source "${_repo}/ros2_ws/install/local_setup.bash"
fi
source "${_repo}/.venv/bin/activate"

# Mandatory on WSL2 for anything that renders. ogre2 on hardware GL aborts in
# GL3PlusTextureGpu::copyTo; ogre v1 runs and silently pins every lidar beam at <min>,
# which would feed SAC constant, plausible-looking observations and never raise an error.
export LIBGL_ALWAYS_SOFTWARE=1

echo "ROS 2 Humble + $(python -c 'import torch;print("torch "+torch.__version__)' 2>/dev/null || echo 'torch MISSING')"
echo "python: $(which python)"
echo "LIBGL_ALWAYS_SOFTWARE=${LIBGL_ALWAYS_SOFTWARE}"
unset _repo
