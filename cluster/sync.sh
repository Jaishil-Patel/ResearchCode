#!/bin/bash
# Copy this repo to the cluster (run in WSL, from anywhere), or bring run results back.
#
#   cluster/sync.sh up      jpatel@<cluster address>    # code, configs, meshes, routes
#   cluster/sync.sh down    jpatel@<cluster address>    # rl/runs/ -> here
#
# `up` never touches rl/runs on the cluster and never deletes anything there, so it is
# safe to re-run while jobs are training. It excludes everything that is machine-specific
# (the .venv, colcon output -- rebuild it there with cluster/build_ws.slurm) or bulky and
# unused by training (raw captures, photogrammetry intermediates).
set -euo pipefail
cd "$(dirname "$0")/.."
DIR="${CLUSTER_REPO_DIR:-pepper-sim2real}"
case "${1:-}" in
  up)
    rsync -az --info=progress2 \
        --exclude=.venv/ --exclude=__pycache__/ \
        --exclude=/ros2_ws/build/ --exclude=/ros2_ws/install/ --exclude=/ros2_ws/log/ \
        --exclude='/ros2_ws/src/*/**/build/' --exclude='/ros2_ws/src/*/**/install/' \
        --exclude='/ros2_ws/src/*/**/log/' \
        --exclude=/rl/runs/ --exclude=/rl/cache/body_randomisation/ \
        --exclude=/simulation/photogrammetry/raw/ --exclude=/simulation/photogrammetry/work/ \
        --exclude=/cluster/logs/ \
        ./ "$2:$DIR/"
    ;;
  down)
    mkdir -p rl/runs
    rsync -az --info=progress2 "$2:$DIR/rl/runs/" rl/runs/
    ;;
  *)
    sed -n 2,10p "$0"; exit 1 ;;
esac
