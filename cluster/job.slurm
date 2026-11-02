#!/bin/bash
# Run one command inside the container on a compute node.
#
#   sbatch --job-name=photogrammetry cluster/job.slurm \
#       python3 rl/scripts/run.py --room final_scan --branch photogrammetry \
#       --spawn "0.6 2.105 0.0" --resume auto
#
# Submit from the repo root (the log path below is relative to it). Flags given to sbatch
# on the command line override the defaults here, e.g. `--time=04:00:00` for a smoke test.
#
# --exclusive is needed for correctness, not just speed: rl/sim_process.shutdown() sweeps
# `pkill -9 -f "ign gazebo"` between launches, which would kill this user's OTHER job if
# the scheduler packed two onto one node. stampede nodes have 16 cores and this job uses
# them -- llvmpipe renders the depth camera on every core it finds.
#SBATCH --partition=stampede
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=3-00:00:00
#SBATCH --output=cluster/logs/%x-%j.out
#SBATCH --error=cluster/logs/%x-%j.out

set -eo pipefail
SIF="${PEPPER_SIF:-$HOME/containers/pepper-sim.sif}"
[ -f "$SIF" ] || { echo "no container at $SIF (see cluster/README.md, step 2)"; exit 1; }
[ $# -gt 0 ] || { echo "usage: sbatch cluster/job.slurm <command...>"; exit 1; }

# PEPPER_RETRIES (default 0): relaunch the whole command -- simulator included -- this many
# more times if it exits non-zero. Only for commands that resume themselves (run.py with
# --resume auto, which continues from the last 25k-step resume point). Ignition can stall
# outright under load: on 7 Oct 2026 /set_pose went unanswered through 3+6+12 s and ended a
# run three hours in. No service-level retry fixes a stalled server; a fresh one does.
# A clean exit is never retried: rerunning a FINISHED run with --resume auto would start
# a new one.
RETRIES="${PEPPER_RETRIES:-0}"

echo "job $SLURM_JOB_ID on $(hostname), started $(date -Is), retries allowed: $RETRIES"
attempt=0
while true; do
    set +e
    apptainer exec --cleanenv \
        --env "SLURM_JOB_ID=$SLURM_JOB_ID" \
        "$SIF" "$SLURM_SUBMIT_DIR/cluster/in_container.sh" "$@"
    rc=$?
    set -e
    [ $rc -eq 0 ] && break
    if [ $attempt -ge "$RETRIES" ]; then
        echo "FAILED with exit $rc after $((attempt + 1)) attempt(s), $(date -Is)"
        exit $rc
    fi
    attempt=$((attempt + 1))
    echo "exit $rc at $(date -Is); relaunching (attempt $((attempt + 1)) of $((RETRIES + 1))) in 30 s"
    pkill -9 -u "$USER" -f "ign gazebo|parameter_bridge|Xvfb|ruby.*ign" || true
    sleep 30
done
echo "finished $(date -Is)"
