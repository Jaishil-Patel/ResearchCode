# Training on mscluster

The full runs take about two days per branch, which ties up the PC. mscluster (Wits
Mathematical Sciences, Slurm) runs them instead: one job per branch on `stampede`
(16 cores, 30 GB, 3-day limit), in parallel, inside an Apptainer container.

There's no GPU here, and none is needed: the depth camera is software-rendered (Mesa
llvmpipe) exactly as on WSL, under a virtual X display (Xvfb).

## One-time setup

**1. Build the image (PC, Docker Desktop running).** From PowerShell, in the repo:

    docker build -t pepper-sim:humble cluster
    docker save pepper-sim:humble -o pepper-sim.tar

**2. Upload it and convert it to Apptainer (cluster).**

    scp pepper-sim.tar jpatel@<cluster>:~/containers/      # mkdir ~/containers first
    # then on the cluster -- as a job, not on the login node:
    sbatch --partition=batch --time=02:00:00 --output=sif-%j.out \
        --wrap "apptainer build $HOME/containers/pepper-sim.sif docker-archive://$HOME/containers/pepper-sim.tar"

Delete `pepper-sim.tar` on both sides once the `.sif` exists. Check your quota first
(`quota -s` or `df -h ~`): the tar and the .sif together are several GB.

**3. Copy the repo and build the workspace.** In WSL:

    cluster/sync.sh up jpatel@<cluster>

On the cluster:

    cd ~/pepper-sim2real && mkdir -p cluster/logs
    sbatch cluster/build_ws.slurm

## Before the real runs: a smoke test

Short jobs prove that rendering, controllers and contacts work on a compute node, and show
how fast that node actually is:

    sbatch --time=00:30:00 --job-name=bench cluster/job.slurm \
        python3 rl/scripts/bench_throughput.py --world room_photogrammetry --spawn "0.6 2.105 0.0"
    for b in photogrammetry standard; do
      sbatch --time=04:00:00 --job-name=smoke-$b cluster/job.slurm \
        python3 rl/scripts/run.py --room final_scan --branch $b \
        --spawn "0.6 2.105 0.0" --smoke
    done

Read `cluster/logs/<name>-<jobid>.out`. Both smokes should finish with a progress line and
`done:`; what matters is that nothing crashed.
Training speed: the log's monitor.csv in the run directory gives steps per wall second;
on the PC it was ~4.5, i.e. ~20-24 h for 300k steps. Over ~60 h won't fit
the 3-day limit in one job -- still fine, see "Resuming".

## The real runs: 3 seeds x 2 branches

One seed per branch can't separate "photogrammetry helps" from RL's run-to-run variance,
so each branch trains with seeds 0, 1 and 2. Six parallel jobs take about the same wall
time as two.

    for s in 0 1 2; do
      for b in photogrammetry standard; do
        sbatch --job-name=$b-s$s cluster/job.slurm \
          python3 rl/scripts/run.py --room final_scan --branch $b --seed $s \
          --spawn "0.6 2.105 0.0" --resume auto
      done
    done

`squeue -u $USER` shows them, and `tail -f cluster/logs/photogrammetry-s0-<jobid>.out`
follows one. A progress line appears every 25k steps.

Which policy goes on the real robot is decided by a rule fixed NOW, before any results:
**per branch, the seed with the highest simulated SR at 300k steps (ties: higher SPL)**.
Choosing after seeing real-world numbers would bias the comparison.

Bring results home with `cluster/sync.sh down jpatel@<cluster>` (in WSL).

## Resuming

Every 25k steps `train.py` writes a resume point: checkpoint, replay buffer, then
`resume.json`. If a job dies (time limit, node failure, pre-emption), submit **the exact
same command again**. `--resume auto` finds the newest unfinished run of that branch and
seed, and
continues it in the same run directory, losing at most the steps since the last resume
point. It refuses to resume if any config hash changed since the run started.

## Things that will bite

- **Don't edit `rl/config/*.yaml` between the two branches' jobs.** Resume and
  `--require-match` both check the hashes, but the cheapest fix is not to need them.
- **`sync.sh up` while jobs run is safe** for code that has already been imported, but a
  resumed job picks up whatever is on disk. Sync before you submit, not during.
- **`--exclusive` in job.slurm is not optional.** `rl/sim_process.shutdown()` runs
  `pkill -9 -f "ign gazebo"`, which would kill your other branch's simulator if both
  jobs landed on one node.
