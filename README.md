# Pepper Sim-to-Real

Code, simulation assets and experiment records for training a Pepper robot to navigate a room
in simulation and testing the trained policies on the real robot. Two simulated versions of
the same room are compared: one built from a photogrammetry scan and one built by hand from
tape measurements.

The method and results are in the [report](report/latex/main.pdf). This file describes what's
in the folder.

## Layout

```
pepper-sim2real/
├── rl/           Learning code, settings, route sets and training output
├── simulation/   Room scans, measurements and the Blender demo room
├── ros2_ws/      ROS 2 workspace: robot and room descriptions, simulator packages
├── tools/        Scripts for the scan pipeline, the real robot and plotting
├── cluster/      Container and job scripts for training on mscluster
├── docs/         Room layouts and presentations
└── report/       The research paper and its LaTeX source
```

## `rl/`

| Path | What it holds |
| --- | --- |
| `env/` | The Gymnasium navigation environment (`pepper_nav_env.py`), plus route sampling, dynamics randomisation, simulator control and pose tracking |
| `scripts/` | Entry points: `run.py` (start the simulator, train, shut down), `train.py`, and checks for the environment, throughput and pose tracker |
| `config/` | Settings for the environment, randomisation and SAC |
| `cache/` | The fixed route sets used for scoring, and scratch files for friction randomisation |
| `runs/` | One folder per training run, with the final model, checkpoints and metadata |
| `reports/` | Throughput benchmarks, and final simulation evaluations in `final/` |
| `metrics.py` | The evaluation metrics, used by simulation scoring |
| `sim_process.py` | Starts, waits for and stops the simulator |

## `simulation/`

| Path | What it holds |
| --- | --- |
| `models/` | `Demo_room.blend`, a Blender room model |
| `photogrammetry/raw/` | Untouched room captures (`final_scan`, `test_scan`, `demo_room`) |
| `photogrammetry/work/` | Intermediate files generated from the captures |
| `photogrammetry/ground_truth/` | Tape measurements, room specifications and blank templates |
| `photogrammetry/manifests/` | A record of how each room mesh was processed |
| `photogrammetry/reports/` | The accuracy check for `final_scan` |

## `ros2_ws/`

| Path | What it holds |
| --- | --- |
| `src/pepper_room_description/` | The two rooms as simulator models and worlds. These files are generated, so don't edit them by hand ([details](ros2_ws/src/pepper_room_description/README.md)) |
| `src/pepper_ign_moveit2/` | The Pepper robot description and simulator setup, ported to ROS 2 Humble and Ignition Fortress |
| `src/gz_ros2_control/`, `src/ros2_controllers/`, `src/ros_ign/`, `src/pymoveit2/` | Third-party packages |
| `src/vendor.repos` | The exact commit of each third-party package |
| `src/patches/` | The changes made to `pepper_ign_moveit2`, as a patch |
| `hello_ros/` | A ROS 2 starter package |
| `build/`, `install/`, `log/` | `colcon` build output |

## `tools/`

| Group | Scripts |
| --- | --- |
| Scan to simulator model | `mesh_import.py`, `blend_export.py`, `blender_room_visual.py`, `mesh_taubin.py`, `mesh_smooth_sweep.py`, `mesh_holes.py`, `mesh_level.py`, `mesh_flatten_floor.py`, `mesh_validate.py`, `mesh_pipeline_selftest.sh` |
| World generation | `gen_world.py` |
| Real-robot connection | `naoqi_shim.py`, `run_shim.sh`, `pepper_bridge_node.py`, `find_pepper.ps1`, `pepper_stop.sh` (stops the base and rests the robot) |
| Robot and sensor checks | `pepper_preflight.sh`, `verify_pepper.sh`, `verify_sensors.sh`, `check_depth.py`, `real_obs_check.py`, `contact_relay.py` |
| Calibration | `calib_session.sh`, `base_sysid.py`, `env_sysid.py` |
| Viewing and plotting | `visualise_policy.py`, `pepper_live_view.py`, `plot_room_layout.py`, `room_planner.html`, `room_plotter.html` |
| Presentation demos | `demo_wave.py`, `demo_wave_real.py`, `demo_dance_real.py` |
| Shell setup | `rl_env.sh` (source this to load ROS 2 and the Python environment) |

## `cluster/`

The Dockerfile, Slurm jobs, and sync and evaluation scripts for running training on mscluster.
Setup and usage are in the [cluster guide](cluster/README.md).

## `docs/`

| Path | What it holds |
| --- | --- |
| `session_20261015/` | Depth captures and logs from the 15 October session |
| `room_layout_*.png`, `room_layouts.pdf` | Printable room layouts for setting up robot sessions |
| `pepper_depth_*.png` | Depth camera captures from the real robot |
| `throughput_20260911.json` | A simulator throughput benchmark |
| `photogrammetry_runbook.md` | Step-by-step instructions for capturing and processing a room |
| `presentation 1/` to `presentation 4/` | Project presentations and their media |

## `report/`

`report/latex/` holds the paper ([PDF](report/latex/main.pdf)), its source (`main.tex`,
`sections/`, `figures/`, `refs.bib`), the IEEE template and the AI declaration.

## Running training

You'll need Ubuntu 22.04, ROS 2 Humble and Ignition Gazebo. To train one policy:

```bash
python3 rl/scripts/run.py --room final_scan --branch photogrammetry --spawn "0.6 2.105 0.0"
```

Use `--branch standard` for the other room, and add `--smoke` for a short test run first.

## Not under version control

`.gitignore` excludes these because they can be regenerated, restored or are specific to this
machine:

- `.venv/`
- `ros2_ws/build/`, `ros2_ws/install/` and `ros2_ws/log/`
- Third-party packages in `ros2_ws/src/`, which are restored from `vendor.repos` and `patches/`
- `rl/runs/`, `rl/reports/` and `rl/cache/body_randomisation/`
- `simulation/photogrammetry/work/`
