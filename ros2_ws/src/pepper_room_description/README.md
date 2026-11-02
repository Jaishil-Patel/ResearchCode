# pepper_room_description

The two environments the experiment compares. Everything here is **generated** -- do not
hand-edit `models/*/model.sdf` or `worlds/*.sdf`, because the next pipeline run overwrites
them and the edit disappears without a trace.

```
models/
  room_photogrammetry/   <- tools/mesh_import.py, from a phone capture
  room_standard/         <- tools/gen_world.py, from tape measurements
worlds/
  room_photogrammetry.sdf
  room_standard.sdf      <- tools/gen_world.py, both from the same template
```

## Why this is a ROS package and not a plain directory

Ignition resolves `model://room_photogrammetry` by scanning `IGN_GAZEBO_RESOURCE_PATH`.
`hooks/pepper_room_description.{dsv,sh}.in` prepend this package's `share/` directory to
that variable at `source install/local_setup.bash` time, which is the same trick
`pepper_robot_description` uses for the robot. A mesh sitting in `simulation/models/` is
invisible to the simulator no matter how correct it is.

## Usage

```bash
colcon build --merge-install --symlink-install --packages-select pepper_room_description
source install/local_setup.bash

export LIBGL_ALWAYS_SOFTWARE=1        # mandatory on WSL2: the only render path with correct sensor data
ros2 launch pepper_robot gz.launch.py \
    world:=$(ros2 pkg prefix --share pepper_room_description)/worlds/room_photogrammetry.sdf
```

`gz.launch.py` already declares `world` as a filepath argument, so no launch file needed
changing to support this.

## The two branches must stay structurally identical

Proposal S4.4.2: the independent variable is the geometry source and nothing else. Both
worlds come out of one template in `tools/gen_world.py` and should differ only in which
model they `<include>`. `diff` them after any regeneration -- if anything else has
drifted, the comparison no longer isolates photogrammetry and the result is not
interpretable.
