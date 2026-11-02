# Presentation media

`checkpoint.html` slide 2 expects two files here:

| File | What it is |
|---|---|
| `ready.mp4` | ~20 s: Pepper waving, then driving. Recorded in Ignition Gazebo. |
| `ready-still.png` | A single frame from the same run, used as the video poster. |

Neither is committed — record them fresh. Do **not** substitute the GIFs in
`ros2_ws/src/pepper_ign_moveit2/vid/`: those are the upstream Sekkat et al. recordings, not this
setup, and presenting them as our own would misattribute the work.

## Recording

`LIBGL_ALWAYS_SOFTWARE=1` is **required**, even for a motion-only clip. The
`Ogre::UnimplementedException` in `GL3PlusTextureGpu::copyTo` is not specific to the depth camera —
it fires from the GUI's own render path, so `sensors:=false` does not avoid it. Measured 2 Aug 2026:
hardware GL crashes Ignition during startup with sensors off.

`enable_rviz:=false` matters too. RViz2 segfaults under WSLg (exit `-11`), and because `ros2 launch`
tears down the whole launch when a node dies, RViz taking itself out kills Gazebo with it.

```bash
cd ~/Research/pepper-sim2real/ros2_ws
export LIBGL_ALWAYS_SOFTWARE=1
source /opt/ros/humble/setup.bash
source install/local_setup.bash
ros2 launch pepper_robot gz.launch.py sensors:=false enable_rviz:=false
```

Software rendering is CPU-bound, so expect a low frame rate and a real-time factor well under 1.
Record generously and speed the clip up in post rather than fighting it live.

Note that `ros2 control list_controllers` tends to hang against this setup even when the controllers
are fine — check the launch output for `Configuring controller` lines and clean spawner exits
instead, or just confirm `/left_arm_controller/joint_trajectory` is present.

With the simulation up:

```bash
# wave — left arm, ~7 s
python3 tools/demo_wave.py

# drive — hold for a few seconds, then stop
ros2 topic pub /pepper_robot/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.2}}"
ros2 topic pub --once /pepper_robot/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0}}"
```

Capture from the Windows side — the Gazebo window is an ordinary Windows window under WSLg, so
Game Bar (`Win+G`) or ScreenToGif is far easier than screen-grabbing inside WSL.

## Assembling

Trim each clip to roughly 10 s, then concatenate and grab the poster frame:

```bash
printf "file 'wave.mp4'\nfile 'drive.mp4'\n" > concat.txt
ffmpeg -f concat -safe 0 -i concat.txt -c:v libx264 -crf 23 -pix_fmt yuv420p -an ready.mp4
ffmpeg -i ready.mp4 -ss 2 -frames:v 1 ready-still.png
rm concat.txt
```

`-an` drops audio: the slide autoplays muted, and browsers block autoplay on unmuted video.
Keep `ready.mp4` well under 10 MB — the upstream `ign.gif` is 8.2 MB for comparison.
