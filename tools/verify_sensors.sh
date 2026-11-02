#!/usr/bin/env bash
# Verify that Pepper's simulated sensors publish AND report true distances.
#
# Why this exists: on WSL the `ogre` v1 render engine runs happily but returns every lidar
# beam pinned at <min> range, with nothing in the log. Checking that /scan merely *publishes*
# is not enough -- it publishes garbage. This places the robot a known distance from a wall
# and asserts the reading matches, which is the only check that catches the silent mode.
#
# Usage:  tools/verify_sensors.sh
# Exit:   0 = sensors correct, 1 = failure

set -o pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/ros2_ws"
DESC="$WS/src/pepper_ign_moveit2/pepper_robot_description"

source /opt/ros/humble/setup.bash
source "$WS/install/local_setup.bash" 2>/dev/null
export IGN_GAZEBO_RESOURCE_PATH="$WS/install/share:${IGN_GAZEBO_RESOURCE_PATH:-}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"

WALL_X=1.5         # wall centre
WALL_THICK=0.2     # so the near face sits at 1.4
LASER_X=0.0562     # front_laser offset along +X of Tibia (= model origin)
# Perpendicular distance from the sensor to the wall face. The scan spans +/-30 deg, so the
# off-axis beams read further (d/cos t); the minimum over all beams is the perpendicular.
EXPECT=$(awk -v w="$WALL_X" -v t="$WALL_THICK" -v l="$LASER_X" 'BEGIN{printf "%.4f", w - t/2 - l}')
TOL=0.05

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "Generating model (LIBGL_ALWAYS_SOFTWARE=$LIBGL_ALWAYS_SOFTWARE)..."
xacro "$DESC/urdf/pepper_robot.urdf.xacro" \
      name:=pepper_robot ros2_control:=false sensors:=true > "$TMP/p.urdf" || exit 1
ign sdf -p "$TMP/p.urdf" > "$TMP/p.sdf" 2>/dev/null
[ -s "$TMP/p.sdf" ] || { echo "FAIL: SDF conversion produced nothing"; exit 1; }

NSENS=$(grep -c '<sensor' "$TMP/p.sdf")
echo "  sensors in SDF: $NSENS (expect 3)"
[ "$NSENS" -eq 3 ] || { echo "FAIL: expected 3 sensors"; exit 1; }

python3 - "$TMP" "$WALL_X" "$WALL_THICK" <<'PY'
import re, sys
tmp, wx, wt = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(f"{tmp}/p.sdf").read()
s = re.sub(r"(<model name=['\"]pepper_robot['\"]>)", r"\1<static>true</static>", s, count=1)
body = re.search(r'<model[\s\S]*</model>', s).group(0)
open(f"{tmp}/w.sdf", "w").write(f"""<?xml version="1.0"?>
<sdf version="1.9"><world name="verify">
<plugin filename="ignition-gazebo-physics-system" name="ignition::gazebo::systems::Physics"/>
<plugin filename="ignition-gazebo-scene-broadcaster-system" name="ignition::gazebo::systems::SceneBroadcaster"/>
<plugin filename="ignition-gazebo-sensors-system" name="ignition::gazebo::systems::Sensors">
<render_engine>ogre2</render_engine></plugin>
<plugin filename="ignition-gazebo-imu-system" name="ignition::gazebo::systems::Imu"/>
<light type="directional" name="sun"><pose>0 0 10 0 0 0</pose>
<diffuse>0.8 0.8 0.8 1</diffuse><direction>-0.3 0.3 -0.9</direction></light>
<model name="wall"><static>true</static><pose>{wx} 0 0.5 0 0 0</pose><link name="wl">
<collision name="c"><geometry><box><size>{wt} 4 1</size></box></geometry></collision>
<visual name="v"><geometry><box><size>{wt} 4 1</size></box></geometry></visual></link></model>
{body}</world></sdf>""")
PY

echo "Starting headless server (software rendering is slow; allow ~30 s)..."
timeout 90 ign gazebo -s -r -v 1 "$TMP/w.sdf" > "$TMP/ign.log" 2>&1 &
SRV=$!
sleep 30

FAIL=0

RANGES=$(timeout 10 ign topic -e -t /scan -n 1 2>/dev/null | grep '^ranges' | sed 's/ranges: //')
if [ -z "$RANGES" ]; then
  echo "FAIL: /scan published nothing"
  grep -Eio 'UnimplementedException|terminate called' "$TMP/ign.log" | sort -u | head -2
  FAIL=1
else
  NBEAM=$(echo "$RANGES" | wc -l)
  CENTRE=$(echo "$RANGES" | awk 'BEGIN{m=1e9} $1<m && $1!="inf" {m=$1} END{print m}')
  OK=$(awk -v c="$CENTRE" -v e="$EXPECT" -v t="$TOL" 'BEGIN{print (c>e-t && c<e+t) ? 1 : 0}')
  echo "  /scan beams: $NBEAM (expect 15)"
  echo "  /scan min (perpendicular) range: $CENTRE  (expect ~$EXPECT +/- $TOL)"
  if [ "$OK" != "1" ]; then
    echo "FAIL: scan does not match ground truth."
    echo "      All beams at minimum range is the known 'ogre v1' silent-garbage mode."
    echo "      Ensure LIBGL_ALWAYS_SOFTWARE=1 and render_engine is ogre2."
    FAIL=1
  fi
fi

for T in /depth_camera /imu; do
  if timeout 8 ign topic -i -t $T 2>&1 | grep -q 'Publishers'; then
    echo "  $T: publishing"
  else
    echo "FAIL: $T has no publisher"; FAIL=1
  fi
done

kill $SRV 2>/dev/null; wait 2>/dev/null

if [ "$FAIL" -eq 0 ]; then
  echo ""
  echo "PASS: sensors publish and report true distances."
else
  echo ""
  echo "FAILED - on WSL2, check that LIBGL_ALWAYS_SOFTWARE=1 is set (tools/rl_env.sh)."
fi
exit $FAIL
