#!/usr/bin/env bash
# One-command real-robot calibration session (a run: docs/calib_20261002_0948/).
# Runs offline: needs only the laptop and Pepper on the same network, no internet.
#
#   tools/calib_session.sh 192.168.1.42     # her IP (press her chest button once)
#   tools/calib_session.sh                  # no IP: finds her on the laptop's Wi-Fi network
#   tools/calib_session.sh --fake           # rehearsal with no robot
#
# Steps: find her -> preflight -> shim -> battery -> wake up -> standing joint angles ->
# bridge at 50 Hz odometry -> base sysid (7 runs, Enter before each) -> depth noise -> rest.
# Everything is saved to ~/Research/pepper-sim2real/docs/calib_<date_time>/.
# Ctrl-C at any point stops the base and shuts everything down.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
source /opt/ros/humble/setup.bash
[ -f .venv/bin/activate ] && source .venv/bin/activate
set -u
OUT="$REPO/docs/calib_$(date +%Y%m%d_%H%M)"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/session.log") 2>&1

FAKE=0; IP="${1:-}"
[ "$IP" = "--fake" ] && { FAKE=1; IP=""; }
SHIM_PID=""; BRIDGE_PID=""
say()  { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
ok()   { printf '   \033[32mOK\033[0m %s\n' "$*"; }
warn() { printf '   \033[33mWARNING\033[0m %s\n' "$*"; }
raw()  { python3 tools/pepper_bridge_node.py --raw "$*" 2>/dev/null | tail -1; }
cleanup() {
  echo; echo "stopping the base and shutting down..."
  [ -n "$SHIM_PID" ] && raw STOP > /dev/null 2>&1
  [ -n "$BRIDGE_PID" ] && kill "$BRIDGE_PID" 2>/dev/null
  [ -n "$SHIM_PID" ] && { raw STOP > /dev/null 2>&1; kill "$SHIM_PID" 2>/dev/null; }
  sleep 1
  echo "results are in: $OUT"
}
trap cleanup EXIT
trap 'exit 130' INT

reachable() { timeout 2 bash -c "</dev/tcp/$1/9559" 2>/dev/null; }

# ---------------------------------------------------------------- 1. find her
say "1/9  Finding Pepper"
if [ $FAKE = 1 ]; then
  ok "rehearsal: fake robot"
else
  if [ -n "$IP" ] && reachable "$IP"; then
    ok "reached $IP on port 9559"
  else
    [ -n "$IP" ] && warn "$IP does not answer on 9559 -- searching the laptop's Wi-Fi network"
    PS1FILE=$(wslpath -w "$REPO/tools/find_pepper.ps1")
    IP=$(powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$PS1FILE" 2>/dev/null | tr -d '\r' | tail -1)
    if [ -z "$IP" ] || ! reachable "$IP"; then
      echo "   Could not find her. Check: laptop Wi-Fi is on HER router (not the phone hotspot),"
      echo "   she has finished booting (eyes steady), then press her chest button and run:"
      echo "       tools/calib_session.sh <the IP she says>"
      exit 1
    fi
    ok "found her at $IP"
  fi
fi
echo "$IP" > "$OUT/robot_ip.txt"

# ---------------------------------------------------------------- 2. preflight
say "2/9  Preflight"
if [ $FAKE = 1 ]; then tools/pepper_preflight.sh --fake > "$OUT/preflight.txt" 2>&1
else tools/pepper_preflight.sh "$IP" > "$OUT/preflight.txt" 2>&1; fi
tail -1 "$OUT/preflight.txt" | grep -q "^Ready" && ok "Ready" || { warn "preflight not Ready -- see $OUT/preflight.txt"; tail -8 "$OUT/preflight.txt"; }

# ---------------------------------------------------------------- 3. shim
say "3/9  Starting the robot connection (shim)"
if [ $FAKE = 1 ]; then tools/run_shim.sh --fake > "$OUT/shim.log" 2>&1 &
else tools/run_shim.sh --robot-ip "$IP" > "$OUT/shim.log" 2>&1 & fi
SHIM_PID=$!
for i in $(seq 1 30); do grep -q "commands on" "$OUT/shim.log" && break; sleep 1; done
grep -q "commands on" "$OUT/shim.log" && ok "shim running" || { warn "shim did not start:"; tail -5 "$OUT/shim.log"; exit 1; }

# ---------------------------------------------------------------- 4. battery + floor
say "4/9  Battery and floor"
V=$(raw VERSION); echo "   $V" | tee "$OUT/version.txt"
BAT=$(echo "$V" | awk '{print $3}' | cut -d. -f1)
[ -n "$BAT" ] && [ "$BAT" -lt 50 ] 2>/dev/null && warn "battery ${BAT}% -- below 50%; plug her in if you can, numbers may read low"
read -r -p "   Type the floor type (carpet / tiles / vinyl / wood / other) and press Enter: " FLOOR
echo "floor: ${FLOOR:-unrecorded}" > "$OUT/floor.txt"

# ---------------------------------------------------------------- 5. wake up
say "5/9  Waking her up (motors on, Autonomous Life off, standing pose) -- takes ~10 s"
W=$(raw WAKEUP); [ "$W" = "OK" ] && ok "awake" || { warn "WAKEUP said: $W"; exit 1; }
sleep 2

# ---------------------------------------------------------------- 6. standing joint angles
say "6/9  Recording her standing joint angles (for the sim's initial pose)"
J=HeadYaw,HeadPitch,LShoulderPitch,LShoulderRoll,LElbowYaw,LElbowRoll,LWristYaw,RShoulderPitch,RShoulderRoll,RElbowYaw,RElbowRoll,RWristYaw,HipRoll,HipPitch,KneePitch
A=$(raw GETANGLES $J)
{ echo "joints: $J"; echo "$A"; } | tee "$OUT/standing_angles.txt"

# ---------------------------------------------------------------- 7. bridge
say "7/9  Starting the ROS bridge (odometry 50 Hz, no depth stream during driving)"
start_bridge() {
  python3 tools/pepper_bridge_node.py --ros-args -p odom_rate:="$1" -p publish_depth:=false \
      > "$OUT/bridge_$1.log" 2>&1 &
  BRIDGE_PID=$!; sleep 6
}
start_bridge 50.0
grep -h "velocity clamp" "$OUT/bridge_50.0.log" | sed 's/^/   /'
HZ=$(timeout 8 ros2 topic hz /pepper_robot/odom 2>/dev/null | grep -m1 "average rate" | awk '{print $3}')
echo "   odometry rate: ${HZ:-none} Hz"
if [ -z "$HZ" ] || [ "${HZ%.*}" -lt 40 ]; then
  warn "odometry under 40 Hz over this link -- falling back to 20 Hz (rise times will read high; noted)"
  kill "$BRIDGE_PID" 2>/dev/null; sleep 2
  start_bridge 20.0
  echo "odom fell back to 20 Hz (50 Hz gave ${HZ:-nothing})" >> "$OUT/floor.txt"
fi

# ---------------------------------------------------------------- 8. base sysid
say "8/9  Drive tests: 7 runs, each waits for Enter"
cat <<'EOF'
   Before EVERY run: check the space in front of her is still clear, hand near her chest
   button, then press Enter. She does NOT need to go back to the tape between runs.
     1 straight 0.3 m/s 4 s (~1.2 m forward)   2 slow 0.1 m/s 5 s (~0.5 m forward)
     3 reverse 0.1 m/s 5 s (~0.5 m back)       4/5 spin left / right in place, 5 s
     6/7 small arc left / right, 6 s
   Ctrl-C stops her immediately.
EOF
python3 tools/base_sysid.py --source real --repeats 1 --out "$OUT/base_sysid_real.json"
python3 tools/base_sysid.py --compare "$OUT/base_sysid_real.json" docs/calib_20261002_0948/base_sysid_sim.json \
    | tee "$OUT/compare.txt"
kill "$BRIDGE_PID" 2>/dev/null; BRIDGE_PID=""; sleep 2

# ---------------------------------------------------------------- 9. depth noise
say "9/9  Depth camera noise: keep everything in front of her STILL for ~15 s"
read -r -p "   Press Enter when nobody is moving in front of her (or type s + Enter to skip): " SKIP
if [ "${SKIP:-}" != "s" ]; then
  timeout 120 python3 tools/check_depth.py --png "$OUT/depth.png" 2>&1 | tee "$OUT/check_depth.txt" | tail -15
fi

say "Done. Putting her to rest."
raw STOP > /dev/null; R=$(raw REST); echo "   REST: $R"
echo
echo "   ALL RESULTS: $OUT"
echo "   Reconnect the laptop to your phone hotspot and tell Claude: 'calibration done'."
