#!/usr/bin/env bash
# Real-robot counterpart to verify_sensors.sh: confirm the physical Pepper responds, and
# measure the two quantities the proposal currently only guesses at.
#
# Why this exists: verify_sensors.sh catches a simulator that publishes confident garbage.
# The equivalent hazard on hardware is a robot that accepts every command and reports
# healthy odometry while actually travelling a different distance than it thinks. Steps 4
# and 5 below are not pass/fail tests -- they are data collection that can only be done
# with the robot in front of you, and they feed directly into the write-up:
#
#   Step 4 (commanded vs achieved velocity) sets the "commanded-velocity behaviour" and
#   "wheel slippage" randomisation ranges of proposal S4.4.2. S4.4.4 currently concedes
#   these ranges are hand-picked; measured ranges are defensible under examination.
#
#   Step 5 (odometry drift) quantifies the Gomez et al. drift claim for *this* robot, and
#   decides how much work the external ground-truth pose reference of S4.4.6 has to do --
#   AprilTags may suffice, or motion capture may be needed. That decision has procurement
#   lead time and Task 6 starts 23 Aug.
#
# Prerequisites: naoqi_shim.py running in another terminal, a tape measure, chalk or tape
# to mark the floor, ~4 m of clear space, and a hand on the robot's stop.
#
# Rehearse the whole thing without a robot first:
#   python3 tools/naoqi_shim.py --fake  &&  tools/verify_pepper.sh
#
# Usage:  tools/verify_pepper.sh
# Exit:   0 = all steps completed, 1 = a step failed

set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="python3 $ROOT/tools/pepper_bridge_node.py --raw"
# Never "MOVE then sleep": the shim's watchdog halts the base after ~0.75 s of silence,
# so a single MOVE followed by a sleep travels for 0.75 s rather than the duration asked
# for -- which would silently under-report achieved velocity in step 5. --drive re-sends
# at 10 Hz and reports the elapsed time it actually held the command for.
DRIVE="python3 $ROOT/tools/pepper_bridge_node.py --drive"
RESULTS="$ROOT/docs/pepper_measurements_$(date +%Y%m%d_%H%M).txt"

TEST_SPEED=0.15      # m/s -- conservative; Pepper's maximum is roughly 0.35
TEST_DURATION=5      # s
TEST_YAW=0.4         # rad/s

# The pose the simulated depth camera is fixed at: pepper_robot.gazebo mounts camera_depth
# on the Head link at rpy 0 0 0, and initial_joint_positions.yaml sets the head here.
SIM_HEAD_YAW=0.0
SIM_HEAD_PITCH=0.45
HEAD_TOL=0.05        # rad

say()  { printf '\n=== %s ===\n' "$1"; }
fail() { printf 'FAIL: %s\n' "$1" >&2; exit 1; }
log()  { printf '%s\n' "$1" | tee -a "$RESULTS"; }

ask() {   # ask <prompt> <varname>
    local reply
    read -r -p "$1 " reply
    printf -v "$2" '%s' "$reply"
}

trap '$RAW "STOP" >/dev/null 2>&1' EXIT INT TERM

log "Pepper measurement session: $(date -Iseconds)"

# ---------------------------------------------------------------------------
say "1/8  Can we reach the shim?"
[ "$($RAW PING)" = "PONG" ] || fail "shim not answering -- start naoqi_shim.py first"
echo "  shim OK"

# ---------------------------------------------------------------------------
say "2/8  What is the robot?"
VERSION_LINE=$($RAW VERSION) || fail "VERSION failed"
read -r _ NAOQI_VERSION BATTERY <<<"$VERSION_LINE"
log "NAOqi version: $NAOQI_VERSION"
log "Battery: ${BATTERY}%"
# 2.5.x is the Pepper 1.8 line the CSSR4Africa manual targets; 2.9.x is the 1.8a with the
# Android tablet, which needs a different SDK download (the ALMotion calls are unchanged).
case "$NAOQI_VERSION" in
    2.5.*) echo "  Pepper 1.8 / NAOqi 2.5 -- matches the pynaoqi 2.5 SDK" ;;
    2.9.*) echo "  NOTE: NAOqi 2.9 (Pepper 1.8a). SDK download differs; this script is fine." ;;
    *)     echo "  NOTE: unrecognised NAOqi version -- record it and check SDK compatibility" ;;
esac
awk -v b="$BATTERY" 'BEGIN{ if (b+0 < 30) print "  WARNING: battery below 30% -- drive tests will be unrepresentative" }'

# ---------------------------------------------------------------------------
say "3/8  Wake up"
echo "Pepper will stiffen her joints and stand. Keep clear."
echo "This takes several seconds -- Autonomous Life off, wakeUp, StandInit, head pose."
ask "Ready? [y/N]" GO
[ "$GO" = "y" ] || fail "aborted by user"
[ "$($RAW WAKEUP)" = "OK" ] || fail "WAKEUP failed"
echo "  awake, Autonomous Life disabled, moveInit done"

# ---------------------------------------------------------------------------
say "4/8  Head pose  (S4.4.2 identical observation)"
# The depth camera is bolted to the Head link in both branches, so head angle *is* camera
# extrinsics. If the real head is not where the simulated head is, the policy sees the
# room from a direction it never trained on -- and nothing about the depth image will look
# obviously wrong, which is what makes this worth asserting rather than assuming.
HEAD_LINE=$($RAW "GETANGLES HeadYaw,HeadPitch") || fail "GETANGLES failed"
read -r _ HEAD_ANGLES <<<"$HEAD_LINE"
HEAD_YAW=${HEAD_ANGLES%%,*}
HEAD_PITCH=${HEAD_ANGLES##*,}

log ""
log "-- Step 4: head pose --"
log "HeadYaw:   $HEAD_YAW   (simulated: $SIM_HEAD_YAW)"
log "HeadPitch: $HEAD_PITCH   (simulated: $SIM_HEAD_PITCH)"
awk -v y="$HEAD_YAW" -v p="$HEAD_PITCH" -v sy="$SIM_HEAD_YAW" -v sp="$SIM_HEAD_PITCH" \
    -v tol="$HEAD_TOL" '
BEGIN {
    dy = y - sy; if (dy < 0) dy = -dy
    dp = p - sp; if (dp < 0) dp = -dp
    if (dy <= tol && dp <= tol)
        print "  head matches the simulated camera pose"
    else
        printf "  MISMATCH: off by %.3f rad yaw, %.3f rad pitch. The real depth image is\n  taken along a different ray than the training data.\n", dy, dp
}' | tee -a "$RESULTS"

# ---------------------------------------------------------------------------
say "5/8  Commanded vs achieved velocity  (S4.4.2 randomisation ranges)"
cat <<EOF
Mark Pepper's current position on the floor. She will drive forward at
${TEST_SPEED} m/s for ${TEST_DURATION} s, so she *should* travel
$(awk -v s="$TEST_SPEED" -v t="$TEST_DURATION" 'BEGIN{printf "%.3f", s*t}') m.
Clear at least 3 m ahead.
EOF
ask "Ready? [y/N]" GO
[ "$GO" = "y" ] || fail "aborted by user"

ODOM_BEFORE=$($RAW ODOM)
DRIVE_LINE=$($DRIVE "$TEST_SPEED" 0 0 "$TEST_DURATION") || fail "drive failed"
read -r _ ACTUAL_DURATION <<<"$DRIVE_LINE"
sleep 1
ODOM_AFTER=$($RAW ODOM)

read -r _ X0 Y0 _ _ _ _ <<<"$ODOM_BEFORE"
read -r _ X1 Y1 _ _ _ _ <<<"$ODOM_AFTER"
ODOM_DIST=$(awk -v a="$X0" -v b="$Y0" -v c="$X1" -v d="$Y1" \
    'BEGIN{printf "%.4f", sqrt((c-a)^2 + (d-b)^2)}')

ask "Measure the distance actually travelled with the tape, in metres:" TAPE_DIST
[ -n "$TAPE_DIST" ] || fail "no measurement entered"

log ""
log "-- Step 5: forward drive --"
log "Commanded:        ${TEST_SPEED} m/s for ${ACTUAL_DURATION} s (asked for ${TEST_DURATION})"
awk -v s="$TEST_SPEED" -v t="$ACTUAL_DURATION" -v o="$ODOM_DIST" -v m="$TAPE_DIST" '
BEGIN {
    expected = s * t
    printf "Expected:         %.3f m\n", expected
    printf "Odometry says:    %.3f m\n", o
    printf "Tape measure:     %.3f m\n", m
    printf "Achieved/commanded ratio: %.3f   <- randomisation range should bracket this\n",
           m / expected
    printf "Achieved speed:   %.3f m/s\n", m / t
    printf "Odometry error:   %+.3f m (%+.1f%% of true distance)\n", o - m, 100*(o - m)/m
}' | tee -a "$RESULTS"

# ---------------------------------------------------------------------------
say "6/8  Odometry drift over a there-and-back  (S4.4.6 pose reference)"
cat <<EOF
Pepper will drive back to roughly her starting mark. Odometry that closes the loop
cleanly means low drift; a large residual means an external pose reference is essential.
EOF
ask "Ready? [y/N]" GO
[ "$GO" = "y" ] || fail "aborted by user"

TURN_TIME=$(awk -v y="$TEST_YAW" 'BEGIN{printf "%.2f", 3.14159/y}')   # ~180 deg
$DRIVE 0 0 "$TEST_YAW" "$TURN_TIME" >/dev/null || fail "rotation failed"
sleep 1
$DRIVE "$TEST_SPEED" 0 0 "$TEST_DURATION" >/dev/null || fail "return drive failed"
sleep 1
ODOM_RETURN=$($RAW ODOM)

read -r _ X2 Y2 TH2 _ _ _ <<<"$ODOM_RETURN"
ask "Distance from the ORIGINAL start mark to where she stopped, in metres:" TAPE_RESIDUAL

log ""
log "-- Step 6: loop closure --"
awk -v x0="$X0" -v y0="$Y0" -v x2="$X2" -v y2="$Y2" -v th="$TH2" -v m="$TAPE_RESIDUAL" '
BEGIN {
    odom_residual = sqrt((x2-x0)^2 + (y2-y0)^2)
    printf "Odometry residual: %.3f m\n", odom_residual
    printf "True residual:     %.3f m\n", m
    printf "Odometry drift:    %+.3f m over the round trip\n", odom_residual - m
    printf "Final heading (odom): %.3f rad\n", th
    if ((odom_residual - m) < 0) d = m - odom_residual; else d = odom_residual - m
    if (d > 0.15)
        print "VERDICT: drift is significant -- external ground truth (mocap or AprilTags) required"
    else
        print "VERDICT: drift is modest over one trip -- but re-check over a full evaluation route"
}' | tee -a "$RESULTS"

# ---------------------------------------------------------------------------
say "7/8  Depth camera  (the observation modality, S4.4.2)"
DEPTH_RAW="$ROOT/docs/pepper_depth_$(date +%Y%m%d_%H%M).raw"
DEPTH_LINE=$($RAW "DEPTH $DEPTH_RAW") || fail "DEPTH failed"
read -r _ W H MIN_MM MAX_MM VALID <<<"$DEPTH_LINE"

log ""
log "-- Step 7: depth frame --"
log "Resolution:  ${W}x${H}   (simulated camera_depth is 320x240)"
log "Range seen:  ${MIN_MM} mm to ${MAX_MM} mm   (simulated clip is 400-8000 mm)"
log "Valid pixels: $(awk -v v="$VALID" 'BEGIN{printf "%.1f%%", v*100}')"
log "Raw 16-bit buffer: $DEPTH_RAW"
[ "$W" = "320" ] && [ "$H" = "240" ] \
    && echo "  matches the simulator" \
    || echo "  MISMATCH vs simulator -- the observation shape must be identical across sim and real"
awk -v v="$VALID" 'BEGIN{ if (v < 0.5) print "  WARNING: over half the frame has no depth return. Glossy or dark surfaces (S4.4.4) -- consider this when choosing the room." }'

# ---------------------------------------------------------------------------
say "8/8  Depth stream throughput"
# A single frame proves the camera works; sustained throughput is a different question.
# getImageRemote pulls the full frame off the robot, so at 320x240x2 and 15 Hz this is
# 2.3 MB/s over the network. Whether Pepper's wifi holds that up is the number that
# decides the real control rate, and it belongs in the write-up rather than in an
# assumption.
STREAM_LINE=$(python3 "$ROOT/tools/pepper_bridge_node.py" --probe-depth 5 2>/dev/null)
if [ -z "$STREAM_LINE" ]; then
    log "Depth stream: NO FRAMES in 5 s -- the stream is not working"
else
    read -r _ SW SH SBYTES SFPS <<<"$STREAM_LINE"
    log ""
    log "-- Step 8: depth stream --"
    log "Frames:      ${SW}x${SH}, ${SBYTES} bytes each"
    log "Achieved:    ${SFPS} fps   (target 15)"
    awk -v f="$SFPS" -v b="$SBYTES" 'BEGIN{
        printf "Throughput:  %.2f MB/s\n", f * b / 1048576
        if (f < 5)  print "  WARNING: below 5 fps. Prefer ethernet over wifi, and state the real rate in the write-up."
        else if (f < 12) print "  NOTE: short of 15 fps. Usable, but set the control rate from this, not from the target."
        else print "  sustains the simulated camera rate"
    }' | tee -a "$RESULTS"
fi

# ---------------------------------------------------------------------------
say "Done"
echo "Results written to: $RESULTS"
echo
echo "Next: put the robot back to rest with"
echo "  tools/pepper_stop.sh"
