#!/usr/bin/env bash
# Stop the base and put Pepper back to rest. The panic button.
#
# Deliberately short and deliberately named something you can type from memory while a
# robot is moving. STOP first and unconditionally -- if REST fails for any reason, the
# wheels are already halted.
#
# This is the software stop. It does not replace the button on the robot.
#
# Usage:  tools/pepper_stop.sh          # stop and rest
#         tools/pepper_stop.sh --stop   # stop the wheels only, stay standing

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="python3 $ROOT/tools/pepper_bridge_node.py --raw"

$RAW STOP || echo "STOP failed -- use the button on the robot" >&2

if [ "${1:-}" != "--stop" ]; then
    $RAW "STREAM OFF" >/dev/null 2>&1
    $RAW REST || echo "REST failed -- she is stopped but still standing" >&2
fi
