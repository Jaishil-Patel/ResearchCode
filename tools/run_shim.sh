#!/usr/bin/env bash
# Launch naoqi_shim.py with the pynaoqi SDK on the path -- for this process only.
#
# Why this exists rather than two lines in ~/.bashrc:
#
# The pynaoqi 2.5 SDK ships its own `qi` package, written in Python 2, which uses `async`
# as an identifier. `async` is a reserved word from Python 3.7. So putting the SDK's
# site-packages on a *global* PYTHONPATH -- which is what every install guide including
# the CSSR4Africa manual tells you to do -- makes every Python 3 process on the machine
# pick up that package and die with `SyntaxError: invalid syntax`. That includes ROS 2
# nodes, and it includes libqi-python, the Python 3 binding needed for a NAOqi 2.9 robot.
#
# The failure is confusing because it appears in programs that have nothing to do with
# the robot, so the SDK is scoped to the one interpreter that needs it: this one.
#
# A second reason the export cannot be PYTHONPATH alone: the SDK's native modules carry no
# RPATH (`readelf -d _inaoqi.so` shows no RPATH/RUNPATH), so Python finds `naoqi.py`,
# imports it, then fails on `libinaoqi.so: cannot open shared object file`.
# LD_LIBRARY_PATH has to be set before the process starts, which a Python-level sys.path
# fix could not do.
#
# Usage:  tools/run_shim.sh --robot-ip 192.168.1.42
#         tools/run_shim.sh --fake            # rehearsal; uses python3, no SDK needed
#
# Override the SDK location with PYNAOQI_SDK=/path/to/pynaoqi-...-linux64

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SDK="${PYNAOQI_SDK:-$HOME/naoqi-sdk/pynaoqi-python2.7-2.5.5.5-linux64}"

# --fake needs no SDK and no python2, so it runs on python3 and skips all of the above.
for arg in "$@"; do
    if [ "$arg" = "--fake" ]; then
        exec python3 "$ROOT/tools/naoqi_shim.py" "$@"
    fi
done

if [ ! -d "$SDK/lib/python2.7/site-packages" ]; then
    echo "No pynaoqi SDK at: $SDK" >&2
    echo "Set PYNAOQI_SDK to the unpacked pynaoqi SDK directory." >&2
    exit 1
fi

if ! command -v python2 >/dev/null 2>&1; then
    echo "python2 is not installed:  sudo apt install python2 libpython2.7" >&2
    exit 1
fi

# Prepended, not appended, and starting from empty: nothing else should be visible to
# this interpreter, and the SDK must win over any stray system packages.
exec env \
    PYTHONPATH="$SDK/lib/python2.7/site-packages" \
    LD_LIBRARY_PATH="$SDK/lib:${LD_LIBRARY_PATH:-}" \
    python2 "$ROOT/tools/naoqi_shim.py" "$@"
