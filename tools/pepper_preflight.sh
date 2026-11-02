#!/usr/bin/env bash
# Check that everything needed to talk to the physical Pepper is installed and reachable.
#
# Why this exists: robot access is the scarce resource. Every check below can be answered
# days in advance, at a desk, and each one that fails on the day costs session time you
# cannot get back. The network check in particular is worth doing early -- WSL2's default
# NAT networking puts this machine on a private subnet (172.x) rather than the LAN, and
# whether that reaches Pepper is much better discovered now than in the room.
#
# Run it twice: once before you go, and again on arrival with the robot's real IP.
#
# Usage:  tools/pepper_preflight.sh [robot_ip]
#         tools/pepper_preflight.sh --fake     # rehearsal: skip the robot-side checks
# Exit:   0 = ready, 1 = something needs fixing

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROBOT_IP="${1:-}"
NAOQI_PORT=9559
CMD_PORT=45455
DEPTH_PORT=45456

FAIL=0
WARN=0

ok()   { printf '  [ ok ] %s\n' "$1"; }
bad()  { printf '  [FAIL] %s\n' "$1"; FAIL=1; }
warn() { printf '  [warn] %s\n' "$1"; WARN=1; }
say()  { printf '\n=== %s ===\n' "$1"; }

# ---------------------------------------------------------------------------
say "Python 3 side (bridge node)"

if python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)' 2>/dev/null; then
    ok "python3 $(python3 -c 'import platform; print(platform.python_version())')"
else
    bad "python3 missing or too old"
fi

if python3 -c 'import numpy' 2>/dev/null; then
    ok "numpy $(python3 -c 'import numpy; print(numpy.__version__)')"
else
    bad "numpy missing -- needed to convert depth mm to metres:  pip3 install numpy"
fi

if python3 -c 'import rclpy' 2>/dev/null; then
    ok "rclpy importable (ROS 2 sourced)"
else
    warn "rclpy not importable -- source /opt/ros/humble/setup.bash before running the bridge"
fi

# ---------------------------------------------------------------------------
say "Python 2 side (NAOqi shim)"

if [ "$ROBOT_IP" = "--fake" ]; then
    ok "--fake: python2 and the NAOqi SDK are not needed"
    if python3 "$ROOT/tools/naoqi_shim.py" --help >/dev/null 2>&1; then
        ok "naoqi_shim.py runs under python3 for rehearsal"
    else
        bad "naoqi_shim.py does not run under python3 -- fake mode is broken"
    fi
else
    if command -v python2 >/dev/null 2>&1; then
        ok "python2 $(python2 -c 'import platform; print(platform.python_version())' 2>&1)"
    else
        bad "python2 missing:  sudo apt install python2 libpython2.7"
    fi

    SDK="${PYNAOQI_SDK:-$HOME/naoqi-sdk/pynaoqi-python2.7-2.5.5.5-linux64}"
    if [ -d "$SDK/lib/python2.7/site-packages" ]; then
        ok "SDK unpacked at ${SDK/#$HOME/\~}"
    else
        bad "no pynaoqi SDK at $SDK -- set PYNAOQI_SDK to where it is unpacked"
    fi

    # The SDK must NOT be on the global PYTHONPATH. It ships a Python 2 `qi` package that
    # uses `async` as an identifier, so any Python 3 process that inherits it -- ROS 2
    # nodes included -- dies with SyntaxError. tools/run_shim.sh scopes it correctly.
    if python3 -c 'import qi' 2>&1 | grep -q SyntaxError; then
        bad "the py2 SDK is on the global PYTHONPATH and is breaking python3 imports."
        printf '         Remove it from ~/.bashrc and use tools/run_shim.sh instead.\n'
    else
        ok "python3 imports are not shadowed by the py2 SDK"
    fi

    # The check that actually matters: can the SDK load under the environment the shim is
    # really launched with? Anything less tests the wrong thing -- "naoqi.py is on the
    # path" succeeds right up until libinaoqi.so fails to load.
    if command -v python2 >/dev/null 2>&1; then
        NAOQI_ERR=$(env PYTHONPATH="$SDK/lib/python2.7/site-packages" \
                        LD_LIBRARY_PATH="$SDK/lib:${LD_LIBRARY_PATH:-}" \
                        python2 -c 'import naoqi' 2>&1)
        if [ -z "$NAOQI_ERR" ]; then
            ok "naoqi imports under tools/run_shim.sh's environment"
        else
            bad "naoqi will not import even with the SDK scoped correctly"
            printf '         %s\n' "$(printf '%s' "$NAOQI_ERR" | tail -1)"
        fi
    fi
fi

# ---------------------------------------------------------------------------
say "NAOqi 2.9 fallback (libqi-python)"
# If the robot turns out to be NAOqi 2.9, the Python 2 SDK above does not apply -- that
# generation uses libqi-python, which is a Python 3 package. Worth knowing before the
# session rather than discovering it in the room.
if python3 -c 'import qi' 2>/dev/null; then
    ok "python3 'qi' $(python3 -c 'import qi; print(getattr(qi, "__version__", "?"))' 2>/dev/null) available -- a 2.9 robot is covered"
else
    warn "python3 'qi' not installed. Only needed if the robot is NAOqi 2.9:  pip3 install --user qi"
fi

# ---------------------------------------------------------------------------
say "Sockets"

for PORT in "$CMD_PORT" "$DEPTH_PORT"; do
    if command -v ss >/dev/null 2>&1 && ss -lun 2>/dev/null | grep -q ":$PORT " ; then
        warn "UDP $PORT already in use -- an old naoqi_shim.py is probably still running"
    elif command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | grep -q ":$PORT " ; then
        warn "TCP $PORT already in use -- an old naoqi_shim.py is probably still running"
    else
        ok "port $PORT free"
    fi
done

# ---------------------------------------------------------------------------
say "Network to the robot"

if [ -z "$ROBOT_IP" ] || [ "$ROBOT_IP" = "--fake" ]; then
    printf '  skipped (no IP given). Press Pepper'\''s chest button once and she says her\n'
    printf '  address, then re-run:  tools/pepper_preflight.sh <ip>\n'
else
    if ping -c 1 -W 2 "$ROBOT_IP" >/dev/null 2>&1; then
        ok "$ROBOT_IP responds to ping"
    else
        warn "$ROBOT_IP does not respond to ping (some networks block ICMP; the port check below is what matters)"
    fi

    # The check that actually decides whether the shim can work. Uses bash's own
    # /dev/tcp so this does not depend on netcat being installed.
    if timeout 3 bash -c "exec 3<>/dev/tcp/$ROBOT_IP/$NAOQI_PORT" 2>/dev/null; then
        ok "NAOqi port $NAOQI_PORT open on $ROBOT_IP -- WSL2 can reach the robot"
    else
        bad "cannot open $ROBOT_IP:$NAOQI_PORT"
        cat <<'EOF'
         Check, in order:
           - the robot is on and finished booting (the eyes settle)
           - this machine and the robot are on the same network
           - WSL2 NAT is not in the way. `ip route get <robot_ip>` from inside WSL
             should leave via eth0. If the robot is reachable from Windows but not
             from WSL, add to C:\Users\<you>\.wslconfig:
                 [wsl2]
                 networkingMode=mirrored
             then `wsl --shutdown` and reopen. (Available on this Windows build.)
EOF
    fi
fi

# ---------------------------------------------------------------------------
say "Result"

if [ "$FAIL" -ne 0 ]; then
    echo "NOT READY -- fix the [FAIL] lines above."
    exit 1
fi
if [ "$WARN" -ne 0 ]; then
    echo "Ready, with warnings. Read the [warn] lines before you rely on this."
else
    echo "Ready."
fi
exit 0
