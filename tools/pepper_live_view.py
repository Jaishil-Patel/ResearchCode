#!/usr/bin/env python3
"""Serve Pepper's camera as a live MJPEG stream you can open in a browser.

A viewing tool, not part of the experiment. It exists because a single frame tells you
very little about whether the head is aimed correctly or the depth sensor is behaving,
and pointing a browser at a URL needs no X server, no WSLg and no GUI toolkit -- which
matters when the whole stack lives inside WSL2.

    tools/run_shim.sh --robot-ip 192.168.1.9        # terminal 1
    python3 tools/pepper_live_view.py               # terminal 2
    # then open http://localhost:8080

Both cameras are available:

    --mode rgb      the head's top camera (default)
    --mode depth    the depth sensor, turbo-colourmapped, black where there is no return

Frames are pulled one at a time through the shim's RGB / DEPTH verbs rather than the TCP
depth stream on 45456. That is slower, but it works for both cameras and needs no second
socket; the achieved rate is printed once a second and is dominated by the link, not by
this process. Over 2.4 GHz Wi-Fi expect a few frames per second at VGA -- fine for aiming
the head, useless as a measurement of what the robot can actually sustain. Use step 8 of
verify_pepper.sh for that.
"""

import argparse
import io as _io
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pepper_bridge_node import DEFAULT_HOST, DEFAULT_PORT, ShimClient  # noqa: E402

PAGE = b"""<!doctype html><html><head><title>Pepper live</title>
<style>
 body{margin:0;background:#111;color:#ddd;font:14px system-ui,sans-serif;
      display:flex;flex-direction:column;align-items:center;justify-content:center;
      min-height:100vh;gap:12px}
 img{max-width:96vw;image-rendering:pixelated;border:1px solid #333}
</style></head><body>
<img src="/stream" alt="Pepper camera">
<div>Pepper live view &mdash; reload if the stream stalls</div>
</body></html>"""


class Grabber(threading.Thread):
    """Pulls frames from the shim into a single latest-JPEG slot.

    One slot, not a queue: a viewer that falls behind should see the newest frame, not
    work through a backlog of stale ones.
    """

    daemon = True

    def __init__(self, host, port, mode, raw_path, resolution):
        threading.Thread.__init__(self)
        self.client = ShimClient(host, port)
        self.mode = mode
        self.raw_path = raw_path
        self.resolution = resolution
        self.lock = threading.Lock()
        self.jpeg = None
        self.frames = 0
        self.width = 0
        self.height = 0
        self.stop_event = threading.Event()
        self.error = None

    def _encode(self, buf):
        if self.mode == "rgb":
            array = np.frombuffer(buf, dtype=np.uint8).reshape(self.height, self.width, 3)
            image = Image.fromarray(array, "RGB")
        else:
            from matplotlib import cm
            raw = np.frombuffer(buf, dtype="<u2").reshape(self.height, self.width)
            metres = raw.astype(np.float32) / 1000.0
            valid = raw != 0
            # Fixed 0.4-8.0 m scale, matching the simulated camera's clip range, so the
            # colours mean the same thing from frame to frame. A per-frame autoscale
            # looks better and makes the image impossible to read distances off.
            norm = np.clip((metres - 0.4) / (8.0 - 0.4), 0.0, 1.0)
            rgb = (cm.turbo(norm)[:, :, :3] * 255).astype(np.uint8)
            rgb[~valid] = 0
            image = Image.fromarray(rgb, "RGB")
        out = _io.BytesIO()
        image.save(out, "JPEG", quality=80)
        return out.getvalue()

    def run(self):
        verb = "RGB" if self.mode == "rgb" else "DEPTH"
        last_report = time.time()
        counted = 0
        try:
            request = "%s %s" % (verb, self.raw_path)
            if self.mode == "rgb":
                request += " %d" % self.resolution
            while not self.stop_event.is_set():
                reply = self.client.request(request)
                parts = reply.split()
                if not parts or parts[0] != verb:
                    self.error = reply
                    time.sleep(0.5)
                    continue
                self.width, self.height = int(parts[1]), int(parts[2])
                with open(self.raw_path, "rb") as handle:
                    buf = handle.read()
                jpeg = self._encode(buf)
                with self.lock:
                    self.jpeg = jpeg
                    self.frames += 1
                counted += 1
                now = time.time()
                if now - last_report >= 1.0:
                    print("  %.1f fps  (%dx%d %s)" % (
                        counted / (now - last_report), self.width, self.height, self.mode))
                    sys.stdout.flush()
                    counted = 0
                    last_report = now
        except Exception as exc:  # the shim dying should not take the server with it
            self.error = str(exc)
            print("grabber stopped: %s" % exc, file=sys.stderr)
        finally:
            self.client.close()

    def latest(self):
        with self.lock:
            return self.jpeg


def make_handler(grabber):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # the fps line is the only output worth having

        def do_GET(self):
            if self.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(PAGE)))
                self.end_headers()
                self.wfile.write(PAGE)
                return
            if self.path != "/stream":
                self.send_error(404)
                return

            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    jpeg = grabber.latest()
                    if jpeg is None:
                        time.sleep(0.1)
                        continue
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(
                        b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n")
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                    time.sleep(0.03)
            except (BrokenPipeError, ConnectionResetError):
                return  # the tab was closed; normal
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--http-port", type=int, default=8080)
    parser.add_argument("--mode", choices=("rgb", "depth"), default="rgb")
    parser.add_argument("--raw-path", default="/tmp/pepper_live.raw")
    parser.add_argument("--resolution", type=int, choices=(0, 1, 2), default=1,
                        help="rgb only: 0=QQVGA 160x120, 1=QVGA 320x240 (default), "
                             "2=VGA 640x480. Wi-Fi bandwidth, not the sensor, is what "
                             "limits the frame rate -- VGA manages about 1.3 fps")
    args = parser.parse_args()

    grabber = Grabber(args.host, args.port, args.mode, args.raw_path, args.resolution)
    if grabber.client.request("PING") != "PONG":
        print("ERR shim did not answer PING", file=sys.stderr)
        return 1
    grabber.start()

    server = ThreadingHTTPServer(("0.0.0.0", args.http_port), make_handler(grabber))
    print("live view (%s) on http://localhost:%d  -- Ctrl-C to stop"
          % (args.mode, args.http_port))
    sys.stdout.flush()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        grabber.stop_event.set()
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
