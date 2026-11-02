#!/usr/bin/env python3
"""Dynamics randomisation, proposal S4.4.2, after Peng et al. [2018].

    python3 -m rl.env.randomisation --selftest

A parameter vector is sampled at the start of each episode and held fixed for it. That is
Peng et al.'s scheme, and the "held fixed" half is the part that is easy to get wrong: a
parameter resampled every step is noise, which a policy learns to average out, whereas a
parameter fixed for an episode is a *world* the policy must cope with, which is what
produces robustness. The two entries that are deliberately per-step -- actuation noise --
are marked as such in randomisation.yaml, because jitter genuinely is noise.

Scope, stated plainly
---------------------
This class implements the command- and observation-level half of the scheme: velocity
scaling, latency, slip, and depth noise, dropout and frame rate. Mass, mass distribution and friction
need the world respawned rather than mutated in place, which happens at world reset, not
here -- see rl/env/body_randomisation.py, which resamples and applies floor friction on
that cadence. Mass and centre of mass are sampled by `sample_body_params` below but not
yet applied to anything; that module's docstring says so plainly, and so should the
write-up. See the MECHANISM note in randomisation.yaml for why respawn is the mechanism at
all, and say it in the write-up: "we randomised at the command level, and friction at
world reset" is a defensible choice, and silence about any part of it is not.

The same config drives both branches. `Randomiser.fingerprint()` is recorded in every run
directory so that claim is checkable after the fact rather than remembered.
"""

import argparse
import hashlib
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "rl" / "config" / "randomisation.yaml"


class CommandDelay:
    """A fixed delay of `steps` control steps on (v, omega), as a FIFO.

    Owned by PepperNavEnv, not by the Randomiser, because the delay is a property of the
    ROBOT that has to be present in evaluation too: the real Pepper reacts 0.4-1.2 s after a
    command (median 0.82 s, measured 2 Oct 2026), and a simulated evaluation without it would
    score the policy in an easier world than the one it is deployed into. The Randomiser
    only contributes a per-episode offset around it (`latency_steps`). Starts full of zeros,
    so the first `steps` commands of an episode are "nothing yet" -- as on the robot, where
    she has not started moving when the first commands go out.
    """

    def __init__(self, steps=0):
        self.reset(steps)

    def reset(self, steps):
        self.steps = max(0, int(steps))
        self._q = deque([(0.0, 0.0)] * self.steps)

    def push(self, v, w):
        if self.steps == 0:
            return float(v), float(w)
        self._q.append((float(v), float(w)))
        return self._q.popleft()


class Randomiser:
    """Per-episode dynamics parameters and the perturbations they imply."""

    def __init__(self, config_path=CONFIG, enabled=True):
        self.path = Path(config_path)
        self.cfg = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        self.enabled = enabled
        self.params = {}
        self._blob = None     # current solid no-return patch: [x0, width_px, top, height_px]
        self._next_frame_t = None   # sim time the next depth frame "arrives"

    def fingerprint(self):
        """Hash of the config as it was actually used.

        Recorded per run so "both branches used identical randomisation" can be verified
        from the artefacts rather than asserted from memory.
        """
        h = hashlib.sha256(self.path.read_bytes()).hexdigest()[:16]
        return f"{h}{'' if self.enabled else '+disabled'}"

    # -- sampling -----------------------------------------------------------------------

    def new_episode(self, rng):
        """Draw the episode's parameter vector. Call once, from env.reset()."""
        self._blob = None
        if not self.enabled:
            self.params = _identity_params()
            return self.params

        c, o = self.cfg["command"], self.cfg["observation"]
        u = lambda pair: float(rng.uniform(*pair))          # noqa: E731
        self.params = {
            "linear_scale": u(c["linear_scale"]),
            "angular_scale": u(c["angular_scale"]),
            "linear_noise_std": float(c["linear_noise_std_mps"]),
            "angular_noise_std": float(c["angular_noise_std_rps"]),
            # An OFFSET, in control steps, around the env's fixed sim delay
            # (env.yaml action.sim_command_delay_steps). May be negative. The env applies
            # max(0, delay + offset); this class no longer queues commands itself.
            "latency_steps": int(rng.integers(c["latency_steps"][0],
                                              c["latency_steps"][1] + 1)),
            "slip_probability": u(c["slip_probability"]),
            "depth_noise_std": float(o["depth_noise_std_m"]),
            "dropout_fraction": u(o["dropout_fraction"]),
        }
        if "depth_rate_hz" in o:
            lo, hi = (float(x) for x in o["depth_rate_hz"])
            self.params["depth_rate_hz"] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        self._next_frame_t = None
        return self.params

    # -- application --------------------------------------------------------------------

    def perturb_command(self, v, w, rng):
        """Scale, delay, drop and jitter one (v, omega) command."""
        if not self.enabled:
            return v, w
        p = self.params
        v = v * p["linear_scale"] + rng.normal(0.0, p["linear_noise_std"])
        w = w * p["angular_scale"] + rng.normal(0.0, p["angular_noise_std"])

        # Slip: the command is not executed at all this step. Modelled as a dropped
        # command rather than a reduced one, because a slipping wheel delivers no useful
        # traction rather than proportionally less.
        if rng.random() < p["slip_probability"]:
            v, w = 0.0, 0.0
        # Latency is applied by the env (CommandDelay), using p["latency_steps"] as an offset.
        return float(v), float(w)

    def depth_frame_due(self, t, rng):
        """Whether a new depth frame has arrived by sim time `t`, at this episode's rate.

        The camera is modelled as a clock ticking at `depth_rate_hz` with a random phase;
        a step sees a new frame only if a tick fell since the last one. False means the env
        shows the policy its previous observation again -- exactly what the real robot does
        when the stream (2-3 fps over Wi-Fi) is slower than the 5 Hz control loop. The first
        call of an episode is always due, so no episode starts on a missing frame.
        """
        rate = self.params.get("depth_rate_hz") if self.enabled else None
        if not rate:
            return True
        period = 1.0 / rate
        if self._next_frame_t is None:
            self._next_frame_t = t + float(rng.uniform(0.0, period))
            return True
        if t < self._next_frame_t:
            return False
        while self._next_frame_t <= t:
            self._next_frame_t += period
        return True

    def perturb_depth(self, depth_m, rng):
        """Additive noise and structured-light dropout on a metric depth image.

        Applied in METRES, before normalisation and before the invalid-value mapping, so
        a dropped pixel follows exactly the same path a genuine no-return does in
        pepper_bridge_node._to_metres. Applying it after normalisation would produce
        dropout values the real sensor can never generate.
        """
        if not self.enabled:
            return depth_m
        p = self.params
        out = depth_m + rng.normal(0.0, p["depth_noise_std"], size=depth_m.shape)
        if p["dropout_fraction"] > 0.0:
            mask = rng.random(depth_m.shape) < p["dropout_fraction"]
            out = np.where(mask, np.inf, out)
        out = self._apply_blob(out, rng)
        return out.astype(np.float32)

    def _apply_blob(self, img, rng):
        """A solid no-return patch that persists and drifts across frames.

        Measured on the real robot, 2 Oct 2026: a window filled a ~25%-wide, full-height
        band of the depth image with no return, steady across 28 frames, while only 1.3% of
        pixels flickered. Per-pixel dropout alone never produces that, and the policy reads
        no-return as FAR -- so glass, screens and black surfaces look like open space. A patch
        starts with `start_probability` per frame, then survives each frame with
        `continue_probability`, drifting a pixel or so sideways as the robot turns.
        """
        cfg = self.cfg["observation"].get("dropout_blob")
        if not cfg:
            return img
        h, w = img.shape[-2:]
        if self._blob is None:
            if rng.random() < float(cfg["start_probability"]):
                bw = max(1, int(round(w * rng.uniform(*cfg["width_fraction"]))))
                bh = max(1, int(round(h * rng.uniform(*cfg["height_fraction"]))))
                self._blob = [int(rng.integers(0, max(1, w - bw + 1))), bw,
                              int(rng.integers(0, max(1, h - bh + 1))), bh]
        elif rng.random() > float(cfg["continue_probability"]):
            self._blob = None
        else:
            self._blob[0] = int(np.clip(self._blob[0] + rng.integers(-1, 2), 0, max(0, w - self._blob[1])))
        if self._blob is not None:
            x0, bw, y0, bh = self._blob
            img = img.copy()
            img[..., y0:y0 + bh, x0:x0 + bw] = np.inf
        return img


def _identity_params():
    return {"linear_scale": 1.0, "angular_scale": 1.0, "linear_noise_std": 0.0,
            "angular_noise_std": 0.0, "latency_steps": 0, "slip_probability": 0.0,
            "depth_noise_std": 0.0, "dropout_fraction": 0.0}


# --------------------------------------------------------------------------------------
# world-reset half: mass, mass distribution, friction
# --------------------------------------------------------------------------------------

def sample_body_params(cfg, rng):
    """Draw the parameters that need the world respawned rather than mutated in place.

    Returned rather than applied: applying `mu`/`mu2` means respawning the room model
    (rl/env/body_randomisation.apply_friction), which belongs to whatever owns the
    world's lifetime, not to an env instance. Keeping it a pure function also makes the
    ranges testable. `mass_scale` and `com_offset_m` are drawn here too, and are NOT
    applied by anything yet -- see the note in randomisation.yaml's `body:` section.

    mu and mu2 share one sample: friction here means the floor's isotropic Coulomb
    coefficient, and the room templates in tools/gen_world.py and tools/mesh_import.py
    both already set mu == mu2 by default, so sampling them independently would invent an
    anisotropy this project has no model for.
    """
    b, s = cfg["body"], cfg["surface"]
    mu = float(rng.uniform(*s["mu"]))
    return {
        "mass_scale": float(rng.uniform(*b["mass_scale"])),
        "com_offset_m": float(rng.uniform(*b["com_offset_m"])),
        "mu": mu,
        "mu2": mu,
    }


# --------------------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------------------

def _selftest():
    fails = []

    def check(label, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")
        if not ok:
            fails.append(label)

    r = Randomiser()
    rng = np.random.default_rng(0)

    # -- the defining property of Peng et al.'s scheme --------------------------------
    r.new_episode(rng)
    p1 = dict(r.params)
    for _ in range(50):
        r.perturb_command(0.2, 0.1, rng)
    check("episode parameters are held fixed across steps", r.params == p1)

    seen = set()
    for _ in range(30):
        r.new_episode(rng)
        seen.add(round(r.params["linear_scale"], 6))
    check("parameters are resampled between episodes", len(seen) > 25,
          f"{len(seen)} distinct scales in 30 episodes")

    # -- ranges are respected ----------------------------------------------------------
    lo, hi = r.cfg["command"]["linear_scale"]
    scales = []
    for _ in range(2000):
        r.new_episode(rng)
        scales.append(r.params["linear_scale"])
    check("linear_scale stays inside its configured range",
          lo <= min(scales) and max(scales) <= hi,
          f"[{min(scales):.3f}, {max(scales):.3f}] vs configured [{lo}, {hi}]")

    # -- latency: the env's CommandDelay, with the randomiser's offset ------------------
    dly = CommandDelay(2)
    outs = [dly.push(1.0, 0.0)[0] for _ in range(4)]
    check("CommandDelay(2) delays a command by exactly two steps",
          outs[0] == 0.0 and outs[1] == 0.0 and outs[2] == 1.0,
          f"outputs {[round(o, 2) for o in outs]}")
    dly.push(9.9, 9.9); dly.reset(2)
    check("CommandDelay.reset clears commands left over from the last episode",
          all(abs(x) < 5.0 for x, _ in dly._q) and len(dly._q) == 2)
    check("CommandDelay(0) is the identity", CommandDelay(0).push(0.3, -0.2) == (0.3, -0.2))
    lo_l, hi_l = r.cfg["command"]["latency_steps"]
    offs = set()
    for _ in range(300):
        r.new_episode(rng); offs.add(r.params["latency_steps"])
    check("latency offset covers its configured range, ends included",
          min(offs) == lo_l and max(offs) == hi_l, f"{sorted(offs)} vs [{lo_l}, {hi_l}]")
    r.params = _identity_params()
    r.params["latency_steps"] = 3
    check("perturb_command no longer delays by itself (the env does)",
          r.perturb_command(1.0, 0.0, rng)[0] == 1.0)

    # -- solid no-return patches persist and are inf -----------------------------------
    rb = Randomiser(); rb.new_episode(rng)
    rb.params.update(depth_noise_std=0.0, dropout_fraction=0.0)
    rb.cfg = dict(rb.cfg); rb.cfg["observation"] = dict(rb.cfg["observation"])
    rb.cfg["observation"]["dropout_blob"] = {"start_probability": 1.0, "continue_probability": 1.0,
                                             "width_fraction": [0.25, 0.25], "height_fraction": [1.0, 1.0]}
    fr = [rb.perturb_depth(np.full((48, 64), 2.0, dtype=np.float32), rng) for _ in range(5)]
    cols = [np.isinf(f).all(axis=0).sum() for f in fr]
    check("a dropout blob is a solid full-height inf band of the configured width, every frame",
          all(c == 16 for c in cols), f"inf columns per frame {cols}")
    rb.new_episode(rng)
    check("new_episode clears a blob from the last episode", rb._blob is None)

    # -- depth frame rate: stale frames below the control rate --------------------------
    lo_r, hi_r = r.cfg["observation"]["depth_rate_hz"]
    rates = []
    for _ in range(2000):
        r.new_episode(rng); rates.append(r.params["depth_rate_hz"])
    rates = np.array(rates)
    check("depth_rate_hz stays inside its configured range",
          lo_r <= rates.min() and rates.max() <= hi_r,
          f"[{rates.min():.2f}, {rates.max():.2f}] vs [{lo_r}, {hi_r}]")
    below = float(np.mean(rates < 5.0))
    check("log-uniform rate: about half the episodes are below the 5 Hz control rate",
          0.4 < below < 0.65, f"{below:.0%}")

    def fresh_fraction(rate, steps=2000, dt=0.2):
        rf = Randomiser(); rf.new_episode(rng); rf.params["depth_rate_hz"] = rate
        return float(np.mean([rf.depth_frame_due(k * dt, rng) for k in range(steps)]))

    f25 = fresh_fraction(2.5)
    check("at 2.5 Hz under a 5 Hz loop, about half the steps get a new frame",
          0.45 < f25 < 0.55, f"{f25:.0%} fresh")
    check("at or above the control rate every step gets a new frame",
          fresh_fraction(5.0) > 0.99 and fresh_fraction(15.0) == 1.0)
    rd = Randomiser(); rd.new_episode(rng); rd.params["depth_rate_hz"] = 1.0
    rd.depth_frame_due(0.0, rng)
    rd.new_episode(rng)
    check("new_episode resets the frame clock, so an episode starts on a fresh frame",
          rd.depth_frame_due(100.0, rng))
    check("disabled randomisation always delivers fresh frames",
          all(Randomiser(enabled=False).depth_frame_due(k * 0.2, rng) for k in range(20)))

    # -- slip drops commands entirely ---------------------------------------------------
    r.params = _identity_params()
    r.params["slip_probability"] = 1.0
    v, w = r.perturb_command(0.3, 0.4, rng)
    check("slip_probability=1 drops the command", v == 0.0 and w == 0.0)

    # -- depth dropout produces inf, the same value a real no-return produces ----------
    r.params = _identity_params()
    r.params["dropout_fraction"] = 0.5
    d = r.perturb_depth(np.full((20, 20), 2.0, dtype=np.float32), rng)
    frac = float(np.mean(~np.isfinite(d)))
    check("dropout marks pixels as inf, matching _to_metres' no-return",
          0.35 < frac < 0.65, f"{frac:.0%} of pixels dropped, expected ~50%")

    # -- disabled means genuinely disabled ----------------------------------------------
    off = Randomiser(enabled=False)
    off.new_episode(rng)
    check("enabled=False is exactly the identity",
          off.perturb_command(0.25, -0.4, rng) == (0.25, -0.4))
    check("fingerprint records that randomisation was off",
          off.fingerprint().endswith("+disabled"), off.fingerprint())

    # -- the world-reset half -----------------------------------------------------------
    b = sample_body_params(r.cfg, rng)
    check("body params cover mass, mass distribution and friction",
          set(b) == {"mass_scale", "com_offset_m", "mu", "mu2"},
          json.dumps(b, default=float))
    check("mu and mu2 are the same sample (isotropic floor friction)",
          b["mu"] == b["mu2"])

    print()
    if fails:
        print(f"{len(fails)} FAILED: {fails}")
        return 1
    print("all randomisation checks passed")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if not a.selftest:
        ap.error("nothing to do; pass --selftest (this module is a library otherwise)")
    sys.exit(_selftest())
