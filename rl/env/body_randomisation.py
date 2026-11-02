#!/usr/bin/env python3
"""The world-reset half of dynamics randomisation: floor friction, via respawn.

    python3 -m rl.env.body_randomisation --selftest --world room_photogrammetry

Why this exists, and what it does not cover
--------------------------------------------
`rl/env/randomisation.py` documents the split explicitly: command-level effects (velocity
scale, latency, slip, observation noise) are resampled every episode in that module, while
"mass, mass distribution and friction are resampled at WORLD RESET, by regenerating the
robot SDF and respawning" -- a promise `sample_body_params()` makes but that nothing in
this codebase carried out until this module.

It is carried out here only for **friction**, not mass or mass distribution, and that is a
real limitation stated plainly rather than quietly implied by a function that "looks done".
Mutating the floor's collision friction is low-risk: the room model is respawned, not the
robot, so a mistake here cannot put Pepper's URDF into a state nothing has verified. Mass
and centre-of-mass live inside Pepper's own model.sdf (in the vendored, third-party
`pepper_ign_moveit2` fork), and changing them correctly needs per-link mass and inertia
tensor edits verified against how DART actually integrates them -- a distinct piece of work,
not a natural extension of this one. `sample_body_params()` still returns `mass_scale` and
`com_offset_m`; they are logged in every run's metadata (so it is visible, not hidden, that
they were sampled and not applied) and are the next thing to implement, not something this
module pretends to have done.

Friction is also the single item Peng et al. single out with a number: removing friction
randomisation alone drops their real-world success from 0.89 to 0.48. Of everything in
`randomisation.yaml`, it is the entry with the most direct evidence behind it, which is
part of why it is the one implemented first.

Mechanism
---------
Fortress has no runtime service to mutate an existing entity's collision surface
properties, so changing the floor's mu means respawning the room model from a modified
copy of its own model.sdf (see `IgnitionControl.respawn`). Both branches' room templates
emit an identical, uniform `<friction><ode><mu>..</mu><mu2>..</mu2></ode></friction>` block
per collision surface -- `tools/gen_world.py`'s `box()` and `tools/mesh_import.py`'s
`write_sdf()` both do -- so one regex substitution, applied to every occurrence, works
against either branch's model unmodified. The modified copy is written to a scratch
location, and its mesh `<uri>` references are rewritten absolute first: the checked-in
models use directory-relative URIs (working because the model.sdf and its meshes/ sit
together), and a copy anywhere else would otherwise silently fail to resolve them.
"""

import argparse
import math
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "ros2_ws" / "src" / "pepper_room_description"
CACHE_DIR = REPO / "rl" / "cache" / "body_randomisation"

# Matches tools/gen_world.py's BOX template and tools/mesh_import.py's MODEL_SDF exactly:
# a bare numeric friction coefficient with no attributes on either element. If either
# template ever grows an attribute or a differently-named friction model, this needs
# updating alongside it -- which is why both are named here rather than paraphrased.
_MU_RE = re.compile(r"<mu>[^<]*</mu>")
_MU2_RE = re.compile(r"<mu2>[^<]*</mu2>")
_URI_RE = re.compile(r"<uri>([^<]+)</uri>")


def die(msg):
    print("ERROR: " + msg, file=sys.stderr)
    sys.exit(1)


def model_sdf_path(model_name):
    p = PKG / "models" / model_name / "model.sdf"
    if not p.is_file():
        die(f"no model.sdf for {model_name!r} at {p}. Run tools/gen_world.py first.")
    return p


def randomised_room_sdf(model_name, mu, mu2, out_dir=CACHE_DIR):
    """Write a friction-modified, mesh-URI-absolutised copy of a room's model.sdf.

    Returns the path to the copy. Deterministic in its output name (`_mu<value>` in the
    filename) purely for debuggability when reading back a run's logs -- it is a scratch
    artefact, regenerated every call, and is not meant to be reused across calls.
    """
    src = model_sdf_path(model_name)
    text = src.read_text(encoding="utf-8")

    n_mu = len(_MU_RE.findall(text))
    n_mu2 = len(_MU2_RE.findall(text))
    if n_mu == 0:
        die(f"{src} has no <mu> element -- either the template changed or this was "
            f"called on something other than a room model")
    if n_mu != n_mu2:
        die(f"{src} has {n_mu} <mu> but {n_mu2} <mu2> elements; the friction blocks are "
            f"not paired the way gen_world.py's box() and mesh_import.py's write_sdf() "
            f"both emit them. Refusing to guess which value belongs to which surface.")

    text = _MU_RE.sub(f"<mu>{mu}</mu>", text)
    text = _MU2_RE.sub(f"<mu2>{mu2}</mu2>", text)

    # Absolutise every mesh URI relative to the ORIGINAL model's directory, before the
    # copy moves anywhere else. A bare `meshes/visual/room.obj` resolves against whatever
    # directory holds the SDF being parsed -- correct for the checked-in model, wrong for
    # a copy sitting in rl/cache/.
    def absolutise(m):
        uri = m.group(1)
        if uri.startswith(("model://", "http://", "https://", "/")):
            return m.group(0)  # already absolute or a resource-path URI; leave it
        return f"<uri>{(src.parent / uri).resolve()}</uri>"

    text = _URI_RE.sub(absolutise, text)

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{model_name}_mu{mu:.4f}.sdf"
    out.write_text(text, encoding="utf-8")
    return out


def apply_friction(ign, model_name, mu, mu2, pose=(0.0, 0.0, 0.0, 0.0)):
    """Respawn `model_name` with its floor friction set to (mu, mu2).

    `ign` is an `IgnitionControl` for the running world; `pose` is where the room model
    sits, which is always the world origin for both branches (tools/gen_world.py includes
    each room at `<pose>0 0 0 0 0 0</pose>` -- the room's own geometry carries whatever
    offset `mesh_import.py --origin-mode` chose, not the include).
    """
    sdf = randomised_room_sdf(model_name, mu, mu2)
    x, y, z, yaw = pose
    # replace_model, not respawn: remove-then-create leaves a moment with no floor and
    # Pepper falls out of the world. See IgnitionControl.replace_model.
    return ign.replace_model(model_name, str(sdf), x=x, y=y, z=z, yaw=yaw)


# --------------------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------------------

def selftest(world, verbose):
    """Prove a friction respawn actually lands: the room model comes back with the new
    mu written into every one of its collision surfaces, and Pepper (spawned separately)
    is undisturbed by it -- respawning the ROOM must not touch the robot entity at all.

    Run against an already-running world:

        export LIBGL_ALWAYS_SOFTWARE=1
        ros2 launch pepper_robot gz.launch.py world:=<path> headless:=true \\
             enable_moveit:=false enable_rviz:=false
    """
    from rl.env.ign_control import IgnitionControl

    ign = IgnitionControl(world, verbose=verbose)
    failures = []

    def check(label, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}{'  ' + detail if detail else ''}")
        if not ok:
            failures.append(label)

    robot_before = ign.pose_of("pepper_robot")
    z_before = ign.height_of("pepper_robot")
    check("robot is present before the respawn", robot_before is not None)

    src = model_sdf_path(world)
    n_before = len(_MU_RE.findall(src.read_text(encoding="utf-8")))
    check("the checked-in model has at least one friction surface", n_before > 0,
          f"{n_before} <mu> elements")

    new_mu = 0.42
    try:
        apply_friction(ign, world, new_mu, new_mu)
    except Exception as e:
        check(f"respawn with mu={new_mu}", False, f"{type(e).__name__}: {e}")
        return 1
    check(f"respawn with mu={new_mu} completed", True)

    time.sleep(2.0)  # let the new model finish loading before reading anything back
    room_after = ign.pose_of(ign.live_name(world))
    check("the room model is present again after respawn", room_after is not None,
          "-- if this fails, /create silently rejected the scratch SDF")

    if robot_before is not None:
        robot_after = ign.pose_of("pepper_robot")
        check("the robot survives a ROOM respawn undisturbed",
              robot_after is not None
              and math.hypot(robot_after[0] - robot_before[0],
                             robot_after[1] - robot_before[1]) < 0.05,
              f"before {robot_before}, after {robot_after}")
        z_after = ign.height_of("pepper_robot")
        check("the robot did not fall during the ROOM respawn (height within 5 cm)",
              z_before is not None and z_after is not None and abs(z_after - z_before) < 0.05,
              f"z before {z_before}, after {z_after}")

    generated = CACHE_DIR / f"{world}_mu{new_mu:.4f}.sdf"
    check("the scratch SDF was written where randomised_room_sdf says it would be",
          generated.is_file())
    if generated.is_file():
        gen_text = generated.read_text(encoding="utf-8")
        mus = set(re.findall(r"<mu>([^<]*)</mu>", gen_text))
        check(f"every <mu> in the scratch SDF reads back as {new_mu}",
              mus == {str(new_mu)}, f"found {mus}")
        check("mesh URIs in the scratch SDF are absolute",
              all(Path(u).is_absolute() for u in _URI_RE.findall(gen_text)
                  if not u.startswith(("model://", "http://", "https://"))))

    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED: {failures}")
        return 1
    print("all checks passed -- friction respawn is usable")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--world", default="room_photogrammetry")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    if not args.selftest:
        ap.error("nothing to do; pass --selftest (this module is a library otherwise)")
    sys.exit(selftest(args.world, args.verbose))


if __name__ == "__main__":
    main()
