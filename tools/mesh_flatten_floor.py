#!/usr/bin/env python3
"""Flatten the floor of a photogrammetric room's COLLISION mesh. The visual mesh is untouched.

    python3 tools/mesh_flatten_floor.py --room final_scan            # report only
    python3 tools/mesh_flatten_floor.py --room final_scan --apply    # rewrite the collision STL

Why. The real floor is flat; the reconstructed one is not. After levelling (mesh_level.py)
final_scan's floor still has +-17 mm bumps and up to ~10 cm of "skirt" where the scan melts
the floor into object bases (the goal pole sits on a 5-10 cm cone). The simulated base cannot
cope with that: its rigid two-wheel drive axle sits at the centre of the three support wheels
and hangs in the air over any dip of more than ~2 mm, ~8% of poses on this floor, and most
photogrammetry smoke-test timeouts were Pepper stuck at such a spot (1 Oct 2026). A sprung
axle was tried and does not work in Fortress/DART (see pepper_drive_wheels.xacro).

What. Every collision vertex below FLOOR_Z_M is moved to z = 0, and a flat plane is added
1 mm under the floor across the room's footprint so a hole in the scan is floor, not a void.
FLOOR_Z_M is routes.FLOOR_CLEARANCE_M, the height the route planner already treats as floor,
not a new number: below it Pepper's wheels see floor, above it an obstacle. Geometry above
it -- walls, furniture, both poles -- is exactly as scanned.

What this means for the experiment, to state in the write-up: the photogrammetric branch's
PHYSICS floor is flat, like the standard branch's and like the real room's; everything the
depth camera sees (the visual mesh) is still the unedited reconstruction, and so are all
obstacles above 0.12 m. The accuracy gate measures the visual mesh and is unaffected.

--apply keeps a `.preflatten` backup and records the operation in manifests/<room>.json.
It refuses to run twice.
"""
import argparse
import datetime
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from rl.env.routes import FLOOR_CLEARANCE_M  # noqa: E402

MODELS = REPO / "ros2_ws" / "src" / "pepper_room_description" / "models"
GT_DIR = REPO / "simulation" / "photogrammetry" / "ground_truth"
MANIFEST_DIR = REPO / "simulation" / "photogrammetry" / "manifests"
FLOOR_Z_M = FLOOR_CLEARANCE_M
UNDERLAY_Z_M = -0.001
UNDERLAY_PAD_M = 0.3


def die(msg):
    print("ERROR: " + msg, file=sys.stderr)
    sys.exit(2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--room", required=True)
    ap.add_argument("--model-name", default="room_photogrammetry")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    import trimesh
    stl = sorted((MODELS / args.model_name / "meshes" / "collision").glob("*.stl"))
    if not stl:
        die("no collision STL")
    stl = stl[0]
    manifest_path = MANIFEST_DIR / f"{args.room}.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    if args.apply and "floor_flattening" in manifest:
        die(f"already flattened ({manifest['floor_flattening'].get('date')}); restore {stl.name}.preflatten to redo it")
    if "levelling" not in manifest:
        print("WARNING: the manifest has no levelling record; flatten a LEVELLED mesh (tools/mesh_level.py) "
              "or the snapped floor will sit at the wrong height across the room")

    mesh = trimesh.load(str(stl), force="mesh")
    v = np.asarray(mesh.vertices).copy()
    low = v[:, 2] < FLOOR_Z_M
    moved = np.abs(v[low, 2])
    print(f"{stl.relative_to(REPO)}: {len(v)} vertices, {low.sum()} below {FLOOR_Z_M:.2f} m "
          f"(moved by median {np.median(moved)*1000:.1f} mm, 95th pct {np.percentile(moved, 95)*1000:.1f} mm, "
          f"max {moved.max()*1000:.1f} mm)")
    if not args.apply:
        print("report only; pass --apply to rewrite the collision mesh")
        return

    v[low, 2] = 0.0
    sys.path.insert(0, str(REPO / "tools"))
    import gen_world
    r = gen_world.load_spec(GT_DIR / f"{args.room}_spec.yaml")["room"]
    x0, y0, x1, y1 = -UNDERLAY_PAD_M, -UNDERLAY_PAD_M, r["length"] + UNDERLAY_PAD_M, r["width"] + UNDERLAY_PAD_M
    n = len(v)
    plane_v = np.array([[x0, y0, UNDERLAY_Z_M], [x1, y0, UNDERLAY_Z_M], [x1, y1, UNDERLAY_Z_M], [x0, y1, UNDERLAY_Z_M]])
    plane_f = np.array([[n, n + 1, n + 2], [n, n + 2, n + 3]])          # counter-clockwise from above: normal +z
    out = trimesh.Trimesh(vertices=np.vstack([v, plane_v]), faces=np.vstack([mesh.faces, plane_f]), process=False)
    shutil.copy2(stl, str(stl) + ".preflatten")
    out.export(str(stl))
    manifest["floor_flattening"] = {
        "tool": "tools/mesh_flatten_floor.py", "date": datetime.date.today().isoformat(),
        "applied_to": str(stl.relative_to(REPO)), "floor_z_threshold_m": FLOOR_Z_M,
        "vertices_snapped": int(low.sum()), "vertices_total": int(len(v)),
        "underlay_plane": {"z": UNDERLAY_Z_M, "xy_min": [x0, y0], "xy_max": [x1, y1]},
        "visual_mesh": "unchanged",
    }
    if manifest_path.is_file():
        shutil.copy2(manifest_path, str(manifest_path) + ".preflatten")
        manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {stl.relative_to(REPO)} ({low.sum()} vertices snapped to z = 0, underlay plane added); "
          f"backup {stl.name}.preflatten")


if __name__ == "__main__":
    main()
