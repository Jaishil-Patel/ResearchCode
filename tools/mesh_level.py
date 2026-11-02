#!/usr/bin/env python3
"""Level an imported photogrammetric room: rotate it so its floor plane is horizontal at z = 0.

    python3 tools/mesh_level.py --room final_scan            # report only
    python3 tools/mesh_level.py --room final_scan --apply    # rotate the model in place

Why this exists. mesh_import.py trusts the capture's up axis ("ARKit/glTF exports are
gravity-aligned"). final_scan's is not quite: its floor plane, fitted on clear floor away
from walls and furniture, slopes 1.35 deg (1 Oct 2026). The real floor is flat, and on a
slope Pepper's frictionless support wheels let gravity pull her downhill. It is not a pure
rigid tilt -- the ceiling slopes the opposite way and both long walls lean outward, the
usual photogrammetric "doming" warp -- so levelling removes the floor's MEAN slope only, not
its local bumps. It is a rigid rotation, so every distance in the mesh is unchanged and the
accuracy gate (mesh_validate.py, Eq. 4.6) measures exactly what it measured before.

What --apply changes, all with `.prelevel` backups beside the originals:
  * meshes/visual/*.obj   -- `v` and `vn` lines only; faces, UVs and materials untouched;
  * meshes/collision/*.stl;
  * ground_truth/<room>.csv -- point_a / point_b, rotated with the mesh so the gate's
    snapped points still sit on the same surfaces;
  * manifests/<room>.json  -- the rotation is recorded under "levelling".
It refuses to run twice (the manifest records that it ran).
"""
import argparse
import csv
import io
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "ros2_ws" / "src" / "pepper_room_description" / "models"
GT_DIR = REPO / "simulation" / "photogrammetry" / "ground_truth"
MANIFEST_DIR = REPO / "simulation" / "photogrammetry" / "manifests"

# Floor samples: only where nothing but floor can be under the ray -- this far from every
# spec obstacle and wall. Near furniture the scan melts floor into object bases (ramps of up
# to 15 cm on final_scan), which would bias the plane.
CLEAR_M = 0.35
STEP_M = 0.05
TRIM_PASSES, KEEP_PCT = 6, 80


def die(msg):
    print("ERROR: " + msg, file=sys.stderr)
    sys.exit(2)


def floor_samples(room, mesh):
    sys.path.insert(0, str(REPO / "tools"))
    import gen_world
    spec = gen_world.load_spec(GT_DIR / f"{room}_spec.yaml")
    L, W = spec["room"]["length"], spec["room"]["width"]
    xs, ys = np.meshgrid(np.arange(CLEAR_M, L - CLEAR_M, STEP_M), np.arange(CLEAR_M, W - CLEAR_M, STEP_M))
    ok = np.ones(xs.shape, bool)
    for f in spec.get("furniture", []):
        cx, cy = f["pose"][:2]
        hx, hy = f["size"][0] / 2 + CLEAR_M, f["size"][1] / 2 + CLEAR_M
        ok &= ~((abs(xs - cx) < hx) & (abs(ys - cy) < hy))
    xy = np.c_[xs[ok], ys[ok]]
    origins = np.c_[xy, np.full(len(xy), 0.6)]
    locs, idx, _ = mesh.ray.intersects_location(origins, np.tile([0.0, 0.0, -1.0], (len(xy), 1)),
                                                 multiple_hits=False)
    return locs


def fit_plane(P):
    keep = np.ones(len(P), bool)
    for _ in range(TRIM_PASSES):
        A = np.c_[P[keep][:, :2], np.ones(keep.sum())]
        coef, *_ = np.linalg.lstsq(A, P[keep][:, 2], rcond=None)
        r = abs(np.c_[P[:, :2], np.ones(len(P))] @ coef - P[:, 2])
        keep = r <= np.percentile(r[keep], KEEP_PCT)
    resid = (np.c_[P[keep][:, :2], np.ones(keep.sum())] @ coef - P[keep][:, 2])
    return coef, int(keep.sum()), float(resid.std())


def levelling_transform(a, b, c):
    """4x4 that maps the plane z = a x + b y + c onto z = 0, rotating about (0, 0, c)."""
    n = np.array([-a, -b, 1.0]); n /= np.linalg.norm(n)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(n, z); s = np.linalg.norm(v); cth = float(n @ z)
    if s < 1e-12:
        R = np.eye(3)
    else:
        K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + K + K @ K * ((1 - cth) / s ** 2)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ np.array([0.0, 0.0, c])
    return T


def rotate_obj(path, T):
    R, t = T[:3, :3], T[:3, 3]
    out = io.StringIO()
    n_v = n_vn = 0
    with open(path, encoding="utf-8", errors="surrogateescape") as f:
        for line in f:
            if line.startswith("v "):
                p = line.split()
                xyz = R @ np.array([float(p[1]), float(p[2]), float(p[3])]) + t
                rest = (" " + " ".join(p[4:])) if len(p) > 4 else ""      # vertex colours, if any
                out.write(f"v {xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}{rest}\n"); n_v += 1
            elif line.startswith("vn "):
                p = line.split()
                nn = R @ np.array([float(p[1]), float(p[2]), float(p[3])])
                out.write(f"vn {nn[0]:.6f} {nn[1]:.6f} {nn[2]:.6f}\n"); n_vn += 1
            else:
                out.write(line)
    Path(path).write_text(out.getvalue(), encoding="utf-8", errors="surrogateescape")
    return n_v, n_vn


def rotate_csv(path, T):
    text = path.read_text()
    lines = text.splitlines(keepends=True)
    header_i = next(i for i, l in enumerate(lines) if l.strip() and not l.lstrip().startswith("#"))
    body = "".join(lines[header_i:])
    rows = list(csv.DictReader(io.StringIO(body)))
    fields = list(csv.DictReader(io.StringIO(body)).fieldnames)
    n = 0
    for r in rows:
        for key in ("point_a", "point_b"):
            if (r.get(key) or "").strip():
                p = np.array([float(v) for v in r[key].split()] + [1.0])
                q = T @ p
                r[key] = f"{q[0]:.4f} {q[1]:.4f} {q[2]:.4f}"; n += 1
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    w.writeheader(); w.writerows(rows)
    path.write_text("".join(lines[:header_i]) + buf.getvalue())
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--room", required=True)
    ap.add_argument("--model-name", default="room_photogrammetry")
    ap.add_argument("--apply", action="store_true", help="rotate the model in place (default: report only)")
    args = ap.parse_args()

    import trimesh
    model = MODELS / args.model_name
    stl = sorted((model / "meshes" / "collision").glob("*.stl"))
    obj = sorted((model / "meshes" / "visual").glob("*.obj"))
    if not stl or not obj:
        die(f"need a collision .stl and a visual .obj under {model}/meshes")
    manifest_path = MANIFEST_DIR / f"{args.room}.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    if args.apply and "levelling" in manifest:
        die(f"{manifest_path.name} says this model was already levelled "
            f"({manifest['levelling'].get('date')}); restore the .prelevel files to redo it")

    coll = trimesh.load(str(stl[0]), force="mesh")
    P = floor_samples(args.room, coll)
    (a, b, c), n, sd = fit_plane(P)
    tilt = float(np.degrees(np.arctan(np.hypot(a, b))))
    print(f"floor plane from {n}/{len(P)} clear-floor samples: dz/dx {a*1000:+.1f} mm/m, dz/dy {b*1000:+.1f} mm/m, "
          f"z(0,0) {c*1000:+.1f} mm  -> tilt {tilt:.2f} deg  (residual {sd*1000:.1f} mm)")
    T = levelling_transform(a, b, c)
    if not args.apply:
        print("report only; pass --apply to level the model")
        return

    for p in (stl[0], obj[0], GT_DIR / f"{args.room}.csv"):
        shutil.copy2(p, str(p) + ".prelevel")
    coll.apply_transform(T)
    coll.export(str(stl[0]))
    n_v, n_vn = rotate_obj(obj[0], T)
    n_pts = rotate_csv(GT_DIR / f"{args.room}.csv", T)
    (a2, b2, c2), _, sd2 = fit_plane(floor_samples(args.room, trimesh.load(str(stl[0]), force="mesh")))
    after = float(np.degrees(np.arctan(np.hypot(a2, b2))))
    manifest["levelling"] = {
        "tool": "tools/mesh_level.py", "date": __import__("datetime").date.today().isoformat(),
        "plane_before": {"dz_dx": a, "dz_dy": b, "z0": c, "tilt_deg": tilt, "samples": n},
        "tilt_after_deg": after, "transform_row_major": T.tolist(),
        "applied_to": [str(stl[0].relative_to(REPO)), str(obj[0].relative_to(REPO)),
                       f"simulation/photogrammetry/ground_truth/{args.room}.csv"],
    }
    if manifest_path.is_file():
        shutil.copy2(manifest_path, str(manifest_path) + ".prelevel")
        manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"levelled: {len(coll.vertices)} collision vertices, {n_v} visual vertices + {n_vn} normals, "
          f"{n_pts} ground-truth points. Floor tilt {tilt:.2f} -> {after:.2f} deg, z(0,0) now {c2*1000:+.1f} mm.")
    print("Next: python3 tools/mesh_validate.py --room", args.room, " (e-bar must be unchanged)")


if __name__ == "__main__":
    main()
