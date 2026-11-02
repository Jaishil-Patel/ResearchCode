#!/usr/bin/env python3
"""The reconstruction accuracy gate: proposal Eq. 4.6, run before anything trains.

    python3 tools/mesh_validate.py --room lab_a --report

Compares K tape-measured room dimensions against the same dimensions measured on the
processed mesh and computes the mean absolute percentage error

    e-bar = (1/K) * sum_k |m_k - m_hat_k| / m_k * 100%

failing if it exceeds 4.97%, the figure Alhamadah et al. [2024] report for
consumer-grade photogrammetry and the number proposal S4.4.3 adopts as the threshold at
which "the reconstruction is revised before training proceeds".

Exits non-zero on failure, so it can gate a build rather than merely print a complaint.
That matters more than it looks: a mis-scaled room does not announce itself. Training
converges, the policy looks fine in simulation, and the gap only appears on the physical
robot in September, by which point there is no schedule left to re-run anything.

Ground truth comes from `simulation/photogrammetry/ground_truth/<room>.csv` -- see
TEMPLATE.csv in that directory for the schema and for how to pick the point coordinates.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
GT_DIR = REPO / "simulation" / "photogrammetry" / "ground_truth"
REPORT_DIR = REPO / "simulation" / "photogrammetry" / "reports"
MANIFEST_DIR = REPO / "simulation" / "photogrammetry" / "manifests"
MODEL_DIR = (
    REPO / "ros2_ws" / "src" / "pepper_room_description" / "models" / "room_photogrammetry"
)

# Alhamadah et al. [2024], via proposal Eq. 4.6.
THRESHOLD_PCT = 4.97

AXES = {"x": 0, "y": 1, "z": 2}


def die(msg):
    print("ERROR: " + msg, file=sys.stderr)
    sys.exit(2)


def read_ground_truth(path):
    """Parse the measurement CSV, tolerating '#' comments and blank lines."""
    if not path.is_file():
        die(
            f"no ground truth at {path}\n"
            f"Copy {GT_DIR / 'TEMPLATE.csv'} to {path.name} and fill it in from the "
            f"tape measurements."
        )
    lines = [
        ln
        for ln in path.read_text().splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    rows = list(csv.DictReader(lines))
    if not rows:
        die(f"{path} has a header but no measurements")

    out = []
    for i, r in enumerate(rows, start=1):
        rid = (r.get("id") or "").strip()
        kind = (r.get("kind") or "").strip().lower()
        if not rid:
            die(f"{path} row {i}: empty id")
        try:
            measured = float(r["measured_m"])
        except (KeyError, TypeError, ValueError):
            die(f"{path} row '{rid}': measured_m is not a number")
        if measured <= 0:
            # Eq. 4.6 divides by m_k, and an unfilled template row is a 0.000 -- catching
            # it here beats a division warning and a nonsense e-bar.
            die(
                f"{path} row '{rid}': measured_m is {measured}. Every row needs a real "
                f"tape reading; delete rows you did not measure rather than zeroing them."
            )

        entry = {
            "id": rid,
            "description": (r.get("description") or "").strip(),
            "kind": kind,
            "measured": measured,
            "notes": (r.get("notes") or "").strip(),
        }
        if kind == "extent":
            axis = (r.get("axis") or "").strip().lower()
            if axis not in AXES:
                die(f"{path} row '{rid}': kind=extent needs axis x, y or z (got '{axis}')")
            entry["axis"] = axis
        elif kind == "points":
            for key in ("point_a", "point_b"):
                try:
                    entry[key] = np.array(
                        [float(v) for v in (r.get(key) or "").split()], dtype=float
                    )
                except ValueError:
                    die(f"{path} row '{rid}': {key} is not three numbers")
                if entry[key].shape != (3,):
                    die(f"{path} row '{rid}': {key} needs exactly three numbers")
        else:
            die(f"{path} row '{rid}': kind must be 'extent' or 'points', got '{kind}'")
        out.append(entry)
    return out


def load_mesh(path):
    try:
        import trimesh
    except ImportError:
        die("trimesh is not installed:  pip3 install 'trimesh[easy]'")
    if not path.is_file():
        die(
            f"no processed mesh at {path}\n"
            f"Run tools/mesh_import.py --room <room> first."
        )
    mesh = trimesh.load(str(path), force="mesh", process=False)
    if not hasattr(mesh, "vertices") or len(mesh.vertices) == 0:
        die(f"{path} has no geometry")
    return mesh


def snap(mesh, point):
    """Nearest vertex to a picked coordinate, plus how far it had to move.

    The snap distance is reported rather than hidden. A pick taken from an older run of
    mesh_import.py -- different scale, different origin -- will still snap to *some*
    vertex and still produce a number, so a large snap distance is the only signal that
    the measurement is against a point nobody chose.
    """
    d = np.linalg.norm(mesh.vertices - point[None, :], axis=1)
    i = int(np.argmin(d))
    return mesh.vertices[i], float(d[i])


def measure(mesh, entry):
    if entry["kind"] == "extent":
        lo, hi = mesh.bounds
        axis = AXES[entry["axis"]]
        return float(hi[axis] - lo[axis]), {}
    a, da = snap(mesh, entry["point_a"])
    b, db = snap(mesh, entry["point_b"])
    return float(np.linalg.norm(b - a)), {"snap_a": da, "snap_b": db}


def evaluate(mesh, entries, snap_warn):
    results = []
    for e in entries:
        model, extra = measure(mesh, e)
        err_pct = abs(e["measured"] - model) / e["measured"] * 100.0
        results.append({**e, "model": model, "error_pct": err_pct, **extra})
    ebar = float(np.mean([r["error_pct"] for r in results]))
    stale = [
        r
        for r in results
        if max(r.get("snap_a", 0.0), r.get("snap_b", 0.0)) > snap_warn
    ]
    return results, ebar, stale


def format_table(results):
    head = (
        "| id | description | tape (m) | mesh (m) | error |\n"
        "|---|---|---:|---:|---:|\n"
    )
    rows = "".join(
        f"| `{r['id']}` | {r['description'] or r['kind']} | "
        f"{r['measured']:.3f} | {r['model']:.3f} | {r['error_pct']:.2f}% |\n"
        for r in results
    )
    return head + rows


def write_report(path, room, mesh_path, results, ebar, threshold, passed, stale):
    manifest_path = MANIFEST_DIR / f"{room}.json"
    manifest_note = ""
    if manifest_path.is_file():
        try:
            m = json.loads(manifest_path.read_text())
            manifest_note = (
                f"- Scale factor applied at import: `{m['arguments'].get('scale')}`\n"
                f"- Up axis: `{m['arguments'].get('up')}`, "
                f"origin mode: `{m['arguments'].get('origin_mode')}`\n"
                f"- Normals flipped at import: `{m.get('normals_flipped')}` "
                f"({m.get('normals_inward_fraction')} of shell area inward)\n"
                f"- Visual mesh: {m['stages']['visual']['faces']:,} faces via "
                f"`{m['stages']['visual']['backend']}`; collision "
                f"{m['stages']['collision']['faces']:,} faces\n"
            )
        except (KeyError, ValueError):
            manifest_note = f"- Manifest at `{manifest_path}` could not be parsed\n"

    verdict = "**PASS**" if passed else "**FAIL**"
    if not manifest_note:
        manifest_note = (
            "- No manifest found; the mesh may not have come from tools/mesh_import.py.\n"
        )
    body = f"""# Reconstruction accuracy -- {room}

Generated {time.strftime('%Y-%m-%d %H:%M:%S')} by `tools/mesh_validate.py`.

Proposal Eq. 4.6, mean absolute percentage error over K = {len(results)} tape-measured
dimensions:

> e-bar = {ebar:.2f}%  (threshold {threshold:.2f}%) -- {verdict}

{format_table(results)}
## Import parameters

{manifest_note}- Mesh measured: `{mesh_path}`

## Reading this

Eq. 4.6 weights every dimension equally, so a single badly-picked point moves e-bar as
much as a genuinely wrong wall. Before concluding the reconstruction is at fault, look
down the error column: a uniform error across all K suggests a scale factor, which is
fixed by re-running the import with `--scale`. Errors concentrated on one axis suggest
the up-axis or yaw estimate is wrong. Errors on scattered individual rows usually mean
the point picks, not the mesh.
"""
    if stale:
        body += (
            "\n## Suspect point picks\n\n"
            "These rows snapped a long way to reach a vertex, which usually means the "
            "coordinates were read off an earlier import and no longer refer to the "
            "feature they were chosen for:\n\n"
            + "".join(
                f"- `{r['id']}`: snapped {max(r.get('snap_a', 0), r.get('snap_b', 0)):.3f} m\n"
                for r in stale
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return path


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Gate the reconstruction against tape-measure ground truth (Eq. 4.6).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--room", required=True)
    p.add_argument("--ground-truth", help=f"default: {GT_DIR}/<room>.csv")
    p.add_argument(
        "--mesh",
        help=f"default: {MODEL_DIR}/meshes/visual/room.obj",
    )
    p.add_argument("--threshold", type=float, default=THRESHOLD_PCT)
    p.add_argument("--report", action="store_true", help="write the markdown report")
    p.add_argument(
        "--snap-warn",
        type=float,
        default=0.05,
        help="flag point picks that snapped further than this, in metres",
    )
    args = p.parse_args(argv)

    gt_path = Path(args.ground_truth) if args.ground_truth else GT_DIR / f"{args.room}.csv"
    mesh_path = (
        Path(args.mesh) if args.mesh else MODEL_DIR / "meshes" / "visual" / "room.obj"
    )

    entries = read_ground_truth(gt_path)
    mesh = load_mesh(mesh_path)
    results, ebar, stale = evaluate(mesh, entries, args.snap_warn)

    width = max(len(r["id"]) for r in results)
    print(f"\nK = {len(results)} dimensions, measured on {mesh_path.name}\n")
    print(f"  {'id':<{width}}  {'tape':>8}  {'mesh':>8}  {'error':>8}")
    for r in results:
        print(
            f"  {r['id']:<{width}}  {r['measured']:8.3f}  {r['model']:8.3f}  "
            f"{r['error_pct']:7.2f}%"
        )

    if stale:
        print("\n  Suspect point picks (snapped a long way to find a vertex):")
        for r in stale:
            worst = max(r.get("snap_a", 0.0), r.get("snap_b", 0.0))
            print(f"    {r['id']}: {worst:.3f} m")

    passed = ebar <= args.threshold
    print(
        f"\n  e-bar = {ebar:.2f}%   threshold {args.threshold:.2f}%   "
        f"{'PASS' if passed else 'FAIL'}\n"
    )

    if args.report:
        out = write_report(
            REPORT_DIR / f"{args.room}_validation.md",
            args.room,
            mesh_path,
            results,
            ebar,
            args.threshold,
            passed,
            stale,
        )
        print(f"  report: {out.relative_to(REPO)}\n")

    if not passed:
        print(
            "Reconstruction is outside the gate. Proposal S4.4.3 says it is revised "
            "before training proceeds -- do not carry on with this mesh.\n",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
