#!/usr/bin/env python3
"""Choose lambda, mu and N by measurement, and leave the evidence behind.

    python3 tools/mesh_smooth_sweep.py --room lab_a --fast
    python3 tools/mesh_smooth_sweep.py --room lab_a --iterations 8 --lambdas 0.5 \
        --mus -0.53 --confirm

Writes simulation/photogrammetry/reports/<room>_smoothing_sweep.md.

The smoothing stage has three free parameters, and a thesis that picks them because the
mesh "looked better" has not established anything. This runs the real pipeline across a
grid of them, scores every result with the real accuracy gate, and prints a table. The
adopted values are whatever that table says, and the table ships with the write-up.

There is deliberately no measurement code in this file. Eq. 4.6, the vertex snapping and
the 4.97% threshold all live in tools/mesh_validate.py, which is imported and called.
Two implementations of the gate would eventually disagree, and the one that disagreed
quietly would be this one.


What the table cannot tell you
------------------------------
Row 0 is the unsmoothed control, and it is the row that matters most: if smoothing does
not improve e-bar it is costing accuracy rather than buying it, and the honest conclusion
is to leave it off for this room.

e-bar also cannot see noise. It is computed from a handful of tape-measured distances, and
a wall can be metrically perfect and still be covered in 5 mm of photogrammetry hash. So
the table carries the displacement and bounding-box drift columns alongside: e-bar says
what smoothing *cost*, and those say what it *did*. Choosing the row with the lowest e-bar
and no displacement would be choosing to do nothing.


Why --fast is not the answer you quote
--------------------------------------
--fast passes --no-blender --allow-uv-loss, which swaps Blender's UV-preserving decimator
for trimesh's. That changes the geometry slightly at the decimation step, so a --fast row
is a ranking, not a measurement. Every number that goes in the write-up comes from a
--confirm run, and the report labels the two kinds of row differently.
"""

import argparse
import itertools
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mesh_validate  # noqa: E402 -- needs the sys.path line above

REPO = Path(__file__).resolve().parent.parent
REPORT_DIR = REPO / "simulation" / "photogrammetry" / "reports"
IMPORTER = Path(__file__).resolve().parent / "mesh_import.py"


def die(msg):
    print("ERROR: " + msg, file=sys.stderr)
    sys.exit(2)


def note(msg):
    print("  " + msg)


def stage(msg):
    print("\n== " + msg)


def glue_negative_lists(argv):
    """Let --mus -0.34,-0.45 work as well as --mus=-0.34,-0.45.

    argparse has a special case for negative numbers, but it only fires on a bare number:
    '-0.45' is a value, '-0.34,-0.45' is not, so a comma-separated list of negative mus
    gets read as an option name and the parser complains that --mus expects an argument.
    Every mu is negative by definition, so that is the normal way to call this and it
    should not require remembering to write '='.
    """
    out = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in ("--mus", "--lambdas", "--iterations") and i + 1 < len(argv):
            nxt = argv[i + 1]
            try:
                [float(v) for v in nxt.split(",") if v.strip()]
            except ValueError:
                out.append(tok)
            else:
                out.append(f"{tok}={nxt}")
                i += 2
                continue
        out.append(tok)
        i += 1
    return out


def floats(text):
    return [float(v) for v in text.split(",") if v.strip()]


def ints(text):
    return [int(v) for v in text.split(",") if v.strip()]


def build_grid(iterations, lambdas, mus):
    """The parameter grid, with the unsmoothed control first and invalid pairs dropped."""
    grid = [{"iterations": 0, "lam": 0.0, "mu": 0.0}]
    for n, lam, mu in itertools.product(iterations, lambdas, mus):
        if n <= 0:
            continue
        # The filter is only a filter when |mu| > lambda; anything else is a Laplacian
        # wearing a hat, and mesh_taubin refuses it anyway.
        if not (lam > 0.0 and mu < 0.0 and abs(mu) > lam):
            continue
        grid.append({"iterations": n, "lam": lam, "mu": mu})
    return grid


def run_import(room, params, out_root, extra_args, fast):
    """One pipeline run into a scratch directory. Returns (visual_mesh, manifest, secs)."""
    tag = f"n{params['iterations']}_l{params['lam']:.2f}_m{params['mu']:.2f}"
    out_dir = out_root / tag
    manifest = out_root / f"{tag}.json"
    cmd = [
        sys.executable,
        str(IMPORTER),
        "--room", room,
        "--output", str(out_dir),
        "--manifest", str(manifest),
        "--work", str(out_root / "work"),
        "--smooth-iterations", str(params["iterations"]),
    ]
    if params["iterations"] > 0:
        cmd += ["--smooth-lambda", repr(params["lam"]),
                "--smooth-mu", repr(params["mu"])]
    if fast:
        cmd += ["--no-blender", "--allow-uv-loss"]
    cmd += extra_args

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    secs = time.time() - t0
    if proc.returncode != 0:
        note(f"{tag}: import failed")
        print(proc.stdout[-2000:], file=sys.stderr)
        print(proc.stderr[-2000:], file=sys.stderr)
        return None, None, secs
    return out_dir / "meshes" / "visual" / "room.obj", manifest, secs


def score(mesh_path, entries, snap_warn):
    """Eq. 4.6 via the gate itself, never a second implementation of it."""
    mesh = mesh_validate.load_mesh(Path(mesh_path))
    return mesh_validate.evaluate(mesh, entries, snap_warn)


def smoothing_figures(manifest_path):
    """Displacement and bounding-box drift, read back out of the manifest."""
    try:
        s = json.loads(Path(manifest_path).read_text())["smoothing"]
    except Exception:  # noqa: BLE001
        return None
    if not s.get("applied"):
        return None
    return s.get("collision_leg") or None


def write_report(path, room, rows, entries, fast, threshold, gt_path):
    best = min((r for r in rows if r["ebar"] is not None), key=lambda r: r["ebar"], default=None)
    lines = [
        f"# Taubin smoothing sweep -- {room}",
        "",
        f"Generated by `tools/mesh_smooth_sweep.py` on {time.strftime('%Y-%m-%d %H:%M:%S')}.",
        "",
        f"Scored with `tools/mesh_validate.py` (Eq. 4.6) against "
        f"`{gt_path}`, {len(entries)} measurements, threshold {threshold:.2f}%.",
        "",
    ]
    if fast:
        lines += [
            "> **Ranking only.** These rows come from `--fast`, which substitutes "
            "trimesh's decimator for Blender's and loses UVs. Re-run the chosen row with "
            "`--confirm` before quoting any number from it.",
            "",
        ]
    lines += [
        "## Table A -- parameter sweep",
        "",
        "| N | lambda | mu | e-bar | vs control | mean disp | max disp | bbox drift x,y,z | secs |",
        "|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    control = rows[0]["ebar"] if rows and rows[0]["ebar"] is not None else None
    for r in rows:
        if r["ebar"] is None:
            lines.append(
                f"| {r['iterations']} | {r['lam']:.2f} | {r['mu']:.2f} | "
                f"FAILED | | | | | {r['secs']:.0f} |"
            )
            continue
        f = r["figures"]
        delta = "--" if control is None or r["iterations"] == 0 else f"{r['ebar'] - control:+.2f}"
        disp_mean = f"{f['mean_displacement_m'] * 1000:.1f} mm" if f else "--"
        disp_max = f"{f['max_displacement_m'] * 1000:.1f} mm" if f else "--"
        drift = (
            ", ".join(f"{d:+.2f}%" for d in f["aabb_drift_pct"]) if f else "--"
        )
        label = "control" if r["iterations"] == 0 else ""
        lines.append(
            f"| {r['iterations']}{(' *' + label + '*') if label else ''} | "
            f"{r['lam']:.2f} | {r['mu']:.2f} | {r['ebar']:.2f}% | {delta} | "
            f"{disp_mean} | {disp_max} | {drift} | {r['secs']:.0f} |"
        )

    lines += ["", "## Table B -- per-dimension, control vs best", ""]
    if best is None or rows[0]["results"] is None:
        lines.append("_No scored rows._")
    else:
        lines += [
            "| id | description | tape (m) | control (m) | best (m) | err control | err best |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for c, b in zip(rows[0]["results"], best["results"]):
            lines.append(
                f"| `{c['id']}` | {c['description'] or c['kind']} | {c['measured']:.3f} | "
                f"{c['model']:.3f} | {b['model']:.3f} | {c['error_pct']:.2f}% | "
                f"{b['error_pct']:.2f}% |"
            )
        lines += [
            "",
            f"Best row: **N={best['iterations']}, lambda={best['lam']:.2f}, "
            f"mu={best['mu']:.2f}**, e-bar {best['ebar']:.2f}%.",
            "",
            "Adopt it by making those the defaults in `tools/mesh_import.py` "
            "(`DEFAULT_SMOOTH_*`) and re-running the self-test.",
        ]

    # Vertex snapping is how kind=points rows are measured, and smoothing moves vertices.
    # If a pick has drifted, the gate is quietly measuring the wrong feature.
    stale = [r for r in rows if r["stale"]]
    lines += ["", "## Table C -- stale point picks", ""]
    lines.append(
        "None; every `kind=points` pick stayed within the snap tolerance at every "
        "parameter setting."
        if not stale
        else "**Point picks drifted past the snap tolerance.** Re-pick them against the "
        "adopted mesh and re-run:\n\n"
        + "\n".join(
            f"- N={r['iterations']}, lambda={r['lam']:.2f}, mu={r['mu']:.2f}: "
            + ", ".join(f"`{s['id']}`" for s in r["stale"])
            for r in stale
        )
    )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return path


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Sweep Taubin parameters and score each with the Eq. 4.6 gate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--room", required=True)
    p.add_argument("--iterations", default="3,5,8,10", help="comma-separated")
    p.add_argument("--lambdas", default="0.30,0.40,0.50", help="comma-separated")
    p.add_argument("--mus", default="-0.34,-0.45,-0.53", help="comma-separated")
    p.add_argument(
        "--fast",
        action="store_true",
        help="skip Blender (--no-blender --allow-uv-loss). Ranking only -- the "
        "decimator differs, so these numbers do not go in the write-up.",
    )
    p.add_argument(
        "--confirm",
        action="store_true",
        help="the opposite of --fast, stated explicitly: run through Blender and mark "
        "the rows as quotable.",
    )
    p.add_argument("--ground-truth", help=f"default: {mesh_validate.GT_DIR}/<room>.csv")
    p.add_argument("--snap-warn", type=float, default=0.05)
    p.add_argument("--report", help=f"default: {REPORT_DIR}/<room>_smoothing_sweep.md")
    p.add_argument(
        "--keep",
        help="directory to keep the intermediate models in (default: a temp dir that "
        "is deleted)",
    )
    p.add_argument(
        "--import-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="extra argument passed through to mesh_import.py; repeatable. Use for "
        "--up, --scale, --origin-mode and anything else the room needs.",
    )
    args = p.parse_args(glue_negative_lists(list(sys.argv[1:] if argv is None else argv)))

    if args.fast and args.confirm:
        die("--fast and --confirm are opposites; pass one")

    gt_path = Path(args.ground_truth) if args.ground_truth else (
        mesh_validate.GT_DIR / f"{args.room}.csv"
    )
    if not gt_path.is_file():
        die(
            f"no ground truth at {gt_path}\n"
            f"The sweep scores every candidate with the accuracy gate, so it needs the "
            f"tape measurements first.\n"
            f"Copy {mesh_validate.GT_DIR / 'TEMPLATE.csv'} to {gt_path.name} and fill it "
            f"in -- tools/room_plotter.html will generate a starter from the numbers on "
            f"your measurement sheet."
        )
    entries = mesh_validate.read_ground_truth(gt_path)

    grid = build_grid(ints(args.iterations), floats(args.lambdas), floats(args.mus))
    stage(f"{len(grid)} settings (including the unsmoothed control)")
    if args.fast:
        note("--fast: ranking run, Blender skipped, UVs discarded")

    tmp = None
    if args.keep:
        out_root = Path(args.keep)
        out_root.mkdir(parents=True, exist_ok=True)
    else:
        tmp = tempfile.TemporaryDirectory(prefix="smooth_sweep_")
        out_root = Path(tmp.name)

    rows = []
    try:
        for i, params in enumerate(grid, start=1):
            label = (
                "control (no smoothing)"
                if params["iterations"] == 0
                else f"N={params['iterations']} lambda={params['lam']:.2f} "
                f"mu={params['mu']:.2f}"
            )
            stage(f"[{i}/{len(grid)}] {label}")
            mesh_path, manifest, secs = run_import(
                args.room, params, out_root, args.import_arg, args.fast
            )
            row = {**params, "secs": secs, "ebar": None, "results": None,
                   "stale": [], "figures": None}
            if mesh_path is not None and mesh_path.is_file():
                results, ebar, stale = score(mesh_path, entries, args.snap_warn)
                row.update(
                    ebar=ebar,
                    results=results,
                    stale=stale,
                    figures=smoothing_figures(manifest),
                )
                note(f"e-bar = {ebar:.2f}%  ({secs:.0f}s)")
                if stale:
                    note(
                        f"WARNING: {len(stale)} point pick(s) snapped further than "
                        f"{args.snap_warn} m -- re-pick them against the adopted mesh"
                    )
            rows.append(row)
    finally:
        if tmp is not None:
            tmp.cleanup()

    report = Path(args.report) if args.report else (
        REPORT_DIR / f"{args.room}_smoothing_sweep.md"
    )
    write_report(
        report, args.room, rows, entries, args.fast, mesh_validate.THRESHOLD_PCT, gt_path
    )

    stage("Done")
    scored = [r for r in rows if r["ebar"] is not None]
    if not scored:
        die("every setting failed to import; nothing to report")
    best = min(scored, key=lambda r: r["ebar"])
    control = rows[0]["ebar"]
    note(f"wrote {report}")
    note(
        f"best: N={best['iterations']} lambda={best['lam']:.2f} mu={best['mu']:.2f} "
        f"-> e-bar {best['ebar']:.2f}%"
    )
    if control is not None and best["iterations"] > 0 and best["ebar"] >= control:
        print(
            "\nNOTE: no smoothed setting beat the unsmoothed control on e-bar. That is a\n"
            "result, not a failure -- but read Table A's displacement column before\n"
            "concluding anything: e-bar measures a handful of distances and cannot see\n"
            "surface noise. Smoothing may still be worth its cost for the depth image.\n"
        )
    return 0 if best["ebar"] <= mesh_validate.THRESHOLD_PCT else 1


if __name__ == "__main__":
    sys.exit(main())
