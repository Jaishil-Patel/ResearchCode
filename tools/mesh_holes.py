#!/usr/bin/env python3
"""Close the small holes a photogrammetry capture leaves, and account for every one.

    python3 tools/mesh_holes.py --report --input simulation/photogrammetry/work/<room>/room.obj

A phone scan does not produce a closed surface. `final_scan` arrives with 732 open boundary
loops, and the ones small enough to be genuine holes -- a patch of wall the camera never saw
square-on, the shadow behind a chair leg -- read as ragged gaps when the room is rendered.
Closing them is cosmetic, and cosmetic is a real requirement when the same mesh produces the
RGB frames a policy trains on and the figures in the write-up.

Closing them is also *inventing geometry that was never measured*, which is why this file is
as much bookkeeping as triangles. Proposal S4.4.2 makes measured geometry the sole
independent variable of the experiment, and docs/photogrammetry_runbook.md:405-409 says a
partial capture is a re-capture rather than a downstream repair. Filling anyway is a
deliberate override, so every synthetic face is tagged, measured, written to the manifest and
printed by the accuracy gate. The one thing that must never happen is a filled room that
looks measured.


Four holes, not seven hundred
-----------------------------
Count boundary edges on the file as it loads and `final_scan` appears to have 22,113 of
them in 715 loops enclosing 188.9 m2 -- on a mesh whose entire surface is 140.3 m2. That
impossibility is the tell: a hole cannot be bigger than the room. The OBJ carries 55,682
`v` entries for 44,666 distinct positions, because a textured mesh splits a vertex wherever
the atlas is cut, and neither trimesh (`process=False`) nor Blender 4.5's importer welds
them. Every texture seam then reads as a boundary.

Welded at 1 um the same mesh has 1,478 boundary edges in exactly four loops:

    loop   boundary edges   fan area
      0              572      6.16 m2     full-height gap, +x wall
      1              527      4.28 m2     full-height gap, +x wall
      2              196      2.76 m2
      3              183      0.41 m2

    filling all four: 13.61 m2, 9.7% of the surface

The capture missed most of one wall; that is the whole story, and it is a very different
story from seven hundred pinholes. So there is no tiering to do and no small-hole
population to be conservative about -- there are four gaps and the only question is whether
to close them.

`--max-area` survives as a safety valve rather than a strategy. Set below 0.41 m2 it fills
nothing at all.


Why this module has no interesting imports
------------------------------------------
numpy only, like tools/mesh_taubin.py and for the same reason: the collision mesh is built
in-process with trimesh and the visual mesh in a separate Blender process, and the two must
agree about which loops are holes. Sharing the *selection* logic rather than merely the
thresholds is what keeps them from drifting -- the repo records a 45% gate error from the
last time those two legs disagreed about geometry.
"""

import argparse
import sys

import numpy as np

import mesh_taubin

# Positions closer than 1e-DIGITS are the same point. See weld_faces().
WELD_DIGITS = 6

# Loops at or below this fan area are treated as holes. See the module docstring.
# No cap by default. On the mesh this was built for there are four loops and the smallest
# is 0.41 m2, so any cap low enough to feel cautious simply turns the stage off. Filling is
# opt-in at the mesh_import.py level, which is where the caution belongs.
DEFAULT_MAX_AREA = float("inf")


def die(msg):
    print("ERROR: " + msg, file=sys.stderr)
    sys.exit(2)


def weld_faces(V, faces, digits=WELD_DIGITS):
    """Re-index faces so coincident positions share one vertex. Returns (inverse, F).

    Skipping this is the difference between four holes and seven hundred, and it is the
    single most important line in this module.

    A textured OBJ splits a vertex wherever the atlas is cut: `final_scan` carries 55,682
    `v` entries for 44,666 distinct positions. Every one of those 11,016 duplicates makes
    the two triangles either side of the seam look unconnected, so an edge that is in the
    middle of a continuous wall counts as used by exactly one face -- a boundary. Counted
    that way the mesh has 22,113 boundary edges in 715 loops enclosing 188.9 m2, which is
    more than the 140.3 m2 the mesh actually has, and that impossibility is the tell.

    Welded, it has 1,478 boundary edges in 4 loops. Those four are the real holes: two of
    them full-height gaps in the same wall, where the capture missed most of one side of
    the room.

    Filling the unwelded reading does not close holes. It stitches flat sheets along every
    texture seam in the room, which is both invisible in a loop count and ruinous.
    """
    V = np.asarray(V, dtype=np.float64)
    _, inverse = mesh_taubin.weld_index(V, digits=digits)
    F = inverse[np.asarray(faces, dtype=np.int64)]
    # A triangle whose corners weld together has no area and no edges worth counting.
    keep = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 2] != F[:, 0])
    return inverse, F[keep]


def boundary_edges(faces):
    """Edges used by exactly one face, as an (E, 2) sorted array.

    Expects faces already welded by weld_faces(); on raw UV-split indices this counts
    every texture seam as a boundary.
    """
    faces = np.asarray(faces)
    e = np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    uniq, counts = np.unique(e, axis=0, return_counts=True)
    return uniq[counts == 1]


def boundary_loops(faces):
    """Walk the boundary edges into loops. Returns a list of (verts, closed) tuples.

    `closed` matters more than it looks. The walk produces an open path, and a path is not a
    hole -- Blender's holes_fill silently fills nothing when handed one, which reads exactly
    like the fill having run and found no work to do. Cost an afternoon. So the closing edge
    back to the start is checked for explicitly and the loop is marked accordingly.
    """
    bnd = boundary_edges(faces)
    adj = {}
    for a, b in bnd:
        adj.setdefault(int(a), []).append(int(b))
        adj.setdefault(int(b), []).append(int(a))

    seen = set()
    loops = []
    for start in adj:
        if start in seen:
            continue
        verts = [start]
        seen.add(start)
        cur, prev = start, None
        while True:
            nxt = [n for n in adj[cur] if n != prev and n not in seen]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            seen.add(cur)
            verts.append(cur)
        closed = len(verts) > 2 and start in adj[cur]
        loops.append((verts, closed))
    return loops


def loop_area(V, verts):
    """Fan area from the loop's centroid -- the area a fill would add.

    Not the planar area: these loops are rarely planar, and the honest question is how much
    new surface filling creates, which is what the fan measures.
    """
    if len(verts) < 3:
        return 0.0
    P = np.asarray(V, dtype=np.float64)[verts]
    c = P.mean(axis=0)
    a, b = P - c, np.roll(P, -1, axis=0) - c
    return float(0.5 * np.linalg.norm(np.cross(a, b), axis=1).sum())


def classify(V, faces, max_area=DEFAULT_MAX_AREA, digits=WELD_DIGITS):
    """Split the boundary loops into fill / leave / not-a-loop.

    Loop vertex indices come back in WELDED numbering. Callers holding an unwelded mesh
    need the `inverse` from weld_faces() to map them back.
    """
    _, F = weld_faces(V, faces, digits)
    fill, leave, open_paths = [], [], []
    for verts, closed in boundary_loops(F):
        if not closed:
            open_paths.append(verts)
        elif loop_area(V, verts) <= max_area:
            fill.append(verts)
        else:
            leave.append(verts)
    return fill, leave, open_paths


def fill_loops(V, faces, max_area=DEFAULT_MAX_AREA):
    """Triangulate every loop at or under `max_area`. Returns (V2, F2, synthetic, stats).

    `synthetic` is one bool per face of F2, True for faces this created -- the provenance
    record the manifest and the gate report are built from.

    Fan triangulation from a new centre vertex, rather than ear-clipping across the loop. A
    scan hole's rim is rarely planar and often non-convex, and ear-clipping a non-convex
    non-planar polygon produces self-intersecting slivers; a centre fan cannot. The centre
    sits at the rim's centroid, so the patch is flat and slightly recessed, which the
    caller's relax pass then pulls into line with the surrounding curvature.
    """
    V = np.asarray(V, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    fill, leave, open_paths = classify(V, faces, max_area)
    # classify() speaks welded indices; the caller's V does not. Weld the positions too so
    # the two agree, and hand back the welded mesh.
    inverse, faces = weld_faces(V, faces)
    uniq = np.zeros((int(inverse.max()) + 1, 3), dtype=np.float64)
    uniq[inverse] = V
    V = uniq

    new_verts, new_faces = [], []
    added_area = 0.0
    for verts in fill:
        centre = V[verts].mean(axis=0)
        ci = len(V) + len(new_verts)
        new_verts.append(centre)
        for i in range(len(verts)):
            new_faces.append([verts[i], verts[(i + 1) % len(verts)], ci])
        added_area += loop_area(V, verts)

    if new_verts:
        V2 = np.vstack([V, np.array(new_verts)])
        F2 = np.vstack([faces, np.array(new_faces, dtype=np.int64)])
    else:
        V2, F2 = V.copy(), faces.copy()

    synthetic = np.zeros(len(F2), dtype=bool)
    synthetic[len(faces):] = True

    stats = {
        "max_area_m2": float(max_area),
        "loops_total": len(fill) + len(leave) + len(open_paths),
        "loops_filled": len(fill),
        "loops_left_closed": len(leave),
        "loops_open_paths": len(open_paths),
        "vertices_added": len(new_verts),
        "faces_added": len(new_faces),
        "synthetic_area_m2": round(added_area, 6),
    }
    return V2, F2, synthetic, stats


BANDS = [
    (0.0, 0.005, "<50 cm2    pinhole"),
    (0.005, 0.05, "50-500 cm2 small"),
    (0.05, 0.20, "0.05-0.2m2 medium"),
    (0.20, 0.50, "0.2-0.5 m2 large"),
    (0.50, float("inf"), ">0.5 m2    silhouette, not a hole"),
]


def census(V, faces):
    """The table --report prints. Ordering by size is the point, not precision."""
    inverse, F = weld_faces(V, faces)
    uniq = np.zeros((int(inverse.max()) + 1, 3), dtype=np.float64)
    uniq[inverse] = np.asarray(V, dtype=np.float64)
    rows = [(loop_area(uniq, v), len(v), closed) for v, closed in boundary_loops(F)]
    out = []
    for lo, hi, label in BANDS:
        sel = [r for r in rows if lo <= r[0] < hi and r[2]]
        out.append((label, len(sel), sum(r[0] for r in sel)))
    return rows, out


def _selftest():
    fails = []

    def check(label, ok, detail=""):
        print(f"  {'PASS' if ok else 'FAIL'}  {label}{('  -- ' + detail) if detail else ''}")
        if not ok:
            fails.append(label)

    V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    F = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    check("a closed surface has no boundary loops", len(boundary_loops(F)) == 0)

    F1 = F[:3]
    loops = boundary_loops(F1)
    check("removing a face leaves one closed loop",
          len(loops) == 1 and loops[0][1], f"{len(loops)} loop(s)")

    V2, F2, syn, st = fill_loops(V, F1, max_area=10.0)
    check("the hole is filled",
          st["loops_filled"] == 1 and st["faces_added"] == 3,
          f"{st['faces_added']} faces added")
    check("filling closes the boundary", len(boundary_loops(F2)) == 0)
    check("synthetic marks exactly the new faces",
          syn.sum() == 3 and not syn[:len(F1)].any())

    _, _, _, st2 = fill_loops(V, F1, max_area=1e-9)
    check("a hole above the cap is left open",
          st2["loops_filled"] == 0 and st2["loops_left_closed"] == 1)

    # The area the fill adds must be the area the census predicted, or the cap is a lie.
    predicted = loop_area(V, loops[0][0])
    check("reported synthetic area matches the loop area",
          abs(st["synthetic_area_m2"] - predicted) < 1e-5,
          f"{st['synthetic_area_m2']:.6f} vs {predicted:.6f}")

    g = 4
    xs = np.linspace(0, 1, g)
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    Vg = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
    idx = np.arange(g * g).reshape(g, g)
    a, b = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel()
    c, d = idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    Fg = np.vstack([np.column_stack([a, b, c]), np.column_stack([a, c, d])])
    _, _, _, stg = fill_loops(Vg, Fg, max_area=10.0)
    check("a flat patch's rim is one closed loop",
          stg["loops_total"] == 1 and stg["loops_filled"] == 1)

    check("determinism", np.array_equal(fill_loops(V, F1, max_area=10.0)[0],
                                        fill_loops(V, F1, max_area=10.0)[0]))
    print()
    if fails:
        print(f"{len(fails)} check(s) failed")
        return 1
    print("all checks passed")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Boundary-loop census and small-hole filling.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", help="mesh to analyse (OBJ/PLY/STL/GLB)")
    p.add_argument("--report", action="store_true", help="print the loop census")
    p.add_argument("--max-area", type=float, default=DEFAULT_MAX_AREA,
                   help="loops at or under this fan area are treated as holes")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args(argv)

    if args.selftest:
        print("== mesh_holes selftest")
        return _selftest()
    if not args.input:
        p.print_help()
        return 0

    import trimesh

    m = trimesh.load(args.input, force="mesh", process=False)
    m.merge_vertices()
    V, F = np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces)
    rows, bands = census(V, F)
    closed = sum(1 for r in rows if r[2])
    print(f"{args.input}")
    print(f"  {len(F):,} faces, {len(V):,} vertices, surface {m.area:.1f} m2")
    print(f"  {len(rows):,} boundary loops "
          f"({closed:,} closed, {len(rows) - closed:,} open paths)\n")
    print(f"  {'band':<32} {'loops':>6} {'fan area':>11}")
    for label, n, area in bands:
        print(f"  {label:<32} {n:>6} {area:>8.2f} m2")
    _, _, _, st = fill_loops(V, F, args.max_area)
    print(f"\n  at --max-area {args.max_area:g} m2: fill {st['loops_filled']} loops, "
          f"{st['faces_added']:,} faces, {st['synthetic_area_m2']:.2f} m2 synthetic "
          f"({st['synthetic_area_m2'] / m.area * 100:.1f}% of surface)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
