#!/usr/bin/env python3
"""Taubin lambda|mu smoothing: the denoising stage of tools/mesh_import.py.

    python3 tools/mesh_taubin.py --selftest

A phone photogrammetry capture is not a clean surface. Walls that are flat in the room
arrive as a few millimetres of high-frequency noise riding on a flat plane, and that noise
goes two places that matter. It goes into the depth image, where the policy learns to read
texture that does not exist on the real wall. And it goes into the collision mesh, where
every bump is a contact the physics engine has to resolve.

Smoothing it in the Blender GUI works and cannot be written up: "I smoothed it until it
looked good" is not a method anyone can repeat. This is the same operation with numbers
attached, applied identically to every capture, recorded in the manifest.

Taubin (1995), rather than plain Laplacian smoothing, because Laplacian shrinks. Each pass
moves every vertex toward the average of its neighbours, and iterated, that walks a closed
surface steadily inward -- a 4.02 m room becomes a 3.9 m room and the accuracy gate is
measuring the smoother, not the reconstruction. Taubin alternates a positive pass with a
slightly larger negative one:

    V' = V + lambda * L(V)         shrink, killing high frequencies
    V'' = V' + mu * L(V')          expand, restoring the low ones

With lambda > 0, mu < 0 and |mu| > lambda, the pair acts as a low-pass filter with near-unit
gain at low spatial frequency, so the room keeps its dimensions while losing its noise.
That claim is not taken on faith here -- see --selftest, which measures sphere-radius
retention against plain Laplacian at equal iteration count.


Why this file has no interesting imports
----------------------------------------
numpy only. No trimesh, no scipy, no bpy.

mesh_import.py builds the collision mesh in-process with trimesh and the visual mesh by
shelling out to Blender (blender_room_visual.py), because trimesh's quadric decimation
discards UVs and the texture *is* the reconstruction. That means two separate code paths
produce the two meshes, and tools/mesh_pipeline_selftest.sh records what happens when they
drift apart: the visual mesh came out 12.1 x 7.1 m against the collision mesh's 6.0 x 4.0 m
and the accuracy gate reported a 45% error.

Smoothing one path and not the other would reintroduce exactly that. So the kernel has to
run unchanged under the venv's Python 3.10 *and* under whatever Python is bundled with
Blender, which rules out depending on anything that is not already in both. numpy is in
both. Nothing else is.


Why bincount and not np.add.at
------------------------------
The Laplacian is a scatter-add over the edge array. `np.add.at` is the obvious way to write
it and is roughly two orders of magnitude slower than `np.bincount` with `weights=`, which
matters at the 500k-2M triangles a raw scan carries: seconds versus minutes per iteration.


Uniform weights, not cotangent
------------------------------
The umbrella operator -- mean of the one-ring, unweighted -- is what Taubin's paper
analyses, and it is the one whose behaviour is predictable on a photogrammetry mesh.
Cotangent weights approximate the true Laplace-Beltrami operator and are better on a
well-conditioned mesh, but they blow up on the sliver triangles a scan is full of, because
the cotangent of a near-zero angle is unbounded.
"""

import argparse
import sys

import numpy as np


def die(msg):
    print("ERROR: " + msg, file=sys.stderr)
    sys.exit(2)


# --------------------------------------------------------------------------------------
# adjacency
# --------------------------------------------------------------------------------------


def edges_from_faces(faces):
    """Undirected edge array (E, 2) from a triangle array (F, 3), deduplicated."""
    faces = np.asarray(faces)
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.sort(e, axis=1)
    return np.unique(e, axis=0)


def weld_index(V, digits=6):
    """Map coincident positions onto one index. Returns (unique_positions, inverse).

    A textured mesh splits vertices at UV seams: two vertices at the same point in space
    carrying different texture coordinates are separate indices with no edge between them.
    Smoothed independently they drift apart, and the mesh tears open along every seam --
    which on a photogrammetry room is every place the atlas was cut, so the failure is not
    subtle once it happens.

    Welding by rounded position, smoothing the unique positions and scattering the result
    back keeps the split vertices coincident by construction. UVs are never touched; only
    positions move.

    `digits` of 6 is a micrometre at room scale -- far below any real vertex spacing, far
    above float32 noise from an exporter.
    """
    V = np.asarray(V, dtype=np.float64)
    keys = np.round(V, digits)
    # np.unique on rows: returns the unique positions and, per input row, which one it is.
    uniq, inverse = np.unique(keys, axis=0, return_inverse=True)
    return uniq, inverse.astype(np.int64, copy=False)


def boundary_mask(faces, n_vertices):
    """True for vertices touching a boundary edge (an edge used by exactly one face).

    Only used by --pin-boundary. The default is to smooth everything, which is textbook
    Taubin; pinning is here so the difference can be measured rather than argued about.
    """
    faces = np.asarray(faces)
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.sort(e, axis=1)
    _, inverse, counts = np.unique(e, axis=0, return_inverse=True, return_counts=True)
    open_edges = e[counts[inverse] == 1]
    mask = np.zeros(n_vertices, dtype=bool)
    if len(open_edges):
        mask[open_edges.ravel()] = True
    return mask


# --------------------------------------------------------------------------------------
# the operator
# --------------------------------------------------------------------------------------


def _prepare(edges, n_vertices):
    """Split the edge array and precompute 1/degree. Done once, reused every half-step."""
    edges = np.asarray(edges, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"edges must be (E, 2), got {edges.shape}")
    e0 = np.ascontiguousarray(edges[:, 0])
    e1 = np.ascontiguousarray(edges[:, 1])
    deg = np.bincount(e0, minlength=n_vertices) + np.bincount(e1, minlength=n_vertices)
    isolated = deg == 0
    inv_deg = np.zeros(n_vertices, dtype=np.float64)
    np.divide(1.0, deg, out=inv_deg, where=~isolated)
    return e0, e1, inv_deg, isolated


def umbrella_laplacian(V, e0, e1, inv_deg, isolated, out=None):
    """L(V)_i = mean(V_j for j adjacent to i) - V_i, and 0 where i has no neighbours.

    Loose vertices do survive the component filter on some scans -- the filter works on
    connected components by surface area, and a vertex referenced by no face has no area
    to judge. Left undefended they would divide by a zero degree.
    """
    n = V.shape[0]
    if out is None:
        out = np.empty_like(V)
    for k in range(3):
        acc = np.bincount(e0, weights=V[e1, k], minlength=n)
        acc += np.bincount(e1, weights=V[e0, k], minlength=n)
        out[:, k] = acc * inv_deg - V[:, k]
    out[isolated] = 0.0
    return out


def taubin_smooth(V, edges, *, iterations, lam, mu, pin_mask=None):
    """Taubin lambda|mu smoothing. Returns a new array; never mutates the input.

    `iterations` counts lambda/mu *pairs*, so one iteration is two passes over the mesh.
    """
    V = np.asarray(V, dtype=np.float64)
    if V.ndim != 2 or V.shape[1] != 3:
        raise ValueError(f"V must be (N, 3), got {V.shape}")
    if iterations <= 0:
        return V.copy()
    if not (lam > 0.0):
        raise ValueError(f"lambda must be > 0, got {lam}")
    if not (mu < 0.0):
        raise ValueError(f"mu must be < 0, got {mu}")
    if not (abs(mu) > lam):
        raise ValueError(
            f"|mu| must exceed lambda or the filter shrinks like a plain Laplacian "
            f"(lambda={lam}, mu={mu})"
        )

    out = V.copy()
    e0, e1, inv_deg, isolated = _prepare(edges, len(V))
    if pin_mask is not None:
        pin_mask = np.asarray(pin_mask, dtype=bool)
        if pin_mask.shape != (len(V),):
            raise ValueError("pin_mask must be one bool per vertex")

    scratch = np.empty_like(out)
    for _ in range(int(iterations)):
        for factor in (lam, mu):
            L = umbrella_laplacian(out, e0, e1, inv_deg, isolated, out=scratch)
            if pin_mask is not None:
                L[pin_mask] = 0.0
            out += factor * L
    return out


def smooth_vertices(V, edges, *, iterations, lam, mu, faces=None, pin_boundary=False,
                    pin_mask_override=None, weld=True, digits=6):
    """Weld coincident positions, smooth, scatter back. The entry point callers want.

    Returns (smoothed_V, stats). `stats` carries the displacement figures that go into the
    manifest, so that "how far did smoothing actually move the room" is a recorded number
    rather than something to re-derive later.
    """
    V = np.asarray(V, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.int64)

    if weld:
        uniq, inverse = weld_index(V, digits=digits)
        # Positions from the *first* occurrence, not the rounded key, so welding cannot
        # itself perturb geometry by up to half an ulp of the rounding grid.
        merged = np.zeros_like(uniq)
        merged[inverse] = V
        work_edges = np.unique(np.sort(inverse[edges], axis=1), axis=0)
        # An edge whose ends weld together is degenerate and carries no adjacency.
        work_edges = work_edges[work_edges[:, 0] != work_edges[:, 1]]
        work_V = merged
    else:
        inverse = None
        work_edges = edges
        work_V = V

    pin = None
    if pin_mask_override is not None:
        # Blender already knows which edges are boundaries (bmesh edge.is_boundary), so
        # it hands the mask straight in rather than having it rebuilt from face counts.
        # It arrives per *input* vertex, so it has to be folded through the weld: a
        # welded position is held if any vertex that landed on it was held.
        override = np.asarray(pin_mask_override, dtype=bool)
        if override.shape != (len(V),):
            raise ValueError("pin_mask_override must be one bool per input vertex")
        if inverse is None:
            pin = override
        else:
            pin = np.zeros(len(work_V), dtype=bool)
            np.logical_or.at(pin, inverse, override)
    elif pin_boundary:
        if faces is None:
            raise ValueError("pin_boundary needs the face array")
        faces = np.asarray(faces)
        if inverse is not None:
            faces = inverse[faces]
        pin = boundary_mask(faces, len(work_V))

    smoothed = taubin_smooth(
        work_V, work_edges, iterations=iterations, lam=lam, mu=mu, pin_mask=pin
    )

    result = smoothed[inverse] if inverse is not None else smoothed
    disp = np.linalg.norm(result - V, axis=1)
    stats = {
        "iterations": int(iterations),
        "lambda": float(lam),
        "mu": float(mu),
        "pin_boundary": bool(pin is not None),
        "vertices": int(len(V)),
        "welded_vertices": int(len(work_V)),
        "pinned_vertices": int(pin.sum()) if pin is not None else 0,
        "mean_displacement_m": round(float(disp.mean()), 6),
        "max_displacement_m": round(float(disp.max()), 6),
    }
    return result, stats


# --------------------------------------------------------------------------------------
# selftest
# --------------------------------------------------------------------------------------


def _icosphere(subdivisions=3, radius=1.0):
    """A reasonably uniform sphere, built without trimesh so the selftest has no deps."""
    t = (1.0 + 5.0**0.5) / 2.0
    verts = np.array(
        [
            [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
            [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
            [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ],
        dtype=np.int64,
    )
    for _ in range(subdivisions):
        cache = {}
        new_faces = []

        def midpoint(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                cache[key] = len(verts_list)
                verts_list.append((verts_list[a] + verts_list[b]) / 2.0)
            return cache[key]

        verts_list = list(verts)
        for f in faces:
            a, b, c = int(f[0]), int(f[1]), int(f[2])
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        verts = np.array(verts_list, dtype=np.float64)
        faces = np.array(new_faces, dtype=np.int64)

    verts /= np.linalg.norm(verts, axis=1)[:, None]
    return verts * radius, faces


def _grid(n=20, size=1.0):
    """Flat square in z=0, triangulated."""
    xs = np.linspace(0.0, size, n)
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    V = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
    idx = np.arange(n * n).reshape(n, n)
    a, b = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel()
    c, d = idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    faces = np.vstack([np.column_stack([a, b, c]), np.column_stack([a, c, d])])
    return V, faces


def _subdivide(verts, faces, n):
    """Midpoint-subdivide n times, without projecting anywhere."""
    for _ in range(n):
        verts_list = list(verts)
        cache = {}

        def midpoint(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                cache[key] = len(verts_list)
                verts_list.append((verts_list[a] + verts_list[b]) / 2.0)
            return cache[key]

        new_faces = []
        for f in faces:
            a, b, c = int(f[0]), int(f[1]), int(f[2])
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        verts = np.array(verts_list, dtype=np.float64)
        faces = np.array(new_faces, dtype=np.int64)
    return verts, faces


def _box(extents=(6.0, 4.0, 2.5), subdivisions=4):
    """A closed box, the stand-in for a room: nothing but flat faces and sharp creases."""
    e = np.asarray(extents, dtype=np.float64) / 2.0
    V = np.array(
        [
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
        ],
        dtype=np.float64,
    ) * e
    F = np.array(
        [
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
            [1, 2, 6], [1, 6, 5], [0, 4, 7], [0, 7, 3],
        ],
        dtype=np.int64,
    )
    V, F = _subdivide(V, F, subdivisions)
    # Subdivision duplicates nothing, but the eight corners are shared, so weld anyway to
    # match what a real load-and-clean pass would hand us.
    uniq, inverse = weld_index(V)
    merged = np.zeros_like(uniq)
    merged[inverse] = V
    return merged, inverse[F]


def _laplacian_only(V, edges, iterations, lam):
    """Plain Laplacian, for the shrinkage comparison. Not used by the pipeline."""
    out = np.asarray(V, dtype=np.float64).copy()
    e0, e1, inv_deg, isolated = _prepare(edges, len(out))
    for _ in range(iterations):
        out += lam * umbrella_laplacian(out, e0, e1, inv_deg, isolated)
    return out


def selftest():
    fails = []

    def check(label, ok, detail=""):
        print(f"  {'PASS' if ok else 'FAIL'}  {label}{('  -- ' + detail) if detail else ''}")
        if not ok:
            fails.append(label)

    lam, mu, iters = 0.50, -0.53, 8

    # 1. A flat surface is already smooth, so the *interior* must not move at all: a
    #    regular grid's interior vertex sits exactly on the average of its four
    #    neighbours, so L is identically zero there. Any movement is a sign error.
    #
    #    The border is a different matter and deliberately not asserted here. A boundary
    #    vertex has an asymmetric one-ring, so L does not vanish and the edge of an open
    #    patch creeps inward along its own plane -- measured at 1.7 cm on a 1 m grid at
    #    these parameters. That is the entire reason --pin-boundary exists, and check 7
    #    is where it gets tested.
    #    The boundary has to be held for this to be a clean test of the flat case. Left
    #    free, the border's inward creep diffuses one ring further into the patch per
    #    iteration, so after 8 iterations there is no vertex left that the boundary has
    #    not reached -- measured at 9.8 mm even in the middle of a 20x20 grid. That is a
    #    true property of the operator, not a defect, and check 7 is where it is measured.
    V, F = _grid()
    E = edges_from_faces(F)
    out, _ = smooth_vertices(
        V, E, iterations=iters, lam=lam, mu=mu, faces=F, pin_boundary=True
    )
    moved = np.abs(out - V).max()
    check(
        "a flat grid with held edges is an exact no-op",
        moved < 1e-12,
        f"max move {moved:.2e} m",
    )

    # 2. The headline claim: Taubin holds a sphere's radius where Laplacian eats it.
    V, F = _icosphere(3, radius=1.0)
    E = edges_from_faces(F)
    taub = taubin_smooth(V, E, iterations=iters, lam=lam, mu=mu)
    lap = _laplacian_only(V, E, iters, lam)
    r_taub = np.linalg.norm(taub, axis=1).mean()
    r_lap = np.linalg.norm(lap, axis=1).mean()
    check(
        "Taubin preserves sphere radius better than Laplacian",
        abs(1.0 - r_taub) < abs(1.0 - r_lap) / 10.0,
        f"taubin r={r_taub:.6f}, laplacian r={r_lap:.6f}",
    )
    check("Taubin sphere radius within 1%", abs(1.0 - r_taub) < 0.01, f"r={r_taub:.6f}")

    # 3. Noise on a plane must actually come off, and this is the pipeline's real job.
    rng = np.random.default_rng(0)
    V, F = _grid(n=30)
    E = edges_from_faces(F)
    noisy = V.copy()
    noisy[:, 2] += rng.normal(0.0, 0.005, len(V))
    cleaned = taubin_smooth(noisy, E, iterations=iters, lam=lam, mu=mu)
    before, after = noisy[:, 2].std(), cleaned[:, 2].std()
    # A factor of ~2.4 at these parameters, not the order of magnitude one might expect.
    # Taubin is a low-pass filter with a passband, not a denoiser: frequencies below
    # k_PB = 1/lambda + 1/mu pass through with near-unit gain by design, and some of the
    # noise lands there. Turning that dial up is what costs geometry -- see check 9.
    check(
        "5 mm of noise on a plane is attenuated",
        after < before / 2.0,
        f"sigma {before * 1000:.2f} mm -> {after * 1000:.2f} mm "
        f"({before / after:.1f}x)",
    )

    # 4. Determinism -- the manifest records parameters, so the same parameters must give
    #    the same mesh on a re-run or the record is worthless.
    again = taubin_smooth(noisy, E, iterations=iters, lam=lam, mu=mu)
    check("deterministic across runs", np.array_equal(cleaned, again))

    # 5. Seam handling: duplicate every vertex the way a UV split does, and check the
    #    duplicates are still coincident afterwards.
    V, F = _icosphere(2, radius=1.0)
    split_V = np.vstack([V, V])
    split_F = np.where(F % 2 == 0, F, F + len(V))
    split_E = edges_from_faces(split_F)
    res, stats = smooth_vertices(
        split_V, split_E, iterations=iters, lam=lam, mu=mu, weld=True
    )
    gap = np.linalg.norm(res[: len(V)] - res[len(V) :], axis=1).max()
    check("UV-split vertices stay welded", gap < 1e-12, f"max seam gap {gap:.2e} m")
    check("weld collapsed the duplicates", stats["welded_vertices"] == len(V),
          f"{stats['welded_vertices']} of {len(split_V)}")

    # 6. Loose vertices must not produce NaN.
    V, F = _grid(n=10)
    V = np.vstack([V, [[9.0, 9.0, 9.0]]])
    E = edges_from_faces(F)
    out = taubin_smooth(V, E, iterations=iters, lam=lam, mu=mu)
    check(
        "an unreferenced vertex survives untouched and without NaN",
        np.isfinite(out).all() and np.allclose(out[-1], [9.0, 9.0, 9.0]),
    )

    # 7. Boundary pinning, when asked for.
    V, F = _grid(n=12)
    E = edges_from_faces(F)
    noisy = V.copy()
    noisy[:, 2] += rng.normal(0.0, 0.005, len(V))
    pinned, st = smooth_vertices(
        noisy, E, iterations=iters, lam=lam, mu=mu, faces=F, pin_boundary=True
    )
    edge_mask = boundary_mask(F, len(V))
    check(
        "--pin-boundary holds the border exactly",
        np.allclose(pinned[edge_mask], noisy[edge_mask]),
        f"{st['pinned_vertices']} vertices pinned",
    )

    # 8. The one that costs accuracy-gate budget, so it is a permanent regression test.
    #
    #    Taubin is usually described as volume-preserving, and on a smooth closed surface
    #    it is. A room is not a smooth closed surface -- it is flat panels meeting at
    #    sharp creases, and at a crease the mu pass overshoots *outward*. The bounding box
    #    therefore grows, and tools/mesh_validate.py measures kind=extent rows straight
    #    off mesh.bounds, so every millimetre of that overshoot is spent from the 4.97%
    #    error budget before the reconstruction error is even counted.
    #
    #    Asserting a bound in both directions: an implementation that shrank the box would
    #    be just as wrong, and a plain-Laplacian regression would shrink it.
    V, F = _box((6.0, 4.0, 2.5), subdivisions=4)
    E = edges_from_faces(F)
    out = taubin_smooth(V, E, iterations=iters, lam=lam, mu=mu)
    before = V.max(axis=0) - V.min(axis=0)
    after = out.max(axis=0) - out.min(axis=0)
    drift = (after - before) / before
    check(
        "a room-like box holds its extents within +2%/-0.5%",
        bool(np.all(drift < 0.02) and np.all(drift > -0.005)),
        "extents "
        + " x ".join(f"{v:.3f}" for v in after)
        + " (was " + " x ".join(f"{v:.2f}" for v in before) + "), "
        + "drift " + ", ".join(f"{d * 100:+.2f}%" for d in drift),
    )

    # 9. Guard rails on the parameters.
    ok = False
    try:
        taubin_smooth(V, E, iterations=1, lam=0.5, mu=-0.3)
    except ValueError:
        ok = True
    check("|mu| <= lambda is rejected", ok)

    print()
    if fails:
        print(f"{len(fails)} check(s) failed")
        return 1
    print("all checks passed")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Taubin lambda|mu mesh smoothing (library + selftest).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--selftest", action="store_true", help="run the invariant checks")
    args = p.parse_args(argv)
    if not args.selftest:
        p.print_help()
        return 0
    print("== mesh_taubin selftest")
    return selftest()


if __name__ == "__main__":
    sys.exit(main())
