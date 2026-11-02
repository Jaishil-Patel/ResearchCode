#!/usr/bin/env python3
"""Free space, start/goal sampling, and the shortest feasible path length l*.

    python3 -m rl.env.routes --room lab_a --build --show
    python3 -m rl.env.routes --room lab_a --routes 20 --seed 0

Why l* cannot be the straight-line distance
-------------------------------------------
SPL (Eq. 4.3) weights each success by l*/max(l, l*), where l* is the shortest *feasible*
path. In an empty room that equals the Euclidean distance; in a room with furniture it
does not, and the error is not a constant offset -- it is largest exactly on the routes
that have to go around something.

Why l* is planned on the STANDARD branch, for both branches
-------------------------------------------------------------
Proposal S4.4.3 evaluates both policies "on the same fixed set of N start/goal routes" --
one set, not two. That set needs one l* per route, and it has to come from one geometry:
if each branch supplied its own l*, SPL would not be comparable between them.

The standard branch's geometry is the tape-measured room, expressed as boxes in
`<room>_spec.yaml` -- metrically exact by construction (it is the ground truth the
photogrammetric reconstruction is itself validated against, Eq. 4.6) and identical for
both branches, since gen_world.py builds both from the same template and the same spec.
The photogrammetric mesh is a candidate too, and a worse one: it can carry reconstruction
holes and leftover scan debris (one capture came out in 147 components), and
using it would let an artefact of one capture distort l* for both branches' scores.

So `--build` here always plans against the standard branch's spec, regardless of which
world is eventually trained or evaluated. This is one thing this module does NOT use the
photogrammetric mesh for -- the mesh is still used elsewhere, by `PepperNavEnv` when
sampling a TRAINING start/goal pair, where each branch legitimately explores its own
geometry (see `build_grid(..., source="mesh")` there).

The route set is cached to disk, keyed by the spec's content hash and every grid
parameter. The cache is what makes both branches score against byte-identical l* values --
a route set re-planned between two evaluations would put a difference into the comparison
that has nothing to do with the policies.

Why a grid rather than the mesh directly
----------------------------------------
A photogrammetric room is not watertight -- 147 connected components on the first real
scan -- so point-in-mesh tests are unreliable and `is_watertight` is False for the mesh
this was developed against. A rasterised occupancy grid degrades gracefully on scan holes
in a way a mesh query does not, and inflating it by the robot radius is both trivial and
exactly the right model for a circular base.
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "ros2_ws" / "src" / "pepper_room_description"
CACHE_DIR = REPO / "rl" / "cache"
GT_DIR = REPO / "simulation" / "photogrammetry" / "ground_truth"

# tools/ is a script directory, not a package -- imported this way rather than turning it
# into one, matching the pattern rl/env/pepper_nav_env.py already uses for
# tools/pepper_bridge_node.py.
sys.path.insert(0, str(REPO / "tools"))
import gen_world  # noqa: E402

# Pepper: base diameter ~0.48 m, so ~0.24 m radius, plus a margin. The margin is not
# cosmetic -- a path planned with zero clearance hugs walls, which makes l* shorter than
# any real robot could achieve and depresses SPL for a policy that is behaving sensibly.
ROBOT_RADIUS_M = 0.28

# Obstacle height band, measured from the LOCAL floor rather than from world z=0.
#
# A fixed band from z=0 is the obvious implementation and it is wrong here, measurably so:
# on the test_scan capture it reported 39.6 m^2 navigable in a room of roughly 125 m^2,
# because a photogrammetric floor is not planar. It undulates by well over 8 cm, so a band
# starting at z=0.08 classifies most of the floor as an obstacle standing on itself. The
# vertex histogram makes it unambiguous -- a 0.08-0.60 band marks 33% of cells blocked
# while a 0.30-0.90 band marks 3.5%, and the difference is the floor.
#
# So occupancy is "does anything stand proud of the floor beneath it", which is what an
# obstacle physically is:
#   FLOOR_CLEARANCE_M  below this above the local floor is floor, not obstacle
#   ROBOT_HEIGHT_M     above this the robot passes underneath (Pepper is ~1.2 m)
FLOOR_CLEARANCE_M = 0.12
ROBOT_HEIGHT_M = 1.20

DEFAULT_RES_M = 0.05


def die(msg):
    print("ERROR: " + msg, file=sys.stderr)
    sys.exit(1)


# --------------------------------------------------------------------------------------
# the grid
# --------------------------------------------------------------------------------------

class OccupancyGrid:
    """2D occupancy over one room, in world coordinates."""

    def __init__(self, occupied, origin, res):
        self.occupied = occupied          # bool array [ny, nx]; True = blocked
        self.origin = np.asarray(origin, dtype=float)   # world (x, y) of cell (0, 0)
        self.res = float(res)
        self._graph = None
        self._free_idx = None

    # -- construction --------------------------------------------------------------------

    @staticmethod
    def from_mesh(mesh_path, res=DEFAULT_RES_M,
                  clearance=FLOOR_CLEARANCE_M, robot_height=ROBOT_HEIGHT_M):
        """Rasterise a room mesh into occupancy, relative to the local floor.

        Two passes over the same triangles. The first finds the floor height of every
        column as the lowest geometry in it; the second marks a column blocked if anything
        sits between floor+clearance and floor+robot_height. A single pass cannot do this,
        because the band a sample is tested against depends on a minimum taken over all
        the samples.

        A column with no geometry at all is blocked, not free. That covers both the space
        outside the room and holes in the reconstruction -- and a hole is exactly where
        there is no floor for the robot to stand on, so "blocked" is the honest answer
        rather than a conservative one.
        """
        import trimesh
        mesh = trimesh.load(str(mesh_path), force="mesh")
        v, f = np.asarray(mesh.vertices), np.asarray(mesh.faces)
        tris = v[f]                                     # [n, 3, 3]
        if len(tris) == 0:
            die(f"{mesh_path} has no faces")

        lo = tris.reshape(-1, 3).min(axis=0)[:2]
        hi = tris.reshape(-1, 3).max(axis=0)[:2]
        pad = res * 2
        origin = lo - pad
        nx = int(np.ceil((hi[0] + pad - origin[0]) / res)) + 1
        ny = int(np.ceil((hi[1] + pad - origin[1]) / res)) + 1

        # Sample density per triangle, from the triangle's own size. A fixed count either
        # leaves gaps in a 3 m wall triangle or wastes time on a 2 cm furniture one.
        e = np.linalg.norm(tris[:, 1] - tris[:, 0], axis=1)
        e = np.maximum(e, np.linalg.norm(tris[:, 2] - tris[:, 0], axis=1))
        e = np.maximum(e, np.linalg.norm(tris[:, 2] - tris[:, 1], axis=1))
        n_samples = np.clip((e / (res * 0.5)).astype(int) + 2, 3, 128)

        def samples(tri, n):
            a, b, c = tri[0], tri[1], tri[2]
            u = np.linspace(0.0, 1.0, n)
            uu, vv = np.meshgrid(u, u)
            m = (uu + vv) <= 1.0
            uu, vv = uu[m], vv[m]
            pts = a + np.outer(uu, b - a) + np.outer(vv, c - a)
            ix = ((pts[:, 0] - origin[0]) / res).astype(int)
            iy = ((pts[:, 1] - origin[1]) / res).astype(int)
            ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
            return ix[ok], iy[ok], pts[ok, 2]

        floor = np.full((ny, nx), np.inf)
        for tri, n in zip(tris, n_samples):
            ix, iy, z = samples(tri, n)
            np.minimum.at(floor, (iy, ix), z)

        # A column's floor is only known where geometry was seen. Elsewhere the grid has
        # no floor, and `blocked` starts as True so those columns stay blocked.
        blocked = ~np.isfinite(floor)
        for tri, n in zip(tris, n_samples):
            ix, iy, z = samples(tri, n)
            base = floor[iy, ix]
            hit = (z > base + clearance) & (z < base + robot_height)
            if hit.any():
                blocked[iy[hit], ix[hit]] = True

        return OccupancyGrid(blocked, origin, res)

    @staticmethod
    def from_boxes(obstacles, res=DEFAULT_RES_M,
                   clearance=FLOOR_CLEARANCE_M, robot_height=ROBOT_HEIGHT_M,
                   room_length=None, room_width=None):
        """Rasterise the standard branch's own geometry -- walls and furniture as boxes.

        This is `standard_room_obstacles()` from tools/gen_world.py, the same list that
        function turns into <collision> elements for the SDF. One source of geometry
        feeds both, so the grid l* is planned on can never silently diverge from the
        collision volume Pepper actually drives against in the standard-branch world.

        No floor-detection pass is needed here, unlike `from_mesh`: this geometry is
        primitives on an exactly flat floor at z = 0, not a noisy scan, so a box is an
        obstacle wherever its own z-range overlaps the band -- there is no floor to
        confuse it with.
        """
        if room_length is not None and room_width is not None:
            lo = np.array([0.0, 0.0])
            hi = np.array([float(room_length), float(room_width)])
        else:
            allc = np.array([o["centre"][:2] for o in obstacles])
            alls = np.array([o["size"][:2] for o in obstacles])
            lo = (allc - alls / 2.0).min(axis=0)
            hi = (allc + alls / 2.0).max(axis=0)
        pad = res * 2
        origin = lo - pad
        nx = int(np.ceil((hi[0] + pad - origin[0]) / res)) + 1
        ny = int(np.ceil((hi[1] + pad - origin[1]) / res)) + 1
        occ = np.zeros((ny, nx), dtype=bool)

        xs = origin[0] + (np.arange(nx) + 0.5) * res
        ys = origin[1] + (np.arange(ny) + 0.5) * res
        gx, gy = np.meshgrid(xs, ys)  # both [ny, nx]

        for o in obstacles:
            cx, cy, cz = o["centre"]
            sx, sy, sz = o["size"]
            z0, z1 = cz - sz / 2.0, cz + sz / 2.0
            if z1 <= clearance or z0 >= robot_height:
                # This box's whole height sits below the clearance band (a low kick-plate)
                # or above it (something mounted overhead) -- the robot's body does not
                # reach it, so it must not block the grid the way a wall does.
                continue
            if o.get("shape", "box") == "cylinder":
                # Round furniture (a pole, a round table): size is (d, d, h), no yaw.
                occ |= (gx - cx) ** 2 + (gy - cy) ** 2 <= (sx / 2.0) ** 2
                continue
            yaw = o.get("yaw", 0.0)
            # Rotate the grid into the box's local frame rather than rotating the box:
            # the grid is what has to be tested pointwise, and this way every point's
            # containment test is a plain axis-aligned comparison regardless of yaw.
            dx, dy = gx - cx, gy - cy
            c, sn = np.cos(-yaw), np.sin(-yaw)
            lx = dx * c - dy * sn
            ly = dx * sn + dy * c
            occ |= (np.abs(lx) <= sx / 2.0) & (np.abs(ly) <= sy / 2.0)

        return OccupancyGrid(occ, origin, res)

    def inflate(self, radius_m):
        """Grow obstacles by the robot radius, so the robot can be planned as a point."""
        from scipy import ndimage
        r = int(np.ceil(radius_m / self.res))
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        disk = (xx ** 2 + yy ** 2) <= r ** 2
        return OccupancyGrid(ndimage.binary_dilation(self.occupied, structure=disk),
                             self.origin, self.res)

    def largest_free_region(self):
        """Keep only the biggest connected free area.

        A photogrammetric scan leaves pockets of "free" space outside the walls and inside
        furniture. They are unreachable, and sampling a goal in one produces an episode
        the policy cannot possibly succeed at -- which would depress SR for reasons that
        have nothing to do with the policy.
        """
        from scipy import ndimage
        free = ~self.occupied
        lab, n = ndimage.label(free)
        if n == 0:
            die("the inflated grid has no free space at all; the room is smaller than the "
                f"robot ({ROBOT_RADIUS_M} m radius) or the mesh is not metrically scaled")
        sizes = ndimage.sum(free, lab, range(1, n + 1))
        keep = int(np.argmax(sizes)) + 1
        return OccupancyGrid(lab != keep, self.origin, self.res)

    # -- coordinates ---------------------------------------------------------------------

    def world_to_cell(self, x, y):
        return (int((y - self.origin[1]) / self.res),
                int((x - self.origin[0]) / self.res))

    def cell_to_world(self, iy, ix):
        return (self.origin[0] + (ix + 0.5) * self.res,
                self.origin[1] + (iy + 0.5) * self.res)

    def is_free(self, x, y):
        iy, ix = self.world_to_cell(x, y)
        ny, nx = self.occupied.shape
        return 0 <= iy < ny and 0 <= ix < nx and not self.occupied[iy, ix]

    # -- planning ------------------------------------------------------------------------

    def _build_graph(self):
        """8-connected sparse adjacency over free cells, diagonals weighted sqrt(2).

        Built once and reused. On a 9 x 14 m room at 5 cm this is ~50k nodes, which
        scipy's Dijkstra clears in milliseconds -- so every route in a set can be planned
        exactly rather than approximated.
        """
        from scipy import sparse
        free = ~self.occupied
        ny, nx = free.shape
        idx = -np.ones((ny, nx), dtype=np.int64)
        ys, xs = np.nonzero(free)
        idx[ys, xs] = np.arange(len(ys))
        rows, cols, data = [], [], []
        for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
            # Only four of the eight offsets: the graph is undirected, so adding the
            # mirror image would double every edge.
            w = self.res * (np.sqrt(2.0) if dy and dx else 1.0)
            y2, x2 = ys + dy, xs + dx
            ok = (y2 >= 0) & (y2 < ny) & (x2 >= 0) & (x2 < nx)
            ok &= free[np.clip(y2, 0, ny - 1), np.clip(x2, 0, nx - 1)]
            rows.append(idx[ys[ok], xs[ok]])
            cols.append(idx[y2[ok], x2[ok]])
            data.append(np.full(ok.sum(), w))
        r, c, d = np.concatenate(rows), np.concatenate(cols), np.concatenate(data)
        n = len(ys)
        g = sparse.csr_matrix((np.concatenate([d, d]),
                               (np.concatenate([r, c]), np.concatenate([c, r]))),
                              shape=(n, n))
        self._graph, self._free_idx = g, idx
        return g

    def shortest_path_length(self, start_xy, goal_xy):
        """l* between two world points, or None if no feasible path exists."""
        from scipy.sparse import csgraph
        if self._graph is None:
            self._build_graph()
        iy0, ix0 = self.world_to_cell(*start_xy)
        iy1, ix1 = self.world_to_cell(*goal_xy)
        ny, nx = self.occupied.shape
        for (iy, ix), label in (((iy0, ix0), "start"), ((iy1, ix1), "goal")):
            if not (0 <= iy < ny and 0 <= ix < nx) or self._free_idx[iy, ix] < 0:
                raise ValueError(f"{label} {start_xy if label == 'start' else goal_xy} "
                                 f"is outside the free space of this room")
        d = csgraph.dijkstra(self._graph, indices=self._free_idx[iy0, ix0],
                             min_only=False)
        v = d[self._free_idx[iy1, ix1]]
        return None if not np.isfinite(v) else float(v)

    def nearest_free(self, x, y):
        """The centre of the free cell nearest (x, y) -- for a robot that ended a route
        pressed against something, i.e. inside the inflated obstacle band."""
        if self._graph is None:
            self._build_graph()
        ys, xs = np.nonzero(self._free_idx >= 0)
        cx = self.origin[0] + (xs + 0.5) * self.res
        cy = self.origin[1] + (ys + 0.5) * self.res
        i = int(np.argmin(np.hypot(cx - x, cy - y)))
        return float(cx[i]), float(cy[i])

    def line_free(self, a, b):
        """Whether the straight segment a-b stays in free cells (sampled at res/2)."""
        n = max(2, int(math.hypot(b[0] - a[0], b[1] - a[1]) / (self.res / 2)) + 1)
        return all(self.is_free(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
                   for t in np.linspace(0.0, 1.0, n))

    def shortest_path(self, start_xy, goal_xy):
        """Waypoints of a shortest free path, simplified to as few straight legs as the
        free space allows. Start and goal are snapped to the nearest free cell first.
        Returns [] if there is no path. Used to drive the robot between evaluation starts
        (rl/env/reposition.py), never for scoring -- l* comes from the length methods."""
        from scipy.sparse import csgraph
        if self._graph is None:
            self._build_graph()
        s = start_xy if self.is_free(*start_xy) else self.nearest_free(*start_xy)
        g = goal_xy if self.is_free(*goal_xy) else self.nearest_free(*goal_xy)
        i0 = self._free_idx[self.world_to_cell(*s)]
        i1 = self._free_idx[self.world_to_cell(*g)]
        if i0 < 0 or i1 < 0:
            return []
        _, pred = csgraph.dijkstra(self._graph, indices=i0, return_predecessors=True)
        if i1 != i0 and pred[i1] < 0:
            return []
        ys, xs = np.nonzero(self._free_idx >= 0)
        chain, k = [], i1
        while k != i0:
            chain.append(k)
            k = pred[k]
        cells = [self.cell_to_world(ys[j], xs[j]) for j in reversed(chain)]
        pts = [tuple(s)] + cells + [tuple(g)]
        # Greedy line-of-sight shortcutting: from each kept point, jump to the furthest
        # later point visible in a straight line.
        out, i = [pts[0]], 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not self.line_free(pts[i], pts[j]):
                j -= 1
            out.append(pts[j])
            i = j
        return out

    def shortest_path_to_zone(self, start_xy, centre_xy, radius_m):
        """l* from a start to the nearest reachable point within `radius_m` of a goal
        object's centre, or None if none is reachable.

        The goal is a solid object, so its centre is an obstacle cell and a point-to-point
        l* to it does not exist. The episode succeeds on entering the tolerance circle
        (`dist <= goal_tol` in PepperNavEnv.step), so the shortest feasible path ends at
        the circle's nearest free cell -- that is the l* SPL has to divide by.
        """
        from scipy.sparse import csgraph
        if self._graph is None:
            self._build_graph()
        iy0, ix0 = self.world_to_cell(*start_xy)
        ny, nx = self.occupied.shape
        if not (0 <= iy0 < ny and 0 <= ix0 < nx) or self._free_idx[iy0, ix0] < 0:
            raise ValueError(f"start {start_xy} is outside the free space of this room")
        ys, xs = np.nonzero(self._free_idx >= 0)
        cx = self.origin[0] + (xs + 0.5) * self.res
        cy = self.origin[1] + (ys + 0.5) * self.res
        inside = np.hypot(cx - centre_xy[0], cy - centre_xy[1]) <= radius_m
        if not inside.any():
            return None
        d = csgraph.dijkstra(self._graph, indices=self._free_idx[iy0, ix0])
        v = d[self._free_idx[ys[inside], xs[inside]]].min()
        return None if not np.isfinite(v) else float(v)

    # -- sampling ------------------------------------------------------------------------

    def sample_free(self, rng, n=1):
        """Uniform over reachable free cells, jittered within the cell.

        Jitter matters: without it every start pose in training sits on a 5 cm lattice,
        and the policy can learn the lattice.
        """
        ys, xs = np.nonzero(~self.occupied)
        pick = rng.integers(0, len(ys), size=n)
        out = []
        for i in pick:
            x, y = self.cell_to_world(ys[i], xs[i])
            out.append((x + rng.uniform(-0.5, 0.5) * self.res,
                        y + rng.uniform(-0.5, 0.5) * self.res))
        return out

    def ascii_map(self, width=110):
        """A quick look. Worth doing before trusting any route set: a scan hole that
        swallows a wall is instantly obvious here and invisible in a length statistic."""
        ny, nx = self.occupied.shape
        step = max(1, int(np.ceil(nx / width)))
        rows = []
        for iy in range(ny - 1, -1, -step):          # +y up, as in a plan view
            band = self.occupied[max(0, iy - step + 1):iy + 1]
            rows.append("".join("#" if band[:, ix:ix + step].any() else "."
                                for ix in range(0, nx, step)))
        return "\n".join(rows)


# --------------------------------------------------------------------------------------
# rooms and caching
# --------------------------------------------------------------------------------------

def collision_mesh_for(room):
    """The photogrammetric branch's own collision mesh -- used only for TRAINING-time
    free-space sampling (`PepperNavEnv._sample_start_goal` on that branch), never for the
    shared evaluation route set. See the module docstring for why the two are planned on
    different geometry."""
    p = PKG / "models" / room / "meshes" / "collision"
    if not p.is_dir():
        avail = sorted(x.name for x in (PKG / "models").glob("room_*"))
        die(f"no model {room!r} in {PKG / 'models'}. Available: {avail or '(none)'}")
    meshes = sorted(p.glob("*.stl")) + sorted(p.glob("*.obj")) + sorted(p.glob("*.dae"))
    if not meshes:
        die(f"no collision mesh under {p}")
    return meshes[0]


def spec_path_for(room):
    p = GT_DIR / f"{room}_spec.yaml"
    if not p.is_file():
        die(f"no room spec at {p}. Copy {GT_DIR / 'TEMPLATE_spec.yaml'} to {p.name} and "
            f"fill it in -- see docs/photogrammetry_runbook.md step 9. The evaluation "
            f"route set is planned on this file, not on the photogrammetric mesh; see "
            f"the module docstring for why.")
    return p


def goal_zone(room, config=None):
    """(x, y) centre and success radius of the room's goal object, for both branches.

    Read from the standard branch's spec for the same reason l* is (module docstring):
    one geometry for both branches. The spec's goal position was taken from the scan
    footprint, so it is also where the pole stands in the photogrammetric world.
    """
    import yaml
    cfg = config or yaml.safe_load((REPO / "rl" / "config" / "env.yaml")
                                   .read_text(encoding="utf-8"))
    name = cfg["task"]["goal_object"]
    spec = gen_world.load_spec(spec_path_for(room))
    for f in spec.get("furniture", []):
        if f["name"] == name:
            return (float(f["pose"][0]), float(f["pose"][1])), \
                float(cfg["task"]["goal_tolerance_m"])
    die(f"env.yaml task.goal_object is {name!r}, but {spec_path_for(room).name} has no "
        f"furniture item with that name.")


def geometry_fingerprint(source_bytes, res, radius, clearance, robot_height):
    """Content hash of whatever the grid was built from, plus every parameter that
    changes it. Keying on a room name alone would let a regenerated room or a rewritten
    spec quietly reuse a stale cached l* -- and the failure is invisible, because the
    numbers would still be plausible lengths."""
    h = hashlib.sha256()
    h.update(source_bytes)
    h.update(f"|{res}|{radius}|{clearance}|{robot_height}|v3".encode())
    return h.hexdigest()[:16]


def build_grid(room, source="spec", res=DEFAULT_RES_M, radius=ROBOT_RADIUS_M,
               clearance=FLOOR_CLEARANCE_M, robot_height=ROBOT_HEIGHT_M,
               verbose=True):
    """Build the occupancy grid, from one of two geometries.

    `source="spec"` (the default, and what the evaluation route set always uses) plans
    against the standard branch's tape-measured `<room>_spec.yaml` -- metrically exact
    and identical for both branches. `source="mesh"` plans against a branch's own
    collision mesh and is for TRAINING-time free-space sampling only, where each branch
    legitimately explores its own geometry; `room` is then that branch's model directory
    name (e.g. `room_photogrammetry`), not the physical-room slug.
    """
    if source == "spec":
        path = spec_path_for(room)
        spec = gen_world.load_spec(path)
        obstacles = gen_world.standard_room_obstacles(spec)
        r = spec["room"]
        if verbose:
            print(f"room spec: {path.relative_to(REPO)}  "
                  f"({r['length']:.2f} x {r['width']:.2f} x {r['height']:.2f} m, "
                  f"{len(obstacles)} obstacle boxes)")
        raw = OccupancyGrid.from_boxes(obstacles, res=res, clearance=clearance,
                                       robot_height=robot_height,
                                       room_length=r["length"], room_width=r["width"])
        source_bytes = path.read_bytes()
    elif source == "mesh":
        path = collision_mesh_for(room)
        if verbose:
            print(f"collision mesh: {path.relative_to(REPO)}")
        raw = OccupancyGrid.from_mesh(path, res=res, clearance=clearance,
                                      robot_height=robot_height)
        source_bytes = path.read_bytes()
    else:
        raise ValueError(f"source must be 'spec' or 'mesh', got {source!r}")

    grid = raw.inflate(radius).largest_free_region()
    if verbose:
        free = int((~grid.occupied).sum())
        area = free * res * res
        ny, nx = grid.occupied.shape
        print(f"grid: {nx} x {ny} cells at {res} m, "
              f"{free} reachable cells = {area:.1f} m^2 of navigable floor")
        if area < 1.0:
            die("under 1 m^2 of navigable floor. Either the geometry is not metrically "
                "scaled, has no recognisable floor, or the robot radius does not fit "
                "anywhere in it. Look at --show before anything else: a plan view makes "
                "all three obvious in a way an area does not.")
    return grid, path, source_bytes


def generate_routes(grid, n, rng, goal_xy, goal_radius_m, min_length_m=1.5,
                    max_tries_per_route=200, also_free=()):
    """A fixed route set: n starts, all heading to the one goal object, each provably
    connected to it.

    Rejects starts less than `min_length_m` of *planned* path from the goal zone. A start
    30 cm from the goal is inside the tolerance before the policy does anything, and it
    would score SR 1.0 and SPL 1.0 for free in both branches, compressing exactly the
    difference the experiment is trying to measure.

    `also_free`: further grids (already inflated by the robot radius) a start must ALSO be
    free in. l* still comes from `grid` alone -- one geometry for every route's l*, as the
    module docstring requires -- but a start has to be a pose the robot can actually stand
    at in every world it is run in. Planned on the spec only, final_scan's r000 put Pepper's
    base against the scanned wall by the door (0.20 m from the mesh, radius 0.24) and it
    "collided" before moving, in the photogrammetric branch only (1 Oct 2026). The real wall
    is where the scan puts it, so the start was bad on hardware too.
    """
    gx, gy = goal_xy
    routes = []
    tries = 0
    while len(routes) < n and tries < n * max_tries_per_route:
        tries += 1
        (sx, sy), = grid.sample_free(rng, 1)
        if not all(g.is_free(sx, sy) for g in also_free):
            continue
        try:
            ell = grid.shortest_path_to_zone((sx, sy), goal_xy, goal_radius_m)
        except ValueError:
            continue
        if ell is None or ell < min_length_m:
            continue
        routes.append({
            "route_id": f"r{len(routes):03d}",
            "start": [round(sx, 4), round(sy, 4)],
            "start_yaw": round(float(rng.uniform(-np.pi, np.pi)), 4),
            "goal": [round(gx, 4), round(gy, 4)],
            "optimal_length_m": round(ell, 4),
        })
    if len(routes) < n:
        die(f"only found {len(routes)} of {n} routes in {tries} tries. The navigable area "
            f"is probably too small or too fragmented for min_length_m={min_length_m}.")
    return routes


def routes_path(room, variant=None):
    """`variant` names a second, distinct route set for the same room -- e.g.
    'perturbed' for the Task 7 / Sub-question 3 robustness set (see B5), which must be
    demonstrably a DIFFERENT set of routes from the nominal one or "before vs after
    perturbation" is not being measured."""
    suffix = f"_{variant}" if variant else ""
    return CACHE_DIR / f"routes_{room}{suffix}.json"


def save_routes(room, routes, source_path, source_bytes, params, goal, variant=None):
    """Write the route set with the fingerprint of the geometry it was planned on.

    The fingerprint is the point. `load_routes` refuses a set planned against a different
    spec, which is what stops a regenerated room from silently rescoring an old policy
    against l* values that no longer describe its world. `goal` ({xy, radius_m}) is
    checked the same way: every l* ends at that circle, so a changed tolerance or a moved
    goal object makes the whole set stale.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "room": room,
        "variant": variant,
        "source_path": str(Path(source_path).relative_to(REPO)),
        "fingerprint": geometry_fingerprint(source_bytes, **params),
        "params": params,
        "goal": goal,
        "n_routes": len(routes),
        "routes": routes,
    }
    p = routes_path(room, variant)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p


def load_routes(room, verify=True, variant=None, **params):
    p = routes_path(room, variant)
    if not p.is_file():
        variant_flag = f" --variant {variant}" if variant else ""
        tag = f" ({variant})" if variant else ""
        die(f"no route set for {room!r}{tag}. Build one first:\n"
            f"    python3 -m rl.env.routes --room {room} --build{variant_flag}")
    d = json.loads(p.read_text(encoding="utf-8"))
    if verify:
        source_path = REPO / d["source_path"]
        if not source_path.is_file():
            die(f"{p.name} was planned against {source_path}, which no longer exists. "
                f"Rebuild the route set.")
        want = geometry_fingerprint(source_path.read_bytes(), **(params or d["params"]))
        if want != d["fingerprint"]:
            die(f"{p.name} was planned against different geometry "
                f"({d['fingerprint']} != {want}). The room spec or the grid parameters "
                f"changed since. Rebuild:\n"
                f"    python3 -m rl.env.routes --room {room} --build\n"
                f"and re-evaluate BOTH branches -- l* values from two different specs "
                f"cannot be compared, and Eq. 4.4 subtracts them.")
        (gx, gy), radius = goal_zone(room)
        want_goal = {"xy": [round(gx, 4), round(gy, 4)], "radius_m": radius}
        if d.get("goal") != want_goal:
            die(f"{p.name} was planned for goal {d.get('goal')}, but env.yaml and the "
                f"spec now give {want_goal}. Every l* ends at the goal circle, so the set "
                f"is stale. Rebuild:\n"
                f"    python3 -m rl.env.routes --room {room} --build")
    return d["routes"]


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--room", default=None,
                    help="physical-room slug matching "
                         "simulation/photogrammetry/ground_truth/<room>_spec.yaml "
                         "(same slug tools/gen_world.py --room takes). Required unless "
                         "--source mesh, where it is a model directory name instead "
                         "(e.g. room_photogrammetry), for inspecting training-time "
                         "free-space sampling rather than the evaluation route set.")
    ap.add_argument("--source", default="spec", choices=["spec", "mesh"],
                    help="'spec' (default) plans on the standard branch's tape-measured "
                         "geometry, which is what the evaluation route set always uses "
                         "-- see the module docstring. 'mesh' is for inspecting a "
                         "branch's own training-time free space and should not be used "
                         "to build a route set that will be compared across branches.")
    ap.add_argument("--build", action="store_true", help="build and cache the route set")
    ap.add_argument("--variant", default=None,
                    help="name a second, distinct route set for the same room -- e.g. "
                         "'perturbed' for the Task 7 robustness trials (B5). Plain "
                         "--build without --variant is always the nominal set.")
    ap.add_argument("--routes", type=int, default=20, help="how many routes (default 20)")
    ap.add_argument("--seed", type=int, default=0,
                    help="the route set must be reproducible; this is how. Use a "
                         "DIFFERENT seed for --variant, or the 'changed routes' set "
                         "would just be the nominal set again.")
    ap.add_argument("--res", type=float, default=DEFAULT_RES_M)
    ap.add_argument("--radius", type=float, default=ROBOT_RADIUS_M)
    ap.add_argument("--clearance", type=float, default=FLOOR_CLEARANCE_M,
                    help="height above the local floor below which geometry is floor")
    ap.add_argument("--robot-height", type=float, default=ROBOT_HEIGHT_M,
                    help="above this the robot passes underneath")
    ap.add_argument("--min-length", type=float, default=1.5)
    ap.add_argument("--starts-clear-of", default="room_photogrammetry",
                    help="model whose collision mesh every start must ALSO be free in "
                         "(robot radius included); l* is still planned on --source only. "
                         "'none' disables. See generate_routes.")
    ap.add_argument("--show", action="store_true", help="print a plan view of free space")
    args = ap.parse_args()

    if not args.room:
        ap.error("--room is required")

    params = dict(res=args.res, radius=args.radius,
                  clearance=args.clearance, robot_height=args.robot_height)
    grid, path, source_bytes = build_grid(args.room, source=args.source, **params)

    if args.show:
        print()
        print(grid.ascii_map())
        print("  ('#' = blocked for a robot of radius "
              f"{args.radius} m, '.' = navigable; +y up, +x right)")
        print()

    if args.build:
        rng = np.random.default_rng(args.seed)
        (gx, gy), radius = goal_zone(args.room)
        also = []
        if args.starts_clear_of != "none" and (PKG / "models" / args.starts_clear_of).is_dir():
            # inflate + largest region, exactly as for the planning grid: a start must be
            # standable AND connected in the other world too.
            mesh_grid, mesh_path, _ = build_grid(args.starts_clear_of, source="mesh", **params)
            also.append(mesh_grid)
            print(f"starts must also be free in {mesh_path.relative_to(REPO)}")
        routes = generate_routes(grid, args.routes, rng, (gx, gy), radius,
                                 min_length_m=args.min_length, also_free=also)
        goal = {"xy": [round(gx, 4), round(gy, 4)], "radius_m": radius}
        p = save_routes(args.room, routes, path, source_bytes, params, goal,
                        variant=args.variant)
        if also:
            # Recorded, not enforced by load_routes: l* depends only on the spec, which the
            # fingerprint already covers; this says which mesh the starts were checked against.
            d = json.loads(p.read_text(encoding="utf-8"))
            d["starts_clear_of"] = {"mesh": str(mesh_path.relative_to(REPO)),
                                    "sha256_16": hashlib.sha256(mesh_path.read_bytes()).hexdigest()[:16]}
            p.write_text(json.dumps(d, indent=2), encoding="utf-8")
        lengths = [r["optimal_length_m"] for r in routes]
        print(f"wrote {p.relative_to(REPO)}: {len(routes)} routes, "
              f"l* from {min(lengths):.2f} to {max(lengths):.2f} m "
              f"(mean {sum(lengths) / len(lengths):.2f})")
        print("This set is now FIXED. Both branches and the real runs must use it, "
              "unchanged.")


if __name__ == "__main__":
    main()
