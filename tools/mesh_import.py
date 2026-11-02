#!/usr/bin/env python3
"""Turn a raw phone photogrammetry capture into an Ignition model (proposal Task 2).

Reads the untouched export from `simulation/photogrammetry/raw/<room>/` and writes
`ros2_ws/src/pepper_room_description/models/room_photogrammetry/`, which is on
IGN_GAZEBO_RESOURCE_PATH and therefore reachable as `model://room_photogrammetry`.

    python3 tools/mesh_import.py --room lab_a --inspect     # look before you touch
    python3 tools/mesh_import.py --room lab_a               # do it
    python3 tools/mesh_validate.py --room lab_a --report    # THEN gate it

Nothing here is allowed to be a judgement call made silently. Every decision -- the
world frame, the scale factor, how much was thrown away -- is written to
`simulation/photogrammetry/manifests/<room>.json`, because Task 6 maps real start and
goal poses onto simulated ones through exactly that frame, and a re-import that lands
the room somewhere else invalidates every pose already measured.


Why the triangle budget is the first-order concern
--------------------------------------------------
WSL2 must run with LIBGL_ALWAYS_SOFTWARE=1 -- it
is the only render path that returns correct sensor data at all. That makes the 320x240
depth camera CPU-rasterised, and it is already the dominant per-step cost with just
Pepper's ~100 meshes present. A raw room scan is 500k-2M triangles. Dropping one in
unprocessed does not make training slow, it makes training arithmetically impossible,
and you find that out three days into a run rather than on launch. Hence --visual-faces
50000 and --collision-faces 5000 as defaults, and hence the RTF check in the runbook.


Why Blender is invoked for the visual mesh
------------------------------------------
Quadric decimation as implemented by trimesh/fast_simplification discards UV
coordinates. On a photogrammetry mesh the texture *is* the reconstruction -- throwing
away the UVs leaves a grey blob that renders a plausible-looking but featureless depth
image. Blender's Decimate modifier carries UVs through the collapse, so the visual mesh
goes through `tools/blender_room_visual.py` under `blender --background`. The collision
mesh has no UVs to lose, so trimesh handles it directly.

If Blender is not installed the visual step falls back to trimesh and says so loudly;
pass --allow-uv-loss to accept that, which is only ever reasonable for a smoke test.
"""

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# The smoothing kernel is a sibling script, not an installed package, and it is imported
# by absolute path on both sides of the pipeline -- here, and again inside Blender by
# blender_room_visual.py. Explicit rather than relying on the script directory being on
# sys.path, because mesh_pipeline_selftest.sh invokes this file by absolute path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mesh_holes  # noqa: E402 -- needs the sys.path line above
import mesh_taubin  # noqa: E402 -- needs the sys.path line above

REPO = Path(__file__).resolve().parent.parent
RAW_DIR = REPO / "simulation" / "photogrammetry" / "raw"
# Derived intermediates -- the OBJ a GLB capture is converted into. Not the capture and
# not a result, so it is regenerated rather than committed.
WORK_DIR = REPO / "simulation" / "photogrammetry" / "work"
MANIFEST_DIR = REPO / "simulation" / "photogrammetry" / "manifests"
MODEL_DIR = (
    REPO / "ros2_ws" / "src" / "pepper_room_description" / "models" / "room_photogrammetry"
)
BLENDER_HELPER = Path(__file__).resolve().parent / "blender_room_visual.py"

# Defaults. See the module docstring for why these are not negotiable upwards without
# measuring real-time factor first.
DEFAULT_VISUAL_FACES = 50_000
DEFAULT_COLLISION_FACES = 5_000

# Taubin smoothing. On by default: a scan that has not been denoised puts its noise into
# the depth image the policy trains on and into every contact the physics engine resolves,
# and leaving that to a hand pass in the Blender GUI is the thing this pipeline exists to
# avoid. Measured cost on the two captures in raw/ is at most 0.35% of any bounding-box
# extent against the 4.97% budget in mesh_validate.py -- see tools/mesh_taubin.py.
#
# These are a starting point, not a result. The room being imported gets its own sweep
# and the adopted figures are whatever reports/<room>_taubin_sweep.md records.
# Hole filling is OFF by default and stays that way. It invents geometry that was never
# measured, so the decision to use it belongs on a command line and in a manifest, not in a
# default. docs/photogrammetry_runbook.md:405-409 is the standing position; --fill-holes is
# the documented override. See tools/mesh_holes.py for why the cap is 0.20 m2.
DEFAULT_FILL_MAX_AREA = mesh_holes.DEFAULT_MAX_AREA

DEFAULT_SMOOTH_ITERATIONS = 8
DEFAULT_SMOOTH_LAMBDA = 0.50
DEFAULT_SMOOTH_MU = -0.53

SOURCE_SUFFIXES = {".obj", ".glb", ".gltf"}

# Faces with |normal.z| above this count as horizontal (floor/ceiling); below
# HORIZONTAL_EPS they count as vertical (walls). The band between is ignored for frame
# estimation -- sloped scan noise should not get a vote on where the floor is.
HORIZONTAL_COS = 0.90
VERTICAL_COS = 0.20


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def die(msg):
    print("ERROR: " + msg, file=sys.stderr)
    sys.exit(1)


def note(msg):
    print("  " + msg)


def stage(msg):
    print("\n== " + msg)


def import_trimesh():
    try:
        import trimesh
    except ImportError:
        die(
            "trimesh is not installed. In WSL:\n"
            "    pip3 install 'trimesh[easy]' pycollada pygltflib"
        )
    return trimesh


def rel(path):
    """Repo-relative for readability, absolute when the path is outside the repo."""
    path = Path(path)
    try:
        return path.relative_to(REPO)
    except ValueError:
        return path


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def find_input(room, explicit):
    """Locate the source mesh: an OBJ if the capture app produced one, else a GLB.

    OBJ is preferred when both are present, because it is what the rest of the pipeline
    consumes -- Ignition Fortress's OGRE2 loader handles glTF unevenly, so the mesh that
    ends up in the model is always OBJ. That is a constraint on the *output*, though, not
    on what a phone is willing to export: several capture apps emit GLB and nothing else.
    A GLB input is converted to OBJ+MTL+texture by convert_glb() before anything else
    touches it, so the whole downstream path stays on the one tested format.
    """
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            die(f"input not found: {p}")
        if p.suffix.lower() not in SOURCE_SUFFIXES:
            die(
                f"unsupported input {p.name} -- expected one of "
                f"{', '.join(sorted(SOURCE_SUFFIXES))}"
            )
        return p

    room_dir = RAW_DIR / room
    if not room_dir.is_dir():
        die(
            f"no capture at {room_dir}\n"
            f"Export the scan there first "
            f"(see simulation/photogrammetry/raw/README.md)."
        )
    for suffix in (".obj", ".glb", ".gltf"):
        found = sorted(room_dir.glob("*" + suffix))
        if len(found) > 1:
            die(
                f"{len(found)} {suffix} files in {room_dir}: "
                f"{', '.join(f.name for f in found)}\nPass --input to say which one."
            )
        if found:
            return found[0]

    present = sorted(p.name for p in room_dir.iterdir())
    die(
        f"no mesh in {room_dir} (found: {', '.join(present) or 'nothing'})\n"
        f"Expected one of {', '.join(sorted(SOURCE_SUFFIXES))}."
    )


def convert_glb(trimesh, src, room, texture_max, work_root=None):
    """Convert a GLB/glTF capture to OBJ + MTL + texture, and return the OBJ path.

    Done as a separate up-front step rather than by teaching the pipeline a second mesh
    format, so that everything after this point -- the crop box, the Blender decimation,
    the MTL rewriting -- runs on the single format that has been tested end to end.

    In particular the Blender leg must keep importing OBJ. `bpy.ops.import_scene.gltf`
    applies its own Y-up -> Z-up conversion with no way to switch it off, which would
    silently compose with up_axis_transform() and rotate the room twice. The OBJ importer
    is pinned to forward_axis="Y", up_axis="Z" precisely to avoid that, and converting
    here keeps us on that path.

    Idempotent: a sidecar records the source hash and the texture cap, and the conversion
    is skipped when both still match. The 4096-square textures these apps emit take a
    while to re-encode.
    """
    work_dir = (work_root or WORK_DIR) / room
    obj_path = work_dir / "room.obj"
    stamp_path = work_dir / ".source.json"
    digest = sha256(src)
    stamp = {"source": str(src), "sha256": digest, "texture_max": texture_max}

    if obj_path.is_file() and stamp_path.is_file():
        try:
            if json.loads(stamp_path.read_text()) == stamp:
                note(f"reusing {rel(obj_path)} (source unchanged)")
                return obj_path
        except (json.JSONDecodeError, OSError):
            pass  # unreadable stamp just means we redo the work

    scene = trimesh.load(str(src), force="mesh", process=False)
    if not hasattr(scene, "faces") or len(scene.faces) == 0:
        die(f"{src} loaded with no faces")

    if texture_max:
        scene = downscale_texture(scene, texture_max)

    from trimesh.exchange.obj import export_obj

    result = export_obj(scene, return_texture=True, mtl_name="room.mtl")
    obj_text, files = result if isinstance(result, tuple) else (result, {})

    work_dir.mkdir(parents=True, exist_ok=True)
    obj_path.write_text(obj_text)
    for name, blob in (files or {}).items():
        (work_dir / Path(name).name).write_bytes(blob)

    written = ", ".join(sorted(Path(n).name for n in (files or {}))) or "no material"
    note(f"converted {src.name} -> {rel(obj_path)} ({written})")
    if "vt " not in obj_text:
        note(
            "WARNING: the converted OBJ carries no texture coordinates. The visual mesh "
            "will be untextured, which on a photogrammetry scan means the reconstruction "
            "is gone -- only the geometry survives."
        )
    stamp_path.write_text(json.dumps(stamp, indent=2))
    return obj_path


def downscale_texture(mesh, cap):
    """Shrink an oversized baked texture in place, returning the mesh.

    Capture apps emit 4096-square textures as a matter of course, which is an 11 MB PNG
    for a room whose visual mesh is a few thousand triangles. Under LIBGL_ALWAYS_SOFTWARE
    that texture is sampled on the CPU for every depth-camera pixel of every step, so the
    cap is a real training-throughput knob and not housekeeping.
    """
    material = getattr(mesh.visual, "material", None)
    image = getattr(material, "image", None) or getattr(material, "baseColorTexture", None)
    if image is None:
        return mesh
    if max(image.size) <= cap:
        note(f"texture {image.size[0]}x{image.size[1]} already within {cap}")
        return mesh

    from PIL import Image

    scale = cap / float(max(image.size))
    size = (max(1, int(round(image.size[0] * scale))), max(1, int(round(image.size[1] * scale))))
    resized = image.convert("RGB").resize(size, Image.LANCZOS)
    note(f"texture {image.size[0]}x{image.size[1]} -> {size[0]}x{size[1]}")

    uv = getattr(mesh.visual, "uv", None)
    import trimesh as _tm

    mesh.visual = _tm.visual.TextureVisuals(
        uv=uv, material=_tm.visual.material.SimpleMaterial(image=resized)
    )
    return mesh


def mesh_stats(mesh):
    lo, hi = mesh.bounds
    return {
        "faces": int(len(mesh.faces)),
        "vertices": int(len(mesh.vertices)),
        "bounds_min": [round(float(v), 4) for v in lo],
        "bounds_max": [round(float(v), 4) for v in hi],
        "extents": [round(float(v), 4) for v in (hi - lo)],
        "area": round(float(mesh.area), 3),
    }


def face_normals(mesh):
    """Unit face normals computed from the triangles, not read off mesh.face_normals.

    trimesh caches normals, and the cache does not reliably invalidate after invert()
    when the mesh was loaded with process=False -- which made the normal check report
    the same value before and after a flip, i.e. it silently stopped being a check.
    Recomputing costs nothing at these mesh sizes and cannot go stale.
    """
    t = mesh.triangles
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    norm = np.linalg.norm(n, axis=1)
    norm[norm == 0.0] = 1.0
    return n / norm[:, None]


def describe(label, mesh):
    s = mesh_stats(mesh)
    note(
        f"{label}: {s['faces']:,} faces, "
        f"extents {s['extents'][0]:.2f} x {s['extents'][1]:.2f} x {s['extents'][2]:.2f} m, "
        f"z range {s['bounds_min'][2]:.2f} .. {s['bounds_max'][2]:.2f}"
    )
    return s


# --------------------------------------------------------------------------------------
# stage 1 -- load and clean
# --------------------------------------------------------------------------------------


def bulk_extent(mesh, pct):
    """The box holding all but the outermost `pct`% of surface area, per axis.

    Used to decide what counts as "inside the room" without relying on connectivity.
    The obvious alternative -- take the largest connected component and call it the room
    shell -- is what this replaced, and it is wrong on real captures: photogrammetry does
    not produce a connected manifold. The first real scan through this pipeline came out
    as 147 disconnected patches with the largest holding 9.2% of the area, so "largest
    component" selected one wall and the filter deleted the other 88% of the room while
    reporting success.

    Area-weighting rather than face count matters: debris arrives as many tiny triangles,
    which would drag a face-count percentile outwards while contributing almost no
    surface. Measured on that scan the result is stable for pct anywhere in 0.5..2.0,
    so this is a threshold with a flat bottom rather than a tuned constant.
    """
    centroids = mesh.triangles.mean(axis=1)
    areas = mesh.area_faces
    lo = np.empty(3)
    hi = np.empty(3)
    for axis in range(3):
        order = np.argsort(centroids[:, axis])
        cum = np.cumsum(areas[order])
        if cum[-1] <= 0:
            lo[axis], hi[axis] = mesh.bounds[0][axis], mesh.bounds[1][axis]
            continue
        cum = cum / cum[-1]
        sorted_coords = centroids[order, axis]
        last = len(order) - 1
        lo[axis] = sorted_coords[min(np.searchsorted(cum, pct / 100.0), last)]
        hi[axis] = sorted_coords[min(np.searchsorted(cum, 1.0 - pct / 100.0), last)]
    return lo, hi


def load_and_clean(trimesh, path, min_area_frac, margin, percentile, max_loss):
    mesh = trimesh.load(str(path), force="mesh", process=False)
    if not hasattr(mesh, "faces") or len(mesh.faces) == 0:
        die(f"{path} loaded with no faces")
    raw = describe("as loaded", mesh)

    mesh.merge_vertices()
    # trimesh 4 renamed the degenerate-face removal; support both.
    if hasattr(mesh, "nondegenerate_faces"):
        mesh.update_faces(mesh.nondegenerate_faces())
    elif hasattr(mesh, "remove_degenerate_faces"):
        mesh.remove_degenerate_faces()
    mesh.remove_unreferenced_vertices()

    # Drop the texture before splitting. trimesh's split() builds a submesh per connected
    # component, and each submesh deep-copies the material -- which means PIL serialising
    # the whole atlas again, per component. That is O(components x atlas) memory for data
    # this function never reads: everything below works on geometry, and the textured
    # visual mesh is built separately by Blender from the untouched source file.
    #
    # It is not a small constant. A two-material scan packs into a 4096-square RGBA atlas
    # (67 MB); at 648 components that is 43 GB of copies, and the importer is killed by
    # the OOM reaper with no traceback and no output, because what it had printed so far
    # was still sitting in a stdio buffer. The single-material test_scan survived only
    # because 147 components of a 2048 atlas happens to fit in 8 GB.
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh)

    # Phone scans reliably produce floating debris -- fragments of the operator, of the
    # room next door seen through a doorway, of nothing identifiable at all. Left in,
    # they wreck the bounding box, and the bounding box is what the accuracy gate
    # measures room length and width from.
    #
    # Two filters, because area alone is not enough: a person-sized blob standing outside
    # the room is several percent of the total surface and survives any area threshold
    # loose enough to keep the furniture. So the room's bulk extent is measured first --
    # see bulk_extent() for why that is done by area percentile and not by connectivity --
    # and anything whose centroid falls outside it is not in the room.
    parts = mesh.split(only_watertight=False)
    small = outside = 0
    if len(parts) > 1:
        areas = np.array([float(p.area) for p in parts])
        total = float(areas.sum())
        lo, hi = bulk_extent(mesh, percentile)
        pad = margin * np.maximum(hi - lo, 1e-6)

        keep = []
        for part, area in zip(parts, areas):
            centroid = np.asarray(part.centroid)
            if np.any(centroid < lo - pad) or np.any(centroid > hi + pad):
                outside += 1
                continue
            if area < min_area_frac * total:
                small += 1
                continue
            keep.append(part)

        if not keep:
            die("component filter removed everything -- check --min-component-area")
        kept_area = float(sum(p.area for p in keep))
        lost = 1.0 - kept_area / total if total > 0 else 0.0
        if outside or small:
            note(
                f"kept {len(keep)} of {len(parts)} disconnected components "
                f"({kept_area / total:.1%} of surface area; dropped {outside} outside "
                f"the room extent, {small} below {min_area_frac:.3%} of total area)"
            )

        # A tripwire, because this filter's failure mode is not an error -- it is a
        # smaller room that still looks like a room. The first real scan lost 88% of its
        # area here and carried on to produce a plausible single-wall model. Anything
        # that eats most of the capture is a wrong assumption about the capture, not a
        # cleaning job that went well.
        if lost > max_loss:
            before = mesh.bounds[1] - mesh.bounds[0]
            after = trimesh.util.concatenate(keep)
            after = after.bounds[1] - after.bounds[0]
            die(
                f"component filter dropped {lost:.1%} of surface area, above the "
                f"--max-filter-loss limit of {max_loss:.0%}.\n"
                f"  extents before: {before[0]:.2f} x {before[1]:.2f} x {before[2]:.2f} m\n"
                f"  extents after:  {after[0]:.2f} x {after[1]:.2f} x {after[2]:.2f} m\n"
                f"This usually means the capture is not one room: a doorway onto another "
                f"space, or two rooms scanned in one pass. Run --inspect and look at the "
                f"extents, then either raise --component-percentile, lower "
                f"--min-component-area, or pass --allow-aggressive-filter if the loss is "
                f"genuinely what you want."
            )
        mesh = trimesh.util.concatenate(keep)

    cleaned = describe("cleaned", mesh)
    return mesh, {
        "raw": raw,
        "cleaned": cleaned,
        "components_dropped_outside": outside,
        "components_dropped_small": small,
    }


# --------------------------------------------------------------------------------------
# stage 2 -- world frame
# --------------------------------------------------------------------------------------


def up_axis_transform(trimesh, up):
    """Rotate so the named source up-axis becomes +Z.

    ARKit-derived captures (Polycam, Scaniverse, RealityScan on an iPhone) are
    gravity-aligned and Y-up, which is also the glTF convention, so 'y' is the default
    and is right far more often than not. It is still worth confirming with --inspect
    rather than assuming: the failure mode is a room lying on its side, which is obvious
    in the GUI but not obvious in a depth image.
    """
    if up == "z":
        return np.eye(4)
    if up == "y":
        # +Y -> +Z, i.e. rotate +90 deg about X.
        return trimesh.transformations.rotation_matrix(math.pi / 2.0, [1, 0, 0])
    if up == "-y":
        return trimesh.transformations.rotation_matrix(-math.pi / 2.0, [1, 0, 0])
    if up == "x":
        return trimesh.transformations.rotation_matrix(-math.pi / 2.0, [0, 1, 0])
    raise ValueError(up)


def estimate_yaw(mesh):
    """Yaw, in radians, that brings the dominant wall direction onto the X/Y axes.

    Walls in a real room are mutually perpendicular, so their normals' headings cluster
    modulo 90 degrees. Take the area-weighted mode of that distribution. Axis-aligning
    the room is not cosmetic: the Task 3 standard branch is built from axis-aligned
    boxes, and if the two branches sit at different yaws then a start/goal pose set
    expressed in world coordinates means two different things in the two branches.
    """
    n = face_normals(mesh)
    a = mesh.area_faces
    vertical = np.abs(n[:, 2]) < VERTICAL_COS
    if vertical.sum() < 16:
        note("too few wall faces to estimate yaw -- leaving rotation alone")
        return 0.0

    nv, av = n[vertical], a[vertical]
    heading = np.arctan2(nv[:, 1], nv[:, 0]) % (math.pi / 2.0)

    bins = 180  # half-degree resolution over the 90 degree period
    hist, edges = np.histogram(
        heading, bins=bins, range=(0.0, math.pi / 2.0), weights=av
    )
    peak = int(np.argmax(hist))

    # Refine within +/- 2 bins of the peak so the answer is not quantised to the bin
    # width. Wrap-around at the period boundary is handled by unrolling around the peak.
    lo, hi = edges[max(peak - 2, 0)], edges[min(peak + 3, bins)]
    sel = (heading >= lo) & (heading < hi)
    dominant = float(np.average(heading[sel], weights=av[sel])) if sel.any() else float(
        0.5 * (edges[peak] + edges[peak + 1])
    )

    # Rotating by -dominant puts that wall family on an axis. Anything more than 45 deg
    # away is better served by rotating the other way.
    yaw = -dominant
    if yaw < -math.pi / 4.0:
        yaw += math.pi / 2.0
    return yaw


def floor_height(mesh):
    """Z of the floor, as the lowest significant horizontal surface.

    The naive answer -- min(z) -- is wrong on essentially every real scan: reconstruction
    bleeds a little geometry through the floor plane, and a single stray vertex 4 cm low
    would sink the whole room by 4 cm and put Pepper's wheels underground.
    """
    n = face_normals(mesh)
    horizontal = np.abs(n[:, 2]) > HORIZONTAL_COS
    if horizontal.sum() < 16:
        note("too few horizontal faces to find the floor -- using 1st percentile of z")
        return float(np.percentile(mesh.vertices[:, 2], 1.0))

    z = mesh.triangles_center[horizontal, 2]
    w = mesh.area_faces[horizontal]
    lo, hi = float(z.min()), float(z.max())
    if hi - lo < 1e-6:
        return lo
    bins = max(int((hi - lo) / 0.02), 8)  # 2 cm bins
    hist, edges = np.histogram(z, bins=bins, range=(lo, hi), weights=w)

    # The floor is the lowest peak carrying a real share of the horizontal area, not
    # necessarily the largest one -- in a room with a big table the table can win.
    threshold = 0.15 * float(hist.max())
    for i, v in enumerate(hist):
        if v >= threshold:
            sel = (z >= edges[i]) & (z < edges[i + 1])
            return float(np.average(z[sel], weights=w[sel]))
    return lo


def build_frame(trimesh, mesh, args):
    """Compose up-axis, yaw, scale and origin into one 4x4, and apply it."""
    T = up_axis_transform(trimesh, args.up)
    mesh.apply_transform(T)
    note(f"up axis: source {args.up} -> +Z")

    if args.yaw is not None:
        yaw = math.radians(args.yaw)
        note(f"yaw: {args.yaw:.2f} deg (forced)")
    elif args.no_align:
        yaw = 0.0
        note("yaw: 0 (--no-align)")
    else:
        yaw = estimate_yaw(mesh)
        note(f"yaw: {math.degrees(yaw):+.2f} deg (estimated from wall normals)")
    if abs(yaw) > 1e-9:
        R = trimesh.transformations.rotation_matrix(yaw, [0, 0, 1])
        mesh.apply_transform(R)
        T = R @ T

    if args.scale != 1.0:
        S = np.eye(4)
        S[:3, :3] *= args.scale
        mesh.apply_transform(S)
        T = S @ T
    note(f"scale: {args.scale:g}")

    # Origin. The default puts the floor at z=0 and the minimum floor corner at the
    # world origin, so "the corner of the room" is a point you can stand in and point at
    # -- which is what Task 6 needs when transferring measured poses between sim and
    # reality. Anything less physical makes that mapping guesswork.
    if args.origin:
        try:
            ox, oy, oz = (float(v) for v in args.origin.split())
        except ValueError:
            die("--origin wants three numbers, e.g. --origin '1.2 0.4 0.0'")
        offset = np.array([-ox, -oy, -oz])
        note(f"origin: explicit, translating by {offset.round(4).tolist()}")
    elif args.origin_mode == "keep":
        offset = np.zeros(3)
        note("origin: unchanged (--origin-mode keep)")
    else:
        zf = floor_height(mesh)
        lo = mesh.bounds[0]
        if args.origin_mode == "floor-min-corner":
            offset = np.array([-lo[0], -lo[1], -zf])
        else:  # floor-centre
            centre = mesh.bounds.mean(axis=0)
            offset = np.array([-centre[0], -centre[1], -zf])
        note(
            f"origin: {args.origin_mode}, floor found at z={zf:+.4f}, "
            f"translating by {offset.round(4).tolist()}"
        )

    if np.any(offset):
        Tr = trimesh.transformations.translation_matrix(offset)
        mesh.apply_transform(Tr)
        T = Tr @ T

    return mesh, T


# --------------------------------------------------------------------------------------
# stage 3 -- normals
# --------------------------------------------------------------------------------------


def inward_fraction(mesh):
    """Area fraction of shell faces whose normals point towards the room interior.

    This is the check that catches the nastiest failure in the whole pipeline. A scanned
    room is a single-sided shell observed from inside. If the winding comes out such that
    normals face outward, the renderer back-face-culls the walls and the depth camera
    looks straight through them into the void, returning inf. The GUI looks fine from
    outside, the model loads without a single warning, and SAC trains against a room with
    no walls.

    Only faces in the outer half of the bounding box vote, so furniture -- whose normals
    legitimately point away from the room centre -- does not skew the result.
    """
    centre = mesh.bounds.mean(axis=0)
    to_centre = centre - mesh.triangles_center
    dist = np.linalg.norm(to_centre, axis=1)
    shell = dist > 0.5 * float(np.median(dist)) * 1.5
    if shell.sum() < 16:
        shell = np.ones(len(mesh.faces), dtype=bool)

    with np.errstate(invalid="ignore", divide="ignore"):
        unit = to_centre[shell] / dist[shell][:, None]
    dots = np.einsum("ij,ij->i", face_normals(mesh)[shell], unit)
    w = mesh.area_faces[shell]
    return float(w[dots > 0].sum() / max(w.sum(), 1e-12))


def fix_normals(mesh, mode):
    """Returns (inward area fraction after any flip, whether a flip was applied).

    The caller has to know whether a flip happened, not merely what was requested: the
    visual mesh is decimated in a separate Blender process from the same source file, so
    an 'auto' decision taken here has to be handed across explicitly or the two meshes
    end up with opposite winding.
    """
    frac = inward_fraction(mesh)
    note(f"normals: {frac:.1%} of shell area faces inward")
    if mode == "flip":
        flip = True
    elif mode == "keep":
        flip = False
    else:  # auto
        flip = frac < 0.5
    if flip:
        note("normals: flipping winding so the camera sees walls from inside")
        mesh.invert()
        frac = inward_fraction(mesh)
        note(f"normals: now {frac:.1%} inward")
    if 0.35 < frac < 0.65:
        note(
            "WARNING: normals are close to evenly split, so neither orientation is "
            "clearly right. Check in the GUI from inside the room before trusting any "
            "depth data -- see the runbook."
        )
    return frac, flip


# --------------------------------------------------------------------------------------
# stage 3a -- hole filling
# --------------------------------------------------------------------------------------


def fill_holes(mesh, args):
    """Close boundary loops under the cap. Returns (mesh, stats) or (mesh, None) if off.

    Runs after fix_normals() and before smooth_mesh(), and both halves of that matter.

    After the normals, because inward_fraction() decides the global winding from the
    inward-facing area fraction of shell faces and warns when the vote is near even. That is
    the check that catches the pipeline's nastiest failure -- a room whose walls the depth
    camera looks straight through -- and synthetic faces have no business voting in it.

    Before the smoothing, because filling removes boundaries, and an unpinned Taubin pass
    creeps inward at every open rim. Fewer rims, less creep. It is also before decimate() so
    the face budget is spent on a surface that is already closed where it is going to be.
    """
    if not args.fill_holes:
        return mesh, None

    trimesh = import_trimesh()
    V, F, synthetic, stats = mesh_holes.fill_loops(
        mesh.vertices, mesh.faces, max_area=args.fill_max_area
    )
    if stats["faces_added"] == 0:
        note(f"no boundary loop at or under {args.fill_max_area:g} m2; nothing filled")
        return mesh, stats

    out = trimesh.Trimesh(vertices=V, faces=F, process=False)
    area = float(out.area)
    stats["surface_area_m2"] = round(area, 4)
    stats["synthetic_fraction"] = round(stats["synthetic_area_m2"] / max(area, 1e-9), 6)

    note(
        f"filled {stats['loops_filled']:,} of {stats['loops_total']:,} boundary loops "
        f"at or under {args.fill_max_area:g} m2"
    )
    note(
        f"left {stats['loops_left_closed']:,} loops open above the cap "
        f"({stats['loops_open_paths']:,} boundaries were not closed loops at all)"
    )
    # The number the write-up has to quote. Printed every run for the same reason the
    # component filter prints how much area it dropped.
    note(
        f"SYNTHETIC: {stats['faces_added']:,} faces, "
        f"{stats['synthetic_area_m2']:.2f} m2 = "
        f"{stats['synthetic_fraction'] * 100:.1f}% of the surface is now invented geometry"
    )
    return out, stats


# --------------------------------------------------------------------------------------
# stage 3b -- smoothing
# --------------------------------------------------------------------------------------


def smooth_mesh(mesh, args):
    """Taubin-smooth in place. Returns (mesh, stats) or (mesh, None) when disabled.

    Runs before decimation, and that ordering is load-bearing rather than incidental.

    The umbrella Laplacian moves a vertex a fraction of the way to its neighbours, so the
    distance it travels scales with how far apart the vertices are -- not with the size of
    the room. On the full-resolution capture (measured: 7 cm mean edge on test_scan, 11 cm
    on demo_room) that is millimetres and the bounding box moves by at most 0.35%. Run the
    same parameters on the 5,000-face collision mesh, where the average edge is tens of
    centimetres, and the identical call rounds the corners off the room: on a synthetic
    box at 60 cm spacing the extents grew 3.8% and the walls came out *worse* than the
    noisy input. Decimating afterwards also means the face budget is spent on the room
    rather than on preserving noise ridges.

    Smoothing is applied after the world transform, which is free to do: the operator is a
    linear combination of positions, so it commutes with rotation, translation and uniform
    scale. lambda and mu are dimensionless and carry between rooms unchanged. Doing it
    here rather than earlier means displacements are reported in metres, and it matches
    blender_room_visual.py, which cannot help but apply the matrix first.
    """
    if args.smooth_iterations <= 0:
        note("smoothing disabled (--smooth-iterations 0)")
        return mesh, None

    before = mesh.bounds[1] - mesh.bounds[0]
    try:
        verts, stats = mesh_taubin.smooth_vertices(
            mesh.vertices,
            mesh_taubin.edges_from_faces(mesh.faces),
            iterations=args.smooth_iterations,
            lam=args.smooth_lambda,
            mu=args.smooth_mu,
            faces=mesh.faces,
            pin_boundary=args.pin_boundary,
        )
    except ValueError as exc:
        die(f"smoothing parameters rejected: {exc}")
    mesh.vertices = verts

    after = mesh.bounds[1] - mesh.bounds[0]
    drift = (after - before) / np.maximum(before, 1e-9) * 100.0
    stats["aabb_before"] = [round(float(v), 4) for v in before]
    stats["aabb_after"] = [round(float(v), 4) for v in after]
    stats["aabb_drift_pct"] = [round(float(v), 4) for v in drift]

    note(
        f"Taubin N={stats['iterations']} lambda={stats['lambda']:g} mu={stats['mu']:g}"
        + (" (boundary pinned)" if stats["pin_boundary"] else "")
    )
    note(
        f"welded {stats['vertices']:,} vertices onto {stats['welded_vertices']:,} "
        f"positions before smoothing"
    )
    # The drift line is the one to read on every run. It is budget spent straight out of
    # the 4.97% the accuracy gate allows, before any reconstruction error is counted.
    note(
        f"moved vertices by {stats['mean_displacement_m'] * 1000:.1f} mm mean, "
        f"{stats['max_displacement_m'] * 1000:.1f} mm max"
    )
    note(
        "bounding box drift "
        + ", ".join(f"{d:+.2f}%" for d in drift)
        + f" (gate allows {mesh_validate_threshold():.2f}% total)"
    )
    return mesh, stats


def mesh_validate_threshold():
    """The gate's threshold, read from the gate rather than duplicated here."""
    try:
        import mesh_validate

        return float(mesh_validate.THRESHOLD_PCT)
    except Exception:  # noqa: BLE001 -- a missing gate must not break the import
        return float("nan")


# --------------------------------------------------------------------------------------
# stage 4 -- decimation
# --------------------------------------------------------------------------------------


def decimate(mesh, target):
    """Quadric decimation to `target` faces. Drops UVs -- collision meshes only."""
    if len(mesh.faces) <= target:
        note(f"already {len(mesh.faces):,} faces, at or under the {target:,} budget")
        return mesh
    try:
        out = mesh.simplify_quadric_decimation(face_count=target)
    except TypeError:
        out = mesh.simplify_quadric_decimation(target)
    except Exception as exc:  # noqa: BLE001 -- the hint matters more than the type
        die(
            f"decimation failed ({exc}).\n"
            f"    pip3 install fast-simplification\n"
            f"is what trimesh 4 uses for this."
        )
    note(f"decimated {len(mesh.faces):,} -> {len(out.faces):,} faces")
    return out


def crop_to_box(mesh, lo, hi):
    """Drop faces whose centre lies outside [lo, hi]. Used to keep the visual mesh in
    step with the cleaned collision mesh -- see export_visual."""
    c = mesh.triangles_center
    keep = np.all((c >= lo) & (c <= hi), axis=1)
    removed = int((~keep).sum())
    if removed:
        mesh = mesh.copy()
        mesh.update_faces(keep)
        mesh.remove_unreferenced_vertices()
        note(f"cropped {removed:,} faces outside the room bounding box")
    return mesh


def strip_above(mesh, z):
    """Remove faces entirely above z. Used to take the ceiling out of the collision mesh.

    Pepper is 1.2 m tall and cannot collide with a ceiling at 2.4 m, so every ceiling
    triangle is pure physics cost. Kept in the visual mesh, where it still bounds the
    depth camera's view.
    """
    keep = ~np.all(mesh.triangles[:, :, 2] > z, axis=1)
    removed = int((~keep).sum())
    if removed:
        mesh = mesh.copy()
        mesh.update_faces(keep)
        mesh.remove_unreferenced_vertices()
        note(f"stripped {removed:,} faces entirely above z={z:g} from collision")
    return mesh


# --------------------------------------------------------------------------------------
# stage 5 -- visual export (Blender, or trimesh under protest)
# --------------------------------------------------------------------------------------


def find_blender(explicit):
    """Locate Blender, preferring a native Linux build but accepting a Windows one.

    Ubuntu 22.04's apt Blender is 2.93, and installing it costs several hundred MB for a
    decimator four years older than the one already sitting on the Windows side of this
    machine. A Windows blender.exe can be launched from WSL directly -- it just needs its
    arguments as Windows paths, which is what to_win() below is for.
    """
    if explicit:
        return explicit
    native = shutil.which("blender")
    if native:
        return native
    candidates = sorted(
        Path("/mnt/c/Program Files/Blender Foundation").glob("Blender */blender.exe")
    ) + sorted(
        Path("/mnt/c/Program Files").glob("Blender*/blender.exe")
    )
    return str(candidates[-1]) if candidates else None


def to_win(path):
    """POSIX path -> Windows path, for handing arguments to a Windows blender.exe."""
    out = subprocess.run(
        ["wslpath", "-w", str(path)], capture_output=True, text=True
    )
    if out.returncode != 0:
        die(f"wslpath failed on {path}: {out.stderr.strip()}")
    return out.stdout.strip()


def export_visual(args, src, transform, flip, crop, out_dir):
    """Produce the visual mesh from the *source* file, so UVs and materials survive.

    The catch is that everything the cleaning stage decided -- which disconnected
    components are debris -- happened on a separate trimesh copy and is not reflected
    here. Left unhandled, the visual and collision meshes end up being different
    geometry: in testing the collision mesh was a 6.0 x 4.0 m room while the visual mesh
    was 12.1 x 7.1 m because a floating blob outside the room came along for the ride.
    That also breaks the accuracy gate, which measures the visual mesh.

    So the cleaned mesh's bounding box is passed through as a crop. It is not the same
    filter -- debris *inside* the room's bounding box survives into the visual mesh --
    but it guarantees the two meshes agree on the extent of the room, which is what the
    gate measures and what the world frame is anchored to.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "room.obj"
    crop_min, crop_max = crop

    blender = None if args.no_blender else find_blender(args.blender)
    if blender:
        # A Windows blender.exe cannot make sense of a POSIX path; a Linux one cannot
        # make sense of a Windows path. Pick the right dialect for whichever we found.
        conv = to_win if str(blender).lower().endswith(".exe") else str
        note(f"visual: Blender at {blender} (UV-preserving decimate)")
        cmd = [
            blender,
            "--background",
            "--factory-startup",
            "--python", conv(BLENDER_HELPER),
            "--",
            "--input", conv(src),
            "--output", conv(target),
            "--faces", str(args.visual_faces),
            "--matrix", " ".join(f"{v!r}" for v in transform.flatten().tolist()),
            "--crop-min", " ".join(f"{v!r}" for v in crop_min.tolist()),
            "--crop-max", " ".join(f"{v!r}" for v in crop_max.tolist()),
            # The same kernel the collision mesh went through. It has to be the same
            # code and not merely the same parameters: these two meshes are built by
            # different libraries in different processes, and the last time they were
            # allowed to disagree about geometry the gate reported a 45% error.
            "--kernel-dir", conv(Path(__file__).resolve().parent),
            "--smooth-iterations", str(args.smooth_iterations),
            "--smooth-lambda", repr(args.smooth_lambda),
            "--smooth-mu", repr(args.smooth_mu),
        ]
        if args.fill_holes:
            cmd += [
                "--fill-holes",
                "--fill-max-area", repr(args.fill_max_area),
                # The gate measures this one. Same smoothing, same face budget, so the
                # fill is the only difference between the two files.
                "--output-unfilled", conv(target.with_name("room_unfilled.obj")),
            ]
        if args.pin_boundary:
            cmd.append("--pin-boundary")
        if flip:
            cmd.append("--flip-normals")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not target.is_file():
            print(proc.stdout[-4000:], file=sys.stderr)
            print(proc.stderr[-4000:], file=sys.stderr)
            die(
                "Blender visual export failed. Re-run with --no-blender "
                "--allow-uv-loss to get an untextured mesh for a smoke test, but do not "
                "train on it."
            )
        visual_stats = {}
        for line in proc.stdout.splitlines():
            if line.startswith("[blender]"):
                note(line.replace("[blender]", "").strip())
            elif line.startswith("[blender-json]"):
                try:
                    visual_stats.update(json.loads(line[len("[blender-json]"):]))
                except json.JSONDecodeError:
                    pass
        return target, "blender", visual_stats

    if not args.allow_uv_loss:
        die(
            "Blender was not found, and trimesh decimation discards UVs -- the texture "
            "is the reconstruction, so losing it silently is not acceptable.\n"
            "Point at an existing install (a Windows blender.exe is fine, it is called "
            "through wslpath):\n"
            "    --blender '/mnt/c/Program Files/Blender Foundation/Blender 4.5/blender.exe'\n"
            "or install one:  sudo apt install -y blender\n"
            "or pass --allow-uv-loss if you genuinely only want an untextured smoke test."
        )

    note("visual: trimesh fallback -- TEXTURE AND UVs WILL BE LOST (--allow-uv-loss)")
    trimesh = import_trimesh()
    mesh = trimesh.load(str(src), force="mesh", process=False)
    mesh.apply_transform(transform)
    if flip:
        mesh.invert()
    mesh = crop_to_box(mesh, crop_min, crop_max)
    # Same stage, same order as the Blender leg -- the fallback is untextured, not
    # differently shaped.
    mesh, fallback_fill = fill_holes(mesh, args)
    mesh, fallback_smoothing = smooth_mesh(mesh, args)
    mesh = decimate(mesh, args.visual_faces)
    mesh.export(str(target))
    return target, "trimesh-fallback", {
        "smoothing": fallback_smoothing,
        "hole_filling": fallback_fill,
    }


# --------------------------------------------------------------------------------------
# stage 6 -- SDF
# --------------------------------------------------------------------------------------

MODEL_CONFIG = """<?xml version='1.0'?>
<model>
    <name>{name}</name>
    <version>1.0.0</version>
    <sdf version='1.9'>model.sdf</sdf>
    <description>
        {description}
    </description>
</model>
"""

MODEL_SDF = """<?xml version="1.0"?>
<!-- GENERATED by tools/mesh_import.py from {source} on {when}.
     Do not edit: the next import overwrites this file. The transform, scale and
     decimation that produced it are recorded in
     simulation/photogrammetry/manifests/{room}.json -->
<sdf version="1.9">
  <model name="{name}">
    <!-- Static: the room is scenery, so no inertial and no dynamics cost. A non-static
         concave trimesh would additionally not be simulable by DART. -->
    <static>true</static>
    <pose>0 0 0 0 0 0</pose>
    <link name="room_link">
      <collision name="room_collision">
        <geometry>
          <mesh>
            <uri>{collision_uri}</uri>
          </mesh>
        </geometry>
        <surface>
          <friction>
            <ode>
              <mu>{mu}</mu>
              <mu2>{mu2}</mu2>
            </ode>
          </friction>
        </surface>
      </collision>
      <visual name="room_visual">
        <geometry>
          <mesh>
            <uri>{visual_uri}</uri>
          </mesh>
        </geometry>
      </visual>
    </link>
  </model>
</sdf>
"""


def write_sdf(out_dir, name, room, source, visual_uri, collision_uri, mu, mu2):
    when = time.strftime("%Y-%m-%d %H:%M:%S")
    (out_dir / "model.config").write_text(
        MODEL_CONFIG.format(
            name=name,
            description=(
                f"Photogrammetric reconstruction of room '{room}'. Generated by "
                f"tools/mesh_import.py; gated by tools/mesh_validate.py."
            ),
        )
    )
    (out_dir / "model.sdf").write_text(
        MODEL_SDF.format(
            name=name,
            room=room,
            source=source,
            when=when,
            visual_uri=visual_uri,
            collision_uri=collision_uri,
            mu=mu,
            mu2=mu2,
        )
    )


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Raw photogrammetry capture -> Ignition model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--room", required=True, help="room slug, e.g. lab_a")
    p.add_argument(
        "--input", help="source .obj/.glb/.gltf (default: the one in raw/<room>/)"
    )
    p.add_argument(
        "--texture-max",
        type=int,
        default=2048,
        help="downscale a GLB's baked texture to at most this many pixels on its long "
        "edge. Capture apps emit 4096; under software rendering that is sampled on the "
        "CPU for every depth-camera pixel. 0 keeps the original",
    )
    p.add_argument("--output", help=f"model directory (default: {MODEL_DIR})")
    p.add_argument(
        "--work",
        help=f"where GLB conversions are written (default: {WORK_DIR})",
    )
    p.add_argument("--manifest", help=f"manifest path (default: {MANIFEST_DIR}/<room>.json)")
    p.add_argument("--model-name", default="room_photogrammetry")

    p.add_argument(
        "--inspect",
        action="store_true",
        help="load, clean and report; write nothing. Run this first, always.",
    )

    p.add_argument(
        "--up",
        default="y",
        choices=["x", "y", "-y", "z"],
        help="up axis of the source mesh. ARKit/glTF exports are y-up.",
    )
    p.add_argument("--yaw", type=float, help="force yaw in degrees instead of estimating")
    p.add_argument("--no-align", action="store_true", help="skip yaw estimation")
    p.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="metric scale factor. 1.0 for LiDAR-assisted capture, which is already "
        "metric; pure photogrammetry is scale-ambiguous and needs this derived from "
        "the tape measurements.",
    )
    p.add_argument(
        "--origin-mode",
        default="floor-min-corner",
        choices=["floor-min-corner", "floor-centre", "keep"],
    )
    p.add_argument("--origin", help="explicit origin as 'x y z' in post-alignment coords")

    p.add_argument(
        "--flip-normals",
        default="auto",
        choices=["auto", "flip", "keep"],
        help="auto flips when most shell normals point outward",
    )

    p.add_argument(
        "--fill-holes",
        action="store_true",
        help="close boundary loops at or under --fill-max-area. OFF by default: this "
        "invents geometry that was never scanned, so it has to be asked for.",
    )
    p.add_argument(
        "--fill-max-area",
        type=float,
        default=DEFAULT_FILL_MAX_AREA,
        help="safety valve: a boundary loop larger than this is left open. The default "
        "is no cap. On a real capture there are only a handful of loops and the smallest "
        "is 0.41 m2, so any cautious-looking cap just turns the stage off entirely -- "
        "see tools/mesh_holes.py.",
    )

    p.add_argument(
        "--smooth-iterations",
        type=int,
        default=DEFAULT_SMOOTH_ITERATIONS,
        help="Taubin lambda|mu iterations; each is one lambda pass and one mu pass. "
        "0 disables smoothing entirely.",
    )
    p.add_argument(
        "--smooth-lambda",
        type=float,
        default=DEFAULT_SMOOTH_LAMBDA,
        help="Taubin lambda, the shrinking pass. Must be > 0.",
    )
    p.add_argument(
        "--smooth-mu",
        type=float,
        default=DEFAULT_SMOOTH_MU,
        help="Taubin mu, the un-shrinking pass. Must be < 0 with |mu| > lambda, or the "
        "filter degenerates into a plain Laplacian and the room shrinks.",
    )
    p.add_argument(
        "--pin-boundary",
        action="store_true",
        help="hold vertices on open mesh borders still. Off by default, which is "
        "textbook Taubin; the flag exists so the difference can be measured.",
    )

    p.add_argument("--visual-faces", type=int, default=DEFAULT_VISUAL_FACES)
    p.add_argument("--collision-faces", type=int, default=DEFAULT_COLLISION_FACES)
    p.add_argument(
        "--collision-max-z",
        type=float,
        default=2.0,
        help="strip collision faces entirely above this height (Pepper is 1.2 m). "
        "Set to a large value to keep the ceiling collidable.",
    )
    p.add_argument(
        "--min-component-area",
        type=float,
        default=0.001,
        help="drop disconnected components below this fraction of total surface area",
    )
    p.add_argument(
        "--component-margin",
        type=float,
        default=0.05,
        help="how far outside the room's bulk extent, as a fraction of its size, "
        "a disconnected component may sit before it is treated as debris",
    )
    p.add_argument(
        "--component-percentile",
        type=float,
        default=1.0,
        help="percentage of surface area, per axis, treated as outliers when measuring "
        "the room's bulk extent. Stable anywhere in 0.5-2.0",
    )
    p.add_argument(
        "--max-filter-loss",
        type=float,
        default=0.4,
        help="fail if the component filter drops more than this fraction of surface area",
    )
    p.add_argument(
        "--allow-aggressive-filter",
        action="store_true",
        help="proceed past --max-filter-loss instead of failing",
    )
    p.add_argument("--mu", type=float, default=1.0, help="collision friction")
    p.add_argument("--mu2", type=float, default=1.0)

    p.add_argument("--blender", help="path to the blender binary")
    p.add_argument("--no-blender", action="store_true")
    p.add_argument(
        "--allow-uv-loss",
        action="store_true",
        help="permit the untextured trimesh fallback for the visual mesh",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    trimesh = import_trimesh()

    capture = find_input(args.room, args.input)
    out_dir = Path(args.output) if args.output else MODEL_DIR

    # The capture stays the provenance record even when it is not what gets loaded.
    src = capture
    if capture.suffix.lower() in (".glb", ".gltf"):
        stage(f"Converting {capture.name}")
        src = convert_glb(
            trimesh,
            capture,
            args.room,
            args.texture_max,
            work_root=Path(args.work) if args.work else None,
        )

    stage(f"Loading {src}")
    mesh, clean_stats = load_and_clean(
        trimesh,
        src,
        args.min_component_area,
        args.component_margin,
        args.component_percentile,
        1.0 if args.allow_aggressive_filter else args.max_filter_loss,
    )

    stage("World frame")
    mesh, transform = build_frame(trimesh, mesh, args)
    aligned_stats = describe("aligned", mesh)

    stage("Normals")
    inward, flipped = fix_normals(mesh, args.flip_normals)

    stage("Hole filling")
    mesh, fill_stats = fill_holes(mesh, args)

    stage("Smoothing")
    mesh, smooth_stats = smooth_mesh(mesh, args)
    smoothed_stats = describe("smoothed", mesh) if smooth_stats else aligned_stats

    if args.inspect:
        stage("--inspect: nothing written")
        ext = smoothed_stats["extents"]
        print(
            f"\nRoom would be {ext[0]:.2f} x {ext[1]:.2f} m with a "
            f"{ext[2]:.2f} m ceiling, floor at z=0.\n"
            f"Sanity-check those against the tape before running for real. If they are "
            f"off by a constant factor, that factor is --scale."
        )
        return 0

    # The extent the visual mesh will be cropped to, padded so decimation cannot nibble
    # the walls away at the edges.
    lo, hi = mesh.bounds
    pad = np.maximum(0.02 * (hi - lo), 0.05)
    crop = (lo - pad, hi + pad)

    stage("Collision mesh")
    collision = strip_above(mesh, args.collision_max_z)
    collision = decimate(collision, args.collision_faces)
    coll_dir = out_dir / "meshes" / "collision"
    coll_dir.mkdir(parents=True, exist_ok=True)
    coll_path = coll_dir / "room_collision.stl"
    collision.export(str(coll_path))
    coll_stats = describe("collision", collision)
    note(f"wrote {rel(coll_path)}")

    stage("Visual mesh")
    vis_path, vis_backend, vis_extra = export_visual(
        args, src, transform, flipped, crop, out_dir / "meshes" / "visual"
    )
    visual = trimesh.load(str(vis_path), force="mesh", process=False)
    vis_stats = describe("visual", visual)
    note(f"wrote {rel(vis_path)}")

    stage("SDF")
    write_sdf(
        out_dir,
        name=args.model_name,
        room=args.room,
        source=str(rel(capture)),
        visual_uri="meshes/visual/" + vis_path.name,
        collision_uri="meshes/collision/" + coll_path.name,
        mu=args.mu,
        mu2=args.mu2,
    )
    note(f"wrote {rel(out_dir / 'model.sdf')}")

    stage("Manifest")
    manifest_path = Path(args.manifest) if args.manifest else MANIFEST_DIR / f"{args.room}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "room": args.room,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tool": "tools/mesh_import.py",
        # The capture is the provenance record. When it was a GLB, `converted` names the
        # OBJ actually read -- an intermediate that is regenerated, not archived, so the
        # hash that matters is still the capture's.
        "source": {
            "path": str(capture),
            "sha256": sha256(capture),
            "bytes": capture.stat().st_size,
            "converted": str(src) if src != capture else None,
        },
        "arguments": vars(args),
        # The transform is the reproducibility record. Re-import with --up z --no-align
        # --origin-mode keep after applying this and you land in the same world frame.
        "transform_row_major": [round(float(v), 9) for v in transform.flatten()],
        "normals_inward_fraction": round(inward, 4),
        "normals_flipped": bool(flipped),
        # The smoothing record. lambda/mu/iterations are the reproducibility pin; the
        # displacement and drift figures are what the methodology chapter cites when it
        # claims the smoother did not move the measured geometry.
        # Both legs are recorded separately and deliberately. The visual mesh is rebuilt
        # from the source file and only ever sees the crop box, not the component filter,
        # so the two legs smooth different vertex sets. That divergence is pre-existing
        # and load-bearing; recording both figures makes it visible instead of assumed
        # away.
        "smoothing": (
            {"applied": False}
            if smooth_stats is None
            else {
                "applied": True,
                "algorithm": "taubin-umbrella",
                "kernel": "tools/mesh_taubin.py",
                "collision_leg": smooth_stats,
                "visual_leg": (vis_extra or {}).get("smoothing"),
            }
        ),
        # Every synthetic face, counted. docs/photogrammetry_runbook.md:405-409 says a
        # partial capture is a re-capture; filling anyway is a deliberate override, and
        # this block plus the gate's coverage section is what keeps that override honest.
        "hole_filling": (
            {"applied": False}
            if fill_stats is None
            else {
                "applied": True,
                "method": "boundary-loop fan/ngon fill under an area cap",
                "kernel": "tools/mesh_holes.py",
                "material_slot": "synthetic",
                "collision_leg": fill_stats,
                "visual_leg": (vis_extra or {}).get("hole_filling"),
            }
        ),
        "stages": {
            **clean_stats,
            "aligned": aligned_stats,
            "smoothed": smoothed_stats,
            "visual": {**vis_stats, "backend": vis_backend},
            "collision": coll_stats,
        },
        "outputs": {
            "model_dir": str(out_dir),
            "visual": str(vis_path),
            "collision": str(coll_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    note(f"wrote {rel(manifest_path)}")

    ext = smoothed_stats["extents"]
    print(
        f"\nDone. Room is {ext[0]:.2f} x {ext[1]:.2f} m, ceiling {ext[2]:.2f} m, "
        f"{vis_stats['faces']:,} visual / {coll_stats['faces']:,} collision faces.\n"
        f"\nNothing has been validated yet. Next:\n"
        f"    python3 tools/mesh_validate.py --room {args.room} --report\n"
    )
    if vis_backend != "blender":
        print(
            "NOTE: the visual mesh came from the untextured fallback. It is fine for a "
            "launch smoke test and not fine for training.\n"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
