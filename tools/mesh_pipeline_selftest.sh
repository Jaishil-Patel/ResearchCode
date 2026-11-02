#!/usr/bin/env bash
# Exercise the whole photogrammetry pipeline against a synthetic room of known size.
#
# Why bother, when the real capture is the thing that matters: every stage of
# mesh_import.py is a heuristic -- where the floor is, which way is up, what counts as
# debris, which way the normals face -- and heuristics fail quietly. On a real scan there
# is no ground truth for "did the yaw estimator work", so a regression would show up as a
# slightly wrong room that still passes the eye test. Here the answers are known exactly:
# 6.00 x 4.00 x 2.50 m, so e-bar must come out at 0.00%.
#
# The synthetic room is deliberately awkward in the same ways a phone capture is: y-up,
# yawed 13 degrees, translated far from the origin, normals facing outward, and a lump of
# floating debris outside the walls.
#
# Writes nothing into the repository -- everything lands in a temp directory.
#
# Usage:  tools/mesh_pipeline_selftest.sh
# Exit:   0 = pipeline correct, 1 = failure

set -o pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

RAW="$TMP/raw"
RAW2="$TMP/raw_frag"
RAW3="$TMP/raw_glb"
PKG="$TMP/pkg"
mkdir -p "$RAW" "$RAW2" "$RAW3" "$PKG"

TRUE_L=6.00
TRUE_W=4.00
TRUE_H=2.50
FAIL=0

check() {  # check <label> <condition-result>
    if [ "$2" = "0" ]; then
        echo "  PASS  $1"
    else
        echo "  FAIL  $1"
        FAIL=1
    fi
}

echo "== building synthetic capture"
python3 - "$RAW/testroom.obj" <<'PY' || exit 1
import math, sys
import trimesh

L, H, W = 6.00, 2.50, 4.00        # y is up in the source, as it is in an ARKit export
room = trimesh.creation.box(extents=[L, H, W])
for _ in range(4):                # ~3k faces, enough for decimation to do something
    room = room.subdivide()

debris = trimesh.creation.box(extents=[0.25, 0.25, 0.25])
debris.apply_translation([9.0, 1.0, 5.0])

mesh = trimesh.util.concatenate([room, debris])
mesh.apply_transform(trimesh.transformations.rotation_matrix(math.radians(13.0), [0, 1, 0]))
mesh.apply_translation([12.34, 5.67, -8.90])
mesh.export(sys.argv[1])
print(f"  {len(mesh.faces)} faces, y-up, yawed 13 deg, offset from origin, debris included")
PY

echo
echo "== mesh_import"
python3 "$REPO/tools/mesh_import.py" \
    --room testroom \
    --input "$RAW/testroom.obj" \
    --output "$PKG/models/room_photogrammetry" \
    --manifest "$TMP/testroom.json" \
    --no-blender --allow-uv-loss --smooth-iterations 0 \
    --visual-faces 800 --collision-faces 300 \
    > "$TMP/import.log" 2>&1
check "mesh_import exits 0" $?
sed 's/^/    /' "$TMP/import.log"

# The frame estimators: each of these is a separate heuristic that can regress alone.
grep -q -- "-13.00 deg" "$TMP/import.log"
check "yaw estimated as -13.00 deg" $?
grep -q "dropped 1 outside the room extent" "$TMP/import.log"
check "floating debris identified and removed" $?
grep -q "now 100.0% inward" "$TMP/import.log"
check "outward normals detected and flipped" $?
grep -qE "aligned: .* extents 6\.00 x 4\.00 x 2\.50 m, z range 0\.00" "$TMP/import.log"
check "room aligned to $TRUE_L x $TRUE_W x $TRUE_H with the floor at z=0" $?
# The visual mesh is produced from the raw file in a separate pass; it has drifted out of
# step with the collision mesh before, which the accuracy gate then reported as a 45% error.
grep -qE "visual: 800 faces, extents 6\.00 x 4\.00 x 2\.50" "$TMP/import.log"
check "visual mesh matches the cleaned extent (not the raw one)" $?

echo
echo "== fragmented capture"
# The room above is one connected box, which is exactly why this suite did not catch the
# filter bug. Real photogrammetry does not produce a connected manifold: the first actual
# scan through this pipeline arrived as 147 disconnected patches whose largest held 9.2%
# of the surface area. The old "the room shell is the largest component" rule then kept
# one wall and deleted 88% of the room while reporting success. Shatter the room here so
# that assumption can never quietly come back.
python3 - "$RAW2/frag.obj" <<'PY' || exit 1
import sys
import numpy as np, trimesh

L, H, W = 6.00, 2.50, 4.00
room = trimesh.creation.box(extents=[L, H, W])
for _ in range(4):
    room = room.subdivide()

# Break the shell into patches by cutting narrow gaps along a grid. The gaps have to be
# real missing faces, not merely split vertices: load_and_clean() calls merge_vertices()
# before it looks at connectivity, so anything still touching gets welded back together.
#
# The grid is offset by half a cell so no cut plane lands on an extreme face of the box.
# The room's outer extent therefore survives exactly, and the assertion below is testing
# the filter rather than the shape of this fixture. The cell is small enough to leave the
# largest patch holding well under a fifth of the area, as on the real scan -- a fixture
# that stayed mostly connected would still pass under the old largest-component rule and
# would not be a regression test for anything.
CELL, BAND = 1.0, 0.03
centroids = room.triangles.mean(axis=1)
gap = np.zeros(len(room.faces), dtype=bool)
for axis in range(3):
    offset = (centroids[:, axis] - CELL / 2) / CELL
    gap |= np.abs(offset - np.round(offset)) * CELL < BAND
room.update_faces(~gap)
room.remove_unreferenced_vertices()

debris = trimesh.creation.box(extents=[0.25, 0.25, 0.25])
debris.apply_translation([9.0, 1.0, 5.0])
mesh = trimesh.util.concatenate([room, debris])
mesh.export(sys.argv[1])

parts = mesh.split(only_watertight=False)
areas = np.array([p.area for p in parts])
print(f"  {len(mesh.faces)} faces in {len(parts)} components, "
      f"largest holds {areas.max() / areas.sum():.1%} of the area")
PY

python3 "$REPO/tools/mesh_import.py" \
    --room frag --input "$RAW2/frag.obj" \
    --output "$TMP/frag_model" --manifest "$TMP/frag.json" \
    --no-align --no-blender --allow-uv-loss --smooth-iterations 0 \
    --visual-faces 800 --collision-faces 300 \
    > "$TMP/frag.log" 2>&1
check "fragmented room imports without tripping the loss limit" $?
grep -qE "aligned: .* extents 6\.00 x 4\.00 x 2\.50 m" "$TMP/frag.log"
check "fragmented room keeps its full $TRUE_L x $TRUE_W x $TRUE_H extent" $?
grep -q "dropped 1 outside the room extent" "$TMP/frag.log"
check "debris still removed when the room itself is fragmented" $?

# The tripwire. Forcing the loss limit to zero makes any drop at all fatal, which is the
# cheapest way to assert that the limit is wired up and can fail the run.
python3 "$REPO/tools/mesh_import.py" \
    --room frag --input "$RAW2/frag.obj" \
    --output "$TMP/frag_model" --manifest "$TMP/frag.json" \
    --no-align --no-blender --allow-uv-loss --smooth-iterations 0 \
    --max-filter-loss 0.0 \
    > "$TMP/frag_trip.log" 2>&1
[ $? -ne 0 ] && grep -q "max-filter-loss" "$TMP/frag_trip.log"
check "--max-filter-loss aborts the import instead of warning" $?

echo
echo "== GLB input"
# Several capture apps export GLB and nothing else. The pipeline consumes OBJ, so a GLB
# is converted up front; this asserts the conversion keeps the geometry, the UVs and the
# texture, since losing the texture on a photogrammetry mesh loses the reconstruction.
python3 - "$RAW3/testroom.glb" <<'PY' || exit 1
import sys
import numpy as np, trimesh
from PIL import Image

L, H, W = 6.00, 2.50, 4.00
room = trimesh.creation.box(extents=[L, H, W])
for _ in range(4):
    room = room.subdivide()

img = Image.fromarray(
    (np.indices((256, 256)).sum(axis=0) // 32 % 2 * 200 + 30).astype("uint8")
).convert("RGB")
v = room.vertices
uv = np.column_stack([(v[:, 0] - v[:, 0].min()) / L, (v[:, 2] - v[:, 2].min()) / W])
room.visual = trimesh.visual.TextureVisuals(
    uv=uv, material=trimesh.visual.material.SimpleMaterial(image=img)
)
room.export(sys.argv[1])
print(f"  {len(room.faces)} faces, {len(uv)} UVs, 256x256 texture, y-up")
PY

python3 "$REPO/tools/mesh_import.py" \
    --room glbroom --input "$RAW3/testroom.glb" \
    --output "$TMP/glb_model" --manifest "$TMP/glbroom.json" \
    --work "$TMP/work" \
    --no-align --no-blender --allow-uv-loss --smooth-iterations 0 \
    --texture-max 128 --visual-faces 800 --collision-faces 300 \
    > "$TMP/glb.log" 2>&1
check "GLB input imports" $?
sed 's/^/    /' "$TMP/glb.log"
grep -qE "aligned: .* extents 6\.00 x 4\.00 x 2\.50 m" "$TMP/glb.log"
check "GLB round-trip preserves the $TRUE_L x $TRUE_W x $TRUE_H extent" $?
grep -q "256x256 -> 128x128" "$TMP/glb.log"
check "--texture-max downscales the baked texture" $?
python3 - "$TMP/work/glbroom/room.obj" <<'PY'
import sys
from pathlib import Path
obj = Path(sys.argv[1])
text = obj.read_text()
assert "vt " in text, "converted OBJ carries no texture coordinates"
mtl = obj.with_suffix(".mtl")
assert mtl.is_file(), "no MTL written next to the converted OBJ"
line = [l for l in mtl.read_text().splitlines() if l.startswith("map_Kd")]
assert line, "MTL references no texture"
name = line[0].split(None, 1)[1].strip()
assert "/" not in name and "\\" not in name, f"texture path is not a bare filename: {name}"
assert (obj.parent / name).is_file(), f"texture {name} missing next to the OBJ"
PY
check "converted OBJ keeps UVs and a bare-filename texture reference" $?
# Re-running must not redo the conversion: the sidecar hash should short-circuit it.
python3 "$REPO/tools/mesh_import.py" \
    --room glbroom --input "$RAW3/testroom.glb" \
    --output "$TMP/glb_model" --manifest "$TMP/glbroom.json" \
    --work "$TMP/work" \
    --no-align --no-blender --allow-uv-loss --smooth-iterations 0 \
    --texture-max 128 --visual-faces 800 --collision-faces 300 \
    > "$TMP/glb2.log" 2>&1
grep -q "source unchanged" "$TMP/glb2.log"
check "re-import reuses the converted OBJ" $?
python3 - "$TMP/glbroom.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
src = m["source"]
assert src["path"].endswith(".glb"), f"manifest source is not the capture: {src['path']}"
assert src["converted"].endswith(".obj"), "manifest does not record the converted OBJ"
assert len(src["sha256"]) == 64
PY
check "manifest records the GLB as the source and names the conversion" $?

echo
echo "== mesh_validate"
cat > "$TMP/testroom.csv" <<CSV
id,description,kind,axis,point_a,point_b,measured_m,notes
room_length,Long wall to long wall,extent,x,,,$TRUE_L,synthetic truth
room_width,Short wall to short wall,extent,y,,,$TRUE_W,synthetic truth
room_height,Floor to ceiling,extent,z,,,$TRUE_H,synthetic truth
diag_floor,Floor diagonal,points,,0 0 0,6 4 0,7.211,synthetic truth
CSV
python3 "$REPO/tools/mesh_validate.py" \
    --room testroom \
    --ground-truth "$TMP/testroom.csv" \
    --mesh "$PKG/models/room_photogrammetry/meshes/visual/room.obj" \
    > "$TMP/validate.log" 2>&1
check "mesh_validate exits 0 (gate passed)" $?
sed 's/^/    /' "$TMP/validate.log"
grep -q "e-bar = 0.00%" "$TMP/validate.log"
check "e-bar is exactly 0.00% against known truth" $?

# And that the gate can actually fail -- a gate that only ever passes is not a gate.
sed 's/,6.00,synthetic/,9.00,synthetic/' "$TMP/testroom.csv" > "$TMP/bad.csv"
python3 "$REPO/tools/mesh_validate.py" \
    --room testroom --ground-truth "$TMP/bad.csv" \
    --mesh "$PKG/models/room_photogrammetry/meshes/visual/room.obj" \
    > "$TMP/validate_bad.log" 2>&1
[ $? -eq 1 ]
check "gate exits non-zero on a deliberately wrong measurement" $?

echo
echo "== taubin smoothing"
# Everything above this point runs with --smooth-iterations 0, because every one of those
# cases asserts an exact extent or an exact 0.00% e-bar and was written to isolate a
# different stage. Smoothing moves vertices by design, so leaving it on there would turn
# five sharp regression tests into blunt ones. It gets its own fixture instead.

# The kernel's own invariants first -- flat surfaces, sphere-radius retention against
# plain Laplacian, UV-seam welding, determinism, the bounding-box drift bound.
python3 "$REPO/tools/mesh_taubin.py" --selftest > "$TMP/taubin.log" 2>&1
check "mesh_taubin kernel invariants hold" $?
sed 's/^/    /' "$TMP/taubin.log"

# A noisy room, and deliberately a finer one than the fixture above. The umbrella
# Laplacian's step scales with the distance between vertices, not with the size of the
# room, so the same lambda that moves a vertex 2 mm on a real 7 cm-spaced capture moves it
# 15 cm on a coarse box. Subdividing 5 times puts this fixture in the density regime a
# real scan actually occupies; at the coarser 4 subdivisions of the fixture above, e-bar
# lands near 2.4% purely from corner rounding and the test would be measuring an artefact
# of the fixture rather than the stage.
RAW4="$TMP/raw_noisy"
mkdir -p "$RAW4"
python3 - "$RAW4/noisy.obj" <<'PY' || exit 1
import sys
import numpy as np
import trimesh

L, W, H = 6.00, 4.00, 2.50
# Built Z-up, unlike the ARKit-shaped y-up fixture above, so the import below passes
# --up z. The point here is the smoothing stage, not the axis heuristic, which the first
# case already covers.
room = trimesh.creation.box(extents=[L, W, H])
for _ in range(5):
    room = room.subdivide()

# Noise along each vertex's dominant axis, i.e. into and out of the wall it sits on,
# which is what a photogrammetry wall actually looks like. Seeded, so a failure here is
# a real regression and not a bad draw.
rng = np.random.default_rng(1234)
V = np.asarray(room.vertices, dtype=np.float64)
half = np.array([L, W, H]) / 2.0
axis = np.argmax(np.abs(V) / half, axis=1)
rows = np.arange(len(V))
V[rows, axis] += np.sign(V[rows, axis]) * rng.normal(0.0, 0.005, len(V))
room.vertices = V
room.export(sys.argv[1])
print(f"  {len(room.faces)} faces, 5 mm of noise normal to the walls")
PY

python3 "$REPO/tools/mesh_import.py" \
    --room noisy --input "$RAW4/noisy.obj" \
    --output "$TMP/noisy_model" --manifest "$TMP/noisy.json" \
    --no-align --no-blender --allow-uv-loss --up z \
    --visual-faces 20000 --collision-faces 3000 \
    > "$TMP/noisy.log" 2>&1
check "smoothed import exits 0" $?
sed 's/^/    /' "$TMP/noisy.log"

grep -q "Taubin N=8" "$TMP/noisy.log"
check "smoothing actually ran" $?

# The extents must survive. This is the number that comes straight out of the accuracy
# gate's budget, so it is asserted here rather than left to the gate to notice.
grep -qE "smoothed: .* extents 6\.0[0-9] x 4\.0[0-9] x 2\.5[0-9] m" "$TMP/noisy.log"
check "smoothing holds $TRUE_L x $TRUE_W x $TRUE_H to within a centimetre" $?

cat > "$TMP/noisy.csv" <<CSV
id,description,kind,axis,point_a,point_b,measured_m,notes
room_length,Long wall to long wall,extent,x,,,$TRUE_L,synthetic truth
room_width,Short wall to short wall,extent,y,,,$TRUE_W,synthetic truth
room_height,Floor to ceiling,extent,z,,,$TRUE_H,synthetic truth
CSV
python3 "$REPO/tools/mesh_validate.py" \
    --room noisy --ground-truth "$TMP/noisy.csv" \
    --mesh "$TMP/noisy_model/meshes/visual/room.obj" \
    > "$TMP/noisy_validate.log" 2>&1
check "smoothed room still passes the gate" $?
sed 's/^/    /' "$TMP/noisy_validate.log"

# Both bounds matter. A smoother that silently did nothing would sail through a "<=
# threshold" assertion, so the lower bound is what makes this a test of the stage rather
# than of the gate.
python3 - "$TMP/noisy_validate.log" <<'PY'
import re, sys
m = re.search(r"e-bar = ([0-9.]+)%", open(sys.argv[1]).read())
sys.exit(0 if m and 0.0 < float(m.group(1)) <= 4.97 else 1)
PY
check "e-bar is non-zero but inside the 4.97% budget" $?

# The manifest is the reproducibility record. If it stops recording the parameters, the
# claim that the preprocessing is reproducible quietly becomes false.
python3 - "$TMP/noisy.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))["smoothing"]
assert s["applied"] is True, "smoothing.applied not recorded"
assert s["algorithm"] == "taubin-umbrella", s["algorithm"]
leg = s["collision_leg"]
assert leg["iterations"] == 8, leg["iterations"]
assert leg["lambda"] == 0.5 and leg["mu"] == -0.53, (leg["lambda"], leg["mu"])
assert leg["welded_vertices"] <= leg["vertices"]
assert leg["max_displacement_m"] > 0.0
PY
check "manifest records the smoothing parameters" $?

# Same parameters in, same mesh out -- otherwise the manifest describes a result it
# cannot reproduce.
python3 "$REPO/tools/mesh_import.py" \
    --room noisy --input "$RAW4/noisy.obj" \
    --output "$TMP/noisy_model2" --manifest "$TMP/noisy2.json" \
    --no-align --no-blender --allow-uv-loss --up z \
    --visual-faces 20000 --collision-faces 3000 \
    > "$TMP/noisy2.log" 2>&1
cmp -s "$TMP/noisy_model/meshes/collision/room_collision.stl" \
       "$TMP/noisy_model2/meshes/collision/room_collision.stl"
check "a repeat import is byte-identical" $?

echo
echo "== gen_world"
cat > "$TMP/testroom_spec.yaml" <<YAML
room:
  length: $TRUE_L
  width: $TRUE_W
  height: $TRUE_H
  wall_thickness: 0.10
  floor_thickness: 0.05
  include_ceiling: true
openings:
  - {name: door,   wall: y_min, start: 1.20, end: 2.10, bottom: 0.00, top: 2.05}
  - {name: window, wall: x_max, start: 1.00, end: 2.60, bottom: 0.90, top: 2.10}
furniture:
  - {name: desk,    size: [1.60, 0.80, 0.74], pose: [1.20, 0.40, 0.37]}
  - {name: cabinet, size: [0.90, 0.45, 1.80], pose: [4.80, 3.55, 0.90, 1.5708]}
YAML
python3 "$REPO/tools/gen_world.py" \
    --room testroom --spec "$TMP/testroom_spec.yaml" --out "$PKG" --branch both --check \
    > "$TMP/world.log" 2>&1
check "gen_world exits 0 and the branch diff is clean" $?
sed 's/^/    /' "$TMP/world.log"
grep -q "differ only in the model they include" "$TMP/world.log"
check "branches differ only in the included model" $?

echo
echo "== SDF parses"
for f in "$PKG"/models/*/model.sdf; do
    ign sdf -k "$f" >/dev/null 2>&1
    check "$(basename "$(dirname "$f")")/model.sdf is valid SDF" $?
done
# Worlds are not checked with `ign sdf -k`: it never calls sdf::setFindCallback(), so it
# cannot resolve model:// at all and reports every include as missing. The real check is
# loading them in the server, which the runbook covers.

echo
if [ "$FAIL" -eq 0 ]; then
    echo "SELFTEST PASSED"
else
    echo "SELFTEST FAILED"
fi
exit $FAIL
