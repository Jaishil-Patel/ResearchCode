# Photogrammetry runbook — Tasks 2 and 3

Follow this in order. **The room visit is the scarce resource**, as robot access
is: the capture and the tape measurements both happen on
site, and discovering afterwards that one of them is short means going back.

One real capture has now been through the pipeline end to end (`test_scan`, 13 Aug 2026):
it converts, cleans, aligns, builds both meshes, loads in Ignition and returns correct
lidar ranges. It has **not** been through the accuracy gate, because no tape measurements
exist for that room. See "What the first real scan changed" below — it broke the importer
in two places, and both fixes are now regression-tested
(`tools/mesh_pipeline_selftest.sh`, 23 checks, all passing).

---

## The three commands

```bash
python3 tools/mesh_import.py     --room lab_a --inspect  # look first, write nothing
python3 tools/mesh_smooth_sweep.py --room lab_a --fast   # pick lambda, mu, N (once per room)
python3 tools/mesh_import.py     --room lab_a            # raw capture -> Ignition model
python3 tools/mesh_validate.py   --room lab_a --report   # the Eq. 4.6 gate
python3 tools/gen_world.py       --room lab_a --branch both --check   # both branches
```

Then build and launch:

```bash
cd ros2_ws
colcon build --merge-install --symlink-install --packages-select pepper_room_description
source install/local_setup.bash
export LIBGL_ALWAYS_SOFTWARE=1
ros2 launch pepper_robot gz.launch.py \
    world:=$(ros2 pkg prefix --share pepper_room_description)/worlds/room_photogrammetry.sdf
```

---

## What the first real scan changed

Both of these were assumptions that held perfectly on the synthetic fixture and failed on
the first phone capture. They are recorded because the same class of assumption is
probably still lurking elsewhere in `mesh_import.py`.

**GLB is now an accepted input.** The importer used to demand OBJ+MTL+PNG and refuse GLB
outright. The reasoning — Ignition Fortress's OGRE2 loader handles glTF unevenly — was
about the mesh Ignition *consumes*, and got mistakenly applied to what the phone is
allowed to *export*. Plenty of apps emit GLB and nothing else. A GLB input is now
converted to OBJ+MTL+texture up front by `convert_glb()` and everything downstream runs
unchanged on the one tested format. The conversion is cached against the source hash.

**The debris filter no longer assumes the room is connected.** It used to take the largest
connected component as the room shell and treat anything outside its bounding box as
debris. Real photogrammetry does not produce a connected manifold: the first scan arrived
as **147 disconnected patches with the largest holding 9.2% of the surface area**, so
"largest component" picked one wall and the filter deleted 88% of the room — while
printing a cheerful note and exiting 0. It now measures the room's bulk extent by
area-weighted percentile over all faces (`bulk_extent()`), ignoring connectivity, and
`--max-filter-loss` makes an excessive drop fatal rather than a warning.

| | before | after |
|---|---|---|
| components kept | 4 of 147 | 74 of 147 |
| surface area kept | 11.9% | 98.0% |
| extents | 0.82 × 2.84 × 2.17 m | 3.81 × 2.97 × 4.30 m |

---

## Before you go

### 1. Pick the room

Three constraints, all binding, and it is worth walking candidate rooms with all three in
mind rather than optimising for one:

- **Reconstructable** — feature-rich and evenly lit. Blank white walls, glossy surfaces
  and windows with strong backlight are what photogrammetry fails on
  (Gómez et al. 2018, cited in proposal §4.4.4).
- **Measurable** — you must be able to get a tape across the K dimensions. A room packed
  wall-to-wall is hard to measure honestly, and the standard branch is built from those
  measurements.
- **Safe to drive Pepper in** — Task 6 puts the physical robot here. Flat floor, no
  steps, room to recover from a bad policy.

### 2. Dependencies

Already installed and verified on this machine:

| Package | Version | For |
|---|---|---|
| `trimesh` | 5.0.0 | load / clean / align / decimate / export |
| `fast-simplification` | 0.1.13 | the quadric decimator trimesh calls |
| `numpy` | 1.26.4 | upgraded from the apt 1.21.5, which was too old for trimesh |
| `scipy` | 1.10.1 | upgraded to match numpy |
| `pycollada`, `pyyaml`, `Pillow` | | mesh I/O and the room spec |
| Blender | 4.5.3 LTS, **on Windows** | UV-preserving decimation |

`rclpy`, `sensor_msgs`, `geometry_msgs` and `cv_bridge` were re-checked after the numpy
upgrade and all still import. numpy was deliberately pinned to 1.26.4 rather than 2.x:
the numpy C ABI is forward compatible, so ROS 2 Humble packages built against 1.21 keep
working, whereas 2.x would break them.

**Blender is found automatically**, including the Windows install — `mesh_import.py`
globs `/mnt/c/Program Files/Blender Foundation/Blender */blender.exe` and converts every
path argument with `wslpath` before handing it over. Nothing needs installing in WSL. To
point at a different one:

```bash
--blender '/mnt/c/Program Files/Blender Foundation/Blender 4.5/blender.exe'
```

---

## On site

### 3. Capture

Standard consumer photogrammetry practice, and proposal §4.4.1 specifies the phone:

- Heavy overlap — each surface visible in many frames from many angles.
- **Multiple heights.** At minimum: floor level, Pepper's camera height (~1.2 m), and
  above head height. The policy sees the room from ~1.2 m, and geometry reconstructed
  only from eye-level passes is worst exactly there.
- Walk the perimeter, then walk the interior, then orbit each piece of furniture.
- Even lighting. Blinds down if the windows are blowing out the exposure.
- Do not move anything mid-capture. If furniture moves, the reconstruction fuses both
  positions into one surface.

Export **OBJ + MTL + texture PNGs** if the app offers it, otherwise **GLB** — both are
accepted, and a GLB is converted to OBJ automatically on import. Export both when you can:
the OBJ skips a conversion step, and having two copies of an irreplaceable capture costs
nothing. See `simulation/photogrammetry/raw/README.md`.

### 4. Tape-measure the K dimensions — same visit, not a second trip

These feed *both* the accuracy gate and the entire Task 3 standard branch, so they are not
a check on the capture, they are half the experiment.

**Take
[`ground_truth/FIELD_MEASUREMENT_SHEET.md`](../simulation/photogrammetry/ground_truth/FIELD_MEASUREMENT_SHEET.md)
into the room** — it's the fillable version of the two files below, meant to be read on a
phone or printed with the tape measure in hand. Transcribe from it afterward into
`simulation/photogrammetry/ground_truth/TEMPLATE.csv` (copied to `<room>.csv`) and
`TEMPLATE_spec.yaml` (copied to `<room>_spec.yaml`) — the sheet says exactly which section
maps to which.

- **K ≥ 8, spread across all three axes.** A gate computed only from floor-plane distances
  cannot detect a vertical scale error, and a vertical scale error puts the depth camera's
  horizon somewhere the real one never is.
- Room length, width, floor-to-ceiling height, door width and height, window dimensions,
  and the bounding dimensions of every piece of furniture Pepper must navigate around.
- Note **which corner of the room** you will use as the world origin. `mesh_import.py`
  puts the origin at the minimum floor corner by default, and Task 6 maps real start and
  goal poses onto simulated ones through that point. It has to be a corner you can stand
  in and point at.

### 5. Record the capture parameters now, not from memory later

Fill this in on the day. If the reconstruction fails the gate, this table is the only
evidence of what to change on the re-shoot.

| | |
|---|---|
| Room | |
| Date | |
| App / device | |
| LiDAR or depth used? | ← decides whether the capture is already metric |
| Number of images / duration | |
| Capture heights | |
| Lighting | |
| Anything that moved | |

---

## Back at the desk

### 6. Look before you touch

```bash
python3 tools/mesh_import.py --room lab_a --inspect
```

Writes nothing. Reports the triangle count and, after alignment, the room's dimensions.
**Check those against the tape immediately.** If they are off by a constant factor, that
factor is `--scale` — LiDAR-assisted captures are already metric and need `--scale 1.0`,
pure photogrammetry is scale-ambiguous and does not.

If the room comes out lying on its side, the source up-axis is not what was assumed;
try `--up z` or `--up -y`. If the walls are visibly skewed rather than axis-aligned, the
yaw estimate failed — force it with `--yaw <degrees>`.

### 7. Import

```bash
python3 tools/mesh_import.py --room lab_a
```

Seven things happen, and the log names each one:

1. **Clean** — merge vertices, drop degenerate faces, remove disconnected debris. Two
   filters: components outside the room's bulk extent, and components below 0.1% of
   total area. Phone scans reliably contain fragments of the operator and of whatever
   was visible through a doorway. **Read the "kept N of M components" line every time**
   — the percentage of surface area retained is printed there, and a filter that quietly
   eats the room is the single most expensive failure in this pipeline.
2. **World frame** — up axis to +Z, yaw so the walls are axis-aligned, floor to z = 0,
   origin at the minimum floor corner.
3. **Normals** — a scanned room is a single-sided shell seen from inside. If the winding
   comes out facing outward, the renderer culls the walls and the depth camera looks
   straight through them into the void. Nothing warns you. The importer measures the
   inward-facing area fraction and flips if needed; if it reports a near-even split,
   neither orientation is clearly right and you must check in the GUI.
4. **Smoothing** — Taubin λ|μ, N = 8, λ = 0.50, μ = −0.53, applied to every vertex.
   Boundaries are *not* pinned; that is textbook Taubin, and `--pin-boundary` exists only
   so the difference can be measured. A phone scan's walls are flat surfaces carrying a
   few millimetres of noise, and that noise goes into the depth image the policy trains on
   and into every contact the physics engine resolves. **Read the "bounding box drift"
   line every time**, for the same reason as the "kept N of M components" line above: it
   is budget spent straight out of the 4.97% the gate allows, before any reconstruction
   error is counted. On the captures in `raw/` it is under 0.5%.

   The same kernel (`tools/mesh_taubin.py`) runs on both the collision mesh here and the
   visual mesh inside Blender. Not merely the same parameters — the same code, imported
   into both interpreters, because the two meshes are built by different libraries and
   the last time they were allowed to disagree about geometry the gate reported 45%.

   λ, μ and N are dimensionless and transfer between rooms, but they are a starting point
   rather than a result. Section 7b is how you pick them for this room.
5. **Collision mesh** — ceiling stripped above 2.0 m (Pepper is 1.2 m tall and every
   ceiling triangle is pure physics cost), decimated to ≤5k triangles, exported as STL.
6. **Visual mesh** — Blender decimates to ≤50k triangles with UVs intact, exported as
   OBJ + MTL + texture.
7. **Manifest** — `simulation/photogrammetry/manifests/<room>.json` records the source
   hash, the exact 4×4 transform, the scale, the smoothing parameters and per-leg
   displacement figures, and every face count. This is the reproducibility record; a
   re-import that lands the room in a different frame invalidates every pose already
   measured against it.

Smoothing runs **before** decimation, and that ordering is load-bearing rather than
incidental. The umbrella Laplacian moves a vertex a fraction of the way toward its
neighbours, so how far it travels scales with how far apart the vertices are — not with
the size of the room. At full capture resolution (7 cm mean edge on `test_scan`, 11 cm on
`demo_room`) that is millimetres. Run the identical parameters on the 5k-face collision
mesh, where edges are tens of centimetres, and it rounds the corners off the room: on a
synthetic box at 60 cm spacing the extents grew 3.8% and the walls came out *worse* than
the noisy input. Decimating afterwards also means the face budget buys room structure
rather than preserved noise ridges.

### 7b. Choosing λ, μ and N

The defaults are a starting point taken from the literature, not a result for this room.
Picking them because the mesh looked better establishes nothing, so sweep them and let
the accuracy gate score every candidate:

```bash
# rank the grid cheaply -- skips Blender, so these numbers do NOT go in the write-up
python3 tools/mesh_smooth_sweep.py --room lab_a --fast

# re-run the winner for real, through Blender
python3 tools/mesh_smooth_sweep.py --room lab_a \
    --iterations 8 --lambdas 0.50 --mus -0.53 --confirm

# and the boundary-pinning ablation, one row
python3 tools/mesh_smooth_sweep.py --room lab_a \
    --iterations 8 --lambdas 0.50 --mus -0.53 --confirm \
    --import-arg --pin-boundary
```

Writes `simulation/photogrammetry/reports/<room>_smoothing_sweep.md`. Table A is the
parameter grid, Table B is the per-dimension before/after that the write-up prints, Table
C flags any `kind=points` pick that drifted past the snap tolerance.

Reading it:

- **Row 0 is the control** — the unsmoothed import. If nothing beats it on ē, smoothing
  is costing accuracy rather than buying it for this room, and the honest answer is
  `--smooth-iterations 0`.
- **ē cannot see noise.** It is computed from a handful of tape-measured distances, and a
  wall can be metrically perfect and still covered in 5 mm of hash. That is why the
  displacement and drift columns sit beside it: ē says what smoothing *cost*, those say
  what it *did*. Choosing the lowest ē with zero displacement is choosing to do nothing.
- **Watch the |μ|/λ ratio, not just the magnitudes.** On `test_scan`, λ = 0.30 with
  μ = −0.53 (ratio 1.77) was the worst row in the grid at +1.40 points of ē, while every
  ratio near 1.05 improved on the control. μ undoes the shrink λ causes; overshoot it and
  the walls bow outward.
- **`--fast` ranks, `--confirm` measures.** `--fast` swaps Blender's UV-preserving
  decimator for trimesh's, which changes the geometry slightly. Quote `--confirm` rows
  only. The report labels which is which.

Adopt the winner by editing `DEFAULT_SMOOTH_ITERATIONS` / `_LAMBDA` / `_MU` in
`tools/mesh_import.py`, then re-run `tools/mesh_pipeline_selftest.sh`.

**Ordering trap.** `kind=points` ground-truth coordinates are vertex-snapped in
processed-mesh coordinates, and smoothing moves vertices. Pick them *after* λ, μ and N are
frozen, or the gate quietly measures a slightly different feature than the one you
measured with the tape. Table C and `mesh_validate.py --snap-warn` both surface this.

### 8. Gate it

```bash
python3 tools/mesh_validate.py --room lab_a --report
```

Exits non-zero if ē > 4.97% (proposal Eq. 4.6). **Do not proceed past a failure.** Reading
the error column:

| Pattern | Means | Fix |
|---|---|---|
| Uniform error across all K | Scale factor | Re-import with `--scale` |
| Concentrated on one axis | Up-axis or yaw estimate wrong | `--up` / `--yaw` |
| Scattered individual rows | The point picks, not the mesh | Re-pick in Blender |
| One dimension wildly off | Debris still in the bounding box | Tighten `--min-component-area` |
| Room too small in every dimension | The filter ate part of it | Check the "kept N of M" line; raise `--component-percentile` |
| Every extent slightly too *large*, `points` rows fine | Smoothing overshooting at the wall creases | Lower `--smooth-iterations` or `--smooth-lambda`; see §7b |

For `kind=points` rows you need coordinates picked off the *processed* mesh: open
`meshes/visual/room.obj` in Blender, select a vertex, `Shift+S` → Cursor to Selected, read
the coordinates from the N panel. The validator prints how far each pick had to snap to
reach a vertex, so a coordinate left over from an earlier import shows up as a large snap
rather than as a silently wrong measurement.

### 9. Generate both branches

Fill in `simulation/photogrammetry/ground_truth/<room>_spec.yaml` from `TEMPLATE_spec.yaml`
— the same measurements, expressed as walls, openings and furniture bounding boxes — then:

```bash
python3 tools/gen_world.py --room lab_a --branch both --check
```

`--check` diffs the two worlds and fails if they differ in anything beyond the model they
include. Run it after every regeneration. Proposal §4.4.2 makes geometry the only
independent variable; the moment the worlds differ in a second respect, the comparison
stops measuring photogrammetry.

Note that the spec's frame must match the import's: interior at x ∈ [0, length],
y ∈ [0, width], floor at z = 0, **origin at the same physical corner**.

---

## Verification

### 10. The mesh loads and the walls are solid from inside

```bash
export LIBGL_ALWAYS_SOFTWARE=1
ign gazebo $(ros2 pkg prefix --share pepper_room_description)/worlds/room_photogrammetry.sdf
```

Fly the camera **inside** the room. Walls must be opaque from within and the texture must
be visible. See-through walls mean the normals are inverted — re-import with
`--flip-normals flip`. A grey untextured surface means the MTL or the PNG did not resolve;
check that `meshes/visual/room.mtl` refers to the texture by bare filename and that the
file is next to it.

### 11. The triangle budget is real — measure RTF

```bash
ros2 launch pepper_robot gz.launch.py world:=<abs path> enable_rviz:=false
ros2 topic hz /pepper_robot/depth_camera/image_raw     # target 15 Hz
```

Read real-time factor off the Gazebo GUI and compare against an empty world. Software
rendering makes the 320×240 depth camera CPU-bound and it is already the dominant
per-step cost with only Pepper in the scene. **If RTF collapses, decimate harder** —
`--visual-faces 25000` — before going any further. This is cheap to fix now and very
expensive to discover three days into a training run.

### 12. Sensors report true geometry — the silent-garbage check

```bash
tools/verify_sensors.sh
ros2 topic echo /pepper_robot/scan --once
```

Park Pepper a tape-measured distance from a wall and confirm `/scan` agrees to a few cm.
This is the strongest end-to-end validation available: it exercises scale, orientation,
normals and the render path in one shot, along the exact code path SAC will consume.
Merely checking that `/scan` *publishes* is not
enough — the `ogre` v1 path publishes plausible-looking constant garbage without error.

### 13. Pepper drives

```bash
ros2 topic pub /pepper_robot/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.2}}"
```

Moves, does not sink through the floor, stops at walls. If it falls through, the collision
mesh has a hole where the floor should be — re-run with `--collision-faces 10000`, or use
`gen_world.py --debug-ground-plane` to confirm that is the cause before chasing anything
else.

Two things that are not bugs in the mesh and will waste an afternoon if you treat them as
such:

**Pepper spawns at the world origin, which is a corner.** `--origin-mode floor-min-corner`
puts (0, 0, 0) at the minimum floor corner, so the default spawn wedges the robot into two
walls. On the first real scan it tipped over immediately (roll 0.85 rad) and Gazebo filled
the log with `ODE Message 2: Trimesh-trimesh contact hash table bucket overflow`. That
message means "these two trimeshes are deeply interpenetrating", not "the collision mesh
is broken". Training needs a spawn pose somewhere sensible in the room's interior; the
corner origin is the right convention for *mapping* real poses to simulated ones and the
wrong place to *start* from.

**A partial capture has holes.** Compare the scan's surface area against roughly
`2(LW + LH + WH)` for the room's dimensions. The first scan came in at 53.8 m² against
~81 m² for a closed room that size, meaning about a third of the surfaces were never
reconstructed. Holes in the floor drop the robot; holes in a wall let the lidar see
through into nothing. Neither is fixable downstream — it is a re-capture.

### 14. Both branches

```bash
ros2 launch pepper_robot gz.launch.py world:=<abs path to room_standard.sdf>
```

---

## Re-verifying the tooling

After any edit to `mesh_import.py`, `mesh_validate.py`, `mesh_taubin.py` or
`gen_world.py`:

```bash
tools/mesh_pipeline_selftest.sh
```

Builds synthetic rooms of exactly known dimensions and asserts that every heuristic
produces the known-correct answer. Writes nothing into the repository. Four fixtures:

- **the awkward room** — y-up, yawed 13°, translated off the origin, normals outward,
  floating debris included. Covers the yaw estimator, the floor finder, the normal flip,
  the alignment, visual/collision consistency and the ē gate.
- **the fragmented room** — the same room cut into 27 disconnected patches with the
  largest holding 11.8% of the area, mirroring a real capture. Covers the component
  filter and the `--max-filter-loss` tripwire.
- **the GLB room** — textured, exported as GLB. Covers the conversion, the UV and texture
  round trip, `--texture-max`, the cache, and the manifest's provenance.
- **the noisy room** — 5 mm of noise normal to the walls, and deliberately subdivided
  finer than the others so it sits in the vertex density a real capture occupies. Covers
  the smoothing stage: that it runs, holds the extents to a centimetre, keeps ē non-zero
  but inside budget, records its parameters in the manifest, and produces a byte-identical
  mesh on a repeat run.

The first three all pass `--smooth-iterations 0`. They assert exact extents and an exact
0.00% ē, and they were written to isolate the yaw, floor, normal and filter heuristics;
leaving smoothing on would turn five sharp regression tests into blunt ones. The kernel's
own invariants — flat surfaces, sphere-radius retention against a plain Laplacian, UV-seam
welding, determinism, the bounding-box drift bound — run separately and are also reachable
on their own:

```bash
python3 tools/mesh_taubin.py --selftest
```

Every stage of the import is a heuristic, and heuristics fail quietly. On a real scan
there is no ground truth for "did the yaw estimator work", so a regression would look like
a slightly wrong room that still passes the eye test.
