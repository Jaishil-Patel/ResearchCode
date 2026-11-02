# Raw captures (not in git)

Drop each capture in its own subdirectory named after the room, e.g. `raw/lab_a/`:

```
raw/lab_a/
  lab_a.obj          <- preferred input
  lab_a.mtl
  lab_a_0.png ...    <- texture atlases referenced by the .mtl
  lab_a.glb          <- also accepted; converted to OBJ on import
```

**OBJ and GLB are both accepted.** OBJ is preferred and is what the pipeline actually
consumes: Ignition Fortress's OGRE2 loader handles glTF unevenly, so the mesh that ends
up in the model is always OBJ. That is a constraint on the output, though, not on what
your phone is willing to export — several capture apps emit GLB and nothing else. When
the input is a GLB, `mesh_import.py` converts it to OBJ+MTL+texture into
`simulation/photogrammetry/work/<room>/` before anything else touches it, caching the
result against the source hash. Export both formats when the app offers both.

These files are gitignored -- they are large and reproducible only by revisiting the
room, so back them up somewhere that is not this repository. Record the capture
parameters (app, device, number of images, lighting, passes, whether depth/LiDAR was
used) below when you capture, not afterwards from memory: if the reconstruction fails
the accuracy gate, those parameters are the only evidence of what to change on the
re-shoot.

---

## Captures

### `test_scan` — 13 Aug 2026

A throwaway capture taken to prove the import path works. **Not a candidate for the
experiment**: no tape measurements were taken, so it has never been through the Eq. 4.6
gate, and its metric scale is unverified.

| | |
|---|---|
| Source | `2026_08_13.glb` (2.3 MB, GLB only — no OBJ export) |
| App / device | not recorded |
| LiDAR or depth used? | unknown — dimensions look metric, but unverified |
| Geometry | 26,784 faces, 16,672 verts, 147 disconnected components |
| Texture | one 4096×4096 PBR atlas, full UVs |
| Extents as captured | 3.81 × 2.97 × 4.30 m, y-up, centred on the origin |
| After import | 3.80 × 4.30 m floor, 2.97 m ceiling, yaw −0.67°, scale 1.0 |
| Surface area | 53.8 m² against ~81 m² for a closed room this size |

Two things to know about it:

- **It is a partial capture.** Two thirds of the expected surface area is present, so
  there are holes — some walls and floor are simply missing. Fine for a load test,
  useless for training.
- **It is what broke the importer.** The 147-component fragmentation defeated the debris
  filter's "largest component is the room shell" assumption, and the GLB-only export was
  rejected outright. Both are fixed and regression-tested; see
  `docs/photogrammetry_runbook.md`, "What the first real scan changed".

Verified: converts, cleans (98.0% of surface area kept), aligns, builds both meshes,
loads in Ignition with no errors, and `/scan` returns varying true ranges from inside the
room rather than the pinned-at-minimum garbage that the wrong render path produces.

### `demo_room` — 23 Aug 2026

A corridor, captured and then cleaned by hand in Blender. Used to prove the whole path
from a scan to a running Ignition world, and recorded for the second progress talk.
**Not a candidate for the experiment**: no tape measurements were taken, so it has never
been through the Eq. 4.6 gate and its metric scale is unverified.

| | |
|---|---|
| Source | `simulation/models/Demo room.blend` (6.6 MB, Blender 4.5.3 LTS) |
| App / device | not recorded |
| LiDAR or depth used? | unknown |
| Texture | two packed JPEG atlases, 4096x4096 and 4096x1344, capped to 2048 on export |
| Geometry exported | 27,262 faces, one mesh object |
| Extents as captured | 9.27 x 14.24 x 6.20 m, z-up |
| After import | 9.03 x 13.94 m floor, 6.16 m ceiling, yaw -0.72 deg, scale 1.0 |
| Visual / collision | 27,255 / 4,999 faces |

Exported with `tools/blend_export.py`, then the standard three commands with two
non-default flags:

```bash
blender --background --factory-startup --python tools/blend_export.py -- \
    --input "simulation/models/Demo room.blend" \
    --output simulation/photogrammetry/raw/demo_room/demo_room.obj \
    --exclude Cube --exclude Mesh_1
python3 tools/mesh_import.py --room demo_room --up z --origin-mode floor-centre
python3 tools/gen_world.py --room demo_room --branch photogrammetry
```

**`--exclude` is the important part.** The .blend holds three mesh objects, and only one
of them is the room:

| Object | Faces | What it is |
|---|---|---|
| `Mesh_0` | 27,262 | the corridor -- floor, walls, doorways |
| `Mesh_1` | 23,348 | floating debris, no floor at all |
| `Cube` | 5 | an open 4.8 x 10 x 6 m box, scaffolding from the cleanup |

Exporting all three and letting Blender join them puts the debris *inside* the room as a
second overlapping surface. It still imports, still passes the debris filter, and still
loads in Ignition -- the mesh is simply wrong, and the only way to notice is to look at
it. Render each object separately before trusting a .blend you did not author.

`--up z` because `blend_export.py` writes Blender's own Z-up coordinates unrotated;
`--origin-mode floor-centre` because Pepper spawns at the world origin and the default
`floor-min-corner` puts that origin in a corner.

Three things to know about it:

- **It is a corridor, not a room**, and a partial one: 53 m^2 of floor-ish surface, with
  holes in the walls and no real ceiling. The 6.16 m "ceiling" is floating geometry above,
  not a surface.
- **The floor is level but bumpy.** Its mean normal is within 0.56 deg of vertical, but
  per-face tilt has a median of 8.2 deg and reaches 32 deg. That roughness is in the scan,
  not in the decimation -- the 27k visual mesh and the 5k collision mesh have the same
  normal spread -- so a finer collision mesh tracks the bumps rather than smoothing them.
- **Pepper spawns below the floor.** Near the origin the collision surface sits at
  z = 0.118 .. 0.154, but the robot is created at z = 0, so she starts inside it and DART
  ejects her. Spawn her above the surface, or stand her on a flat pad.

Verified: exports, imports, aligns, builds both meshes, loads in `ign gazebo` with no
errors or warnings, renders textured, and holds Pepper upright (roll and pitch under
1 deg) at RTF 0.94 under `LIBGL_ALWAYS_SOFTWARE=1`.
