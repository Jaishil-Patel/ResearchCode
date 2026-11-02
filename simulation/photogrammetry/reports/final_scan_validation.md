# Reconstruction accuracy -- final_scan

Generated 2026-09-27 11:44:39 by `tools/mesh_validate.py`.

Proposal Eq. 4.6, mean absolute percentage error over K = 10 tape-measured
dimensions:

> e-bar = 2.12%  (threshold 4.97%) -- **PASS**

| id | description | tape (m) | mesh (m) | error |
|---|---|---:|---:|---:|
| `room_length` | Long wall interior face to interior face | 5.500 | 5.449 | 0.92% |
| `room_width` | Short wall interior face to interior face | 4.200 | 4.154 | 1.08% |
| `room_height` | Floor to ceiling at the room centre | 3.100 | 3.114 | 0.45% |
| `Box_height` | Floor to Box top surface | 0.710 | 0.694 | 2.26% |
| `Couch_length` | Couch long edge (outer faces of the arms at arm height) | 1.330 | 1.290 | 3.03% |
| `Couch_height` | Floor to Couch top surface (backrest) | 0.810 | 0.865 | 6.85% |
| `Box2_height` | Floor to Box2 top surface | 0.840 | 0.841 | 0.13% |
| `Bag_height` | Floor to Bag top surface | 0.580 | 0.613 | 5.70% |
| `goal_height` | Floor to goal (red-topped pole) top | 1.000 | 1.000 | 0.01% |
| `green_pole_height` | Floor to green-topped pole top | 0.700 | 0.695 | 0.72% |

## Import parameters

- Scale factor applied at import: `1.0`
- Up axis: `y`, origin mode: `floor-min-corner`
- Normals flipped at import: `True` (0.8843 of shell area inward)
- Visual mesh: 50,000 faces via `blender`; collision 5,000 faces
- Mesh measured: `/home/jaishil/Research/pepper-sim2real/ros2_ws/src/pepper_room_description/models/room_photogrammetry/meshes/visual/room.obj`

## Reading this

Eq. 4.6 weights every dimension equally, so a single badly-picked point moves e-bar as
much as a genuinely wrong wall. Before concluding the reconstruction is at fault, look
down the error column: a uniform error across all K suggests a scale factor, which is
fixed by re-running the import with `--scale`. Errors concentrated on one axis suggest
the up-axis or yaw estimate is wrong. Errors on scattered individual rows usually mean
the point picks, not the mesh.
