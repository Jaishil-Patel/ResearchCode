"""Floor height under a start pose, per branch -- so a teleport lands ON the floor.

Episode resets teleport Pepper to (x, y, z). The original code used z = 0.0 everywhere,
on the assumption that a floor-aligned room has its floor at 0. That holds for the
standard branch (a flat box with its top at exactly z = 0) and does NOT hold for a
reconstruction: final_scan's collision floor runs from about -0.01 m to +0.11 m across
the route starts (it rises ~1.9 cm/m along x). A teleport to z = 0 there puts the base up
to ~11 cm inside a one-sided floor mesh, and the first seconds of every photogrammetric
episode are the solver pushing her back out -- a branch-specific artefact in exactly the
comparison the experiment exists to make. Measured 27 Sep 2026: spawned at (5.07, 3.15),
z = 0, the model sat at z = 0.011-0.023 m under a collision floor at 0.114 m.

The rule, identical for both branches: drop from DROP_M above the local floor.
  * A world whose model has a collision MESH (the photogrammetric room) -> ray-cast it.
  * A world without one (the standard room, box primitives) -> its floor top is z = 0.
"""
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODELS = REPO / "ros2_ws" / "src" / "pepper_room_description" / "models"

# Small enough that the landing does not disturb the settle, large enough to clear the
# scan's floor noise. Applied to both branches so the drop itself is not a difference.
DROP_M = 0.02
# Cast from well above the floor but below any furniture top a start pose could sit
# under; start poses are planned in free space, so the first hit below this is floor.
_CAST_FROM_Z = 0.6


class FloorHeight:
    def __init__(self, world_name):
        self.world_name = world_name
        self.mesh = None
        coll = MODELS / world_name / "meshes" / "collision"
        found = (sorted(coll.glob("*.stl")) + sorted(coll.glob("*.obj"))) if coll.is_dir() else []
        if found:
            import trimesh
            self.mesh = trimesh.load(str(found[0]), force="mesh")
            self.source = str(found[0])
        else:
            self.source = "flat floor at z=0 (no collision mesh for this world)"

    def floor_z(self, x, y):
        if self.mesh is None:
            return 0.0
        import numpy as np
        locs, _, _ = self.mesh.ray.intersects_location(
            np.array([[x, y, _CAST_FROM_Z]]), np.array([[0.0, 0.0, -1.0]]))
        if len(locs) == 0:
            # A hole in the scan under the start pose. Fall back to the old behaviour
            # rather than dropping her into the void, and say so.
            print(f"[floor_height] no floor under ({x:.2f}, {y:.2f}) in {self.world_name}; using 0.0")
            return 0.0
        return float(locs[:, 2].max())

    def spawn_z(self, x, y):
        return self.floor_z(x, y) + DROP_M
