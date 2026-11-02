#!/usr/bin/env python3
"""Blender-side half of tools/mesh_import.py: UV-preserving decimation of the room mesh.

Not run directly. `mesh_import.py` invokes it as

    blender --background --factory-startup --python tools/blender_room_visual.py -- \
        --input raw/lab_a/lab_a.obj --output .../meshes/visual/room.obj \
        --faces 50000 --matrix "<16 floats, row major>"

It exists because trimesh's quadric decimation throws UVs away. On a photogrammetry mesh
the texture carries the reconstruction -- an untextured 50k-triangle room renders a
plausible but featureless depth image, which is worse than an obvious failure because
training would proceed. Blender's Decimate modifier interpolates UVs through the
collapse, so the texture survives.

The transform is passed in rather than recomputed here so that the visual and collision
meshes are guaranteed to land in the same world frame. They are produced by different
libraries; agreeing on the frame by construction is the only way to be sure they agree.
"""

import sys

import bpy  # noqa: E402 -- only importable inside Blender

# --factory-startup means no add-ons and no user prefs, so behaviour does not depend on
# whatever this machine's Blender happens to be configured with.


def log(msg):
    print("[blender] " + msg)
    sys.stdout.flush()


def parse_args():
    argv = sys.argv
    if "--" not in argv:
        raise SystemExit("expected arguments after '--'")
    argv = argv[argv.index("--") + 1 :]

    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--faces", type=int, required=True)
    p.add_argument("--matrix", required=True, help="16 floats, row major")
    p.add_argument("--crop-min", help="'x y z' lower corner, post-transform")
    p.add_argument("--crop-max", help="'x y z' upper corner, post-transform")
    p.add_argument("--flip-normals", action="store_true")
    p.add_argument("--kernel-dir", help="directory holding mesh_taubin.py / mesh_holes.py")
    p.add_argument("--fill-holes", action="store_true")
    p.add_argument("--fill-max-area", type=float, default=0.20)
    p.add_argument("--output-unfilled", help="also export the mesh without hole filling")
    p.add_argument("--smooth-iterations", type=int, default=0)
    p.add_argument("--smooth-lambda", type=float, default=0.5)
    p.add_argument("--smooth-mu", type=float, default=-0.53)
    p.add_argument("--pin-boundary", action="store_true")
    return p.parse_args(argv)


def crop(obj, lo, hi):
    """Delete faces whose centre falls outside the box.

    Mirrors the bounding box mesh_import.py's cleaning stage settled on, so the visual
    and collision meshes describe the same room. Done with bmesh rather than the Bisect
    operator because bmesh keeps the UV layers intact on the faces that survive, and the
    UVs are the entire reason this script exists.
    """
    import bmesh

    bm = bmesh.new()
    bm.from_mesh(obj.data)
    doomed = [
        f
        for f in bm.faces
        if not all(lo[i] <= f.calc_center_median()[i] <= hi[i] for i in range(3))
    ]
    if doomed:
        bmesh.ops.delete(bm, geom=doomed, context="FACES")
        bm.to_mesh(obj.data)
        obj.data.update()
    bm.free()
    return len(doomed)


def smooth(obj, args, emit_json=True):
    """Taubin-smooth the mesh with the same kernel mesh_import.py used on the collision
    mesh, in the same process order: after the transform and crop, before decimation.

    Imported rather than reimplemented. The two meshes are produced by different
    libraries in different processes, and blender_room_visual.py's whole reason to exist
    is that they must still describe the same room -- matching parameters would not be
    enough, it has to be the same arithmetic.

    Blender's vertices are already welded by position on import and its UVs live on loops
    rather than on vertices, so moving a vertex cannot disturb a texture coordinate. The
    kernel welds again anyway, which is a no-op here and the thing that saves the OBJ leg
    where 79% of the vertices in a real capture are UV-seam duplicates.
    """
    if args.smooth_iterations <= 0:
        return None

    import json  # noqa: F401 -- used below when emit_json

    sys.path.insert(0, args.kernel_dir)
    try:
        import numpy as np

        import mesh_taubin
    except ImportError as exc:
        raise SystemExit(
            f"cannot import the smoothing kernel from {args.kernel_dir!r} ({exc}).\n"
            f"mesh_taubin.py needs numpy, which Blender bundles. Check with:\n"
            f"    blender --background --python-expr 'import numpy'"
        )

    me = obj.data
    nv, ne = len(me.vertices), len(me.edges)
    V = np.empty(nv * 3, dtype=np.float64)
    me.vertices.foreach_get("co", V)
    E = np.empty(ne * 2, dtype=np.int32)
    # Blender maintains an edge list for arbitrary polygon topology, so connectivity does
    # not have to be re-derived from faces -- and this runs before triangulation, where
    # the faces are not guaranteed to be triangles.
    me.edges.foreach_get("vertices", E)

    pin = None
    if args.pin_boundary:
        import bmesh

        bm = bmesh.new()
        bm.from_mesh(me)
        pin = np.zeros(nv, dtype=bool)
        for e in bm.edges:
            if e.is_boundary:
                pin[e.verts[0].index] = True
                pin[e.verts[1].index] = True
        bm.free()

    V = V.reshape(-1, 3)
    before = V.max(axis=0) - V.min(axis=0)
    out, stats = mesh_taubin.smooth_vertices(
        V,
        E.reshape(-1, 2).astype(np.int64),
        iterations=args.smooth_iterations,
        lam=args.smooth_lambda,
        mu=args.smooth_mu,
        pin_mask_override=pin,
    )
    after = out.max(axis=0) - out.min(axis=0)
    stats["aabb_before"] = [round(float(v), 4) for v in before]
    stats["aabb_after"] = [round(float(v), 4) for v in after]
    stats["aabb_drift_pct"] = [
        round(float(v), 4) for v in (after - before) / np.maximum(before, 1e-9) * 100.0
    ]

    me.vertices.foreach_set("co", out.ravel())
    me.update()

    log(
        f"Taubin N={stats['iterations']} lambda={stats['lambda']:g} "
        f"mu={stats['mu']:g}: moved {stats['mean_displacement_m'] * 1000:.1f} mm mean, "
        f"{stats['max_displacement_m'] * 1000:.1f} mm max, bbox drift "
        + ", ".join(f"{d:+.2f}%" for d in stats["aabb_drift_pct"])
    )
    # A machine-readable copy, so mesh_import.py can put the visual leg's own figures in
    # the manifest instead of assuming they match the collision leg's.
    if emit_json:
        print("[blender-json] " + json.dumps({"smoothing": stats}))
        sys.stdout.flush()
    return stats


def fill(obj, args):
    """Close boundary loops at or under --fill-max-area, and tag what was invented.

    Which loops count as holes is decided by tools/mesh_holes.py, the same module the
    collision leg uses, so the two legs cannot drift apart about that. How the loop is then
    closed differs by necessity: here bmesh spans the rim with a single n-gon, which the
    decimator later triangulates while interpolating UVs across it; the trimesh leg has no
    n-gons and fans the rim to a centre vertex instead. Same loops, different realised area,
    which is why the manifest records the two legs separately.

    Synthetic faces go into their own material slot rather than a vertex group or a custom
    attribute. A material slot survives the Decimate modifier and the OBJ export without any
    help, so the invented surface is visible in Blender, in Gazebo and in any other viewer,
    and a provenance render costs nothing. A vertex group survives neither.
    """
    if not args.fill_holes:
        return None

    import json

    import bmesh
    import numpy as np

    sys.path.insert(0, args.kernel_dir)
    import mesh_holes

    me = obj.data

    bm = bmesh.new()
    bm.from_mesh(me)

    # Weld first, and this is the whole ballgame. A textured mesh splits a vertex at every
    # atlas cut, so the triangles either side of a seam look unconnected and every seam
    # reads as an open boundary: 22,113 boundary edges in 715 loops on this capture,
    # enclosing more area than the room has. Welded at 1 um it is 1,478 edges in 4 loops,
    # and those four are the real holes.
    #
    # Safe for the texture because Blender keeps UVs on loops rather than on vertices, so
    # merging two coincident vertices cannot merge two texture coordinates. That is not
    # true of the OBJ on disk, which is why the file has the duplicates in the first place.
    before_verts = len(bm.verts)
    bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=1e-6)
    bmesh.ops.dissolve_degenerate(bm, dist=1e-5, edges=list(bm.edges))
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    # remove_doubles renumbers nothing on its own: v.index is stale straight afterwards, so
    # the face array built below would reference vertices that no longer exist and the loop
    # trace would match no real edge. It fails by filling nothing, quietly.
    bm.verts.index_update()
    bm.edges.index_update()
    bm.faces.index_update()
    log(f"welded {before_verts:,} -> {len(bm.verts):,} vertices before tracing boundaries")

    before = sum(1 for e in bm.edges if e.is_boundary)
    V = np.array([v.co[:] for v in bm.verts], dtype=np.float64)
    F = np.array([[v.index for v in f.verts] for f in bm.faces if len(f.verts) == 3],
                 dtype=np.int64)
    if len(F) == 0:
        log("fill: no triangles to analyse, skipped")
        bm.free()
        return None

    # classify() welds internally, and weld_index sorts the unique positions, so the loop
    # indices it returns are a PERMUTATION of bmesh's numbering rather than the same
    # numbering -- even though this mesh is already welded. Every lookup has to go through
    # `inverse` or it matches nothing: the symptom is a correct loop count and an empty
    # edge selection.
    inverse, _ = mesh_holes.weld_faces(V, F)
    keep, leave, paths = mesh_holes.classify(V, F, max_area=args.fill_max_area)
    wanted = set()
    for loop in keep:
        for i in range(len(loop)):
            a_, b_ = loop[i], loop[(i + 1) % len(loop)]
            wanted.add((min(a_, b_), max(a_, b_)))
    def welded_key(e):
        a_, b_ = int(inverse[e.verts[0].index]), int(inverse[e.verts[1].index])
        return (min(a_, b_), max(a_, b_))

    edges = [e for e in bm.edges if e.is_boundary and welded_key(e) in wanted]

    log(f"  {before:,} boundary edges, {len(keep):,} loops to fill "
        f"({len(leave):,} above the cap, {len(paths):,} not closed), "
        f"{len(edges):,} edges selected")
    res = bmesh.ops.holes_fill(bm, edges=edges, sides=0) if edges else {"faces": []}
    new = list(res.get("faces", []))
    # holes_fill declines some loops outright and says nothing about it. triangle_fill
    # picks up the ones it dropped; without this, a loop is silently left open.
    if edges and not new:
        res = bmesh.ops.triangle_fill(bm, edges=edges, use_beauty=True, use_dissolve=False)
        new = [g for g in res.get("geom", []) if isinstance(g, bmesh.types.BMFace)]
        if new:
            log(f"  holes_fill declined; triangle_fill closed {len(new):,} faces")

    # An n-gon spanning a crumpled 572-vertex rim is meaningless: its area is mis-reported
    # and the exporter would tessellate it arbitrarily. Triangulate on our terms instead.
    if new:
        tri = bmesh.ops.triangulate(bm, faces=new, ngon_method="BEAUTY")
        new = list(tri.get("faces", new))

    bm.normal_update()

    # holes_fill does not take a side: it winds each patch whichever way it likes, and a
    # patch wound the other way is back-face culled -- so the hole it just closed still
    # renders as a hole, the one outcome this stage exists to prevent. Measured on this
    # capture, only 27 of 241 patches came out agreeing with the surface around them.
    #
    # Agreement is with the patch's own rim, not with the room. The global orientation was
    # decided by mesh_import.fix_normals() and handed here as --flip-normals; a global
    # recalculation would overrule it. It is also the only well-posed question, since a
    # patch in a ceiling corner has no meaningful "towards the middle of the room".
    flipped = 0
    for f in new:
        ref = [g.normal for e in f.edges for g in e.link_faces if g is not f]
        if not ref:
            continue
        avg = ref[0].copy()
        for v in ref[1:]:
            avg += v
        if f.normal.dot(avg) < 0.0:
            f.normal_flip()
            flipped += 1
    if flipped:
        log(f"  re-wound {flipped:,} of {len(new):,} patches to match their surroundings")

    syn_area = float(sum(f.calc_area() for f in new))
    total = float(sum(f.calc_area() for f in bm.faces))
    after = sum(1 for e in bm.edges if e.is_boundary)
    bm.faces.index_update()
    new_idx = set(f.index for f in new)
    bm.to_mesh(me)
    bm.free()
    me.update()

    if not new_idx:
        log("fill: nothing filled")
        return None

    mat = bpy.data.materials.new("synthetic")
    mat.diffuse_color = (0.85, 0.15, 0.15, 1.0)
    me.materials.append(mat)
    slot = len(me.materials) - 1
    for poly in me.polygons:
        if poly.index in new_idx:
            poly.material_index = slot

    stats = {
        "max_area_m2": float(args.fill_max_area),
        "loops_filled": len(keep),
        "loops_left_closed": len(leave),
        "loops_open_paths": len(paths),
        "faces_added": len(new_idx),
        "synthetic_area_m2": round(syn_area, 6),
        "synthetic_fraction": round(syn_area / max(total, 1e-9), 6),
        "boundary_edges_before": before,
        "boundary_edges_after": after,
        "material_slot": "synthetic",
        "patches_rewound": flipped,
    }
    log(f"filled {len(keep):,} loops at or under {args.fill_max_area:g} m2: "
        f"{len(new_idx):,} faces, {syn_area:.2f} m2 synthetic "
        f"({stats['synthetic_fraction'] * 100:.1f}% of surface), "
        f"boundary edges {before:,} -> {after:,}")
    print("[blender-json] " + json.dumps({"hole_filling": stats}))
    sys.stdout.flush()
    return stats


def transfer_uvs(obj, donor):
    """Give the synthetic faces the nearest real surface's texture coordinates.

    Without this they carry whatever the fill left behind -- in practice a collapsed UV that
    samples one texel, so the patch renders as a flat colour and reads instantly as a hole
    that someone puttied over. Nearest-polygon interpolation continues the surrounding wall
    across the patch instead. It is a smear, not a reconstruction, and at Pepper's 1.2 m
    viewing distance that is the difference between a wall and an obvious defect; close up
    it is still a smear.
    """
    bpy.ops.object.select_all(action="DESELECT")
    donor.select_set(True)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    try:
        bpy.ops.object.data_transfer(
            use_reverse_transfer=True,
            data_type="UV",
            loop_mapping="POLYINTERP_NEAREST",
            use_create=False,
        )
        log("transferred UVs onto the filled patches from the unfilled copy")
    except RuntimeError as exc:
        log(f"UV transfer failed ({exc}); patches will be untextured")
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def decimate(obj, budget):
    n = len(obj.data.polygons)
    if n > budget:
        mod = obj.modifiers.new(name="decimate", type="DECIMATE")
        mod.decimate_type = "COLLAPSE"
        mod.ratio = budget / float(n)
        mod.use_collapse_triangulate = True
        bpy.ops.object.modifier_apply(modifier=mod.name)
        log(f"decimated {n:,} -> {len(obj.data.polygons):,} faces (UVs preserved)")
    else:
        log(f"{n:,} faces already within the {budget:,} budget")


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_obj(path):
    """Import with axis conversion disabled.

    Blender's OBJ importer defaults to forward=-Z / up=Y, i.e. it silently rotates Y-up
    OBJ files into its own Z-up world. mesh_import.py has already decided what the up
    axis is and baked that into the matrix, so a second conversion here would apply the
    rotation twice and lay the room on its side.
    """
    if hasattr(bpy.ops.wm, "obj_import"):  # Blender 3.3+
        bpy.ops.wm.obj_import(filepath=path, forward_axis="Y", up_axis="Z")
    else:  # Blender 2.8 - 3.2
        bpy.ops.import_scene.obj(filepath=path, axis_forward="Y", axis_up="Z")


def export_obj(path):
    if hasattr(bpy.ops.wm, "obj_export"):
        bpy.ops.wm.obj_export(
            filepath=path,
            export_selected_objects=True,
            export_materials=True,
            export_uv=True,
            export_normals=True,
            # COPY writes the texture images next to the .obj and makes the .mtl refer
            # to them by bare filename. Ignition resolves mesh-relative paths, so this
            # is what makes the texture show up in Gazebo rather than a grey surface.
            path_mode="COPY",
            forward_axis="Y",
            up_axis="Z",
        )
    else:
        bpy.ops.export_scene.obj(
            filepath=path,
            use_selection=True,
            use_materials=True,
            use_uvs=True,
            use_normals=True,
            path_mode="COPY",
            axis_forward="Y",
            axis_up="Z",
        )


def join_meshes():
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise SystemExit("no mesh objects imported")
    for o in bpy.context.scene.objects:
        o.select_set(o.type == "MESH")
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
        log(f"joined {len(meshes)} mesh objects")
    return bpy.context.view_layer.objects.active


def main():
    args = parse_args()
    from mathutils import Matrix

    vals = [float(v) for v in args.matrix.split()]
    if len(vals) != 16:
        raise SystemExit(f"--matrix wants 16 floats, got {len(vals)}")
    M = Matrix([vals[0:4], vals[4:8], vals[8:12], vals[12:16]])

    clear_scene()
    import_obj(args.input)
    obj = join_meshes()
    log(f"imported {len(obj.data.polygons):,} faces")

    obj.matrix_world = M @ obj.matrix_world
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    log("applied world transform from mesh_import.py")

    if args.crop_min and args.crop_max:
        lo = [float(v) for v in args.crop_min.split()]
        hi = [float(v) for v in args.crop_max.split()]
        n = crop(obj, lo, hi)
        log(f"cropped {n:,} faces outside the room bounding box")

    if args.flip_normals:
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.flip_normals()
        bpy.ops.object.mode_set(mode="OBJECT")
        log("flipped normals")

    def solo(o):
        bpy.ops.object.select_all(action="DESELECT")
        o.select_set(True)
        bpy.context.view_layer.objects.active = o

    def duplicate(o, name):
        d = o.copy()
        d.data = o.data.copy()
        d.name = name
        bpy.context.scene.collection.objects.link(d)
        return d

    # Both copies are taken here, before anything is filled: the donor supplies untouched
    # UVs for the patches, and the unfilled branch is what the accuracy gate measures, so
    # that e-bar keeps scoring the reconstruction rather than scoring the interpolation.
    donor = duplicate(obj, "UVDONOR") if args.fill_holes else None
    unfilled = (
        duplicate(obj, "UNFILLED")
        if args.fill_holes and args.output_unfilled
        else None
    )

    solo(obj)
    fill(obj, args)
    if donor is not None:
        transfer_uvs(obj, donor)
        bpy.data.objects.remove(donor, do_unlink=True)
    solo(obj)
    smooth(obj, args)
    decimate(obj, args.faces)
    solo(obj)
    export_obj(args.output)
    log(f"exported {args.output}")

    if unfilled is not None:
        # Same smoothing and the same face budget, so the only difference between the two
        # files is the fill itself.
        solo(unfilled)
        smooth(unfilled, args, emit_json=False)
        decimate(unfilled, args.faces)
        solo(unfilled)
        export_obj(args.output_unfilled)
        log(f"exported {args.output_unfilled} (unfilled, for the gate)")
        bpy.data.objects.remove(unfilled, do_unlink=True)


main()
