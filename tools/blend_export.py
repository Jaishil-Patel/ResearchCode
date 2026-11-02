#!/usr/bin/env python3
"""Export a hand-cleaned .blend room scan to OBJ + MTL + textures for mesh_import.py.

The pipeline's front door is `simulation/photogrammetry/raw/<room>/`, which wants what a
capture app emits: OBJ+MTL+texture, or GLB. A .blend is neither, so a scan that has been
tidied up by hand in Blender has no way in. This is that way in.

Not run directly -- `blender` runs it:

    blender --background --factory-startup --python tools/blend_export.py -- \
        --input "docs/presentation 2/Demo room.blend" \
        --output simulation/photogrammetry/raw/demo_room/demo_room.obj \
        --exclude Cube

Three things here are load-bearing.

**Packed textures have to be written out first.** A scan .blend almost always carries its
images packed, and a packed image has an empty `filepath`. Blender's OBJ exporter copies
the file named by `filepath`, so with nothing there it silently emits an .mtl with no
`map_Kd` at all -- and a room whose texture did not survive renders as a plausible
featureless grey box rather than as an obvious failure. So each image is saved beside the
.obj and unpacked onto that path before the export runs.

**`forward_axis="Y", up_axis="Z"` disables Blender's OBJ axis conversion**, so the file
carries Blender's own Z-up coordinates unrotated. Blender's default would rotate the room
into Y-up on the way out and mesh_import.py's `--up` would then rotate it again. Export
like this and import with `--up z`. tools/blender_room_visual.py pins the same axes for
the same reason.

**`--exclude` earns its place.** A scan .blend collects scaffolding -- a backdrop box, a
reference cube, a bounding volume drawn while cleaning up. None of it is the room, and all
of it would be fed to the importer as though it were.

Prefer GLB straight from the capture app where there is a choice; this is for meshes that
have been edited, where the .blend is the only copy of the edit.
"""

import os
import sys

import bpy  # noqa: E402 -- only importable inside Blender


def log(msg):
    print("[blender] " + msg)
    sys.stdout.flush()


def parse_args():
    argv = sys.argv
    if "--" not in argv:
        raise SystemExit("expected arguments after '--'")
    argv = argv[argv.index("--") + 1 :]

    import argparse

    parser = argparse.ArgumentParser(prog="blend_export.py")
    parser.add_argument("--input", required=True, help=".blend to export")
    parser.add_argument("--output", required=True, help=".obj to write")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="NAME",
        help="mesh object to leave out; repeatable",
    )
    parser.add_argument(
        "--texture-max",
        type=int,
        default=2048,
        help="downscale texture long edge to this; 0 keeps the original size",
    )
    return parser.parse_args(argv)


def write_textures(out_dir, stem, cap):
    """Save every used image beside the .obj, giving each one a real filepath.

    Saving is enough on its own: it sets `filepath`, which is all the exporter needs to
    find something to copy. Do *not* reach for `Image.unpack()` to finish the job -- it
    ignores the filepath just set and writes the original bytes next to the .blend
    instead, littering whatever directory the scan happens to live in.

    Saved as JPEG because that is what these atlases already are -- the packed data is a
    JPEG stream, so PNG would re-encode a photograph into something several times larger
    for no gain. The cap matters for the same reason it does in mesh_import.py: under
    LIBGL_ALWAYS_SOFTWARE the texture is sampled on the CPU, so its size is a throughput
    knob rather than housekeeping.
    """
    written = 0
    for index, img in enumerate(bpy.data.images):
        if img.size[0] == 0 or img.size[1] == 0:
            continue  # 'Render Result' and friends
        if not img.users:
            continue

        if cap and max(img.size) > cap:
            scale = cap / float(max(img.size))
            size = (max(1, int(round(img.size[0] * scale))),
                    max(1, int(round(img.size[1] * scale))))
            log(f"texture {img.size[0]}x{img.size[1]} -> {size[0]}x{size[1]}")
            img.scale(size[0], size[1])

        target = os.path.join(out_dir, f"{stem}_{index}.jpg")
        img.filepath_raw = target
        img.file_format = "JPEG"
        img.save()
        log(f"wrote {os.path.basename(target)} ({img.size[0]}x{img.size[1]})")
        written += 1
    return written


def main():
    args = parse_args()

    bpy.ops.wm.open_mainfile(filepath=args.input)
    log(f"opened {os.path.basename(args.input)}")

    excluded = set(args.exclude)
    meshes = []
    for o in bpy.context.scene.objects:
        if o.type != "MESH":
            continue
        if o.name in excluded:
            log(f"excluding {o.name} ({len(o.data.polygons):,} faces)")
            continue
        meshes.append(o)

    missing = excluded - {o.name for o in bpy.context.scene.objects}
    if missing:
        raise SystemExit(f"--exclude named objects that are not in the file: {sorted(missing)}")
    if not meshes:
        raise SystemExit("no mesh objects left to export")

    faces = sum(len(o.data.polygons) for o in meshes)
    names = ", ".join(o.name for o in meshes)
    log(f"exporting {len(meshes)} mesh object(s), {faces:,} faces: {names}")

    out_dir = os.path.dirname(args.output) or "."
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.output))[0]
    if write_textures(out_dir, stem, args.texture_max) == 0:
        log("WARNING: no textures were written -- the export will be untextured")

    for o in bpy.context.scene.objects:
        o.select_set(o in meshes)
    bpy.context.view_layer.objects.active = meshes[0]

    bpy.ops.wm.obj_export(
        filepath=args.output,
        export_selected_objects=True,
        export_materials=True,
        export_uv=True,
        export_normals=True,
        # COPY writes the images next to the .obj and makes the .mtl name them by bare
        # filename. Ignition resolves mesh-relative paths, so this is what makes the
        # texture show up in Gazebo rather than a grey surface.
        path_mode="COPY",
        forward_axis="Y",
        up_axis="Z",
    )
    log(f"wrote {args.output}")


if __name__ == "__main__":
    main()
