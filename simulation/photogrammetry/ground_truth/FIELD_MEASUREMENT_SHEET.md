# Field measurement sheet

Print this, or open it on a phone, and fill it in with the tape measure in hand. It is
the human-readable version of `TEMPLATE.csv` and `TEMPLATE_spec.yaml` — transcribe from
here into those two machine-readable files back at the desk (see the bottom of this
sheet). Filling in this sheet correctly is what `tools/mesh_validate.py`'s accuracy gate
and the entire standard (non-photogrammetric) branch are built from — see "why measurements
matter" below if that's not obvious.

**Do this on the same visit as the photo capture. Not a second trip.**

---

## 0. Pick your origin corner — do this first, before anything else

Stand in one corner of the room. This is world (0, 0). Every measurement on this sheet is
taken **from that corner**, along the two walls that meet there.

> **My origin corner is:** _______________________________________________
> (describe it well enough to stand in the exact same spot on a second visit — e.g.
> "the corner where the door wall meets the window wall")

This has to be the same corner `tools/mesh_import.py` uses for the scan (it defaults to
the floor's lowest corner) and the same corner Task 6's real-to-sim pose mapping goes
through. Get it wrong and both branches silently describe two different rooms — nothing
errors, the comparison just stops meaning anything.

---

## 1. Room shell

| Dimension | Value (m) |
|---|---|
| Room **length** — along the long wall from your origin corner (this is the x axis) | |
| Room **width** — along the short wall from your origin corner (this is the y axis) | |
| **Height** — floor to ceiling, measured at the room centre | |

---

## 2. Doors — one row per door Pepper must pass through

`wall` is whichever of the room's four walls the door is set into, named by which shell
edge it's on: `x_min` (the wall at x=0, through your origin corner), `x_max` (the far
long wall), `y_min` (through your origin corner, the other direction), `y_max` (the far
short wall). `start`/`end` are distances **along that wall**, measured from your origin
corner, to each edge of the door frame.

| Door | width (m) | height, floor to lintel (m) | wall (x_min / x_max / y_min / y_max) | start (m) | end (m) |
|---|---|---|---|---|---|
| s| | | | | |
| | | | | | |

---

## 3. Windows — only if low enough to matter (roughly 0.4–1.2 m off the floor)

Same layout as doors, plus how high off the floor the opening starts and ends.

| Window | width (m) | bottom height (m) | top height (m) | wall | start (m) | end (m) |
|---|---|---|---|---|---|---|
| | | | | | | |

---

## 4. Furniture — every piece Pepper must navigate around

This is a **bounding box**, not the real shape — the standard branch deliberately
represents furniture as a box; the photogrammetric scan is what captures the real
surface. Don't try to be more precise than a box.

`centre x, y` is the middle of the footprint, measured from your origin corner.
`yaw` only matters if the piece sits at an angle to the walls — leave blank if it's
square to the room.

| Item | length, x (m) | depth, y (m) | height, z (m) | centre x (m) | centre y (m) | yaw (deg, if angled) |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

---

## 5. Extra measurements, to reach K ≥ 8 total

Count what you have so far: room shell (3) + each door (2 each) + each window (2 each) +
... You need **at least 8 measurements total, spread across all three axes** — not just
floor-plan distances, because a gate built only from floor-plane numbers can't catch a
vertical scale error, and a vertical scale error is exactly what would put the depth
camera's simulated horizon somewhere the real one never is.

If you're short, add point-to-point measurements here — a diagonal across the room, a
second furniture edge, anything repeatable. **At least one more vertical (height)
measurement** is the cheapest way to strengthen the gate if you're light on the z axis.

| What | Value (m) | Notes (enough detail to repeat it on a second visit) |
|---|---|---|
| | | |
| | | |

---

## 6. Capture metadata — fill in the same day, not from memory later

If the reconstruction fails the accuracy gate, this is the only record of what to change
on the re-shoot.

| | |
|---|---|
| Room name / slug (becomes your filenames, e.g. `lab_a`) | |
| Date | |
| App / device | |
| LiDAR or depth sensor used? (decides if the capture is already metric) | |
| Number of images / duration | |
| Heights photographed (need floor level, ~1.2 m, and above head — all three) | |
| Lighting | |
| Anything that moved mid-capture? | |

---

## Why these measurements matter (short version)

Two reasons, not one:

1. **They check the scan isn't lying.** A phone photogrammetry scan reconstructs shape
   accurately but estimates absolute scale, and that estimate can be subtly wrong —
   glossy surfaces, blank walls, and uneven lighting all distort it in ways that look
   fine to the eye. These measurements are compared against the same dimensions on the
   mesh (Eq. 4.6); if the error is too large, the reconstruction is redone before it's
   ever used for training.
2. **They're the only source of geometry for the standard branch.** The standard branch
   is a hand-built room made of primitive boxes — not derived from the scan at all — and
   the whole comparison only works if it's a *faithful* model of the same real room, not
   an eyeballed one. Otherwise a difference in policy performance could just mean "the
   standard branch happened to be a worse room," not "photogrammetry helps" — and the
   experiment's independent variable stops being isolated.

## What happens to this sheet next

1. Copy `TEMPLATE.csv` → `<slug>.csv` in this same directory. Sections 1, 2 and 3 above
   map onto its `extent`/`points` rows; section 5 fills in the rest of the K ≥ 8.
2. Copy `TEMPLATE_spec.yaml` → `<slug>_spec.yaml`. Section 1 → its `room:` block,
   section 2/3 → `openings:`, section 4 → `furniture:`.
3. Run the pipeline in `docs/photogrammetry_runbook.md`: `mesh_import.py` →
   `mesh_validate.py` (the gate) → `gen_world.py --branch both --check`.
