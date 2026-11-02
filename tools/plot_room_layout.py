#!/usr/bin/env python3
"""Printable plan views of the final_scan room for the real-robot sessions.

    python3 tools/plot_room_layout.py            # -> docs/room_layout_*.png + room_layouts.pdf

Two sheets: the normal layout with the 20 route-set-v2 starts, and the robustness layout
(Sub-question 3) with the moved pieces and its 10 starts. Everything is read from the
room specs and the cached route sets, so the pictures cannot drift from what the routes
are scored against -- regenerate after any change to either.
"""

import json
import math
import textwrap
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle  # noqa: E402
import yaml                                          # noqa: E402

REPO = Path(__file__).resolve().parents[1]
GT = REPO / "simulation" / "photogrammetry" / "ground_truth"
CACHE = REPO / "rl" / "cache"
DOCS = REPO / "docs"

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
FURN_FILL, FURN_EDGE = "#e4e3dc", "#52514e"
START = "#2a78d6"        # categorical slot 1
MOVED = "#eb6834"        # categorical slot 2
GOAL = "#e34948"         # the red-topped pole
ROBOT_R = 0.24           # Pepper's base radius, for the placement footprint


def furniture(spec):
    for f in spec.get("furniture", []):
        if f["name"] in ("goal", "Door_leaf_closed"):
            continue                      # goal drawn separately; the leaf is in the wall
        yield f


def draw_piece(ax, f, face, edge, ls="-", lw=1.0, z=2, alpha=1.0):
    (x, y), size = f["pose"][:2], f["size"]
    if f.get("shape") == "cylinder":
        ax.add_patch(Circle((x, y), size[0] / 2, fc=face, ec=edge, lw=lw, ls=ls, zorder=z,
                            alpha=alpha))
    else:
        w, h = size[0], size[1]
        ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, fc=face, ec=edge, lw=lw, ls=ls,
                               zorder=z, alpha=alpha))


def label_piece(ax, f, text=None, color=INK2, dy=0.0):
    x, y = f["pose"][:2]
    small = f.get("shape") == "cylinder" or max(f["size"][:2]) < 0.45
    ax.text(x + (0.12 if small else 0), y + dy, text or f["name"].replace("_", " "),
            ha="left" if small else "center", va="center", fontsize=7.5, color=color,
            zorder=6)


def room_frame(ax, spec, title):
    r = spec["room"]
    L, W = r["length"], r["width"]
    ax.set_xlim(-0.45, L + 0.25)
    ax.set_ylim(-0.45, W + 0.3)
    ax.set_aspect("equal")
    for v in [i * 0.5 for i in range(int(L / 0.5) + 1)]:
        ax.axvline(v, color="#ecebe6", lw=0.6, zorder=0)
    for v in [i * 0.5 for i in range(int(W / 0.5) + 1)]:
        ax.axhline(v, color="#ecebe6", lw=0.6, zorder=0)
    ax.set_xticks([i * 0.5 for i in range(int(L / 0.5) + 1)])
    ax.set_yticks([i * 0.5 for i in range(int(W / 0.5) + 1)])
    ax.tick_params(labelsize=7, colors=INK2, length=2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("x (m) — along the 5.5 m wall", fontsize=8, color=INK2)
    ax.set_ylabel("y (m) — along the 4.2 m wall", fontsize=8, color=INK2)
    ax.set_title(title, fontsize=11, color=INK, loc="left", pad=8)

    # Walls, leaving the door gap on the x = 0 wall.
    door = next(o for o in spec.get("openings", []) if o["wall"] == "x_min")
    kw = dict(color=INK, lw=2.4, solid_capstyle="butt", zorder=3)
    ax.plot([0, L, L, 0], [0, 0, W, W], **kw)
    ax.plot([0, 0], [0, door["start"]], **kw)
    ax.plot([0, 0], [door["end"], W], **kw)
    ax.plot([-0.06, -0.06], [door["start"], door["end"]], color=INK2, lw=1, ls=(0, (2, 2)))
    ax.text(-0.12, (door["start"] + door["end"]) / 2, "DOOR\n(keep\nclosed)", ha="right",
            va="center", fontsize=6.5, color=INK2)
    ax.plot(0, 0, marker="o", ms=5, color=INK, zorder=7)
    ax.text(0.05, -0.13, "origin (0, 0)", fontsize=7, color=INK, va="top")

    g = next(f for f in spec["furniture"] if f["name"] == "goal")
    gx, gy = g["pose"][:2]
    ax.add_patch(Circle((gx, gy), 0.6, fc="none", ec=GOAL, lw=1, ls=(0, (3, 2)), zorder=4))
    ax.add_patch(Circle((gx, gy), 0.06, fc=GOAL, ec="white", lw=1, zorder=5))
    ax.text(gx, gy - 0.13, f"GOAL: red pole\n({gx:.2f}, {gy:.2f})\nreached = base\nwithin 0.6 m",
            ha="center", va="top", fontsize=6.5, color=INK, zorder=6, linespacing=1.2,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))


def draw_starts(ax, routes):
    # The number sits INSIDE the dot marking the base centre: with starts a few cm apart, a
    # label beside a dot can be read as belonging to its neighbour, and on the day that
    # means a route driven from the wrong mark.
    for r in routes:
        x, y = r["start"]
        yaw = r["start_yaw"]
        ax.add_patch(Circle((x, y), ROBOT_R, fc=START, ec="none", alpha=0.10, zorder=3))
        ax.add_patch(FancyArrowPatch((x, y), (x + 0.36 * math.cos(yaw), y + 0.36 * math.sin(yaw)),
                                     arrowstyle="-|>", mutation_scale=8, color=START, lw=1.4,
                                     shrinkA=7, shrinkB=0, zorder=5))
        ax.add_patch(Circle((x, y), 0.075, fc=START, ec="white", lw=0.8, zorder=6))
        ax.text(x, y, str(int(r["route_id"][1:])), ha="center", va="center", fontsize=6,
                color="white", fontweight="bold", zorder=7)


def start_table(ax, routes, note):
    ax.axis("off")
    rows = [[r["route_id"], f"{r['start'][0]:.2f}", f"{r['start'][1]:.2f}",
             f"{math.degrees(r['start_yaw']):+.0f}°", f"{r['optimal_length_m']:.2f}", ""]
            for r in routes]
    t = ax.table(cellText=rows, colLabels=["route", "x (m)", "y (m)", "heading", "l* (m)", "done"],
                 loc="upper center", cellLoc="center", colLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(7.5)
    t.scale(1, 1.25)
    for (i, _), c in t.get_celld().items():
        c.set_edgecolor("#d6d5ce")
        if i == 0:
            c.set_text_props(color=INK, fontweight="bold")
            c.set_facecolor("#f3f2ee")
    close = [f"{int(a['route_id'][1:])}/{int(b['route_id'][1:])}"
             for i, a in enumerate(routes) for b in routes[i + 1:]
             if math.hypot(a["start"][0] - b["start"][0], a["start"][1] - b["start"][1]) < 0.15]
    if close:
        note += (f" Overlapping dots ({', '.join(close)}) are starts a few cm apart, not a "
                 f"drawing error: tape those from the table's coordinates.")
    # Below the table whatever its length (20 rows reaches the bottom of the axes).
    ax.text(0.0, -0.03, textwrap.fill(note, 62), transform=ax.transAxes, fontsize=7.5,
            color=INK2, va="top", linespacing=1.4)


HOW_TO = ("Viewed from above. Numbered blue dot = centre of Pepper's base (route number); arrow = the way she faces "
          "(heading measured from +x, anticlockwise positive); pale disc = her footprint. "
          "Tape a cross on each dot and a 40 cm line along each arrow.")


def sheet(fig, spec, routes, title, note, moved=None):
    gs = fig.add_gridspec(1, 2, width_ratios=[2.25, 1], wspace=0.05)
    ax = fig.add_subplot(gs[0])
    room_frame(ax, spec, title)
    for f in furniture(spec):
        if moved and f["name"] in moved:
            continue
        draw_piece(ax, f, FURN_FILL, FURN_EDGE)
        label_piece(ax, f)
    if moved:
        for name, m in moved.items():
            f = next(x for x in spec["furniture"] if x["name"] == name)
            old = dict(f, pose=[*m["from"], 0])
            draw_piece(ax, old, "none", MUTED, ls=(0, (2, 2)), z=2)
            below = -(f["size"][1] / 2 + 0.12) if f.get("shape") != "cylinder" else -0.2
            ax.text(m["from"][0], m["from"][1] + below, "normal spot", ha="center",
                    va="top", fontsize=6.5, color=MUTED, zorder=6)
            ax.add_patch(FancyArrowPatch(tuple(m["from"]), tuple(m["to"]), arrowstyle="-|>",
                                         mutation_scale=10, color=MOVED, lw=1.3,
                                         ls=(0, (4, 2)), zorder=4, shrinkA=8, shrinkB=10))
            draw_piece(ax, f, MOVED, INK, lw=1.0, z=5, alpha=0.9)
            label_piece(ax, f, f"{name.replace('_', ' ').upper()} → ({m['to'][0]:.2f}, "
                               f"{m['to'][1]:.2f})", color=INK, dy=0.27)
    draw_starts(ax, routes)
    start_table(fig.add_subplot(gs[1]), routes, note)


def main():
    nominal = yaml.safe_load((GT / "final_scan_spec.yaml").read_text())
    perturbed = yaml.safe_load((GT / "final_scan_perturbed_spec.yaml").read_text())
    v2 = json.loads((CACHE / "routes_final_scan_v2.json").read_text())
    pt = json.loads((CACHE / "routes_final_scan_perturbed.json").read_text())

    DOCS.mkdir(exist_ok=True)
    out_pdf = DOCS / "room_layouts.pdf"
    with PdfPages(out_pdf) as pdf:
        for name, spec, rs, title, note, moved in (
            ("room_layout_nominal.png", nominal, v2["routes"],
             "final_scan — NORMAL layout · route set v2 (20 routes)",
             HOW_TO + " Use with: route set v2.", None),
            ("room_layout_robustness.png", perturbed, pt["routes"],
             "final_scan — ROBUSTNESS layout (Sub-question 3) · 10 routes",
             HOW_TO + " Move ONLY the two orange pieces, tape-measuring their centres; "
             "everything else stays. Run after both normal passes. Use with: "
             "route set perturbed. Put them back afterwards.",
             pt["layout"]["moved"]),
        ):
            fig = plt.figure(figsize=(11.69, 8.27))      # A4 landscape
            sheet(fig, spec, rs, title, note, moved)
            fig.savefig(DOCS / name, dpi=200, bbox_inches="tight", facecolor="white")
            pdf.savefig(fig, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            print(f"wrote docs/{name}")
    print(f"wrote {out_pdf.relative_to(REPO)} (both sheets, for printing)")


if __name__ == "__main__":
    sys.exit(main())
