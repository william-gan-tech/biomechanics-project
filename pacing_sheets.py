"""
Phase 8e (11/4): one pacing sheet per skater.

Lap time, stroke rate and stroke length by lap, from stroke_rate/lap_times.csv
and stroke_rate/stretch_rates.csv. Each stroke point is one race stretch:
straight-only is the main version, the counter's first answer is shown as a
grey ring so the effect of the corrections stays visible. Laps with no point
had no usable stretch, and are left empty. Axes are the same on every sheet so
sheets can be compared. The other skater in each pair has no lap times, so the
comparison panel is across the six skaters.

Output: reports/pacing/pacing_<skater>.png

Usage:
    python -m pacing_sheets
"""

import os
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from stroke_rate_laps import EARLY, LATE

OUT_DIR = "reports/pacing"
MAIN, GREY, BAND, INK, MUTED = "#2a78d6", "#8a8984", "#efeeea", "#0b0b0b", "#52514e"
LAPS = range(2, 14)
RATE_LIM, LEN_LIM, LAP_LIM = (50, 135), (6, 16), (28, 33)

HOW_TO_READ = [
    "Each dot is one race stretch where the skater was large enough to measure.",
    "Blue is the straight-only stroke rate. A diamond means it comes from hand-marked set-downs.",
    "A grey ring is the counter's first answer, before corner strokes were cut and hand checks made.",
    "Shaded laps are the early (2-5) and late (9-12) groups. Lines show each group's median.",
    "Empty laps had no usable stretch. Lap 1 is a 200 m half lap and is left out.",
]
LIMITS = ("Stroke length is worked out from lap speed and stroke rate, not measured. Lap speeds are close, so it mostly "
          "mirrors stroke rate. Only high rates were hand-checked, so corrections went one way, and other rates may "
          "still hold counter errors.")


def wrapped(ax, y, head, lines, width=52):
    ax.text(0, y, head, fontsize=11, fontweight="bold", color=INK, va="top")
    y -= 0.05
    for line in lines:
        w = textwrap.wrap(line, width)
        ax.text(0, y, "\n".join(w), fontsize=9, color=MUTED, va="top", linespacing=1.3)
        y -= 0.034 * len(w) + 0.012
    return y - 0.015


def result_lines(s, skater):
    g = s[s.skater == skater].set_index("version")
    st = g.loc["straight-only"]
    out = []
    if pd.notna(st.early_spm) and pd.notna(st.late_spm):
        word = "past" if st.rate_real else "not past"
        out.append(f"Straight-only: {st.early_spm:.1f} to {st.late_spm:.1f} strokes/min ({st.rate_change:+.1%}), {word} the 10% bar. "
                   f"Stroke length {st.len_change:+.1%}.")
    else:
        out.append("Straight-only: early vs late can't be measured "
                   f"({int(st.early_n)} early and {int(st.late_n)} late stretches left).")
    for v in ("counter", "hand-checked"):
        if pd.notna(g.loc[v].get("rate_change")):
            out.append(f"{v.capitalize()} version: {g.loc[v].rate_change:+.1%}.")
    return out


def group_medians(ax, d, col):
    for laps in (EARLY, LATE):
        g = d[d.lap.isin(laps)][col]
        if len(g):
            ax.plot([laps[0] - 0.45, laps[-1] + 0.45], [g.median()] * 2, color=MAIN, lw=2, zorder=2)


def base(ax, title, ylim, unit):
    for laps in (EARLY, LATE):
        ax.axvspan(laps[0] - 0.5, laps[-1] + 0.5, color=BAND, zorder=0, lw=0)
    ax.set_xlim(1.5, 13.5); ax.set_ylim(*ylim); ax.set_xticks(list(LAPS))
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK)
    ax.set_ylabel(unit, fontsize=9, color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(axis="y", color="#e2e1dc", lw=0.7); ax.set_axisbelow(True)
    for k in ("top", "right", "left"):
        ax.spines[k].set_visible(False)
    ax.spines["bottom"].set_color("#c9c8c2")


def strokes(ax, d, col, ring):
    ax.scatter(ring.lap, ring[col + "_counter"], s=80, facecolors="none", edgecolors=GREY, linewidths=1.3, zorder=3)
    for hand, marker in ((False, "o"), (True, "D")):
        g = d[d.hand == hand]
        ax.scatter(g.lap, g[col], s=46, marker=marker, color=MAIN, edgecolors="white", linewidths=0.8, zorder=4)
    group_medians(ax, d, col)


def sheet(skater, r, laps, s, pos):
    a = r[r.skater == skater]
    ring = a[a.rate_spm.notna()].assign(rate_counter=lambda x: x.rate_spm, length_counter=lambda x: x.length_m)
    d = a[a.rate_straight.notna()].copy()
    d["rate"] = d.rate_straight
    d["length"] = ((400 / d.lap_s) / (d.rate / 60)).round(2)
    d["hand"] = d.rate_hand.notna() | d.stretch.isin(HAND)
    lt = laps[(laps.skater == skater) & laps.lap.isin(LAPS)]

    fig = plt.figure(figsize=(13, 10), facecolor="white")
    gs = fig.add_gridspec(3, 2, width_ratios=[1.75, 1], left=0.06, right=0.97, top=0.91, bottom=0.06, hspace=0.34, wspace=0.14)
    fig.text(0.06, 0.955, f"{skater}: pacing and strokes, Beijing 2022 5000 m", fontsize=16, fontweight="bold", color=INK)

    ax = fig.add_subplot(gs[0, 0]); base(ax, "Lap time", LAP_LIM, "seconds per 400 m")
    ax.plot(lt.lap, lt.lap_s, color=MAIN, lw=2, marker="o", ms=5, zorder=3)
    ax = fig.add_subplot(gs[1, 0]); base(ax, "Stroke rate", RATE_LIM, "strokes per minute")
    strokes(ax, d, "rate", ring)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=MAIN, label="straight-only, counter"),
                       Line2D([], [], marker="D", ls="", color=MAIN, label="straight-only, hand-marked"),
                       Line2D([], [], marker="o", ls="", mfc="none", mec=GREY, label="counter, first answer"),
                       Line2D([], [], color=MAIN, lw=2, label="group median")],
              loc="upper right", ncol=2, fontsize=8.5, frameon=False)
    ax = fig.add_subplot(gs[2, 0]); base(ax, "Stroke length", LEN_LIM, "metres per stroke")
    strokes(ax, d, "length", ring)
    ax.set_xlabel("lap", fontsize=9, color=MUTED)

    ax = fig.add_subplot(gs[0:2, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    y = wrapped(ax, 1.0, "Result", result_lines(s, skater))
    y = wrapped(ax, y, "How to read this sheet", HOW_TO_READ)
    wrapped(ax, y, "Limits", [LIMITS])

    ax = fig.add_subplot(gs[2, 1])
    ax.set_title("Compared with the other skaters", loc="left", fontsize=11, fontweight="bold", color=INK)
    for name, p in pos.iterrows():
        me = name == skater
        ax.scatter(p.rate, p.length, s=70 if me else 36, color=MAIN if me else GREY, zorder=3 if me else 2)
        ax.annotate(f"{name.split(maxsplit=1)[1]} ({int(p.n)})", (p.rate, p.length), xytext=(6, 4), textcoords="offset points",
                    fontsize=8.5, color=INK if me else MUTED, fontweight="bold" if me else "normal")
    ax.set_xlim(52, 80); ax.set_ylim(10.5, 15.5)
    ax.set_xlabel("median stroke rate, strokes/min (stretches in brackets)", fontsize=9, color=MUTED)
    ax.set_ylabel("median stroke length, m", fontsize=9, color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(color="#e2e1dc", lw=0.7); ax.set_axisbelow(True)
    for k in ("top", "right"):
        ax.spines[k].set_visible(False)
    for k in ("left", "bottom"):
        ax.spines[k].set_color("#c9c8c2")

    path = os.path.join(OUT_DIR, f"pacing_{skater.split(maxsplit=1)[1].replace(' ', '').replace('-', '').lower()}.png")
    fig.savefig(path, dpi=100); plt.close(fig)
    return path


chk = pd.read_csv("stroke_rate/race_checks.csv", dtype=str).dropna(subset=["setdown_frames"])
HAND = set(chk["clip"].str.replace("check_", ""))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    r = pd.read_csv("stroke_rate/stretch_rates.csv")
    laps = pd.read_csv("stroke_rate/lap_times.csv")
    s = pd.read_csv("stroke_rate/skater_summary.csv")
    d = r[r.rate_straight.notna()].copy()
    d["length"] = (400 / d.lap_s) / (d.rate_straight / 60)
    pos = d.groupby("skater").agg(rate=("rate_straight", "median"), length=("length", "median"), n=("stretch", "size"))
    print(pos.round(2).to_string())
    for skater in sorted(r.skater.unique()):
        print(sheet(skater, r, laps, s, pos))


if __name__ == "__main__":
    main()
