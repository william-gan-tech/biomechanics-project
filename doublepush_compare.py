"""
Phase 7d, visual comparison tool (10/27).

The automatic feedback report failed its test (10/26), so this tool doesn't
judge: it shows a skater's measured double-push movement next to a reference
clip, with the researcher's coaching text as guidance on what to look for.
The skater or coach makes the judgement.

One sheet (PNG) per comparison, two columns (clip A, clip B):
  1. Still frames at two touchdowns, with feet, shins and hips drawn
  2. Feet sideways paths over one 4-second window (hip centre = 0, leg lengths)
  3. Where the recovering foot lands, every touchdown in the clip
  4. Hips up-down over the same window
Plus a notes panel from doublepush/feedback_text.json.

Input clips are rows of a labels CSV (video, start_frame, end_frame, ...)
with saved per-frame data from doublepush_measures.track (or tracked here).
Only windows with the skater large enough (leg >= 150 px) are used.

Usage:
    python -m doublepush_compare        # demo: Thorup single push vs Thorup double push
"""

import os
import json
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from doublepush_measures import track, MIN_LEG_PX, WINDOW_S
from recovery_measure import touchdowns

FEEDBACK = "doublepush/feedback_text.json"
OUT_DIR = "reports/doublepush"


def load_clip(rows):
    """rows: list of (labels row, frames cache path). Returns frames (with stretch id), fps, video."""
    parts, fps, video = [], 30.0, None
    for k, (r, cache) in enumerate(rows):
        video = r.video
        fps = cv2.VideoCapture(r.video).get(cv2.CAP_PROP_FPS) or 30.0
        if os.path.exists(cache):
            d = pd.read_csv(cache)
        else:
            d, fps = track(r.video, int(r.start_frame), int(r.end_frame)); d.to_csv(cache, index=False)
        parts.append(d.assign(part=k))
    return pd.concat(parts, ignore_index=True), fps, video


def best_window(d, fps):
    """The 4-second window with the most frames where the skater is large enough."""
    w = int(WINDOW_S * fps)
    best, best_n = None, 0
    for part, g in d.groupby("part"):
        g = g[g.leg_px >= MIN_LEG_PX]
        for a in range(int(g.frame.min()), int(g.frame.max()) - w + 2, max(1, w // 4)):
            n = ((g.frame >= a) & (g.frame < a + w)).sum()
            if n > best_n:
                best, best_n = (part, a, a + w - 1), n
    return best


def all_touchdowns(d, fps):
    out = []
    for part, g in d.groupby("part"):
        g = g[g.leg_px >= MIN_LEG_PX]
        for _, seq in g.groupby("cut_seq"):
            if len(seq) >= fps:
                t = touchdowns(seq, fps)
                if len(t):
                    out.append(t.assign(part=part))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["frame", "foot", "setdown", "part"])


def still(video, frame, row):
    cap = cv2.VideoCapture(video); cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame)); ok, img = cap.read()
    if not ok:
        return None
    h = img.shape[0]
    hx, hy, s = row.hip_x, row.hip_y, row.leg_px
    # feet points from the per-frame data (sideways relative to hips, heights absolute)
    for xf, yf, col in ((row.lf + hx, row.lf_y, (14, 127, 255)), (row.rf + hx, row.rf_y, (180, 119, 31))):  # BGR: orange = left, blue = right, as in the plots
        cv2.circle(img, (int(xf), int(yf)), max(6, int(s * 0.05)), col, -1)
        cv2.line(img, (int(hx), int(hy)), (int(xf), int(yf)), col, max(2, int(s * 0.02)))
    cv2.circle(img, (int(hx), int(hy)), max(6, int(s * 0.05)), (0, 255, 0), -1)
    cv2.line(img, (int(hx), 0), (int(hx), h), (0, 255, 0), 2)
    x0 = int(max(0, hx - 1.3 * s)); x1 = int(min(img.shape[1], hx + 1.3 * s))
    y0 = int(max(0, hy - 1.4 * s)); y1 = int(min(h, hy + 1.3 * s))
    return cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)


UNDER = 0.15   # landing within this many leg lengths of the hip line = "under the hips"
WIDE = 0.30    # landing further out than this = "out to the side"


def clip_stats(d, t, g, s):
    big = d[d.leg_px >= MIN_LEG_PX]
    side = np.sign((big.lf / big.leg_px).median() - (big.rf / big.leg_px).median()) or 1.0
    L, R = side * big.lf / big.leg_px, -side * big.rf / big.leg_px
    hy = (g.hip_y - g.hip_y.rolling(15, center=True, min_periods=3).mean()) / s
    return {"under": (t.setdown.abs() <= UNDER).mean() if len(t) else np.nan,
            "wide": (t.setdown > WIDE).mean() if len(t) else np.nan,
            "crossed": float(((L < -0.1) | (R < -0.1)).mean()),
            "hips": float(hy.quantile(0.95) - hy.quantile(0.05))}


def caption(fig, gs_cell, title, body):
    ax = fig.add_subplot(gs_cell); ax.axis("off")
    ax.text(0, 0.95, title, fontsize=12, fontweight="bold", va="top")
    ax.text(0, 0.45, body, fontsize=9.5, va="top", wrap=True)


def compare(a_rows, b_rows, a_name, b_name, out_name, same_camera=True):
    text = json.load(open(FEEDBACK, encoding="utf-8"))
    clips = []
    for rows, name in ((a_rows, a_name), (b_rows, b_name)):
        d, fps, video = load_clip(rows)
        win = best_window(d, fps)
        t = all_touchdowns(d, fps)
        part, a, b = win
        g = d[(d.part == part) & (d.frame >= a) & (d.frame <= b)].copy()
        s = g.leg_px.median()
        clips.append((name, d, fps, video, win, t, g, s, clip_stats(d, t, g, s)))

    fig = plt.figure(figsize=(14, 25))
    gs = fig.add_gridspec(10, 2, height_ratios=[1.25, 0.32, 1.35, 0.42, 1.0, 0.42, 1.0, 0.42, 0.8, 1.05])

    # summary box
    sa, sb = clips[0][-1], clips[1][-1]
    ax = fig.add_subplot(gs[0, :]); ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, color="#f3f3f3", zorder=0))
    cam = "Same camera, front view." if same_camera else "Front view; different cameras, so compare patterns rather than exact values."
    summary = (
        "HOW TO READ THIS SHEET\n"
        f"Left column: {a_name}.   Right column: {b_name}.   {cam}\n"
        "Each row shows one part of the stride. Read the caption above each row, then compare left and right.\n\n"
        "KEY DIFFERENCES, IN NUMBERS\n"
        f"  - Recovering foot lands UNDER THE HIPS (within {UNDER:.2f} leg lengths):   {a_name} {sa['under']:.0%}   vs   {b_name} {sb['under']:.0%}\n"
        f"  - Recovering foot lands OUT TO THE SIDE (more than {WIDE:.2f} leg lengths):   {a_name} {sa['wide']:.0%}   vs   {b_name} {sb['wide']:.0%}\n"
        f"  - A foot is CROSSED UNDER THE BODY (the underpush):   {a_name} {sa['crossed']:.0%} of the time   vs   {b_name} {sb['crossed']:.0%}\n"
        f"  - Hips up-down range in one stride:   {a_name} {sa['hips']:.2f}   vs   {b_name} {sb['hips']:.2f} leg lengths\n\n"
        "In a good double push: more landings under the hips, few out to the side, the foot crossing under the body more often, and quiet hips.")
    ax.text(0.02, 0.95, summary, fontsize=10.5, va="top", family="monospace", transform=ax.transAxes)

    caption(fig, gs[1, :], "1. THE MOMENT THE FOOT LANDS",
            "Green line = straight down from the hips (the centre of the body). Dots = the feet (orange = left, blue = right).\n"
            "LOOK FOR: in double push, the landing foot touches down close to the green line, under the body; in single push it lands further out to the side.")
    caption(fig, gs[3, :], "2. HOW THE FEET MOVE SIDEWAYS (4 seconds)",
            "Each line is one foot. Up = out to that foot's own side (the push). Down below the green line = the foot has crossed under the body.\n"
            "LOOK FOR: in double push, after each push the line dips BELOW the green line (the underpush) before rising again. In single push it stays mostly above.")
    caption(fig, gs[5, :], "3. WHERE THE RECOVERING FOOT LANDS (every stride in the clip)",
            "Each bar counts strides. Green band = landed under the hips. Red band = landed out to the side.\n"
            "LOOK FOR: double push = a tall, narrow pile inside the green band. Single push = spread out, with many strides in the red band.")
    caption(fig, gs[7, :], "4. HIPS UP AND DOWN (4 seconds)",
            "The line shows the hips rising and dropping during the strides.\n"
            "LOOK FOR: a fairly flat line (energy goes forward). Big waves = bouncing, energy lost up and down.")

    ylim = 0
    for c, (name, d, fps, video, win, t, g, s, st) in enumerate(clips):
        part, a, b = win
        sd = np.sign((g.lf / s).median() - (g.rf / s).median()) or 1.0
        g["L"] = sd * g.lf / s; g["R"] = -sd * g.rf / s
        g["sec"] = (g.frame - a) / fps
        tw = t[(t.part == part) & (t.frame >= a) & (t.frame <= b)].sort_values("frame")
        # 1. stills
        picks = tw.iloc[[0, len(tw) // 2]] if len(tw) >= 2 else tw
        sub = gs[2, c].subgridspec(1, 2)
        for k, (_, p) in enumerate(picks.iterrows()):
            ax = fig.add_subplot(sub[0, k])
            row = g[g.frame == p.frame].iloc[0]
            im = still(video, row.frame, row)
            if im is not None:
                ax.imshow(im)
            where = "under the hips" if abs(p.setdown) <= UNDER else ("crossed under" if p.setdown < 0 else "out to the side")
            ax.set_title(f"{name}\n{'left' if p.foot == 'l' else 'right'} foot landing", fontsize=10)
            ax.text(0.5, -0.04, f"lands {abs(p.setdown):.2f} leg lengths\nfrom the hip line\n= {where.upper()}",
                    transform=ax.transAxes, ha="center", va="top", fontsize=9.5, fontweight="bold",
                    color="green" if where != "out to the side" else "firebrick")
            ax.axis("off")
        # 2. feet sideways
        ax = fig.add_subplot(gs[4, c])
        ax.axhspan(-1.5, 0, color="green", alpha=0.07)
        ax.plot(g.sec, g.L, color="tab:orange", label="left foot")
        ax.plot(g.sec, g.R, color="tab:blue", label="right foot")
        ax.axhline(0, color="green", lw=1.5, label="under the hips")
        low = np.minimum(g.L.values, g.R.values); i = int(np.nanargmin(low))
        if low[i] < -0.1:
            ax.annotate("foot crossed under the body\n(the underpush)", xy=(g.sec.iloc[i], low[i]), xytext=(g.sec.iloc[i], -0.75),
                        ha="center", fontsize=9, color="darkgreen", arrowprops=dict(arrowstyle="->", color="darkgreen"))
        else:
            ax.text(0.5, 0.08, "feet barely cross under the body", transform=ax.transAxes, ha="center", fontsize=9, color="firebrick")
        ax.set_title(f"{name}", fontsize=10.5, fontweight="bold")
        ax.set_ylabel("out to own side  (leg lengths)\nbelow 0 = under the body")
        ax.set_xlabel("seconds"); ax.legend(fontsize=8, loc="upper right")
        ylim = max(ylim, np.nanmax(np.abs(np.r_[g.L, g.R])))
        # 3. landing positions
        ax = fig.add_subplot(gs[6, c])
        ax.axvspan(-UNDER, UNDER, color="green", alpha=0.15, label="under the hips")
        ax.axvspan(WIDE, 0.85, color="red", alpha=0.08, label="out to the side")
        if len(t):
            ax.hist(t.setdown, bins=np.arange(-0.8, 0.85, 0.05), color="tab:purple", alpha=0.85)
        ax.axvline(0, color="green", lw=1.5)
        ax.text(0.02, 0.95, f"under the hips: {st['under']:.0%}\nout to the side: {st['wide']:.0%}\n({len(t)} strides)",
                transform=ax.transAxes, va="top", fontsize=10, fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="grey"))
        ax.set_title(f"{name}", fontsize=10.5, fontweight="bold")
        ax.set_xlabel("landing position, leg lengths out to own side (0 = under the hips)"); ax.set_ylabel("strides")
        ax.legend(fontsize=8, loc="upper right")
        # 4. hips
        ax = fig.add_subplot(gs[8, c])
        hy = (g.hip_y - g.hip_y.mean()) / s
        ax.plot(g.sec, -hy, color="tab:red")
        ax.axhline(0, color="grey", lw=0.8)
        ax.text(0.02, 0.92, f"up-down range: {st['hips']:.2f} leg lengths", transform=ax.transAxes, va="top", fontsize=9.5,
                fontweight="bold", bbox=dict(facecolor="white", alpha=0.85, edgecolor="grey"))
        ax.set_title(f"{name}", fontsize=10.5, fontweight="bold"); ax.set_xlabel("seconds"); ax.set_ylabel("leg lengths")
        ax.set_ylim(-0.3, 0.3)
    for ax in fig.axes:
        if ax.get_ylabel().startswith("out to own side"):
            ax.set_ylim(-max(ylim, 0.9) * 1.1, ylim * 1.1)

    notes = fig.add_subplot(gs[9, :]); notes.axis("off")
    NL = "\n"
    msg = ("COACHING NOTES (guidance, not a verdict)" + NL + NL +
           f"If the foot doesn't come under the body: {text['no_underpush']['explain']}" + NL +
           f"   Cue: {text['no_underpush']['cue']}   Drill: {text['no_underpush']['drill']}" + NL + NL +
           f"If the foot lands out to the side: {text['wide_setdown']['explain']}" + NL +
           f"   Cue: {text['wide_setdown']['cue']}   Drill: {text['wide_setdown']['drill']}" + NL + NL +
           f"If the hips bounce: {text['hip_bounce']['explain']}" + NL +
           f"   Cue: {text['hip_bounce']['cue']}   Drill: {text['hip_bounce']['drill']}")
    notes.text(0, 1, msg, va="top", fontsize=9, wrap=True)
    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, out_name)
    fig.savefig(path, dpi=85); plt.close(fig)
    return path


def to_seconds(t):
    """'1:05' or '65' or '65.5' -> seconds."""
    parts = [float(x) for x in str(t).split(":")]
    return parts[0] * 60 + parts[1] if len(parts) == 2 else parts[0]


def main():
    import argparse
    from types import SimpleNamespace
    p = argparse.ArgumentParser(description="Compare a skater's double push with Thorup's (or run the Thorup demo).")
    p.add_argument("--video", help="the skater's clip (front or back view, whole body in frame)")
    p.add_argument("--start", help="start time, e.g. 1:05")
    p.add_argument("--end", help="end time, e.g. 1:20")
    p.add_argument("--name", default="Your skating", help="label for the skater's column")
    a = p.parse_args()
    lab = pd.read_csv("doublepush/fresh_test_labels.csv")
    rows = list(lab.itertuples())
    pick = lambda cond: [(r, f"doublepush/fresh_frames_{n:02d}.csv") for n, r in enumerate(rows) if cond(r)]
    double = pick(lambda r: r.skater_note.startswith("Thorup skate-faster") and r.label == "double_push")
    if not a.video:
        single = pick(lambda r: r.skater_note.startswith("Thorup skate-faster") and r.label == "normal")
        print(compare(single, double, "Thorup single push", "Thorup double push", "compare_thorup_single_vs_double.png"))
        return
    fps = cv2.VideoCapture(a.video).get(cv2.CAP_PROP_FPS) or 30.0
    s0, s1 = int(to_seconds(a.start) * fps), int(to_seconds(a.end) * fps)
    base = os.path.splitext(os.path.basename(a.video))[0]
    clip = SimpleNamespace(video=a.video, start_frame=s0, end_frame=s1)
    cache = f"doublepush/user_frames_{base}_{s0}_{s1}.csv"
    out = f"compare_{base}_{a.start.replace(':', '')}-{a.end.replace(':', '')}_vs_thorup.png"
    print(compare([(clip, cache)], double, a.name, "Thorup double push (elite example)", out, same_camera=False))


if __name__ == "__main__":
    main()
