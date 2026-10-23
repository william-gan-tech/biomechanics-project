"""
Phase 7c, second detection measure (10/23): where the recovering foot lands.

Researcher (10/22): much of a double push looks like a normal push; what sets
it apart is the leg after the push, on its way back. In a double push the
recovering foot is set down under the body (near the centreline); in a normal
stride it comes back and lands out on its own side.

Touchdown timing: from a front-facing camera the foot furthest forward appears
lowest in the picture, so each foot's touchdown is taken as a local maximum of
(foot height - hip height) / leg length: the moment it is planted ahead under
the skater, before the body rolls over it (peaks at least 0.4 s apart,
prominence >= 0.05 leg lengths).

Measure: setdown = median, over touchdowns in the window, of the foot's
sideways position at touchdown relative to the hip midline, in leg lengths,
signed so that positive = out on the foot's own side, near zero or negative =
under or across the body.

Secondary measure (added 10/23, before any fresh results; reported, not the
pass/fail): twist. Researcher: in a double push the inside foot twists for
the second push, while in a single push it just glides out slowly. From the
heel-to-toe direction in the image: for each touchdown, the range (max - min)
of that foot's direction over the 0.5 s after touchdown, in degrees; window
value = median over touchdowns. Higher = more twist = double push. Its
cut-off is also set on Mantia only.

Rule: cut-off set on Joey Mantia only (halfway between his normal-stride and
double-push medians); windows below it are called double push.
Fresh test (kept aside 10/23, never used to set the rule):
doublepush/fresh_test_labels.csv. Target: >= 85% of fresh windows correct.

Usage:
    python -m recovery_measure            # rule on Mantia, report on all labelled stretches
    python -m recovery_measure --fresh    # also track and score the fresh test clips
"""

import sys
import os
import numpy as np
import pandas as pd
import cv2
from scipy.signal import find_peaks

from doublepush_measures import track, MIN_LEG_PX, WINDOW_S, MIN_TRACKED

LABELS = "doublepush/style_labels.csv"
FRESH = "doublepush/fresh_test_labels.csv"
TARGET = 0.85


def touchdowns(d, fps):
    s = d.leg_px.median()
    side = np.sign((d.lf / s).median() - (d.rf / s).median()) or 1.0
    out = []
    for _, g in d.groupby("cut_seq"):
        if len(g) < int(fps):
            continue
        for foot, sign in (("l", side), ("r", -side)):
            depth = ((g[f"{foot}f_y"] - g.hip_y) / s).rolling(3, center=True, min_periods=1).mean().values
            peaks, _ = find_peaks(depth, distance=max(1, int(0.4 * fps)), prominence=0.05)
            ang = None
            if f"{foot}_dir" in g.columns:
                ang = pd.Series(np.degrees(np.unwrap(np.radians(g[f"{foot}_dir"].values)))).rolling(3, center=True, min_periods=1).mean().values
            for p in peaks:
                twist = np.nan
                if ang is not None:
                    seg = ang[p:p + max(2, int(0.5 * fps))]
                    twist = float(seg.max() - seg.min()) if len(seg) >= 2 else np.nan
                out.append({"frame": int(g.frame.iloc[p]), "foot": foot,
                            "setdown": float(sign * g[f"{foot}f"].iloc[p] / s), "twist": twist})
    return pd.DataFrame(out)


def window_scores(d, fps, start, end):
    w = int(WINDOW_S * fps)
    span = end - start + 1
    starts = [start] if span <= 1.5 * w else list(range(start, end - w + 2, w))
    rows = []
    for k, a in enumerate(starts):
        b = end if len(starts) == 1 else a + w - 1
        g = d[(d.frame >= a) & (d.frame <= b)]
        if len(g) < MIN_TRACKED * (b - a + 1) or g.leg_px.median() < MIN_LEG_PX:
            continue
        t = touchdowns(g, fps)
        if len(t) < 2:
            continue
        rows.append({"window": k, "touchdowns": len(t), "setdown": t.setdown.median(),
                     "twist": t.twist.median(), "leg_px": g.leg_px.median()})
    return rows


def score_labels(path, frames_prefix, track_missing):
    lab = pd.read_csv(path)
    rows = []
    for n, r in enumerate(lab.itertuples()):
        cache = f"{frames_prefix}{n:02d}.csv"
        fps = cv2.VideoCapture(r.video).get(cv2.CAP_PROP_FPS) or 30.0
        if os.path.exists(cache) and "lf_y" in pd.read_csv(cache, nrows=1).columns:
            d = pd.read_csv(cache)
        elif track_missing:
            d, fps = track(r.video, int(r.start_frame), int(r.end_frame)); d.to_csv(cache, index=False)
        else:
            continue
        for w in window_scores(d, fps, int(r.start_frame), int(r.end_frame)):
            rows.append({"stretch": r.skater_note, "label": r.label, **w})
    return pd.DataFrame(rows)


def main():
    m = score_labels(LABELS, "doublepush/frames_", track_missing=False)
    mantia = m[m.stretch.str.startswith("Mantia") & m.label.isin(["normal", "double_push"])]
    med = mantia.groupby("label").setdown.median()
    cut = (med["normal"] + med["double_push"]) / 2
    print(f"Rule set on Mantia only: normal median {med['normal']:+.2f} ({(mantia.label == 'normal').sum()} windows), "
          f"double push median {med['double_push']:+.2f} ({(mantia.label == 'double_push').sum()} windows); "
          f"cut-off {cut:+.2f}, below = double push")
    pred = lambda df: np.where(df.setdown < cut, "double_push", "normal")
    m["truth"] = m.label.replace({"double_push_flawed": "double_push"})
    m["correct"] = pred(m) == m.truth
    print("\nAll labelled stretches (Mantia set the rule; others were seen on 10/22):")
    for (st, lab), g in m.groupby(["stretch", "label"]):
        if not st.startswith("Mantia clinic"):
            print(f"  {st[:42]:42s} {lab:18s} {g.correct.mean():4.0%} ({len(g)} windows, setdown median {g.setdown.median():+.2f})")
    cl = m[m.stretch.str.startswith("Mantia clinic")]
    print(f"  {'Mantia clinic (all 50 stretches)':42s} {'normal':18s} {cl.correct.mean():4.0%} ({len(cl)} windows, setdown median {cl.setdown.median():+.2f})")
    tmed = mantia.groupby("label").twist.median()
    tcut = (tmed["normal"] + tmed["double_push"]) / 2
    print(f"\nSecondary (twist), rule on Mantia: normal median {tmed['normal']:.1f} deg, double push median "
          f"{tmed['double_push']:.1f} deg; cut-off {tcut:.1f}, above = double push")
    tpred = lambda df: np.where(df.twist > tcut, "double_push", "normal")
    m["twist_correct"] = tpred(m) == m.truth
    m.to_csv("doublepush/recovery_results.csv", index=False)
    if "--fresh" in sys.argv:
        f = score_labels(FRESH, "doublepush/fresh_frames_", track_missing=True)
        f["correct"] = pred(f) == f.label.replace({"double_push_flawed": "double_push"})
        f["twist_correct"] = tpred(f) == f.label.replace({"double_push_flawed": "double_push"})
        print("\nFRESH TEST (kept aside, never used to set the rule):")
        for st, g in f.groupby("stretch"):
            print(f"  {st[:42]:42s} {g.label.iloc[0]:12s} {g.correct.mean():4.0%} ({len(g)} windows, setdown median {g.setdown.median():+.2f})")
        acc = f.correct.mean()
        print(f"\nFresh test accuracy {acc:.1%} on {len(f)} windows; target {TARGET:.0%}: {'MET' if acc >= TARGET else 'NOT MET'}")
        print(f"Secondary (twist) on the fresh test: {f.twist_correct.mean():.1%} (reported only, not the pass/fail)")
        f.to_csv("doublepush/recovery_fresh_results.csv", index=False)


if __name__ == "__main__":
    main()
