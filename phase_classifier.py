"""
Phase 5g: automatic corner vs straightaway detection (10/7).

The form report (form_report.py) needs every clip hand-labelled into corner and
straightaway segments. This tests whether that labelling can be automated,
using the researcher's 188 hand-labelled long-track segments as training data.

Approach (kept simple and inspectable):
  - Each labelled segment is cut into 1-second windows (25 frames, step 12).
  - Each window gets summary features of pose signals we already extract:
    torso lean (abs and signed), its left/right difference, both knee angles,
    knee asymmetry, pelvic tilt, projected hip width, and how much each varies
    within the window.
  - A random forest classifies each window as corner or straightaway; a
    segment's prediction is the majority vote of its windows.

Evaluation: LEAVE-ONE-SKATER-OUT. The model is never tested on a skater it
was trained on (each skater's footage comes from a different race/camera
setup), which is the realistic case for a new clip. Accuracy is reported per
window and per segment, with a majority-class baseline for comparison.

Only full-race skaters are used (the lap-labelled corner/straightaway
segments); start phases are excluded.

Usage:
    python -m phase_classifier
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from compare_to_elite_reference import load_labeled_segments, get_segment_frames

SIGNALS = ["torso_lean_angle_deg", "torso_lean_angle_deg_signed", "torso_lean_lr_diff",
           "right_knee_filtered", "left_knee_filtered", "knee_angle_asymmetry",
           "hip_height_asymmetry", "hip_lateral_asymmetry"]
# Phase 6d (10/8): the 5g model leaned on projected hip width, which tracks the
# camera shot rather than the track section. Distance-based signals (also
# affected by camera-scale calibration) can be excluded via PHASE_CLF_ANGLES_ONLY=1.
if os.environ.get("PHASE_CLF_ANGLES_ONLY") == "1":
    SIGNALS = [s for s in SIGNALS if s not in ("hip_height_asymmetry", "hip_lateral_asymmetry")]
# PHASE_CLF_WIN = window length in frames; 0 = one window per whole segment.
WIN = int(os.environ.get("PHASE_CLF_WIN", "25"))
STEP = 12
TARGET = 0.85  # accuracy needed to call automatic detection usable
OUTPUT_PATH = "phase5g_phase_classifier_results.csv"


def windows_for(seg_frames):
    f = seg_frames.sort_values("frame")
    f = f[[c for c in SIGNALS if c in f.columns]].dropna()
    feats = []
    starts = [0] if WIN == 0 else range(0, len(f) - WIN + 1, STEP)
    for i in starts:
        w = f if WIN == 0 else f.iloc[i:i + WIN]
        if len(w) < 10:
            continue
        row = {}
        for c in w.columns:
            row[f"{c}_mean"] = w[c].mean()
            row[f"{c}_std"] = w[c].std()
        # knees alternate bending; in crossovers (corners) the pattern differs
        row["knee_diff_signed_mean"] = (w["left_knee_filtered"] - w["right_knee_filtered"]).mean()
        feats.append(row)
    return feats


# Phase 6d (10/15): optionally add segments recovered from both-skaters footage
# (dual_tracking/recovered_segments.csv, all shot types), which brings in new
# skaters. Results then go to a separate file so the 5g numbers stay as they were.
RECOVERED = os.environ.get("PHASE_CLF_RECOVERED") == "1"
RECOVERED_PATH = "dual_tracking/recovered_segments.csv"
if RECOVERED:
    OUTPUT_PATH = "phase6d_phase_classifier_with_recovered.csv"


def build_dataset():
    rows = []
    segs = load_labeled_segments()
    if RECOVERED and os.path.exists(RECOVERED_PATH):
        rec = pd.read_csv(RECOVERED_PATH, dtype=str).fillna("")
        segs = segs + [{**r, "source": "recovered"} for r in rec.to_dict("records")]
    for r in segs:
        if r["phase"] not in ("corner", "straightaway") or "lap" not in r.get("notes", ""):
            continue
        f = get_segment_frames(r)
        if f is None:
            continue
        for w in windows_for(f):
            rows.append({**w, "skater": r["skater"], "phase": r["phase"],
                         "source": r.get("source", "phase5") + (f"-{r.get('shot')}" if r.get("source") == "recovered" else ""),
                         "segment": f"{r['video_path']}|{r['start_frame']}-{r['end_frame']}|{r.get('piece_no', '')}"})
    return pd.DataFrame(rows)


def main():
    df = build_dataset()
    feature_cols = [c for c in df.columns if c not in ("skater", "phase", "segment", "source")]
    df = df.dropna(subset=feature_cols)
    print(f"{len(df)} one-second windows from {df.segment.nunique()} segments, {df.skater.nunique()} skaters")
    print(df.groupby("phase").segment.nunique().to_string())

    results, all_seg = [], []
    for skater in sorted(df.skater.unique()):
        train, test = df[df.skater != skater], df[df.skater == skater]
        if test.phase.nunique() < 1 or train.phase.nunique() < 2:
            continue
        clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=0,
                                     class_weight="balanced")
        clf.fit(train[feature_cols], train.phase)
        test = test.assign(pred=clf.predict(test[feature_cols]))
        seg = test.groupby("segment").agg(phase=("phase", "first"), source=("source", "first"),
                                          pred=("pred", lambda s: s.value_counts().idxmax()))
        all_seg.append(seg)
        majority = train.phase.value_counts().idxmax()
        results.append({"held_out_skater": skater, "windows": len(test), "segments": len(seg),
                        "window_acc": (test.pred == test.phase).mean(),
                        "segment_acc": (seg.pred == seg.phase).mean(),
                        "baseline_acc": (seg.phase == majority).mean()})

    res = pd.DataFrame(results)
    res.to_csv(OUTPUT_PATH, index=False)
    pd.set_option("display.width", 200)
    print("\nLEAVE-ONE-SKATER-OUT (never tested on a skater it trained on):")
    print(res.round(3).to_string(index=False))
    seg_total = (res.segment_acc * res.segments).sum() / res.segments.sum()
    base_total = (res.baseline_acc * res.segments).sum() / res.segments.sum()
    win_total = (res.window_acc * res.windows).sum() / res.windows.sum()
    print(f"\nOverall: segment accuracy {seg_total:.1%}, window accuracy {win_total:.1%}, "
          f"majority-class baseline {base_total:.1%}")
    print(f"Usable for automatic labelling (>= {TARGET:.0%} segment accuracy): {seg_total >= TARGET}")
    segs = pd.concat(all_seg)
    for src, g in segs.groupby("source"):
        print(f"  {src} segments: {(g.pred == g.phase).mean():.1%} correct ({len(g)} segments)")

    clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=0, class_weight="balanced")
    clf.fit(df[feature_cols], df.phase)
    imp = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nMost useful features:")
    print(imp.head(8).round(3).to_string())


if __name__ == "__main__":
    main()
