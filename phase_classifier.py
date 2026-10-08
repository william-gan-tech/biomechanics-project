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

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from compare_to_elite_reference import load_labeled_segments, get_segment_frames

SIGNALS = ["torso_lean_angle_deg", "torso_lean_angle_deg_signed", "torso_lean_lr_diff",
           "right_knee_filtered", "left_knee_filtered", "knee_angle_asymmetry",
           "hip_height_asymmetry", "hip_lateral_asymmetry"]
WIN, STEP = 25, 12
TARGET = 0.85  # accuracy needed to call automatic detection usable
OUTPUT_PATH = "phase5g_phase_classifier_results.csv"


def windows_for(seg_frames):
    f = seg_frames.sort_values("frame")
    f = f[[c for c in SIGNALS if c in f.columns]].dropna()
    feats = []
    for i in range(0, len(f) - WIN + 1, STEP):
        w = f.iloc[i:i + WIN]
        row = {}
        for c in w.columns:
            row[f"{c}_mean"] = w[c].mean()
            row[f"{c}_std"] = w[c].std()
        # knees alternate bending; in crossovers (corners) the pattern differs
        row["knee_diff_signed_mean"] = (w["left_knee_filtered"] - w["right_knee_filtered"]).mean()
        feats.append(row)
    return feats


def build_dataset():
    rows = []
    for r in load_labeled_segments():
        if r["phase"] not in ("corner", "straightaway") or "lap" not in r.get("notes", ""):
            continue
        f = get_segment_frames(r)
        if f is None:
            continue
        for w in windows_for(f):
            rows.append({**w, "skater": r["skater"], "phase": r["phase"],
                         "segment": f"{r['video_path']}|{r['start_frame']}-{r['end_frame']}"})
    return pd.DataFrame(rows)


def main():
    df = build_dataset()
    feature_cols = [c for c in df.columns if c not in ("skater", "phase", "segment")]
    df = df.dropna(subset=feature_cols)
    print(f"{len(df)} one-second windows from {df.segment.nunique()} segments, {df.skater.nunique()} skaters")
    print(df.groupby("phase").segment.nunique().to_string())

    results = []
    for skater in sorted(df.skater.unique()):
        train, test = df[df.skater != skater], df[df.skater == skater]
        if test.phase.nunique() < 1 or train.phase.nunique() < 2:
            continue
        clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=0,
                                     class_weight="balanced")
        clf.fit(train[feature_cols], train.phase)
        test = test.assign(pred=clf.predict(test[feature_cols]))
        seg = test.groupby("segment").agg(phase=("phase", "first"),
                                          pred=("pred", lambda s: s.value_counts().idxmax()))
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

    clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=0, class_weight="balanced")
    clf.fit(df[feature_cols], df.phase)
    imp = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nMost useful features:")
    print(imp.head(8).round(3).to_string())


if __name__ == "__main__":
    main()
