"""
Phase 7c, third and final detection attempt (planned 10/24, before testing).

Two measures per 4-second window, combined:
  cross_frac  share of frames with a foot past the hip midline (first measure)
  setdown     median sideways landing position of the recovering foot relative
              to the hips, leg lengths (second measure, recovery_measure.py)
Foot twist is dropped (chance level on 10/23).

Rule: logistic regression on the two measures (standardised), trained on every
labelled window from Joey Mantia, Viktor Hald Thorup, Bart Swings, the three
skaters in Thorup's styles short and the junior 300 m time trial (flawed double
push counts as double push). Probability >= 0.5 = double push.

Preview (not the test): leave-one-skater-out on the training skaters.
Test: doublepush/new_test_labels.csv, skaters never used anywhere, chosen and
labelled by the researcher before any results. Target: >= 85% of test windows
correct. If not met, detection becomes a stated limit of Phase 7.

Usage:
    python -m detect_combined            # preview (leave-one-skater-out)
    python -m detect_combined --test     # train on everything, score the new test set once
"""

import os
import sys
import numpy as np
import pandas as pd
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from doublepush_measures import track, measures, MIN_LEG_PX, WINDOW_S, MIN_TRACKED
from recovery_measure import touchdowns

SOURCES = [("doublepush/style_labels.csv", "doublepush/frames_"),
           ("doublepush/fresh_test_labels.csv", "doublepush/fresh_frames_")]
TEST = ("doublepush/new_test_labels.csv", "doublepush/new_frames_")
TARGET = 0.85
FEATURES = ["cross_frac", "setdown"]


def skater_of(note):
    for key, name in [("Mantia", "Mantia"), ("Thorup", "Thorup"), ("Swings", "Swings"), ("skater 1", "short skater 1"),
                      ("skater 2", "short skater 2"), ("skater 3", "short skater 3"), ("Junior", "junior 300m")]:
        if key in note:
            return name
    return note


def windows(path, prefix, track_missing):
    lab = pd.read_csv(path)
    rows = []
    for n, r in enumerate(lab.itertuples()):
        cache = f"{prefix}{n:02d}.csv"
        fps = cv2.VideoCapture(r.video).get(cv2.CAP_PROP_FPS) or 30.0
        if os.path.exists(cache) and "lf_y" in pd.read_csv(cache, nrows=1).columns:
            d = pd.read_csv(cache)
        elif track_missing:
            d, fps = track(r.video, int(r.start_frame), int(r.end_frame)); d.to_csv(cache, index=False)
        else:
            continue
        a0, a1 = int(r.start_frame), int(r.end_frame)
        w = int(WINDOW_S * fps)
        starts = [a0] if a1 - a0 + 1 <= 1.5 * w else list(range(a0, a1 - w + 2, w))
        for k, a in enumerate(starts):
            b = a1 if len(starts) == 1 else a + w - 1
            g = d[(d.frame >= a) & (d.frame <= b)]
            if len(g) < MIN_TRACKED * (b - a + 1) or len(g) < 15 or g.leg_px.median() < MIN_LEG_PX:
                continue
            t = touchdowns(g, fps)
            if len(t) < 2:
                continue
            rows.append({"stretch": r.skater_note, "skater": skater_of(r.skater_note),
                         "truth": "double_push" if "double_push" in r.label else "normal",
                         "cross_frac": measures(g, fps)["cross_frac"], "setdown": t.setdown.median()})
    return pd.DataFrame(rows)


def model():
    return make_pipeline(StandardScaler(), LogisticRegression())


def main():
    train = pd.concat([windows(p, f, track_missing=False) for p, f in SOURCES], ignore_index=True)
    y = (train.truth == "double_push").astype(int)
    print(f"Training windows: {len(train)} from {train.skater.nunique()} skaters "
          f"({y.sum()} double push, {len(y) - y.sum()} single push)")
    if "--test" not in sys.argv:
        print("\nPREVIEW, leave-one-skater-out (train on the others, test on this skater):")
        ok = []
        for sk in sorted(train.skater.unique()):
            tr, te = train[train.skater != sk], train[train.skater == sk]
            if tr.truth.nunique() < 2:
                continue
            m = model().fit(tr[FEATURES], tr.truth == "double_push")
            c = (m.predict(te[FEATURES]) == (te.truth == "double_push")).mean()
            ok.extend([c] * len(te))
            print(f"  {sk:16s} {c:5.0%}  ({len(te)} windows; styles: {', '.join(sorted(te.truth.unique()))})")
        print(f"  weighted: {np.mean(ok):.1%}")
        return
    m = model().fit(train[FEATURES], y)
    coef = m[-1].coef_[0]
    print(f"Weights (standardised): cross_frac {coef[0]:+.2f}, setdown {coef[1]:+.2f}")
    test = windows(*TEST, track_missing=True)
    test["pred_dp"] = m.predict(test[FEATURES]).astype(bool)
    test["correct"] = test.pred_dp == (test.truth == "double_push")
    print("\nNEW TEST (skaters never used anywhere):")
    for (st, tr_), g in test.groupby(["stretch", "truth"]):
        print(f"  {st[:45]:45s} {tr_:12s} {g.correct.mean():4.0%} ({len(g)} windows)")
    acc = test.correct.mean()
    print(f"\nTest accuracy {acc:.1%} on {len(test)} windows; target {TARGET:.0%}: {'MET' if acc >= TARGET else 'NOT MET'}")
    test.to_csv("doublepush/combined_test_results.csv", index=False)


if __name__ == "__main__":
    main()
