"""
Phase 7c (10/20): double-push detection, rule set on one elite skater and
tested on others.

Detection measure: cross_frac, the share of frames with a foot past the hip
midline (under the body), per 4-second window or short stretch
(doublepush_measures.py). Only judgeable windows (leg >= 150 px) are used.

Rule (set before looking at the test skaters): the cut-off is halfway between
the median cross_frac of Joey Mantia's normal strides (stroke clinic) and his
double push (tutorial). Windows above the cut-off are called double push.

Test: every judgeable window from other skaters (Bart Swings; the three
skaters in Viktor Thorup's styles short; the junior 300 m time trial).
Target (Phase 7 plan, set before testing): >= 85% of test windows correct.
Mantia's flawed double push is reported separately (it should still count as
double push).

Usage:
    python -m validate_7c_detection
"""

import pandas as pd

MEASURES = "doublepush/measures.csv"
TARGET = 0.85


def main():
    m = pd.read_csv(MEASURES)
    m = m[m.judgeable == True].copy()
    is_mantia = m.stretch.str.startswith("Mantia")
    train = m[is_mantia & m.label.isin(["normal", "double_push"])]
    med = train.groupby("label").cross_frac.median()
    cut = (med["normal"] + med["double_push"]) / 2
    print("Rule set on Mantia only:")
    print(f"  normal (clinic) median {med['normal']:.2f} ({(train.label == 'normal').sum()} windows), "
          f"double push (tutorial) median {med['double_push']:.2f} ({(train.label == 'double_push').sum()} windows)")
    print(f"  cut-off {cut:.2f}: above = double push")
    m["pred"] = (m.cross_frac > cut).map({True: "double_push", False: "normal"})
    m["truth"] = m.label.replace({"double_push_flawed": "double_push"})
    m["correct"] = m.pred == m.truth

    print("\nMantia (the skater the rule was set on), for reference:")
    for lab, g in m[is_mantia].groupby("label"):
        print(f"  {lab:20s} {g.correct.mean():.0%} correct ({len(g)} windows)")
    test = m[~is_mantia]
    print("\nTEST (other skaters):")
    for st, g in test.groupby("stretch"):
        print(f"  {st[:45]:45s} {g.label.iloc[0]:12s} {g.correct.mean():.0%} correct ({len(g)} windows; "
              f"cross_frac median {g.cross_frac.median():.2f})")
    acc = test.correct.mean()
    print(f"\nTest accuracy {acc:.1%} on {len(test)} windows from {test.stretch.nunique()} stretches; "
          f"target {TARGET:.0%}: {'MET' if acc >= TARGET else 'NOT MET'}")
    m.to_csv("doublepush/detection_results.csv", index=False)


if __name__ == "__main__":
    main()
