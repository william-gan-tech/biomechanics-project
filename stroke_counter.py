"""
Phase 8c (10/29): automatic stroke counter, scored against hand-marked
set-downs (stroke_rate/hand_counts.csv).

Signal: how far apart the two feet are sideways in the image
(|left foot x - right foot x| / leg length, smoothed). Each push ends with
the feet furthest apart; the recovering skate sets down about OFFSET frames
later. Compared on the tune clips (Kramer, Roest) against foot height
(Phase 7 touchdowns), feet closest together, and foot-to-foot distance; the
widest-apart moment matched the hand marks best and was stable across
settings (75.7% of tune set-downs at prominence 0.15-0.40, smoothing 3-7).

Settings below were fixed on the tune clips before the test clips were
scored. Scoring rules, also fixed before the test:
  - a set-down is found if an automatic one is within TOL frames (one to one)
  - only hand marks the counter can see both sides of are scored: at least
    OFFSET + TOL frames after the clip start and TOL frames before its end
    (most hand-marked straightaways end on a set-down, the step into the
    corner); automatic set-downs outside the same window are ignored
  - target: >= 85% of scored hand marks found, extras <= 15% of them,
    judged on test clips with leg >= MIN_LEG_PX; smaller clips reported
    separately

Usage:
    python -m stroke_counter          # tune clips only
    python -m stroke_counter --test   # the one pre-set test
"""

import argparse
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from doublepush_measures import MIN_LEG_PX

COUNTS = "stroke_rate/hand_counts.csv"
PROM = 0.2
MIN_GAP = 12        # frames between strokes (0.4 s)
SMOOTH = 5
OFFSET = 7          # widest-apart moment to set-down, frames
TOL = 3
TARGET, MAX_EXTRA = 0.85, 0.15


def setdowns(d):
    out = []
    for _, g in d.groupby("cut_seq"):
        if len(g) < 10:
            continue
        sep = pd.Series(np.abs(g.lf.values - g.rf.values) / g.leg_px.median())
        sep = sep.rolling(SMOOTH, center=True, min_periods=1).mean().values
        p, _ = find_peaks(sep, distance=MIN_GAP, prominence=PROM)
        out += [int(f) + OFFSET for f in g.frame.values[p]]
    return sorted(out)


def score(marks, auto, start, end):
    lo, hi = start + OFFSET + TOL, end - TOL
    marks = [m for m in marks if lo <= m <= hi]
    auto = [a for a in auto if lo - TOL <= a <= hi + TOL]
    used, hit = set(), 0
    for m in marks:
        c = [a for a in auto if abs(a - m) <= TOL and a not in used]
        if c:
            used.add(min(c, key=lambda a: abs(a - m)))
            hit += 1
    extra = len([a for a in auto if a not in used and lo <= a <= hi])
    return hit, len(marks), extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    h = pd.read_csv(COUNTS, dtype=str)
    h = h[h["set"] == ("test" if args.test else "tune")]
    rows = []
    for _, r in h.iterrows():
        d = pd.read_csv(f"stroke_rate/frames_{r['clip']}.csv")
        marks = [int(x) for x in r.setdown_frames.split()] if pd.notna(r.setdown_frames) else []
        auto = setdowns(d)
        hit, n, extra = score(marks, auto, int(r.start_frame), int(r.end_frame))
        rows.append({"clip": r["clip"], "skater": r.skater, "lap": r.lap, "leg_px": round(d.leg_px.median()),
                     "hand": " ".join(map(str, marks)), "auto": " ".join(map(str, auto)),
                     "scored": n, "found": hit, "extra": extra})
    t = pd.DataFrame(rows)
    t.to_csv(f"stroke_rate/counter_{'test' if args.test else 'tune'}.csv", index=False)
    print(t[["clip", "skater", "lap", "leg_px", "scored", "found", "extra"]].to_string(index=False))
    for name, g in (("normal size (leg >= %d px)" % MIN_LEG_PX, t[t.leg_px >= MIN_LEG_PX]),
                    ("small (leg < %d px)" % MIN_LEG_PX, t[t.leg_px < MIN_LEG_PX])):
        n = g.scored.sum()
        if n:
            print(f"{name}: found {g.found.sum()}/{n} = {g.found.sum() / n:.1%}, extra {g.extra.sum()} = {g.extra.sum() / n:.1%}")
    if args.test:
        g = t[t.leg_px >= MIN_LEG_PX]
        ok = g.found.sum() / g.scored.sum() >= TARGET and g.extra.sum() / g.scored.sum() <= MAX_EXTRA
        print("TARGET", "MET" if ok else "NOT MET")


if __name__ == "__main__":
    main()
