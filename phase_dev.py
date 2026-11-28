"""
Phase 9 (11/26): improvement work on the races already marked, each of the
six marked skaters left out in turn (Swings, van der Poel, Bloemen, Bergsma,
Ghiotto, Eitrem). Results chosen this way are optimistic, so a fresh race
stays the final check (decided 11/26).

  --settings   window length and smoothing of the pose model (#8)
  --feet       the foot-visibility cutoff for "can't judge" (#4)
  --bounds     corner boundary timing: smaller skaters allowed only near entries
               and exits, so more boundaries can be scored (#1)
  --combo      11/27: the best of the above together (smaller skaters near
               boundaries down to 100 px, smoothing 0.8 s, window 0.7, 1.0 or
               1.5 s: the trade-off between frame accuracy and timing)

All use the round 4 scoring: average of corner and straight scores, 0.25 s
margin around marks, boundaries within 0.5 s (two-sided and one-sided).

Usage:
    python -m phase_dev --settings
"""

import argparse
import numpy as np
import pandas as pd

import phase_model as pm
from label_audit import frame_labels
from mark_phases import race
from skater_pose import usable

SKATERS = pm.ROUNDS[4]["train"] + pm.ROUNDS[4]["test"]
GAP = 5


def load_near(pair, sk, near_leg, near_s):
    """Frames the model normally uses, plus frames down to near_leg px (720p) within near_s of an entry or exit,
    kept only if they belong to a tracked person the researcher confirmed as this skater."""
    d, marks = pm.load(pair, sk, GAP)
    d["near_only"] = False
    if near_leg is None:
        return d, marks
    pose, runs = pm.DATA[(pair, sk)]
    sc = race(pair)["scale"]
    small = usable(pd.read_csv(pose), marks, GAP, sc * near_leg / 150)
    r = pd.read_csv(runs)
    ok_tracks = set(r[(r.skater == sk) & (r.identity == "ok")].track_id)
    edges = marks[marks.event.isin(["entry", "exit"])].frame.values
    fps = race(pair)["fps"]
    near = small.frame.apply(lambda f: np.abs(edges - f).min() <= near_s * fps)
    small = small[near & small.track_id.isin(ok_tracks) & ~small.frame.isin(d.frame)].copy()
    lab = frame_labels(marks)
    small["label"] = small.frame.map(lab)
    small = small[small.label.notna()]
    small["skater"], small["near_only"] = sk, True
    return pd.concat([d, small], ignore_index=True).sort_values("frame"), marks


def build(win_s=None, near_leg=None, near_s=1.5):
    fs, ms = [], {}
    for pair, sk in SKATERS:
        d, marks = load_near(pair, sk, near_leg, near_s)
        fps = race(pair)["fps"]
        t = pm.frames_at(fps)
        win = t["win"] if win_s is None else (lambda x: x if x % 2 else x + 1)(int(win_s * fps))
        # runs must be rebuilt after adding frames, so each run is a stretch of the same track
        d = d.sort_values("frame")
        d["run"] = (d.track_id != d.track_id.shift()).cumsum()
        f = pm.features(d, False, win, t["min_win"], fps)
        f["skater"], f["pair"] = sk, pair
        f = f.merge(d[["frame", "foot_vis", "near_only"]], on="frame", how="left")
        fs.append(f); ms[sk] = marks
    return pd.concat(fs, ignore_index=True), ms


def run(data, marks, smooth_s=0.5, feet=pm.FEET_MIN, label="", verbose=False):
    rows = []
    cols = [c for c in data.columns if c not in ("frame", "label", "run", "skater", "pair", "foot_vis", "near_only")]
    for pair, sk in SKATERS:
        tr, te = data[data.skater != sk], data[data.skater == sk]
        w = tr.label.map(len(tr) / (2 * tr.label.value_counts()))
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(random_state=0).fit(tr[cols], tr.label, sample_weight=w)
        fps = race(pair)["fps"]
        pred = model.predict(te[cols])
        keep = (te.foot_vis >= feet).values if feet is not None else np.ones(len(te), bool)
        t = pm.frames_at(fps)
        k = (lambda x: x if x % 2 else x + 1)(int(smooth_s * fps))
        tk, pk = te[keep], pred[keep]
        sm = pm.smooth(tk.frame.values, tk.run.values, pk, k)
        b = pm.boundaries(tk.frame.values, sm, marks[sk], True, t["tol"])
        # frame scoring: normal-size frames only, outside the 0.25 s margin
        edges = marks[sk][marks[sk].event.isin(["entry", "exit"])].frame.values
        far = np.array([np.abs(edges - f).min() > 0.25 * fps for f in tk.frame.values])
        fr = far & ~tk.near_only.values
        y, s_ = tk.label.values[fr], sm[fr]
        rows.append({"skater": sk, "judged": round(float(keep[~te.near_only.values].mean()), 3),
                     "corner": float((s_[y == "corner"] == "corner").mean()), "n_c": int((y == "corner").sum()),
                     "straight": float((s_[y == "straight"] == "straight").mean()), "n_s": int((y == "straight").sum()),
                     "b_found": int(b.found.sum()) if len(b) else 0, "b_n": len(b)})
    t = pd.DataFrame(rows)
    c = (t.corner * t.n_c).sum() / t.n_c.sum()
    s = (t.straight * t.n_s).sum() / t.n_s.sum()
    if verbose:
        print(t.round(3).to_string(index=False))
    print(f"{label:38s} corner {c:.3f}  straight {s:.3f}  average {(c + s) / 2:.3f}  judged {t.judged.mean():.2f}  "
          f"boundaries {t.b_found.sum()}/{t.b_n.sum()}"
          + (f" ({t.b_found.sum() / t.b_n.sum():.0%})" if t.b_n.sum() else ""), flush=True)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", action="store_true")
    ap.add_argument("--feet", action="store_true")
    ap.add_argument("--bounds", action="store_true")
    ap.add_argument("--combo", action="store_true")
    a = ap.parse_args()
    if a.settings:
        for win_s in (0.7, 1.0, 1.5):
            data, marks = build(win_s)
            for smooth_s in (0.3, 0.5, 0.8):
                run(data, marks, smooth_s, label=f"window {win_s} s, smoothing {smooth_s} s")
    if a.feet:
        data, marks = build()
        for feet in (None, 0.1, 0.2, 0.3, 0.4, 0.5):
            run(data, marks, feet=feet, label=f"feet cutoff {feet}")
    if a.combo:
        for win_s in (0.7, 1.0, 1.5):
            data, marks = build(win_s, near_leg=100)
            run(data, marks, 0.8, label=f"near 100 px, window {win_s} s, smoothing 0.8 s", verbose=True)
    if a.bounds:
        for near_leg in (None, 120, 100, 80):
            data, marks = build(near_leg=near_leg)
            run(data, marks, label=f"near boundaries down to {near_leg} px", verbose=near_leg is None)


if __name__ == "__main__":
    main()
