"""
Phase 9b (11/16): the skating phase model (corner vs straight per frame).

Setup fixed before training, in docs/CAPABILITIES_PHASE9.md ("Round 1
setup"): frames inside a skater's marked race, live, leg >= 150 px, steady
track, in a stretch the researcher confirmed as the right skater; labels from
the marks (straight strokes inside a corner count as corner); inputs are the
skater's own pose over a centred 31-frame window; a gradient-boosted tree
classifier with classes weighted equally, then a 15-frame majority filter
within each stretch.

Scoring: share of frames correct (target 80%), and corner entries/exits found
within 15 frames (target 85%; scorable only where the test frames cover 80%
of the 15 frames each side).

Usage:
    python -m phase_model --loso      # pair 10: train on one skater, test on the other (allowed for choices)
    python -m phase_model --round 1   # train on pair 10, test once on pair 09 (refuses to run twice)
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from label_audit import frame_labels
from skater_pose import usable, short

MARKS = "phase9/phase_marks.csv"
WIN, MIN_WIN, SMOOTH, TOL = 31, 10, 15, 15
DATA = {  # pair, skater -> (pose file, researcher's stretch check)
    ("pair10_vanderpoel", "Bart Swings"): ("phase9/pose_pair10_vanderpoel_swings_v1.csv", "phase9/runs_pair10_vanderpoel.csv"),
    ("pair10_vanderpoel", "Nils van der Poel"): ("phase9/pose_pair10_vanderpoel_poel_v1.csv", "phase9/runs_pair10_vanderpoel.csv"),
    ("pair09_bloemen", "Ted-Jan Bloemen"): ("phase9/pose_pair09_bloemen_bloemen.csv", "phase9/runs_pair09_bloemen.csv"),
}
ROUNDS = {1: {"train": [("pair10_vanderpoel", "Bart Swings"), ("pair10_vanderpoel", "Nils van der Poel")],
              "test": [("pair09_bloemen", "Ted-Jan Bloemen")]}}


def load(pair, skater):
    pose, runs = DATA[(pair, skater)]
    marks = pd.read_csv(MARKS)
    marks = marks[(marks.pair == pair) & (marks.skater == skater) & (marks.use == "yes")]
    d = usable(pd.read_csv(pose), marks)
    r = pd.read_csv(runs)
    r = r[(r.skater == skater) & (r.identity == "ok")]
    keep = np.zeros(len(d), bool)
    for _, x in r.iterrows():
        keep |= (d.frame.between(x["first"], x["last"]) & (d.track_id == x.track_id)).values
    d = d[keep].copy()
    lab = frame_labels(marks)
    d["label"] = d.frame.map(lab)
    d = d[d.label.notna()]
    d["skater"] = skater
    return d, marks


def features(d):
    out = []
    for run, g in d.groupby("run"):
        g = g.set_index("frame").reindex(range(g.frame.min(), g.frame.max() + 1))
        leg = g.leg_px
        sig = pd.DataFrame({"lf": g.lf / leg, "rf": g.rf / leg, "gap": (g.lf - g.rf).abs() / leg,
                            "cross": (g.lf < g.rf).astype(float).where(g.lf.notna()),
                            "lfy": (g.lf_y - g.hip_y) / leg, "rfy": (g.rf_y - g.hip_y) / leg,
                            "dhx": g.hip_x.diff() / leg, "dhy": g.hip_y.diff() / leg})
        roll = sig.rolling(WIN, center=True, min_periods=MIN_WIN)
        f = pd.concat([roll.mean().add_suffix("_mean"), roll.std().add_suffix("_std"),
                       roll.min().add_suffix("_min"), roll.max().add_suffix("_max")], axis=1)
        f["n_win"] = sig.lf.notna().astype(int).rolling(WIN, center=True, min_periods=1).sum()
        f = f[g.label.notna() & (f.n_win >= MIN_WIN)].drop(columns="n_win")
        f["label"] = g.label[f.index]
        f["run"] = run
        out.append(f.reset_index().rename(columns={"index": "frame"}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def smooth(frames, runs, pred):
    s = pd.Series(pred == "corner", index=frames).astype(float)
    out = []
    for r in pd.unique(runs):
        m = runs == r
        x = s[m].rolling(SMOOTH, center=True, min_periods=1).mean()
        out.append(np.where(x >= 0.5, "corner", "straight"))
    return np.concatenate(out)


def boundaries(frames, pred, marks):
    have = dict(zip(frames, pred))
    rows = []
    for _, m in marks[marks.event.isin(["entry", "exit"])].iterrows():
        f = int(m.frame)
        before = [have.get(x) for x in range(f - TOL, f)]
        after = [have.get(x) for x in range(f, f + TOL)]
        if sum(x is not None for x in before) < 0.8 * TOL or sum(x is not None for x in after) < 0.8 * TOL:
            continue
        want = ("straight", "corner") if m.event == "entry" else ("corner", "straight")
        seq = [have.get(x) for x in range(f - TOL, f + TOL + 1)]
        found = any(seq[i] == want[0] and seq[i + 1] == want[1] for i in range(len(seq) - 1))
        rows.append({"frame": f, "event": m.event, "found": found})
    return pd.DataFrame(rows)


def fit(train):
    X = train.drop(columns=["frame", "label", "run", "skater"])
    w = train.label.map(len(train) / (2 * train.label.value_counts()))
    return HistGradientBoostingClassifier(random_state=0).fit(X, train.label, sample_weight=w), list(X.columns)


def evaluate(model, cols, test, marks):
    pred = model.predict(test[cols])
    sm = smooth(test.frame.values, test.run.values, pred)
    acc = float((sm == test.label.values).mean())
    b = boundaries(test.frame.values, sm, marks)
    return {"frames": len(test), "frames_correct": round(acc, 3), "raw_correct": round(float((pred == test.label.values).mean()), 3),
            "boundaries_scorable": len(b), "boundaries_found": int(b.found.sum()) if len(b) else 0,
            "boundary_share": round(float(b.found.mean()), 3) if len(b) else None}


def build(keys):
    fs, ms = [], {}
    for pair, sk in keys:
        d, marks = load(pair, sk)
        f = features(d)
        f["skater"] = sk
        fs.append(f); ms[sk] = marks
    return pd.concat(fs, ignore_index=True), ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loso", action="store_true")
    ap.add_argument("--round", type=int)
    args = ap.parse_args()
    if args.loso:
        data, marks = build(ROUNDS[1]["train"])
        for sk in data.skater.unique():
            model, cols = fit(data[data.skater != sk])
            print("test on", sk, evaluate(model, cols, data[data.skater == sk], marks[sk]))
    if args.round:
        out = f"phase9/round{args.round}_result.json"
        if os.path.exists(out):
            raise SystemExit(f"{out} exists: round {args.round} was already run once")
        train, _ = build(ROUNDS[args.round]["train"])
        test, marks = build(ROUNDS[args.round]["test"])
        model, cols = fit(train)
        res = {"round": args.round, "train_frames": len(train)}
        for sk in test.skater.unique():
            res[sk] = evaluate(model, cols, test[test.skater == sk], marks[sk])
        json.dump(res, open(out, "w"), indent=2)
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
