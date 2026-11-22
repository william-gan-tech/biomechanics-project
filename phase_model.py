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

Round 2 rules (fixed 11/17 and 11/19, in the same doc): frames scored as the
average of the corner and straight scores (target 85%); a boundary is also
scorable with one side covered, and then found if no frame on that side more
than 15 frames from the mark has the wrong label; stretches continue through
misses of up to 5 frames (skater_pose.GAP).

Usage:
    python -m phase_model --loso      # pair 10: train on one skater, test on the other (allowed for choices)
    python -m phase_model --round 1   # train on pair 10, test once on pair 09 (refuses to run twice)
    python -m phase_model --round 2   # train on pair 10 and Bloemen, test once on pair 06
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
    ("pair06_bergsma", "Jorrit Bergsma"): ("phase9/pose_pair06_bergsma_bergsma.csv", "phase9/runs_pair06_bergsma.csv"),
    ("pair06_bergsma", "Seitaro Ichinohe"): ("phase9/pose_pair06_bergsma_ichinohe.csv", "phase9/runs_pair06_bergsma.csv"),
}
ROUNDS = {1: {"train": [("pair10_vanderpoel", "Bart Swings"), ("pair10_vanderpoel", "Nils van der Poel")],
              "test": [("pair09_bloemen", "Ted-Jan Bloemen")], "gap": 0},
          2: {"train": [("pair10_vanderpoel", "Bart Swings"), ("pair10_vanderpoel", "Nils van der Poel"),
                        ("pair09_bloemen", "Ted-Jan Bloemen")],
              "test": [("pair06_bergsma", "Jorrit Bergsma"), ("pair06_bergsma", "Seitaro Ichinohe")], "gap": 5}}


def load(pair, skater, gap=0):
    pose, runs = DATA[(pair, skater)]
    marks = pd.read_csv(MARKS)
    marks = marks[(marks.pair == pair) & (marks.skater == skater) & (marks.use == "yes")]
    from mark_phases import race
    d = usable(pd.read_csv(pose), marks, gap, race(pair)["scale"])
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


def cue_features(g, leg, win):
    """11/21, the researcher's straight cues: on straights the feet do not cross (more crossing = corner);
    crossovers have much more lean; on straights a lean comes with the other leg pushing out wide, in corners
    the legs stay close during the lean. Needs shoulder points (augment_pose.py)."""
    side = np.sign(g.lf - g.rf)
    swap = (side != side.shift()).astype(float).where(side.notna() & side.shift().notna())
    sh_x, sh_y = (g.lsh_x + g.rsh_x) / 2, (g.lsh_y + g.rsh_y) / 2
    hx, hy = (g.lhip_x + g.rhip_x) / 2, (g.lhip_y + g.rhip_y) / 2
    lean = np.degrees(np.arctan2((sh_x - hx).abs(), (hy - sh_y)))          # 0 = upright, sideways in the image
    gap = (g.lf - g.rf).abs() / leg
    r = lambda x, fn: getattr(x.rolling(win, center=True, min_periods=MIN_WIN), fn)()
    hi = lean >= r(lean, "median")                                          # the more-leaning half of the window
    gap_lean = gap.where(hi)
    return pd.DataFrame({"swap_rate": r(swap, "mean"), "lean_mean": r(lean, "mean"), "lean_max": r(lean, "max"),
                         "lean_std": r(lean, "std"), "gap_when_leaning": r(gap_lean, "mean"),
                         "gap_lean_ratio": r(gap_lean, "mean") / (r(lean, "mean") + 1)})


def features(d, cues=False, win=WIN):
    out = []
    for run, g in d.groupby("run"):
        g = g.set_index("frame").reindex(range(g.frame.min(), g.frame.max() + 1))
        leg = g.leg_px
        sig = pd.DataFrame({"lf": g.lf / leg, "rf": g.rf / leg, "gap": (g.lf - g.rf).abs() / leg,
                            "cross": (g.lf < g.rf).astype(float).where(g.lf.notna()),
                            "lfy": (g.lf_y - g.hip_y) / leg, "rfy": (g.rf_y - g.hip_y) / leg,
                            "dhx": g.hip_x.diff() / leg, "dhy": g.hip_y.diff() / leg})
        roll = sig.rolling(win, center=True, min_periods=MIN_WIN)
        f = pd.concat([roll.mean().add_suffix("_mean"), roll.std().add_suffix("_std"),
                       roll.min().add_suffix("_min"), roll.max().add_suffix("_max")], axis=1)
        if cues:
            f = pd.concat([f, cue_features(g, leg, win)], axis=1)
        f["n_win"] = sig.lf.notna().astype(int).rolling(win, center=True, min_periods=1).sum()
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


def boundaries(frames, pred, marks, one_sided=False):
    """Two-sided (round 1): both sides covered, found if the label changes the right way within TOL.
    One-sided (round 2 adds): one side covered, found if no frame on that side beyond TOL has the wrong label."""
    have = dict(zip(frames, pred))
    rows = []
    for _, m in marks[marks.event.isin(["entry", "exit"])].iterrows():
        f = int(m.frame)
        want = ("straight", "corner") if m.event == "entry" else ("corner", "straight")
        cov_b = sum(have.get(x) is not None for x in range(f - TOL, f)) >= 0.8 * TOL
        cov_a = sum(have.get(x) is not None for x in range(f, f + TOL)) >= 0.8 * TOL
        if cov_b and cov_a:
            seq = [have.get(x) for x in range(f - TOL, f + TOL + 1)]
            found = any(seq[i] == want[0] and seq[i + 1] == want[1] for i in range(len(seq) - 1))
            rows.append({"frame": f, "event": m.event, "sides": 2, "found": found})
        elif one_sided and (cov_b or cov_a):
            # the covered side, out to the next mark: frames more than TOL from this mark must have that side's label
            nxt = marks.frame[marks.frame > f].min() if cov_a else marks.frame[marks.frame < f].max()
            span = range(f + TOL, int(nxt) if pd.notna(nxt) else f + 300) if cov_a else range(int(nxt) if pd.notna(nxt) else f - 300, f - TOL)
            side = want[1] if cov_a else want[0]
            labs = [have[x] for x in span if x in have]
            if not labs:
                continue
            rows.append({"frame": f, "event": m.event, "sides": 1, "found": all(x == side for x in labs)})
    return pd.DataFrame(rows)


def fit(train):
    X = train.drop(columns=["frame", "label", "run", "skater"])
    w = train.label.map(len(train) / (2 * train.label.value_counts()))
    return HistGradientBoostingClassifier(random_state=0).fit(X, train.label, sample_weight=w), list(X.columns)


def evaluate(model, cols, test, marks, one_sided=False):
    pred = model.predict(test[cols])
    sm = smooth(test.frame.values, test.run.values, pred)
    y = test.label.values
    acc = float((sm == y).mean())
    per = {lab: round(float((sm[y == lab] == lab).mean()), 3) if (y == lab).any() else None for lab in ("corner", "straight")}
    b = boundaries(test.frame.values, sm, marks, one_sided)
    return {"frames": len(test), "labels": {lab: int((y == lab).sum()) for lab in ("corner", "straight")},
            "frames_correct": round(acc, 3), "corner_correct": per["corner"], "straight_correct": per["straight"],
            "balanced_correct": round(np.mean([v for v in per.values() if v is not None]), 3),
            "raw_correct": round(float((pred == y).mean()), 3),
            "boundaries_scorable": len(b), "boundaries_two_sided": int((b.sides == 2).sum()) if len(b) else 0,
            "boundaries_found": int(b.found.sum()) if len(b) else 0,
            "boundary_share": round(float(b.found.mean()), 3) if len(b) else None,
            "boundary_verdict": "too little data" if len(b) < 10 else ("met" if b.found.mean() >= 0.85 else "not met")}


def pooled(rs):
    """Test skaters together (a pair's skaters share frame numbers, so counts are added, not frames merged)."""
    n = {lab: sum(r["labels"][lab] for r in rs) for lab in ("corner", "straight")}
    per = {lab: round(sum((r[f"{lab}_correct"] or 0) * r["labels"][lab] for r in rs) / n[lab], 3) if n[lab] else None
           for lab in ("corner", "straight")}
    sc, fo = sum(r["boundaries_scorable"] for r in rs), sum(r["boundaries_found"] for r in rs)
    return {"frames": sum(r["frames"] for r in rs), "labels": n, "corner_correct": per["corner"],
            "straight_correct": per["straight"], "balanced_correct": round(np.mean([v for v in per.values() if v is not None]), 3),
            "boundaries_scorable": sc, "boundaries_found": fo, "boundary_share": round(fo / sc, 3) if sc else None,
            "boundary_verdict": "too little data" if sc < 10 else ("met" if fo / sc >= 0.85 else "not met")}


def build(keys, gap=0, cues=False):
    fs, ms = [], {}
    for pair, sk in keys:
        d, marks = load(pair, sk, gap)
        f = features(d, cues)
        f["skater"] = sk
        fs.append(f); ms[sk] = marks
    return pd.concat(fs, ignore_index=True), ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loso", action="store_true")
    ap.add_argument("--loso4", action="store_true", help="leave one of the four Beijing skaters out, with and without the cues")
    ap.add_argument("--round", type=int)
    args = ap.parse_args()
    if args.loso4:
        keys = ROUNDS[2]["train"] + [("pair06_bergsma", "Jorrit Bergsma")]
        for cues in (False, True):
            data, marks = build(keys, gap=5, cues=cues)
            rs = []
            for sk in data.skater.unique():
                model, cols = fit(data[data.skater != sk])
                r = evaluate(model, cols, data[data.skater == sk], marks[sk], one_sided=True)
                rs.append(r)
                print("cues" if cues else "base", f"{sk:18s}", {k: r[k] for k in ("labels", "corner_correct", "straight_correct", "balanced_correct")})
            print("cues" if cues else "base", "POOLED", {k: v for k, v in pooled(rs).items() if k in ("labels", "corner_correct", "straight_correct", "balanced_correct")})
    if args.loso:
        data, marks = build(ROUNDS[1]["train"])
        for sk in data.skater.unique():
            model, cols = fit(data[data.skater != sk])
            print("test on", sk, evaluate(model, cols, data[data.skater == sk], marks[sk]))
    if args.round:
        out = f"phase9/round{args.round}_result.json"
        if os.path.exists(out):
            raise SystemExit(f"{out} exists: round {args.round} was already run once")
        cfg = ROUNDS[args.round]
        train, _ = build(cfg["train"], cfg["gap"])
        test, marks = build(cfg["test"], cfg["gap"])
        model, cols = fit(train)
        res = {"round": args.round, "train_frames": len(train)}
        for sk in test.skater.unique():
            res[sk] = evaluate(model, cols, test[test.skater == sk], marks[sk], one_sided=args.round >= 2)
        if args.round >= 2:
            res["all test skaters"] = pooled([res[sk] for sk in test.skater.unique()])
        json.dump(res, open(out, "w"), indent=2)
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
