"""
Phase 6f (10/17): corner/straight classifier (6d) on 3D vs 2D angles.

Same segments, same windows, same random forest and leave-one-skater-out
test as phase_classifier.py (angles-only variant), once with 2D angles and
once with the 3D estimates from extract_3d_segments.py. Signals: trunk lean
right and left, their difference, knee angles right and left, their
difference. Target unchanged: >= 85% segment accuracy.

Usage:
    python -m validate_6f_classifier
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

FRAMES = "phase6f_5f_segments_3d.csv"
WIN = int(os.environ.get("PHASE_CLF_WIN", "50"))
STEP = 12
TARGET = 0.85


def signals(f, dim):
    return pd.DataFrame({
        "lean_r": f[f"lean{dim}_r"], "lean_l": f[f"lean{dim}_l"],
        "lean_lr": (f[f"lean{dim}_r"] - f[f"lean{dim}_l"]).abs(),
        "knee_r": f[f"knee{dim}_r"], "knee_l": f[f"knee{dim}_l"],
        "knee_lr": (f[f"knee{dim}_r"] - f[f"knee{dim}_l"]).abs(),
    })


def windows(fr, dim):
    rows = []
    for key, g in fr.groupby(["skater", "video_path", "start_frame", "phase"]):
        s = signals(g.sort_values("frame"), dim).dropna()
        for i in range(0, len(s) - WIN + 1, STEP):
            w = s.iloc[i:i + WIN]
            row = {f"{c}_mean": w[c].mean() for c in w.columns}
            row.update({f"{c}_std": w[c].std() for c in w.columns})
            rows.append({**row, "skater": key[0], "phase": key[3], "segment": f"{key[1]}|{key[2]}"})
    return pd.DataFrame(rows)


def loso(df):
    feats = [c for c in df.columns if c not in ("skater", "phase", "segment")]
    seg_ok, n = 0, 0
    for sk in sorted(df.skater.unique()):
        tr, te = df[df.skater != sk], df[df.skater == sk]
        clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=0, class_weight="balanced")
        clf.fit(tr[feats], tr.phase)
        te = te.assign(pred=clf.predict(te[feats]))
        seg = te.groupby("segment").agg(phase=("phase", "first"), pred=("pred", lambda s: s.value_counts().idxmax()))
        seg_ok += (seg.pred == seg.phase).sum(); n += len(seg)
    return seg_ok / n, n


def main():
    fr = pd.read_csv(FRAMES)
    for dim in ("2d", "3d"):
        acc, n = loso(windows(fr, dim))
        print(f"{dim.upper()} angles, {WIN}-frame windows: segment accuracy {acc:.1%} on {n} segments "
              f"(target {TARGET:.0%}: {'MET' if acc >= TARGET else 'NOT MET'})")


if __name__ == "__main__":
    main()
