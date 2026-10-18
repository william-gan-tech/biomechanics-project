"""
Phase 6f step 3 (10/17): the 5f late-race trunk-lean asymmetry, re-run on
3D pose estimates.

Phase 5: late-race trunk-lean left/right difference rose for 8 of 9 skaters
across all shots (p = 0.002) but was small and unstable within camera shot
type (6 of 9, p = 0.11), because controlling for shot left little data. 6f
step 1 showed 3D trunk lean depends much less on camera view, so all shots
may be pooled.

Target (fixed 10/17, before running): with 3D angles on all shots, one
late-minus-early value per skater, the lead counts as camera-robust if it is
positive for at least 7 of 9 skaters with p < 0.05 (one-sample t-test across
skaters). Anything weaker is reported as not confirmed.

Input: phase6f_5f_segments_3d.csv (extract_3d_segments.py), shot_labels.csv.

Usage:
    python -m validate_6f_5f
"""

import numpy as np
import pandas as pd
from scipy import stats

FRAMES = "phase6f_5f_segments_3d.csv"
SHOT_LABELS = "shot_labels.csv"
OUTPUT = "phase6f_5f_results.csv"


def segments():
    f = pd.read_csv(FRAMES)
    f["lr2d"] = (f.lean2d_r - f.lean2d_l).abs()
    f["lr3d"] = (f.lean3d_r - f.lean3d_l).abs()
    f["knee2d"] = (f.knee2d_r + f.knee2d_l) / 2
    f["knee3d"] = (f.knee3d_r + f.knee3d_l) / 2
    key = ["skater", "video_path", "phase", "stage", "lap", "start_frame", "end_frame"]
    seg = f.groupby(key)[["lr2d", "lr3d", "knee2d", "knee3d"]].mean().reset_index()
    seg["frames"] = f.groupby(key).size().values
    lab = pd.read_csv(SHOT_LABELS, encoding="utf-8-sig")[["video_path", "start_frame", "end_frame", "shot_type"]]
    seg = seg.merge(lab, on=["video_path", "start_frame", "end_frame"], how="left")
    return seg[seg.frames >= 10]


def per_skater(seg, keys, metric):
    out = []
    for k, g in seg.groupby(keys):
        e, l = g[g.stage == "early"], g[g.stage == "late"]
        if len(e) and len(l):
            out.append({"skater": k[0], "d": l[metric].mean() - e[metric].mean()})
    d = pd.DataFrame(out)
    return d.groupby("skater").d.mean() if not d.empty else pd.Series(dtype=float)


def summarize(v):
    if len(v) < 2:
        return {"skaters": len(v)}
    t, p = stats.ttest_1samp(v, 0.0)
    return {"skaters": len(v), "mean": v.mean(), "positive": int((v > 0).sum()), "p_ttest": p,
            "p_sign": stats.binomtest(int((v > 0).sum()), len(v), 0.5).pvalue}


def main():
    seg = segments()
    print(f"{len(seg)} segments, {seg.skater.nunique()} skaters; shot labels for {seg.shot_type.notna().sum()}\n")
    rows = []
    for metric, label in [("lr2d", "trunk-lean L/R difference, 2D"), ("lr3d", "trunk-lean L/R difference, 3D"),
                          ("knee2d", "knee angle, 2D"), ("knee3d", "knee angle, 3D")]:
        for variant, data, keys in [("all shots", seg, ["skater", "phase"]),
                                    ("within shot type", seg.dropna(subset=["shot_type"]), ["skater", "phase", "shot_type"]),
                                    ("side-on only", seg[seg.shot_type == "side"], ["skater", "phase", "shot_type"])]:
            v = per_skater(data, keys, metric)
            rows.append({"metric": label, "variant": variant, **summarize(v)})
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))
    res.to_csv(OUTPUT, index=False)

    target = res[(res.metric == "trunk-lean L/R difference, 3D") & (res.variant == "all shots")].iloc[0]
    met = target.positive >= 7 and target.p_ttest < 0.05
    print(f"\nTARGET (3D, all shots: >= 7 of 9 skaters positive and p < 0.05): "
          f"{int(target.positive)} of {int(target.skaters)}, p = {target.p_ttest:.3f} -> {'MET' if met else 'NOT MET'}")
    print("\nPer skater, all shots (late minus early, trunk-lean L/R difference):")
    print(pd.DataFrame({"2D": per_skater(seg, ["skater", "phase"], "lr2d"),
                        "3D": per_skater(seg, ["skater", "phase"], "lr3d")}).round(1).to_string())


if __name__ == "__main__":
    main()
