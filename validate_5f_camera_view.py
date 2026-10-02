"""
Phase 5f robustness check (9/30): is the "more upright late in the race"
pattern real, or a camera-viewpoint artifact?

The four-skater 5f run found straighter knees late in the race in 7/7
skater x phase comparisons. Early and late laps are matched by PHASE, but
not by camera shot. If broadcasters film late laps from a different angle
(e.g. more frontal), 2D knee angles and sit height would change even if the
skater's form did not.

View proxy: projected hip width (`hip_lateral_asymmetry`, body-scale units).
Seen from the side, the two hips overlap (small); seen from the front or
back, they are far apart (large). It changes with camera angle, which is
exactly why it is flagged as unreliable as a technique metric -- and exactly
why it works as a measure of viewpoint.

Three checks, each within skater x phase (never pooling across skaters):
  1. Do early and late segments differ in viewpoint at all?
  2. Segment level: knee angle ~ late + viewpoint (+ one intercept per
     skater x phase). Does "late" still predict straighter knees once
     viewpoint is accounted for?
  3. Frame level, matched viewpoint: compare early vs late knee angles only
     on frames whose viewpoint falls in the range BOTH early and late frames
     cover (common support).

Usage:
    python -m validate_5f_camera_view
"""

import numpy as np
import pandas as pd
from scipy import stats

from compare_to_elite_reference import get_segment_frames
from validate_fatigue_form_degradation import lap_segments

VIEW = "hip_lateral_asymmetry"
KNEE = "knee_mean"
OUTPUT_PATH = "phase5f_camera_view_check.csv"


def rank_biserial(a, b):
    u, _ = stats.mannwhitneyu(a, b, alternative="two-sided")
    return 2 * u / (len(a) * len(b)) - 1  # +1 = every a above every b


def load():
    segs, frames = [], []
    for _, r in lap_segments().iterrows():
        f = get_segment_frames(r)
        if f is None or VIEW not in f.columns:
            continue
        f = f.copy()
        f[KNEE] = f[["left_knee_filtered", "right_knee_filtered"]].mean(axis=1)
        f = f.dropna(subset=[VIEW, KNEE])
        if f.empty:
            continue
        key = dict(skater=r["skater"], phase=r["phase"], stage=r["stage"], lap=r["lap"],
                   segment=f"{r['start_frame']}-{r['end_frame']}")
        segs.append({**key, "view": f[VIEW].median(), "knee": f[KNEE].mean(),
                     "sit": f["hip_to_ankle_vertical_right"].mean(), "frames": len(f)})
        frames.append(f[[VIEW, KNEE]].assign(**key))
    return pd.DataFrame(segs), pd.concat(frames, ignore_index=True)


def check1_view_differs(seg):
    rows = []
    for (sk, ph), g in seg.groupby(["skater", "phase"]):
        e, l = g[g.stage == "early"], g[g.stage == "late"]
        if len(e) and len(l):
            rows.append({"skater": sk, "phase": ph, "n_early": len(e), "n_late": len(l),
                         "view_early": e.view.mean(), "view_late": l.view.mean(),
                         "view_pct_change": 100 * (l.view.mean() - e.view.mean()) / e.view.mean()})
    return pd.DataFrame(rows)


def check2_segment_regression(seg):
    g = seg[seg.groupby(["skater", "phase"]).stage.transform(lambda s: {"early", "late"} <= set(s))].copy()
    g["late"] = (g.stage == "late").astype(float)
    groups = pd.get_dummies(g.skater + "|" + g.phase, dtype=float)

    def fit(cols):
        X = np.column_stack([groups.values] + [g[c].values for c in cols])
        y = g["knee"].values
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        dof = len(y) - X.shape[1]
        cov = (resid @ resid / dof) * np.linalg.pinv(X.T @ X)
        k = groups.shape[1]
        out = {}
        for i, c in enumerate(cols):
            b, se = beta[k + i], np.sqrt(cov[k + i, k + i])
            out[c] = (b, se, 2 * stats.t.sf(abs(b / se), dof))
        return out, len(y), dof

    without, n, _ = fit(["late"])
    with_view, _, dof = fit(["late", "view"])
    return without, with_view, n, dof


def check3_matched_frames(fr):
    rows = []
    for (sk, ph), g in fr.groupby(["skater", "phase"]):
        e, l = g[g.stage == "early"], g[g.stage == "late"]
        if e.empty or l.empty:
            continue
        lo = max(e[VIEW].quantile(0.05), l[VIEW].quantile(0.05))
        hi = min(e[VIEW].quantile(0.95), l[VIEW].quantile(0.95))
        em, lm = e[(e[VIEW] >= lo) & (e[VIEW] <= hi)], l[(l[VIEW] >= lo) & (l[VIEW] <= hi)]
        row = {"skater": sk, "phase": ph,
               "knee_change_all": l[KNEE].mean() - e[KNEE].mean(),
               "effect_all": rank_biserial(l[KNEE], e[KNEE]),
               "matched_early_frames": len(em), "matched_late_frames": len(lm)}
        if len(em) >= 20 and len(lm) >= 20:
            row.update({"knee_change_matched": lm[KNEE].mean() - em[KNEE].mean(),
                        "effect_matched": rank_biserial(lm[KNEE], em[KNEE]),
                        "view_effect_matched": rank_biserial(lm[VIEW], em[VIEW])})
        rows.append(row)
    return pd.DataFrame(rows)


OTHER_METRICS = ["torso_lean_lr_diff", "hip_height_asymmetry", "torso_lean_angle_deg",
                 "knee_angle_asymmetry", "hip_to_ankle_vertical_right"]


def check4_other_metrics():
    """9/30: the same segment-level test (metric ~ late [+ viewpoint], one
    intercept per skater x phase) for the other 5f patterns."""
    rows = []
    recs = []
    for _, r in lap_segments().iterrows():
        f = get_segment_frames(r)
        if f is None or VIEW not in f.columns:
            continue
        rec = dict(skater=r["skater"], phase=r["phase"], stage=r["stage"], view=f[VIEW].median())
        for m in OTHER_METRICS:
            rec[m] = f[m].mean() if m in f.columns else np.nan
        recs.append(rec)
    seg = pd.DataFrame(recs)
    seg = seg[seg.groupby(["skater", "phase"]).stage.transform(lambda s: {"early", "late"} <= set(s))]
    seg["late"] = (seg.stage == "late").astype(float)
    groups = pd.get_dummies(seg.skater + "|" + seg.phase, dtype=float)
    k = groups.shape[1]
    for m in OTHER_METRICS:
        d = seg.dropna(subset=[m])
        gd = groups.loc[d.index]
        out = {"metric": m, "segments": len(d)}
        for label, cols in (("without_view", ["late"]), ("with_view", ["late", "view"])):
            X = np.column_stack([gd.values] + [d[c].values for c in cols])
            y = d[m].values
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ beta
            dof = len(y) - X.shape[1]
            cov = (resid @ resid / dof) * np.linalg.pinv(X.T @ X)
            b, se = beta[k], np.sqrt(cov[k, k])
            out[f"late_{label}"] = b
            out[f"p_{label}"] = 2 * stats.t.sf(abs(b / se), dof)
        out["pct_remaining"] = 100 * out["late_with_view"] / out["late_without_view"] if out["late_without_view"] else np.nan
        rows.append(out)
    return pd.DataFrame(rows)


SHOT_LABELS = "shot_labels.csv"
SHOT_METRICS = ["knee", "torso_lean_lr_diff", "hip_height_asymmetry", "sit"]


def check5_shot_type():
    """9/30: viewpoint from shot-type labels made by eye (make_shot_label_sheets.py),
    which, unlike projected hip width, don't depend on the skater's body.
    (a) Are late segments more often front-on shots?
    (b) metric ~ late + shot type (+ one intercept per skater x phase)
    (c) side-on shots only: metric ~ late (+ intercepts)."""
    import os
    if not os.path.exists(SHOT_LABELS):
        return None
    lab = pd.read_csv(SHOT_LABELS, encoding="utf-8-sig")
    lab = lab[lab.shot_type.isin(["side", "front", "wide"])]
    recs = []
    for _, r in lap_segments().iterrows():
        m = lab[(lab.video_path == r["video_path"]) & (lab.start_frame == int(r["start_frame"]))
                & (lab.end_frame == int(r["end_frame"]))]
        if m.empty:
            continue
        f = get_segment_frames(r)
        if f is None:
            continue
        recs.append({"skater": r["skater"], "phase": r["phase"], "stage": r["stage"],
                     "shot": m.iloc[0].shot_type, "labeler": m.iloc[0].labeler,
                     "knee": f[["left_knee_filtered", "right_knee_filtered"]].mean(axis=1).mean(),
                     "torso_lean_lr_diff": f["torso_lean_lr_diff"].mean(),
                     "hip_height_asymmetry": f["hip_height_asymmetry"].mean(),
                     "sit": f["hip_to_ankle_vertical_right"].mean()})
    seg = pd.DataFrame(recs)
    seg = seg[seg.groupby(["skater", "phase"]).stage.transform(lambda s: {"early", "late"} <= set(s))].copy()
    seg["late"] = (seg.stage == "late").astype(float)

    mix = pd.crosstab(seg.stage, seg.shot, normalize="index").round(2)

    def fit(d, extra_dummies):
        groups = pd.get_dummies(d.skater + "|" + d.phase, dtype=float)
        parts = [groups.values]
        if extra_dummies:
            parts.append(pd.get_dummies(d.shot, drop_first=True, dtype=float).values)
        parts.append(d["late"].values[:, None])
        X = np.column_stack(parts)
        out = {}
        for m in SHOT_METRICS:
            y = d[m].values
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ beta
            dof = len(y) - np.linalg.matrix_rank(X)
            if dof <= 0:
                out[m] = (np.nan, np.nan)
                continue
            cov = (resid @ resid / dof) * np.linalg.pinv(X.T @ X)
            b, se = beta[-1], np.sqrt(cov[-1, -1])
            out[m] = (b, 2 * stats.t.sf(abs(b / se), dof))
        return out

    plain = fit(seg, False)
    with_shot = fit(seg, True)
    side = seg[seg.shot == "side"]
    side = side[side.groupby(["skater", "phase"]).stage.transform(lambda s: {"early", "late"} <= set(s))]
    side_fit = fit(side, False) if len(side) else {}
    table = pd.DataFrame([{
        "metric": m,
        "late_plain": plain[m][0], "p_plain": plain[m][1],
        "late_with_shot_type": with_shot[m][0], "p_with_shot_type": with_shot[m][1],
        "late_side_only": side_fit.get(m, (np.nan, np.nan))[0], "p_side_only": side_fit.get(m, (np.nan, np.nan))[1],
    } for m in SHOT_METRICS])
    return mix, table, len(seg), len(side), seg.labeler.unique().tolist()


def main():
    seg, fr = load()
    pd.set_option("display.width", 220)
    seg.to_csv(OUTPUT_PATH, index=False)

    print(f"{'='*78}\n1. DO EARLY AND LATE SEGMENTS DIFFER IN VIEWPOINT? (projected hip width)\n{'='*78}")
    c1 = check1_view_differs(seg)
    print(c1.round(3).to_string(index=False))
    up = int((c1.view_pct_change > 0).sum())
    print(f"\nLate segments show a WIDER view (more frontal) in {up}/{len(c1)} comparisons.")

    print(f"\n{'='*78}\n2. SEGMENT LEVEL: knee angle ~ late (+ viewpoint), one intercept per skater x phase\n{'='*78}")
    without, with_view, n, dof = check2_segment_regression(seg)
    b, se, p = without["late"]
    print(f"Without viewpoint: late = {b:+.2f} deg (SE {se:.2f}, p = {p:.3f})   [{n} segments]")
    b2, se2, p2 = with_view["late"]
    bv, sev, pv = with_view["view"]
    print(f"With viewpoint:    late = {b2:+.2f} deg (SE {se2:.2f}, p = {p2:.3f});  "
          f"view = {bv:+.2f} deg per unit (p = {pv:.3f})")
    kept = 100 * b2 / b if b else float("nan")
    print(f"-> {kept:.0f}% of the late-race knee change remains after accounting for viewpoint.")

    print(f"\n{'='*78}\n3. FRAME LEVEL, MATCHED VIEWPOINT (common-support frames only)\n{'='*78}")
    c3 = check3_matched_frames(fr)
    print(c3.round(2).to_string(index=False))
    m = c3.dropna(subset=["knee_change_matched"])
    if not m.empty:
        print(f"\nKnees straighter late with viewpoint matched: {int((m.knee_change_matched > 0).sum())}/{len(m)} comparisons "
              f"(moderate effect, >= 0.3: {int((m.effect_matched >= 0.3).sum())})")
        print("view_effect_matched near 0 confirms the matching worked (early and late now seen from similar angles).")

    print(f"\n{'='*78}\n4. OTHER 5f PATTERNS: metric ~ late (+ viewpoint), segment level\n{'='*78}")
    print(check4_other_metrics().round(3).to_string(index=False))

    c5 = check5_shot_type()
    if c5 is not None:
        mix, table, n, n_side, labelers = c5
        print(f"\n{'='*78}\n5. SHOT TYPE (labelled by eye; labelers: {labelers})\n{'='*78}")
        print("Shot mix by stage (share of segments):")
        print(mix.to_string())
        print(f"\nmetric ~ late, three ways ({n} segments; {n_side} side-on segments in groups with early+late side shots):")
        print(table.round(3).to_string(index=False))
        if any("provisional" in str(l) for l in labelers):
            print("NOTE: some shot labels are provisional (made by Claude) and need the researcher's check.")

    print("\nREADING THIS:")
    print("  - If the late effect survives checks 2 and 3, the upright-late pattern is not explained by viewpoint.")
    print("  - If it shrinks toward 0 once viewpoint is controlled, it was (at least partly) a camera artifact.")
    print("  - Projected hip width is an imperfect viewpoint proxy (it also changes with hip rotation in a stride),")
    print("    so this reduces, but cannot fully remove, the viewpoint concern.")
    print(f"\nPer-segment data saved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
