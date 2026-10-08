"""
Phase 5e: Bilateral Asymmetry Validation.

Question: do corners show genuinely more left-right asymmetry than
straightaways? Motivated by the 9/19 finding (straightaway knees ~2 deg
apart vs corner ~14 deg, n=1 video).

Asymmetry measures (all from start_phase_features.py):
  - knee_angle_asymmetry   |right - left knee angle|, degrees (scale-free)
  - hip_height_asymmetry   |right - left hip height| = pelvic tilt
  - torso_lean_lr_diff     |right-side - left-side trunk lean|, degrees
  - hip_lateral_asymmetry  2D left-right hip distance. CAUTION: this is
                           really PROJECTED hip width, which changes with
                           body rotation relative to the camera -- and
                           skaters rotate through corners. Reported for
                           continuity with 9/22, but not a clean
                           technique measure.

Three levels of evidence, weakest-assumption first:
  1. Within-skater, per-frame: for each skater with BOTH corner and
     straightaway segments, Mann-Whitney U on per-frame values + a
     rank-biserial effect size. Frames are autocorrelated, so p-values here
     are optimistic; the effect size and direction are what matter.
  2. Within-skater, segment means: corner / straightaway ratio per skater.
     Direction consistency across skaters is the honest headline.
  3. Between-skater elite reference (5a): corner vs straightaway profile
     means, with all skaters that have either phase.

HONEST SCOPE: only 2 skaters (start_candidate_3, Patrick Meek) have both
phases -- this is a preliminary directional result, not a population claim.

Usage:
    python -m validate_asymmetry_by_phase
"""

import numpy as np
import pandas as pd
from scipy import stats

from compare_to_elite_reference import load_labeled_segments, get_segment_frames

ASYMMETRY_METRICS = [
    ("knee_angle_asymmetry", "deg"),
    ("hip_height_asymmetry", "torso"),
    ("torso_lean_lr_diff", "deg"),
    ("hip_lateral_asymmetry", "torso, view-dependent"),
]
PHASES = ["corner", "straightaway"]
OUTPUT_PATH = "phase5e_asymmetry_results.csv"


def collect_frames():
    """Per-frame asymmetry values for every corner/straightaway segment."""
    frames = []
    for row in load_labeled_segments():
        if row["phase"] not in PHASES:
            continue
        seg = get_segment_frames(row)
        if seg is None:
            print(f"  [SKIP] {row['skater']} | {row['phase']}")
            continue
        cols = [m for m, _ in ASYMMETRY_METRICS if m in seg.columns]
        part = seg[cols].copy()
        part["skater"] = row["skater"]
        part["phase"] = row["phase"]
        part["segment"] = f"{row['start_frame']}-{row['end_frame']}"
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def within_skater(frames):
    both = [s for s, g in frames.groupby("skater") if set(PHASES) <= set(g["phase"])]
    rows = []
    for skater in both:
        g = frames[frames["skater"] == skater]
        for metric, unit in ASYMMETRY_METRICS:
            c = g.loc[g["phase"] == "corner", metric].dropna()
            s = g.loc[g["phase"] == "straightaway", metric].dropna()
            if len(c) < 5 or len(s) < 5:
                continue
            u, p = stats.mannwhitneyu(c, s, alternative="two-sided")
            # rank-biserial: +1 = every corner frame more asymmetric than every straightaway frame
            r_rb = 2 * u / (len(c) * len(s)) - 1
            rows.append({
                "skater": skater, "metric": metric, "unit": unit,
                "corner_mean": c.mean(), "straight_mean": s.mean(),
                "ratio_corner_over_straight": c.mean() / s.mean() if s.mean() else np.nan,
                "rank_biserial": r_rb, "p_frames": p,
                "n_corner_frames": len(c), "n_straight_frames": len(s),
            })
    return pd.DataFrame(rows), both


def between_skater(frames):
    # True mean per skater per phase (same as the elite profile), then pooled
    per_skater = frames.groupby(["phase", "skater"])[[m for m, _ in ASYMMETRY_METRICS]].mean()
    out = per_skater.groupby("phase").agg(["mean", "std", "count"])
    return per_skater, out


def within_shot_per_skater(frames):
    """10/7: corners and straightaways are often filmed differently, so compare
    them only WITHIN the same camera shot type (researcher-checked labels in
    shot_labels.csv). One corner-minus-straightaway value per skater (averaged
    over their shot-type cells), tested across skaters."""
    import os
    if not os.path.exists("shot_labels.csv"):
        return None
    lab = pd.read_csv("shot_labels.csv", encoding="utf-8-sig")
    lab = lab[lab.shot_type.isin(["side", "front", "wide"])]
    lab["segment"] = lab.start_frame.astype(str) + "-" + lab.end_frame.astype(str)
    f = frames.merge(lab[["skater", "segment", "shot_type"]], on=["skater", "segment"], how="inner")
    metrics = [m for m, _ in ASYMMETRY_METRICS]
    per_cell = []
    for (sk, shot), g in f.groupby(["skater", "shot_type"]):
        c, s = g[g.phase == "corner"], g[g.phase == "straightaway"]
        if len(c) >= 5 and len(s) >= 5:
            per_cell.append({"skater": sk, "shot": shot, **{m: c[m].mean() - s[m].mean() for m in metrics}})
    cells = pd.DataFrame(per_cell)
    if cells.empty:
        return None
    per_skater = cells.groupby("skater")[metrics].mean()
    rows = []
    for m in metrics:
        v = per_skater[m].dropna()
        if len(v) >= 2:
            rows.append({"metric": m, "skaters": len(v), "mean_corner_minus_straight": v.mean(),
                         "skaters_corner_higher": int((v > 0).sum()),
                         "p_ttest": stats.ttest_1samp(v, 0.0).pvalue})
    return pd.DataFrame(rows), per_skater


def main():
    print("Loading corner and straightaway segments...\n")
    frames = collect_frames()

    ws, both = within_skater(frames)
    print(f"{'='*78}")
    print(f"1+2. WITHIN-SKATER: corner vs straightaway (skaters with both: {', '.join(both)})")
    print(f"{'='*78}")
    if ws.empty:
        print("No skater has both phases -- cannot run a within-skater test.")
    else:
        show = ws[["skater", "metric", "corner_mean", "straight_mean",
                   "ratio_corner_over_straight", "rank_biserial", "p_frames"]]
        print(show.round(3).to_string(index=False))
        ws.to_csv(OUTPUT_PATH, index=False)

        print("\nDirection consistency (does EVERY skater show corner > straightaway?):")
        for metric, unit in ASYMMETRY_METRICS:
            m = ws[ws["metric"] == metric]
            if m.empty:
                continue
            n_up = int((m["ratio_corner_over_straight"] > 1).sum())
            mean_rb = m["rank_biserial"].mean()
            verdict = ("CONSISTENT: corner more asymmetric" if n_up == len(m)
                       else "CONSISTENT: corner LESS asymmetric" if n_up == 0
                       else "MIXED across skaters")
            print(f"  {metric:24s} ({unit}): {n_up}/{len(m)} skaters corner>straight, "
                  f"mean effect size {mean_rb:+.2f}  -> {verdict}")

    per_skater, pooled = between_skater(frames)
    print(f"\n{'='*78}")
    print("3. BETWEEN-SKATER (elite-reference style: per-skater mean, then pooled)")
    print(f"{'='*78}")
    print(per_skater.round(3).to_string())
    print()
    print(pooled.round(3).to_string())

    ws_shot = within_shot_per_skater(frames)
    if ws_shot is not None:
        summary, per_skater = ws_shot
        print(f"\n{'='*78}\n4. WITHIN THE SAME CAMERA SHOT TYPE (per skater, corner minus straightaway)\n{'='*78}")
        print(per_skater.round(3).to_string())
        print()
        print(summary.round(3).to_string(index=False))

    print("\nREADING THIS:")
    print("  - rank_biserial: +1 = all corner frames more asymmetric than all straightaway")
    print("    frames, 0 = no difference, -1 = the reverse. |r| > 0.3 is a moderate effect.")
    print("  - p_frames treats frames as independent; they aren't (autocorrelated), so")
    print("    these p-values overstate certainty. Trust direction + effect size.")
    print("  - hip_lateral_asymmetry is projected hip width: a corner effect here may be")
    print("    body rotation relative to the camera, not technique.")
    print(f"\nWithin-skater results saved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
