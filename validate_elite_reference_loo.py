"""
Phase 5a/5m: Leave-One-Out Validation of the Elite-Anchor Reference.

Replaces the 9/23 Sven Kramer self-comparison as the real test of
compare_to_elite_reference.py. That self-check was weak: a skater who is
already IN a 3-skater reference can't score beyond about +/-1.4 with
population std, so small z-scores were near-guaranteed.

Here, for every phase and every reference skater, the skater is HELD OUT,
the reference is rebuilt from the remaining skaters, and the held-out
skater is compared against it -- the same situation a genuinely new
skater is in.

HONEST NOTE ON INTERPRETATION: with n=3 per phase, each held-out reference
has only n=2 skaters, so its std comes from 2 values and is very noisy.
Under the ideal assumption (skaters drawn from one normal population),
the plain z-score is NOT standard-normal here; the prediction-adjusted
score  t = (x - mean) / (std * sqrt(1 + 1/n))  follows a Student-t with
n-1 = 1 degree of freedom, which has very heavy tails. So |z| > 2 is
expected fairly often by chance alone. This script reports both, plus the
fraction of held-out scores beyond the t(1) 80% and 90% cutoffs, which is
the actual calibration check: roughly 20% and 10% should exceed them if
the reference behaves as assumed.

Usage:
    python -m validate_elite_reference_loo
"""

import numpy as np
import pandas as pd
from scipy import stats

from compare_to_elite_reference import (
    load_labeled_segments, extract_segment_means, VALID_PHASES, METRICS,
)

OUTPUT_PATH = "loo_validation_results.csv"


def collect_skater_means():
    """Extracts every labeled segment ONCE and averages per skater per
    phase (true mean, same methodology as build_elite_profile)."""
    per_phase = {phase: {} for phase in VALID_PHASES}
    for row in load_labeled_segments():
        means = extract_segment_means(row)
        if means is None:
            print(f"  [SKIP] {row['skater']} | {row['phase']}")
            continue
        per_phase[row["phase"]].setdefault(row["skater"], []).append(means)

    skater_means = {}
    for phase, skaters in per_phase.items():
        skater_means[phase] = {
            s: {m: float(np.mean([seg[m] for seg in segs])) for m in METRICS}
            for s, segs in skaters.items()
        }
    return skater_means


def run_loo(skater_means):
    rows = []
    for phase, skaters in skater_means.items():
        names = list(skaters)
        if len(names) < 3:
            print(f"  [SKIP] {phase}: need >=3 skaters for LOO, have {len(names)}")
            continue
        for held_out in names:
            others = [s for s in names if s != held_out]
            n_ref = len(others)
            for metric in METRICS:
                ref_values = np.array([skaters[s][metric] for s in others])
                ref_mean = ref_values.mean()
                ref_std = ref_values.std(ddof=1)
                x = skaters[held_out][metric]
                z = (x - ref_mean) / ref_std if ref_std > 0 else np.nan
                t_pred = z / np.sqrt(1 + 1 / n_ref)
                rows.append({
                    "phase": phase, "held_out": held_out, "metric": metric,
                    "n_ref": n_ref, "value": x, "ref_mean": ref_mean,
                    "ref_std": ref_std, "z": z, "t_pred": t_pred,
                    "p_two_sided": 2 * stats.t.sf(abs(t_pred), df=n_ref - 1),
                })
    return pd.DataFrame(rows)


def summarize(df):
    print(f"\n{'='*78}")
    print("LEAVE-ONE-OUT RESULTS (each skater vs. reference built WITHOUT them)")
    print(f"{'='*78}")
    for phase, g in df.groupby("phase", sort=False):
        print(f"\n--- {phase} ---")
        table = g.pivot(index="metric", columns="held_out", values="z")
        print(table.round(2).to_string())

    df_valid = df.dropna(subset=["t_pred"])
    n = len(df_valid)
    # FIXED 9/29: phases now have different reference sizes (corner 5 skaters,
    # start phases 3), so one shared t cutoff is wrong. Each row's p-value
    # already uses its own df (n_ref - 1); calibrate on those.
    frac80 = (df_valid["p_two_sided"] < 0.20).mean()
    frac90 = (df_valid["p_two_sided"] < 0.10).mean()
    sizes = ", ".join(f"{p}: n_ref={int(g['n_ref'].iloc[0])}" for p, g in df_valid.groupby("phase", sort=False))

    print(f"\n{'='*78}")
    print(f"CALIBRATION ({n} held-out scores; each scored with its own t df -- {sizes})")
    print(f"{'='*78}")
    print(f"|z| > 2 (naive normal reading):        {(df_valid['z'].abs() > 2).mean():6.1%}")
    print(f"p < 0.20 (expect ~20% by chance):      {frac80:6.1%}")
    print(f"p < 0.10 (expect ~10% by chance):      {frac90:6.1%}")
    for phase, g in df_valid.groupby("phase", sort=False):
        print(f"  {phase:18s}: p<0.20 {(g['p_two_sided'] < 0.20).mean():6.1%}   p<0.10 {(g['p_two_sided'] < 0.10).mean():6.1%}")

    print("\nMost extreme held-out scores (inspect these by hand):")
    worst = df_valid.reindex(df_valid["t_pred"].abs().sort_values(ascending=False).index).head(8)
    print(worst[["phase", "held_out", "metric", "value", "ref_mean", "ref_std", "z", "p_two_sided"]]
          .round(3).to_string(index=False))

    print("\nHONEST NOTE: with n=2 per held-out reference, this checks whether the")
    print("tool is roughly calibrated and flags which metric/skater combinations")
    print("drive disagreement -- it cannot give precise percentile rankings.")


def main():
    print("Extracting per-skater segment means (one pass over all labeled segments)...\n")
    skater_means = collect_skater_means()
    df = run_loo(skater_means)
    if df.empty:
        print("No phase has >=3 skaters -- nothing to validate.")
        return
    df.to_csv(OUTPUT_PATH, index=False)
    summarize(df)
    print(f"\nFull results saved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
