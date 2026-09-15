"""
Outlier sensitivity check: Mia Manganello Kilburg's results are visibly the
largest outlier in both Phase 3a and 3b (e.g. 3b separability gap of 2.60 vs.
the next-highest value of 0.33). This script re-runs the same paired
comparisons with her excluded, to check whether the "scaling increases
variance" finding holds for the other 6 skaters or is being driven almost
entirely by her one data point.

This is standard practice for small-n outlier sensitivity checking, NOT
selective exclusion to get a preferred answer -- both versions (with and
without Mia) should be reported side by side in your write-up.

Usage:
    python -m outlier_sensitivity_check
"""

import pandas as pd
import numpy as np
from scipy import stats

ABLATION_RESULTS = "ablation_results/results.csv"
SEPARABILITY_RESULTS = "fatigue_separability_results/results.csv"
EXCLUDE_SKATER = "Mia Manganello Kilburg"


def compare_with_without(df, value_col, condition_a, condition_b, exclude_skater):
    pivot = df.pivot(index="skater", columns="condition", values=value_col)
    if condition_a not in pivot.columns or condition_b not in pivot.columns:
        print(f"  Missing condition(s): {condition_a}/{condition_b}")
        return

    paired_full = pivot[[condition_a, condition_b]].dropna()
    paired_excl = paired_full.drop(index=exclude_skater, errors="ignore")

    for label, data in [("FULL (n={})".format(len(paired_full)), paired_full),
                         ("EXCLUDING {} (n={})".format(exclude_skater, len(paired_excl)), paired_excl)]:
        a_vals = data[condition_a].values
        b_vals = data[condition_b].values
        mean_diff = np.mean(b_vals - a_vals)
        var_a = np.var(a_vals)
        var_b = np.var(b_vals)
        try:
            stat, p = stats.wilcoxon(a_vals, b_vals)
            p_str = f"p={p:.4f}"
        except ValueError:
            p_str = "p=N/A (too few non-zero diffs)"

        print(f"  [{label}] mean_diff={mean_diff:+.4f}  var({condition_a})={var_a:.4f}  "
              f"var({condition_b})={var_b:.4f}  {p_str}")


def main():
    print("=" * 70)
    print(f"OUTLIER SENSITIVITY CHECK: excluding '{EXCLUDE_SKATER}'")
    print("=" * 70)

    print("\n--- Phase 3a: unscaled vs. scaled (held_out_loss) ---")
    df_3a = pd.read_csv(ABLATION_RESULTS)
    compare_with_without(df_3a, "held_out_loss", "unscaled", "scaled", EXCLUDE_SKATER)

    print("\n--- Phase 3a: zscore_only vs. scaled (held_out_loss) ---")
    compare_with_without(df_3a, "held_out_loss", "zscore_only", "scaled", EXCLUDE_SKATER)

    print("\n--- Phase 3b: unscaled vs. scaled (separability_gap) ---")
    df_3b = pd.read_csv(SEPARABILITY_RESULTS)
    compare_with_without(df_3b, "separability_gap", "unscaled", "scaled", EXCLUDE_SKATER)

    print("\n--- Phase 3b: zscore_only vs. scaled (separability_gap) ---")
    compare_with_without(df_3b, "separability_gap", "zscore_only", "scaled", EXCLUDE_SKATER)

    print("\n" + "=" * 70)
    print("If the direction/magnitude of the result changes substantially when")
    print(f"{EXCLUDE_SKATER} is excluded, that means the original 7-skater result was")
    print("largely driven by one athlete's data -- worth reporting BOTH versions")
    print("in your write-up rather than treating the 7-skater number as the final word.")
    print("=" * 70)


if __name__ == "__main__":
    main()
