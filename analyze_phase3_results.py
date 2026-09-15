"""
Phase 3 statistical analysis: combines 3a (general reconstruction variance)
and 3b (fatigue separability) results and runs proper significance tests.

Why this matters: "variance went up" or "the gap was bigger" is a
descriptive observation, not a statistical claim. With only n=7 skaters,
it's easy for a real effect to look identical to noise. This script:

  1. Reports EFFECTIVE sample size per condition (some skaters may have
     been [SKIP]'d in 3b due to insufficient fresh/fatigued windows --
     this silently shrinks n if not checked).
  2. Runs a paired Wilcoxon signed-rank test (appropriate for small,
     non-parametric paired samples -- safer than a paired t-test at n=7,
     which assumes normality that's hard to justify with this few points)
     comparing scaled vs. unscaled, and scaled vs. zscore_only, for BOTH
     3a's per-skater loss and 3b's per-skater separability gap.
  3. Writes one combined markdown summary tying 3a and 3b together, ready
     to drop into your Phase 3 write-up.

Usage:
    python -m analyze_phase3_results

Requires ablation_results/results.csv (3a) and
fatigue_separability_results/results.csv (3b) to already exist.
"""

import os
import pandas as pd
import numpy as np
from scipy import stats

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE_3A_RESULTS = os.path.join(ROOT_DIR, "ablation_results", "results.csv")
PHASE_3B_RESULTS = os.path.join(ROOT_DIR, "fatigue_separability_results", "results.csv")
OUTPUT_MD = os.path.join(ROOT_DIR, "phase3_statistical_summary.md")


def paired_comparison(df, value_col, condition_a, condition_b, skater_col="skater"):
    """Returns a dict with the paired samples (only skaters present in BOTH
    conditions -- unpaired skaters are dropped and reported), plus a
    Wilcoxon signed-rank test result."""
    pivot = df.pivot(index=skater_col, columns="condition", values=value_col)

    if condition_a not in pivot.columns or condition_b not in pivot.columns:
        return {"error": f"Missing condition(s): {condition_a} and/or {condition_b} not in data"}

    paired = pivot[[condition_a, condition_b]].dropna()
    n_paired = len(paired)
    n_total = len(pivot)
    n_dropped = n_total - n_paired

    if n_paired < 3:
        return {
            "error": f"Only {n_paired} skaters have BOTH conditions -- too few for a meaningful test",
            "n_paired": n_paired,
            "n_dropped_unpaired": n_dropped,
        }

    a_vals = paired[condition_a].values
    b_vals = paired[condition_b].values
    diffs = b_vals - a_vals

    try:
        stat, p_value = stats.wilcoxon(a_vals, b_vals)
    except ValueError as e:
        # Wilcoxon fails if all differences are zero, or too few non-zero diffs
        stat, p_value = None, None
        wilcoxon_error = str(e)
    else:
        wilcoxon_error = None

    return {
        "n_paired": n_paired,
        "n_dropped_unpaired": n_dropped,
        "mean_diff": float(np.mean(diffs)),
        "median_diff": float(np.median(diffs)),
        "wilcoxon_stat": stat,
        "p_value": p_value,
        "wilcoxon_error": wilcoxon_error,
        "per_skater_values": paired.to_dict("index"),
    }


def format_result_block(title, result):
    lines = [f"### {title}", ""]
    if "error" in result and result.get("n_paired") is None:
        lines.append(f"**Could not run:** {result['error']}")
        lines.append("")
        return "\n".join(lines)

    lines.append(f"- Paired skaters: **{result['n_paired']}** "
                 f"(dropped {result['n_dropped_unpaired']} not present in both conditions)")
    lines.append(f"- Mean difference: `{result['mean_diff']:.5f}`")
    lines.append(f"- Median difference: `{result['median_diff']:.5f}`")

    if result["p_value"] is not None:
        significant = "YES" if result["p_value"] < 0.05 else "NO"
        lines.append(f"- Wilcoxon signed-rank test: statistic=`{result['wilcoxon_stat']:.3f}`, "
                     f"p=`{result['p_value']:.4f}`")
        lines.append(f"- **Statistically significant at α=0.05: {significant}**")
        if result["n_paired"] < 8:
            lines.append(f"  - ⚠️ n={result['n_paired']} is very small -- treat this p-value as "
                         f"suggestive, not conclusive. A non-significant result here does NOT "
                         f"prove there's no effect, only that this sample can't detect one reliably.")
    else:
        lines.append(f"- Wilcoxon test could not be computed: {result['wilcoxon_error']}")

    lines.append("")
    lines.append("| Skater | " + " | ".join(result["per_skater_values"][list(result["per_skater_values"].keys())[0]].keys()) + " |")
    lines.append("|---|" + "---|" * len(result["per_skater_values"][list(result["per_skater_values"].keys())[0]]))
    for skater, vals in result["per_skater_values"].items():
        lines.append(f"| {skater} | " + " | ".join(f"{v:.4f}" for v in vals.values()) + " |")
    lines.append("")

    return "\n".join(lines)


def main():
    report_sections = ["# Phase 3 Statistical Summary\n", "*Auto-generated -- verify before citing.*\n"]

    # ---------------- Phase 3a ----------------
    report_sections.append("## Phase 3a: General Motion-Reconstruction Variance\n")
    report_sections.append("*Question: does bone-length scaling reduce cross-subject variance in "
                            "general reconstruction loss?*\n")

    if os.path.exists(PHASE_3A_RESULTS):
        df_3a = pd.read_csv(PHASE_3A_RESULTS)
        conditions_present = df_3a["condition"].unique().tolist()
        report_sections.append(f"Conditions found: {conditions_present}\n")

        if "unscaled" in conditions_present and "scaled" in conditions_present:
            result = paired_comparison(df_3a, "held_out_loss", "unscaled", "scaled")
            report_sections.append(format_result_block("Unscaled vs. Scaled", result))

        if "unscaled" in conditions_present and "zscore_only" in conditions_present:
            result = paired_comparison(df_3a, "held_out_loss", "unscaled", "zscore_only")
            report_sections.append(format_result_block("Unscaled vs. Z-Score-Only", result))

        if "zscore_only" in conditions_present and "scaled" in conditions_present:
            result = paired_comparison(df_3a, "held_out_loss", "zscore_only", "scaled")
            report_sections.append(format_result_block("Z-Score-Only vs. Scaled "
                                                          "(does bone geometry add anything beyond standardization?)", result))
    else:
        report_sections.append(f"⚠️ No results found at `{PHASE_3A_RESULTS}` -- run "
                               f"`python -m run_bone_scaling_ablation` first.\n")

    # ---------------- Phase 3b ----------------
    report_sections.append("## Phase 3b: Cross-Subject Fatigue-Detection Separability\n")
    report_sections.append("*Question: does bone-length scaling improve an autoencoder's ability to "
                            "separate fresh from fatigued movement in an unseen athlete?*\n")

    if os.path.exists(PHASE_3B_RESULTS):
        df_3b = pd.read_csv(PHASE_3B_RESULTS)
        conditions_present = df_3b["condition"].unique().tolist()
        report_sections.append(f"Conditions found: {conditions_present}\n")

        skaters_per_condition = df_3b.groupby("condition")["skater"].apply(list).to_dict()
        report_sections.append("**Effective sample size per condition:**\n")
        for cond, skaters in skaters_per_condition.items():
            report_sections.append(f"- `{cond}`: n={len(skaters)} ({', '.join(skaters)})")
        report_sections.append("")

        if "unscaled" in conditions_present and "scaled" in conditions_present:
            result = paired_comparison(df_3b, "separability_gap", "unscaled", "scaled")
            report_sections.append(format_result_block("Unscaled vs. Scaled (separability gap)", result))

        if "unscaled" in conditions_present and "zscore_only" in conditions_present:
            result = paired_comparison(df_3b, "separability_gap", "unscaled", "zscore_only")
            report_sections.append(format_result_block("Unscaled vs. Z-Score-Only (separability gap)", result))

        if "zscore_only" in conditions_present and "scaled" in conditions_present:
            result = paired_comparison(df_3b, "separability_gap", "zscore_only", "scaled")
            report_sections.append(format_result_block("Z-Score-Only vs. Scaled (separability gap)", result))

        # Also report: what fraction of skaters showed a positive separability
        # gap at all (fatigue signal detected, regardless of magnitude), per condition
        report_sections.append("### Fraction of skaters with a positive separability gap (any fatigue signal)\n")
        for cond in conditions_present:
            subset = df_3b[df_3b["condition"] == cond]
            pct_positive = (subset["separability_gap"] > 0).mean() * 100
            report_sections.append(f"- `{cond}`: {pct_positive:.0f}% of held-out skaters "
                                   f"({int((subset['separability_gap'] > 0).sum())}/{len(subset)})")
        report_sections.append("")
    else:
        report_sections.append(f"⚠️ No results found at `{PHASE_3B_RESULTS}` -- run "
                               f"`python -m run_fatigue_separability_ablation` first.\n")

    report_sections.append("## How to read this report\n")
    report_sections.append(
        "- A non-significant p-value with n≈7 is NOT proof of 'no effect' -- it means this sample "
        "is too small to distinguish a real small/moderate effect from noise. Report p-values "
        "honestly alongside the sample size, don't claim a null result is proven.\n"
        "- The 'Z-Score-Only vs. Scaled' comparisons are the most directly informative for your "
        "core question: they isolate whether bone-length GEOMETRY specifically adds value beyond "
        "what simple image-size normalization and standard z-scoring already provide.\n"
    )

    full_report = "\n".join(report_sections)
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write(full_report)

    print(full_report)
    print(f"\n\nSaved full report -> {OUTPUT_MD}")


if __name__ == "__main__":
    main()
