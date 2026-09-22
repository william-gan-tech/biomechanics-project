"""
Builds real reference statistics for each technique phase, pooled across
ALL verified skaters in labeled_segments.csv -- not just one skater's
numbers, but a genuine small reference dataset with per-phase mean/std
computed across multiple athletes.

Usage:
    python -m build_reference_stats
"""

import csv
import pandas as pd
import numpy as np

from run_bone_scaling_ablation import get_skater_features, filter_implausible_frames
from start_phase_features import add_start_phase_features

LOG_PATH = "labeled_segments.csv"
OUTPUT_PATH = "reference_stats.csv"

VALID_PHASES = ["start_rest", "start_acceleration", "corner", "straightaway"]


def load_labeled_segments():
    rows = []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["phase"] in VALID_PHASES:
                rows.append(row)
    return rows


def extract_features_for_segment(row):
    skater = row["skater"]
    video_path = row["video_path"]
    start_f = int(row["start_frame"])
    end_f = int(row["end_frame"])

    df = get_skater_features(skater, video_path, "scaled")
    if df is None:
        print(f"  [SKIP] Could not load features for {skater} ({video_path})")
        return None

    df = filter_implausible_frames(df)
    df = add_start_phase_features(df)

    segment = df[(df["frame"] >= start_f) & (df["frame"] <= end_f)]
    if segment.empty:
        print(f"  [SKIP] No frames found in {skater} range {start_f}-{end_f}")
        return None

    return segment


def main():
    segments = load_labeled_segments()
    print(f"Loaded {len(segments)} valid labeled segments from {LOG_PATH}\n")

    per_phase_skater_segments = {phase: {} for phase in VALID_PHASES}

    for row in segments:
        phase = row["phase"]
        skater = row["skater"]
        print(f"Processing: {skater} | {phase} | frames {row['start_frame']}-{row['end_frame']}")

        segment_df = extract_features_for_segment(row)
        if segment_df is None:
            continue

        segment_means = {
            "torso_lean_angle_deg": segment_df["torso_lean_angle_deg"].mean(),
            "hip_velocity": segment_df["hip_velocity"].mean(),
            "hip_acceleration_abs": segment_df["hip_acceleration"].abs().mean(),
            "right_knee_filtered": segment_df["right_knee_filtered"].mean(),
            "left_knee_filtered": segment_df["left_knee_filtered"].mean(),
        }

        per_phase_skater_segments[phase].setdefault(skater, []).append(segment_means)

    # Collapse each skater's segment list to a TRUE mean per metric (fixes a
    # bug where a sequential pairwise running average incorrectly overweighted
    # earlier segments when a skater had 3+ segments for the same phase)
    per_phase_skater_means = {phase: {} for phase in VALID_PHASES}
    for phase, skater_segments in per_phase_skater_segments.items():
        for skater, segment_list in skater_segments.items():
            keys = segment_list[0].keys()
            true_mean = {k: float(np.mean([seg[k] for seg in segment_list])) for k in keys}
            per_phase_skater_means[phase][skater] = true_mean

    print("\n" + "=" * 70)
    print("REFERENCE STATISTICS (pooled across skaters, one data point per skater per phase)")
    print("=" * 70)

    results = []
    for phase in VALID_PHASES:
        skater_data = per_phase_skater_means[phase]
        n_skaters = len(skater_data)
        print(f"\n{phase} (n={n_skaters} skaters: {', '.join(skater_data.keys())})")

        if n_skaters == 0:
            continue

        metrics_df = pd.DataFrame(list(skater_data.values()))
        for col in metrics_df.columns:
            mean_val = metrics_df[col].mean()
            std_val = metrics_df[col].std() if n_skaters > 1 else float("nan")
            print(f"  {col:25s}: mean={mean_val:8.3f}  std={std_val:8.3f} (across-skater variation)")
            results.append({
                "phase": phase,
                "metric": col,
                "n_skaters": n_skaters,
                "mean": mean_val,
                "std": std_val,
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved -> {OUTPUT_PATH}")

    print("\nHONEST NOTE: n=3 per phase is a small reference sample. These stats")
    print("show real cross-skater central tendency and spread, but should be")
    print("treated as preliminary, not a definitive 'normal range' -- more")
    print("skaters per phase would meaningfully strengthen this.")


if __name__ == "__main__":
    main()
