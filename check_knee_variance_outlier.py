"""
Sensitivity check on the surprising finding from build_reference_stats.py:
straightaway left-knee angle showed the highest across-skater std (39.0)
of any metric/phase combination. This checks whether that's driven by
one outlier skater (same leave-one-out logic as Phase 3's
outlier_sensitivity_check.py) or genuinely spread across all three.

Usage:
    python -m check_knee_variance_outlier
"""

import argparse
import csv
import numpy as np

from run_bone_scaling_ablation import get_skater_features, filter_implausible_frames
from start_phase_features import add_start_phase_features

LOG_PATH = "labeled_segments.csv"

VALID_METRICS = [
    "torso_lean_angle_deg", "hip_velocity", "hip_acceleration",
    "right_knee_filtered", "left_knee_filtered",
]
VALID_PHASES = ["start_rest", "start_acceleration", "corner", "straightaway"]


def load_segments_for_phase(phase):
    rows = []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["phase"] == phase:
                rows.append(row)
    return rows


def get_segment_mean(row, metric):
    skater = row["skater"]
    video_path = row["video_path"]
    start_f = int(row["start_frame"])
    end_f = int(row["end_frame"])

    df = get_skater_features(skater, video_path, "scaled")
    if df is None:
        return None
    df = filter_implausible_frames(df)
    df = add_start_phase_features(df)

    segment = df[(df["frame"] >= start_f) & (df["frame"] <= end_f)]
    if segment.empty or metric not in segment.columns:
        return None

    return segment[metric].mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=VALID_PHASES, default="straightaway")
    parser.add_argument("--metric", choices=VALID_METRICS, default="left_knee_filtered")
    parser.add_argument("--abs", action="store_true",
                         help="Use absolute value of the metric -- important for signed "
                              "metrics like torso_lean_angle_deg, where sign depends on "
                              "camera angle/turn direction rather than real technique "
                              "difference, and would otherwise inflate apparent variance.")
    args = parser.parse_args()
    target_phase = args.phase
    target_metric = args.metric
    use_abs = args.abs

    rows = load_segments_for_phase(target_phase)
    print(f"Found {len(rows)} labeled '{target_phase}' segments")
    if use_abs:
        print(f"(Using ABSOLUTE VALUE of {target_metric} -- sign is being ignored)\n")
    else:
        print()

    per_skater_values = {}
    for row in rows:
        skater = row["skater"]
        val = get_segment_mean(row, target_metric)
        if val is None:
            print(f"  [SKIP] {skater}: could not extract {target_metric}")
            continue
        if use_abs:
            val = abs(val)
        per_skater_values.setdefault(skater, []).append(val)
        print(f"  {skater}: {target_metric} = {val:.2f} (segment {len(per_skater_values[skater])})")

    # Collapse each skater's segment list to a TRUE mean (not a sequential
    # pairwise running average, which incorrectly overweights earlier values)
    per_skater_final = {skater: float(np.mean(vals)) for skater, vals in per_skater_values.items()}
    print(f"\nPer-skater TRUE means for {target_phase}/{target_metric} (all segments properly averaged):")
    for skater, val in per_skater_final.items():
        n_segs = len(per_skater_values[skater])
        print(f"  {skater}: {val:.2f} (from {n_segs} segment(s))")

    if len(per_skater_final) < 3:
        print("\nNeed at least 3 skaters for a meaningful leave-one-out check.")
        return

    skaters = list(per_skater_final.keys())
    values = np.array(list(per_skater_final.values()))

    print(f"\n{'='*60}")
    print(f"FULL SAMPLE: mean={values.mean():.2f}, std={values.std():.2f}")
    print(f"{'='*60}\n")

    print("Leave-one-out sensitivity:")
    for i, excluded_skater in enumerate(skaters):
        remaining_values = np.delete(values, i)
        remaining_std = remaining_values.std()
        print(f"  Excluding {excluded_skater:20s}: remaining std = {remaining_std:.2f} "
              f"(full std was {values.std():.2f})")

    print("\nInterpretation:")
    print("If excluding any ONE skater dramatically reduces the std, that skater is")
    print("driving the variance (an outlier). If std stays roughly similar regardless")
    print("of which skater is excluded, the variation is genuinely spread across all")
    print("three -- a real, not artifact-driven, finding.")


if __name__ == "__main__":
    main()
