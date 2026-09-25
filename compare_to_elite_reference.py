"""
Phase 5a: Elite-Anchor Reference Methodology.

Formalizes the existing labeled_segments.csv dataset as an explicit
"elite anchor" reference profile (mean + std per phase + metric, computed
per-skater-then-pooled, same corrected methodology as build_reference_stats.py),
and provides a real comparison tool: given a NEW video segment, extract
the same features and report how many standard deviations it falls from
the elite reference, per metric.

Usage:
    python -m compare_to_elite_reference --build-profile
    python -m compare_to_elite_reference --compare --video "data/new_skater.mp4" --start 100 --end 150 --phase straightaway --name "New Skater"
"""

import os
import csv
import argparse
import numpy as np
import pandas as pd

from run_bone_scaling_ablation import get_skater_features, filter_implausible_frames
from start_phase_features import add_start_phase_features

LOG_PATH = "labeled_segments.csv"
PROFILE_PATH = "elite_reference_profile.csv"
VALID_PHASES = ["start_rest", "start_acceleration", "corner", "straightaway"]
METRICS = [
    "torso_lean_angle_deg", "hip_velocity", "hip_acceleration_abs",
    "right_knee_filtered", "left_knee_filtered", "hip_to_ankle_vertical_right",
]

def load_labeled_segments():
    rows = []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["phase"] in VALID_PHASES:
                rows.append(row)
    return rows


def extract_segment_means(row):
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
    if segment.empty:
        return None
    return {
        "torso_lean_angle_deg": segment["torso_lean_angle_deg"].mean(),
        "hip_velocity": segment["hip_velocity"].mean(),
        "hip_acceleration_abs": segment["hip_acceleration"].abs().mean(),
        "right_knee_filtered": segment["right_knee_filtered"].mean(),
        "left_knee_filtered": segment["left_knee_filtered"].mean(),
        "hip_to_ankle_vertical_right": segment["hip_to_ankle_vertical_right"].mean()
            if "hip_to_ankle_vertical_right" in segment.columns else None,
    }

def build_elite_profile():
    segments = load_labeled_segments()
    print(f"Loaded {len(segments)} valid labeled segments\n")

    per_phase_skater_segments = {phase: {} for phase in VALID_PHASES}

    for row in segments:
        phase = row["phase"]
        skater = row["skater"]
        means = extract_segment_means(row)
        if means is None:
            print(f"  [SKIP] {skater} | {phase}")
            continue
        per_phase_skater_segments[phase].setdefault(skater, []).append(means)
        print(f"  Processed: {skater} | {phase}")

    profile_rows = []
    for phase, skater_segments in per_phase_skater_segments.items():
        skater_true_means = {}
        for skater, segment_list in skater_segments.items():
            skater_true_means[skater] = {
                m: float(np.mean([seg[m] for seg in segment_list])) for m in METRICS
            }

        n_skaters = len(skater_true_means)
        if n_skaters == 0:
            continue

        for metric in METRICS:
            values = np.array([skater_true_means[s][metric] for s in skater_true_means])
            profile_rows.append({
                "phase": phase,
                "metric": metric,
                "n_skaters": n_skaters,
                "elite_mean": values.mean(),
                "elite_std": values.std() if n_skaters > 1 else 0.0,
            })

    profile_df = pd.DataFrame(profile_rows)
    profile_df.to_csv(PROFILE_PATH, index=False)
    print(f"\nSaved elite reference profile -> {PROFILE_PATH}")
    print(profile_df.to_string(index=False))


def compare_segment_to_profile(video_path, start_f, end_f, phase, skater_name):
    if not os.path.exists(PROFILE_PATH):
        print(f"{PROFILE_PATH} not found -- run with --build-profile first.")
        return

    profile_df = pd.read_csv(PROFILE_PATH)
    phase_profile = profile_df[profile_df["phase"] == phase]
    if phase_profile.empty:
        print(f"No elite reference data for phase '{phase}'.")
        return

    fake_row = {"skater": skater_name, "video_path": video_path,
                "start_frame": start_f, "end_frame": end_f}
    means = extract_segment_means(fake_row)
    if means is None:
        print("Could not extract features for this segment.")
        return

    print(f"\n{'='*70}")
    print(f"COMPARISON: {skater_name} vs. elite reference ({phase})")
    print(f"{'='*70}")

    for _, ref_row in phase_profile.iterrows():
        metric = ref_row["metric"]
        elite_mean = ref_row["elite_mean"]
        elite_std = ref_row["elite_std"]
        your_value = means.get(metric)

        if your_value is None:
            continue

        if elite_std > 0:
            z_score = (your_value - elite_mean) / elite_std
            z_str = f"z={z_score:+.2f}"
        else:
            z_str = "z=N/A (elite std is 0, likely n=1 reference)"

        print(f"{metric:25s}: you={your_value:8.3f}  elite_mean={elite_mean:8.3f}  "
              f"elite_std={elite_std:8.3f}  {z_str}")

    print(f"\nn_skaters in elite reference for this phase: {int(phase_profile['n_skaters'].iloc[0])}")
    print("HONEST NOTE: with a small elite-reference sample, treat z-scores as")
    print("directional signals, not precise percentile rankings.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-profile", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--video", type=str)
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--phase", type=str, choices=VALID_PHASES)
    parser.add_argument("--name", type=str, default="Comparison Subject")
    args = parser.parse_args()

    if args.build_profile:
        build_elite_profile()
    elif args.compare:
        if not all([args.video, args.start is not None, args.end is not None, args.phase]):
            print("--compare requires --video, --start, --end, --phase")
            return
        compare_segment_to_profile(args.video, args.start, args.end, args.phase, args.name)
    else:
        print("Specify --build-profile or --compare. See docstring for usage.")


if __name__ == "__main__":
    main()
