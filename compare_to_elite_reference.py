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
from functools import lru_cache
import cv2
import numpy as np
import pandas as pd
from scipy import stats

from run_bone_scaling_ablation import get_skater_features, filter_implausible_frames
from start_phase_features import add_start_phase_features

LOG_PATH = "labeled_segments.csv"
PROFILE_PATH = "elite_reference_profile.csv"
VALID_PHASES = ["start_rest", "start_acceleration", "corner", "straightaway"]
METRICS = [
    "torso_lean_angle_deg", "hip_velocity", "hip_acceleration_abs",
    "right_knee_filtered", "left_knee_filtered", "hip_to_ankle_vertical_right",
    "hip_to_ankle_2d_right", "hip_to_ankle_lateral_right",
    # Phase 5e bilateral asymmetry (9/28)
    "knee_angle_asymmetry", "hip_height_asymmetry", "torso_lean_lr_diff",
    "hip_lateral_asymmetry",
]

# Distance-based metrics (everything measured in body-scale units rather than
# degrees). FIXED 9/28: these were divided by ONE fixed per-video scale set by
# early-frame calibration. diagnose_sit_height_scale.py found that scale was
# 2-26x too large for start_candidate_3 (a close-up at the start of the
# broadcast), shrinking every distance in that video, and drifting ~2x
# within Haralds Silovs' video from camera zoom. They're now rescaled per
# SEGMENT by the segment's 90th-percentile torso length (the torso at its least
# foreshortened, at that shot's zoom level). Angle metrics are unaffected.
# Set False to reproduce pre-9/28 numbers.
SEGMENT_RESCALE = True
DISTANCE_METRICS = [
    "hip_velocity", "hip_acceleration_abs", "hip_to_ankle_vertical_right",
    "hip_to_ankle_2d_right", "hip_to_ankle_lateral_right",
    "hip_height_asymmetry", "hip_lateral_asymmetry",
]

# FIXED 9/28: hip_velocity / hip_acceleration were per-FRAME, but the videos
# mix 25 fps (Sven Kramer, Silovs, Ragne Wiklund) and 29.97 fps (the rest),
# biasing 25 fps velocity ~1.2x high and acceleration ~1.44x high. Converted
# to per-second units using each video's real frame rate.
# Set False to reproduce pre-9/28 numbers.
PER_SECOND_UNITS = True


@lru_cache(maxsize=None)
def get_video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps and fps > 0 else None


def load_labeled_segments():
    rows = []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["phase"] in VALID_PHASES:
                rows.append(row)
    return rows


# 10/7: frames trimmed off BOTH ends of every segment before analysis (default 0).
# Used as a robustness check: corner entry/exit can look like a straightaway, so
# phase calls are least certain near segment boundaries. Set via the
# SEGMENT_TRIM_FRAMES environment variable, e.g. 6 (~0.25 s at 25 fps).
SEGMENT_TRIM_FRAMES = int(os.environ.get("SEGMENT_TRIM_FRAMES", "0"))


def get_segment_frames(row):
    """Per-frame features for one labeled segment, with the 9/28 corrections
    applied (per-second velocity units, per-segment distance rescaling).
    Every metric in METRICS is a column. Shared by the profile builder and
    the per-frame analyses (e.g. validate_asymmetry_by_phase.py)."""
    skater = row["skater"]
    video_path = row["video_path"]
    start_f = int(row["start_frame"]) + SEGMENT_TRIM_FRAMES
    end_f = int(row["end_frame"]) - SEGMENT_TRIM_FRAMES
    if end_f <= start_f:
        return None

    df = get_skater_features(skater, video_path, "scaled")
    if df is None:
        return None
    df = filter_implausible_frames(df)
    df = add_start_phase_features(df)

    segment = df[(df["frame"] >= start_f) & (df["frame"] <= end_f)].copy()
    if segment.empty:
        return None
    segment["hip_acceleration_abs"] = segment["hip_acceleration"].abs()

    if PER_SECOND_UNITS:
        fps = get_video_fps(video_path)
        if fps is None:
            print(f"  [WARN] {skater}: could not read fps -- velocity left per-frame")
        else:
            segment["hip_velocity"] *= fps
            segment["hip_acceleration_abs"] *= fps ** 2

    if SEGMENT_RESCALE:
        if "scale_used_px" not in segment.columns or "frame_torso_length_px" not in segment.columns:
            print(f"  [WARN] {skater}: no scale columns cached -- distance metrics left on fixed per-video scale")
        else:
            # Back to pixels (x fixed scale), then into segment torso units
            factor = segment["scale_used_px"].median() / segment["frame_torso_length_px"].quantile(0.9)
            for m in DISTANCE_METRICS:
                if m in segment.columns:
                    segment[m] *= factor
    return segment


def extract_segment_means(row):
    segment = get_segment_frames(row)
    if segment is None:
        return None
    return {m: (segment[m].mean() if m in segment.columns else None) for m in METRICS}

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
                # ddof=1 (sample std): population std underestimates spread at n=3-4
                "elite_std": values.std(ddof=1) if n_skaters > 1 else 0.0,
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

    n_ref = int(phase_profile["n_skaters"].iloc[0])

    for _, ref_row in phase_profile.iterrows():
        metric = ref_row["metric"]
        elite_mean = ref_row["elite_mean"]
        elite_std = ref_row["elite_std"]
        your_value = means.get(metric)

        if your_value is None:
            continue

        if elite_std > 0:
            z_score = (your_value - elite_mean) / elite_std
            # A NEW skater isn't part of the reference, so use the prediction
            # score t = z / sqrt(1 + 1/n), which follows Student-t with n-1 df
            # (validated by validate_elite_reference_loo.py, 9/28). Plain z read
            # as normal flagged 33% of held-out elite skaters as outliers.
            t_pred = z_score / np.sqrt(1 + 1 / n_ref)
            p_value = 2 * stats.t.sf(abs(t_pred), df=n_ref - 1)
            flag = "  <-- unusual (p<0.10)" if p_value < 0.10 else ""
            z_str = f"z={z_score:+.2f}  t_pred={t_pred:+.2f}  p={p_value:.2f}{flag}"
        else:
            z_str = "z=N/A (elite std is 0, likely n=1 reference)"

        print(f"{metric:27s}: you={your_value:8.3f}  elite_mean={elite_mean:8.3f}  "
              f"elite_std={elite_std:8.3f}  {z_str}")

    print(f"\nn_skaters in elite reference for this phase: {n_ref}")
    if n_ref > 1:
        print(f"A score is only flagged if |t_pred| > {stats.t.ppf(0.95, df=n_ref - 1):.2f} "
              f"(t with {n_ref - 1} df, p<0.10).")
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
