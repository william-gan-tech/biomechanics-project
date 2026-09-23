"""
Fixes a real methodological limitation identified 9/21: acceleration
segments were manually picked per skater as "looked visually reasonable,"
not at a consistent RELATIVE position within each skater's own
acceleration ramp. This meant cross-skater velocity comparisons could be
comparing, e.g., skater A's early ramp against skater B's late ramp,
inflating apparent variance for reasons unrelated to real technique.

This script standardizes by:
  1. For each skater's labeled start_acceleration segment, finding the
     frame where velocity first crosses a RELATIVE threshold (a percentage
     of that skater's own max velocity within the segment) -- this defines
     a consistent "ramp onset" point regardless of each skater's absolute
     speed scale.
  2. Extracting a fixed-duration window of frames starting from that onset
     point for every skater.
  3. Recomputing cross-skater stats on this standardized window and
     comparing against the original (raw, manually-picked) segment stats.

Usage:
    python -m standardize_ramp_segments
"""

import csv
import numpy as np

from run_bone_scaling_ablation import get_skater_features, filter_implausible_frames
from start_phase_features import add_start_phase_features

LOG_PATH = "labeled_segments.csv"
ONSET_THRESHOLD_PCT = 0.20
WINDOW_LENGTH = 20


def load_acceleration_segments():
    rows = []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["phase"] == "start_acceleration":
                rows.append(row)
    return rows


def get_segment_df(row):
    skater = row["skater"]
    video_path = row["video_path"]
    start_f = int(row["start_frame"])
    end_f = int(row["end_frame"])

    df = get_skater_features(skater, video_path, "scaled")
    if df is None:
        return None
    df = filter_implausible_frames(df)
    df = add_start_phase_features(df)

    segment = df[(df["frame"] >= start_f) & (df["frame"] <= end_f)].reset_index(drop=True)
    if segment.empty:
        return None
    return segment


def find_ramp_onset(segment_df, threshold_pct=ONSET_THRESHOLD_PCT):
    velocity = segment_df["hip_velocity"].fillna(0).values
    max_v = velocity.max()
    if max_v <= 0:
        return 0
    threshold = max_v * threshold_pct
    onset_candidates = np.where(velocity >= threshold)[0]
    if len(onset_candidates) == 0:
        return 0
    return onset_candidates[0]


def main():
    rows = load_acceleration_segments()
    print(f"Found {len(rows)} labeled 'start_acceleration' segments\n")

    per_skater_raw = {}
    per_skater_standardized = {}

    for row in rows:
        skater = row["skater"]
        segment_df = get_segment_df(row)
        if segment_df is None:
            print(f"  [SKIP] {skater}: could not load segment data")
            continue

        raw_velocity_mean = segment_df["hip_velocity"].mean()
        per_skater_raw.setdefault(skater, []).append(raw_velocity_mean)

        onset_idx = find_ramp_onset(segment_df)
        window = segment_df.iloc[onset_idx:onset_idx + WINDOW_LENGTH]
        std_velocity_mean = window["hip_velocity"].mean()

        print(f"{skater}: raw segment mean={raw_velocity_mean:.4f} | "
              f"ramp onset at index {onset_idx} (frame {segment_df.iloc[onset_idx]['frame']:.0f}) | "
              f"standardized {WINDOW_LENGTH}-frame window mean={std_velocity_mean:.4f}")

        per_skater_standardized.setdefault(skater, []).append(std_velocity_mean)

    per_skater_raw_final = {s: float(np.mean(v)) for s, v in per_skater_raw.items()}
    per_skater_std_final = {s: float(np.mean(v)) for s, v in per_skater_standardized.items()}

    raw_values = np.array(list(per_skater_raw_final.values()))
    std_values = np.array(list(per_skater_std_final.values()))

    print("\n" + "=" * 70)
    print("COMPARISON: raw (manually-picked ranges) vs standardized (onset-relative)")
    print("=" * 70)
    print(f"RAW segments:           mean={raw_values.mean():.4f}, std={raw_values.std():.4f}")
    print(f"STANDARDIZED segments:  mean={std_values.mean():.4f}, std={std_values.std():.4f}")

    if raw_values.std() > 0:
        pct_change = (std_values.std() - raw_values.std()) / raw_values.std() * 100
        print(f"\nStd change from standardizing: {pct_change:+.1f}%")
        if pct_change < -15:
            print("Standardizing MEANINGFULLY reduced cross-skater variance -- supports the")
            print("hypothesis that inconsistent segment placement (not real technique")
            print("differences) was inflating the original raw-segment variance.")
        elif pct_change > 15:
            print("Standardizing INCREASED variance -- the raw segments may have coincidentally")
            print("been more comparable than this onset-detection method, or onset detection")
            print("itself may be noisy for this data. Worth reviewing individual onset points above.")
        else:
            print("Std is similar either way -- segment placement inconsistency does not appear")
            print("to be a major driver of the original cross-skater variance for this metric.")


if __name__ == "__main__":
    main()
