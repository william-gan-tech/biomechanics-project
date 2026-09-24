"""
Phase 4a/5b: biomechanical features for distinguishing technique phases
(start, corner, straightaway) and, as of Phase 5b, arm swing.

  1. Torso-lean/crouch angle: angle between the shoulder-hip vector and
     vertical.
     FIXED 9/22: the primary `torso_lean_angle_deg` column is the ABSOLUTE
     VALUE of the raw signed angle. Sensitivity testing on 9/21 found that
     raw signed angle direction depends on camera orientation and turn
     direction, not real technique -- this was inflating apparent
     cross-skater variance (42.28 degrees std, mostly artifact, vs. 17.63
     real). The raw signed value is preserved as `torso_lean_angle_deg_signed`.
  2. Hip velocity: frame-to-frame displacement of hip position.
  3. Hip acceleration: frame-to-frame change in velocity.
  4. Bilateral (left-side) tracking, added 9/22: `torso_lean_angle_deg_left`
     and `hip_lateral_asymmetry` (real distance between left/right hip
     position, bone-scaled) -- only computed if the underlying data has
     left-side columns (requires re-running preprocess_video.py after the
     9/22 update; older cached CSVs won't have this).
  5. Arm swing, added 9/22 (Phase 5b): `right_arm_swing_amplitude` and
     `left_arm_swing_amplitude` (frame-to-frame elbow displacement).
     `right_elbow_angle`/`left_elbow_angle` (arm bend) are saved directly
     by preprocess_video.py. A spike filter is applied to both elbow-angle
     columns: direct inspection confirmed ~2.9% of frames show an isolated,
     physically implausible single-frame jump (>60 degrees in 1/25s --
     no human arm moves that fast), almost certainly wrist-detection
     glitches. Surrounding data is clean and smooth (spot-checked), so
     these are isolated spikes, interpolated over -- NOT a systemic
     problem with the feature.

IMPORTANT LIMITATION, stated honestly: hip/torso features are computed
from the RIGHT side plus (as of 9/22) the LEFT side; skating is not
perfectly symmetric, especially at push-off, so left/right differences
can reflect real technique, not just noise -- see hip_lateral_asymmetry.

Usage:
    python -m start_phase_features
"""

import os
import numpy as np
import pandas as pd


def add_start_phase_features(df):
    df = df.sort_values("frame").reset_index(drop=True)

    dx = df["norm_right_shoulder_x"] - df["norm_right_hip_x"]
    dy = df["norm_right_shoulder_y"] - df["norm_right_hip_y"]
    torso_lean_rad = np.arctan2(dx, -dy)
    df["torso_lean_angle_deg_signed"] = np.degrees(torso_lean_rad)
    df["torso_lean_angle_deg"] = df["torso_lean_angle_deg_signed"].abs()

    hip_dx = df["norm_right_hip_x"].diff()
    hip_dy = df["norm_right_hip_y"].diff()
    df["hip_velocity"] = np.sqrt(hip_dx**2 + hip_dy**2)

    if "norm_left_hip_x" in df.columns and "norm_left_shoulder_x" in df.columns:
        l_dx = df["norm_left_shoulder_x"] - df["norm_left_hip_x"]
        l_dy = df["norm_left_shoulder_y"] - df["norm_left_hip_y"]
        left_torso_lean_rad = np.arctan2(l_dx, -l_dy)
        df["torso_lean_angle_deg_left_signed"] = np.degrees(left_torso_lean_rad)
        df["torso_lean_angle_deg_left"] = df["torso_lean_angle_deg_left_signed"].abs()

        hip_asymmetry_dx = df["norm_right_hip_x"] - df["norm_left_hip_x"]
        hip_asymmetry_dy = df["norm_right_hip_y"] - df["norm_left_hip_y"]
        df["hip_lateral_asymmetry"] = np.sqrt(hip_asymmetry_dx**2 + hip_asymmetry_dy**2)

    if "norm_right_elbow_x" in df.columns:
        r_elbow_dx = df["norm_right_elbow_x"].diff()
        r_elbow_dy = df["norm_right_elbow_y"].diff()
        df["right_arm_swing_amplitude"] = np.sqrt(r_elbow_dx**2 + r_elbow_dy**2)

    if "norm_left_elbow_x" in df.columns:
        l_elbow_dx = df["norm_left_elbow_x"].diff()
        l_elbow_dy = df["norm_left_elbow_y"].diff()
        df["left_arm_swing_amplitude"] = np.sqrt(l_elbow_dx**2 + l_elbow_dy**2)

    for col in ["right_elbow_angle", "left_elbow_angle"]:
        if col not in df.columns:
            continue
        values = df[col].copy()
        jump_in = values.diff().abs()
        jump_out = values.diff(-1).abs()
        isolated_spike = (jump_in > 60) & (jump_out > 60)
        values[isolated_spike] = np.nan
        df[col] = values.interpolate(limit=2)

    return df


def summarize_start_vs_rest(df, start_frame_count=30):
    if len(df) <= start_frame_count:
        return None

    start_phase = df.iloc[:start_frame_count]
    rest_phase = df.iloc[start_frame_count:]

    summary = {}
    for col in ["torso_lean_angle_deg", "hip_velocity", "hip_acceleration"]:
        if col not in df.columns:
            continue
        summary[col] = {
            "start_phase_mean": start_phase[col].mean(),
            "start_phase_std": start_phase[col].std(),
            "rest_phase_mean": rest_phase[col].mean(),
            "rest_phase_std": rest_phase[col].std(),
        }
    return summary


def main():
    cache_dir = "ablation_feature_cache"
    test_files = []
    if os.path.isdir(cache_dir):
        test_files = [f for f in os.listdir(cache_dir) if f.endswith("_scaled_features.csv")]

    if test_files:
        test_file = os.path.join(cache_dir, test_files[0])
        print(f"Testing feature math against cached file: {test_file}\n")
        df = pd.read_csv(test_file)
    else:
        print(f"No cached feature files found in {cache_dir} -- extracting fresh from a video instead.\n")
        from pipeline_engine import compute_video_reference_scale
        from preprocess_video import process_skating_video_multivariate
        import cv2

        video_path = "data/sven_kramer_ref.mp4"
        if not os.path.exists(video_path):
            print(f"{video_path} not found either -- edit this script's video_path to point at a real video.")
            return

        print(f"Extracting features from {video_path} (this takes a minute)...")
        reference_scale = compute_video_reference_scale(video_path)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        df = process_skating_video_multivariate(video_path, fps=fps, reference_scale=reference_scale)
        if df is None or df.empty:
            print("Feature extraction failed.")
            return
        print(f"Extracted {len(df)} frames.\n")

    df = add_start_phase_features(df)

    print("New columns added:")
    print(df[["frame", "torso_lean_angle_deg", "hip_velocity", "hip_acceleration"]].head(10))

    print("\n--- Descriptive summary: first 30 frames vs. rest ---")
    summary = summarize_start_vs_rest(df)
    if summary:
        for feature, stats in summary.items():
            print(f"{feature}:")
            print(f"  Start-phase (first 30 frames): mean={stats['start_phase_mean']:.4f}, std={stats['start_phase_std']:.4f}")
            print(f"  Rest of clip:                  mean={stats['rest_phase_mean']:.4f}, std={stats['rest_phase_std']:.4f}")
            print()


if __name__ == "__main__":
    main()
