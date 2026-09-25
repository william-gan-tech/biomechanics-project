"""
Phase 4a/5b/5c: biomechanical features for distinguishing technique phases
(start, corner, straightaway) and characterizing arm swing and sit height.

  1. Torso-lean/crouch angle: angle between the shoulder-hip vector and
     vertical.
     FIXED 9/22: the primary `torso_lean_angle_deg` column is the ABSOLUTE
     VALUE of the raw signed angle (see torso_lean_angle_deg_signed for
     the raw version). Sensitivity testing on 9/21 found raw signed angle
     direction depends on camera orientation/turn direction, not real
     technique -- inflated cross-skater variance (42.28 vs real 17.63).
  2. Hip velocity / acceleration: frame-to-frame hip displacement.
  3. Bilateral (left-side) tracking, added 9/22: `torso_lean_angle_deg_left`
     and `hip_lateral_asymmetry` (bone-scaled left/right hip distance).
     Requires left-side columns in the source data (re-run
     preprocess_video.py after the 9/22 update).
  4. Arm swing, added 9/22 (Phase 5b): `right_arm_swing_amplitude`,
     `left_arm_swing_amplitude`. `right_elbow_angle`/`left_elbow_angle`
     are saved directly by preprocess_video.py.
     KNOWN LIMITATION (documented, unresolved as of 9/23): elbow-angle
     data can be confidently WRONG (not just noisy) in some frames --
     confirmed via direct visual inspection that computed angle swung
     150+ degrees despite the visible arm being nearly static across the
     same ~0.1s window. Attempted fix using MediaPipe's own landmark
     visibility scores; tested directly and found it does NOT reliably
     separate bad frames (known-bad frames scored 0.76-0.81, within the
     clip's normal 0.58-0.97 range). No spike filter is applied here
     since a jump-based filter was already shown to miss this exact
     failure mode. Practical mitigation: manually spot-check arm-swing
     segments against visible video before trusting them, and prefer
     frames well past the start of a clip.
  5. Sit height / knee-bend depth, added 9/24 (Phase 5c): `sit_height_ratio`
     -- vertical hip-to-ankle distance, normalized by the skater's own
     bone-scale. Lower ratio = more crouched. Requires ankle position
     columns (re-run preprocess_video.py after the 9/24 update).

IMPORTANT LIMITATION, stated honestly: all position-based features here
are 2D projections from a single camera view, not true 3D measurement.
This is a permanent, stated limitation of the approach.

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
    df["hip_acceleration"] = df["hip_velocity"].diff()

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

    # Sit height / knee-bend depth (Phase 5c, 9/24). Vertical hip-to-ankle
    # distance, already bone-scale-normalized (same origin/scale as every
    # other position feature). Standing upright = larger distance; deep
    # crouch = smaller distance (hip drops toward ankle).
    if "norm_right_ankle_y" in df.columns:
        df["hip_to_ankle_vertical_right"] = (df["norm_right_ankle_y"] - df["norm_right_hip_y"]).abs()
    if "norm_left_ankle_y" in df.columns and "norm_left_hip_y" in df.columns:
        df["hip_to_ankle_vertical_left"] = (df["norm_left_ankle_y"] - df["norm_left_hip_y"]).abs()

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


def summarize_sit_height_vs_standing(df, standing_frame_range=None):
    """Computes a sit-height RATIO relative to a standing reference, so
    the metric is comparable across skaters of different heights/camera
    distances. If `standing_frame_range` (start_frame, end_frame) is
    given, uses the mean hip-to-ankle distance in that range as the
    standing reference -- intended to be a skater's own start_rest
    segment. If not given, uses this clip's own maximum hip-to-ankle
    distance as a proxy standing reference.
    """
    if "hip_to_ankle_vertical_right" not in df.columns:
        return None

    if standing_frame_range is not None:
        start_f, end_f = standing_frame_range
        standing_segment = df[(df["frame"] >= start_f) & (df["frame"] <= end_f)]
        if standing_segment.empty:
            return None
        standing_reference = standing_segment["hip_to_ankle_vertical_right"].mean()
    else:
        standing_reference = df["hip_to_ankle_vertical_right"].max()

    if standing_reference <= 0:
        return None

    df = df.copy()
    df["sit_height_ratio"] = df["hip_to_ankle_vertical_right"] / standing_reference
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
    cols_to_show = ["frame", "torso_lean_angle_deg", "hip_velocity", "hip_acceleration"]
    if "hip_to_ankle_vertical_right" in df.columns:
        cols_to_show.append("hip_to_ankle_vertical_right")
    print(df[cols_to_show].head(10))

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
