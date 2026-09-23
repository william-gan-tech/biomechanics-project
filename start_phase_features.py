
"""
Phase 4a: new biomechanical features for distinguishing start-phase
acceleration mechanics from steady-state cruising form.

These did NOT exist in the Phase 3 feature set (which only had knee
angles + hip/shoulder position). This module adds:

  1. Torso-lean/crouch angle: angle between the shoulder-hip vector and
     vertical. A crouched start position should show a much larger lean
     angle than upright cruising form.
     FIXED 9/22: the primary `torso_lean_angle_deg` column is now the
     ABSOLUTE VALUE of the raw signed angle. Sensitivity testing on 9/21
     found that raw signed angle direction depends on camera orientation
     and turn direction, not real technique -- this was inflating
     apparent cross-skater variance (42.28 degrees std, mostly artifact,
     vs. 17.63 real). The raw signed value is still available as
     `torso_lean_angle_deg_signed` for anyone specifically studying
     left-vs-right turn asymmetry within one consistent camera setup.
  2. Hip velocity: frame-to-frame displacement of hip position (proxy for
     speed, since we don't have real-world distance calibration).
  3. Hip acceleration: frame-to-frame change in velocity -- this is the
     actual "explosiveness" signal a start phase should show a spike in,
     that steady-state cruising should not.

IMPORTANT LIMITATION, stated honestly: these features are computed from
norm_right_hip_x/y and norm_right_shoulder_x/y -- the RIGHT side only,
since that's what the existing pipeline saves. This is a real asymmetry
worth being aware of (skating is not symmetric, especially at push-off);
extending to compute left-side or averaged features would need pipeline
changes and is a legitimate future improvement, not done here.

Can be run RIGHT NOW against already-cached Phase 3 feature data --
doesn't require new start-line footage to test the math itself, only to
validate whether it captures a REAL start-vs-cruising distinction (which
does require real start footage, per Phase 4's footage audit).

Usage:
    python -m start_phase_features
"""

import os
import numpy as np
import pandas as pd


def add_start_phase_features(df):
    """Takes a feature DataFrame (as produced by
    process_skating_video_multivariate, or loaded from the ablation
    feature cache) and adds torso-lean angle, hip velocity, and hip
    acceleration columns.

    Assumes df is already sorted by frame (ablation cache files are).
    """
    df = df.sort_values("frame").reset_index(drop=True)

    dx = df["norm_right_shoulder_x"] - df["norm_right_hip_x"]
    dy = df["norm_right_shoulder_y"] - df["norm_right_hip_y"]
    torso_lean_rad = np.arctan2(dx, -dy)
    df["torso_lean_angle_deg_signed"] = np.degrees(torso_lean_rad)

    # PRIMARY metric for cross-skater/cross-video comparisons. Raw signed
    # angle direction depends on camera orientation (which side of the rink)
    # and turn direction (left-hand vs right-hand oval), NOT on real
    # technique differences -- confirmed 9/21 via check_knee_variance_outlier.py:
    # using magnitude instead of signed value reduced a false cross-skater
    # std of 42.28 degrees (mostly a sign artifact) down to a real 17.63.
    # The signed column above is kept for anyone specifically studying
    # left-vs-right turn asymmetry within one consistent camera setup, where
    # sign IS meaningful -- but it should NOT be used for cross-video
    # comparisons without controlling for camera/turn-direction first.
    df["torso_lean_angle_deg"] = df["torso_lean_angle_deg_signed"].abs()

    hip_dx = df["norm_right_hip_x"].diff()
    hip_dy = df["norm_right_hip_y"].diff()
    df["hip_velocity"] = np.sqrt(hip_dx**2 + hip_dy**2)

    # NEW: real bilateral asymmetry metric, only possible now that left-side
    # position is extracted. Gracefully skipped (columns not added) if the
    # underlying data doesn't have left-side columns yet (e.g. older cached
    # CSVs from before this feature existed) -- re-run extraction to get it.
    if "norm_left_hip_x" in df.columns and "norm_left_shoulder_x" in df.columns:
        l_dx = df["norm_left_shoulder_x"] - df["norm_left_hip_x"]
        l_dy = df["norm_left_shoulder_y"] - df["norm_left_hip_y"]
        left_torso_lean_rad = np.arctan2(l_dx, -l_dy)
        df["torso_lean_angle_deg_left_signed"] = np.degrees(left_torso_lean_rad)
        df["torso_lean_angle_deg_left"] = df["torso_lean_angle_deg_left_signed"].abs()

        # The actual asymmetry signal: how far apart are left and right hip
        # position, relative to bone-scaled body size. A skater standing/
        # moving perfectly symmetrically would show near-zero; corner
        # technique (inherently asymmetric -- inside vs outside leg do
        # different things) should show a real, larger value.
        hip_asymmetry_dx = df["norm_right_hip_x"] - df["norm_left_hip_x"]
        hip_asymmetry_dy = df["norm_right_hip_y"] - df["norm_left_hip_y"]
        df["hip_lateral_asymmetry"] = np.sqrt(hip_asymmetry_dx**2 + hip_asymmetry_dy**2)

    df["hip_acceleration"] = df["hip_velocity"].diff()

    return df


def summarize_start_vs_rest(df, start_frame_count=30):
    """Quick descriptive comparison: first `start_frame_count` frames
    (proxy for 'start phase') vs. the rest of the clip (proxy for
    'cruising phase'). Same honesty standard as the Phase 3b fresh/fatigued
    proxy -- this assumes the clip begins at or near the actual start,
    which is NOT verified here and needs to be confirmed per-video before
    trusting the comparison.
    """
    if len(df) <= start_frame_count:
        return None

    start_phase = df.iloc[:start_frame_count]
    rest_phase = df.iloc[start_frame_count:]

    summary = {}
    for col in ["torso_lean_angle_deg", "hip_velocity", "hip_acceleration"]:
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
    print("(NOTE: this is a PROXY comparison since this clip is not confirmed")
    print(" to actually start at a start-line moment -- treat as a math sanity")
    print(" check only, not a real start-phase finding.)\n")
    summary = summarize_start_vs_rest(df)
    if summary:
        for feature, stats in summary.items():
            print(f"{feature}:")
            print(f"  Start-phase (first 30 frames): mean={stats['start_phase_mean']:.4f}, std={stats['start_phase_std']:.4f}")
            print(f"  Rest of clip:                  mean={stats['rest_phase_mean']:.4f}, std={stats['rest_phase_std']:.4f}")
            print()


if __name__ == "__main__":
    main()
