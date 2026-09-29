"""
Diagnoses WHY start_candidate_3's hip-to-ankle score is extreme in every
phase (LOO validation, 9/28: -8.9 rest, -5.6 accel, -4.3 corner, -6.3
straightaway). A consistent offset across all four phases points at the
video, not technique.

Hypothesis: preprocess_video.py normalizes every frame by ONE fixed
per-video scale (reference_scale * image height, calibrated from early
frames). If that calibration is off for a video -- or the camera zooms
during it -- every normalized distance in that video is off by the same
factor.

Test: for each labeled segment, compare the fixed scale against the
skater's torso length actually measured in those frames. If the fixed
scale is right, scale_used_px / frame_torso_length_px should be similar
across videos. Then recompute vertical hip-to-ankle normalized by the
PER-FRAME torso length instead, and see whether start_candidate_3 stops
standing out.

Usage:
    python -m diagnose_sit_height_scale
"""

import numpy as np
import pandas as pd

from compare_to_elite_reference import load_labeled_segments
from run_bone_scaling_ablation import get_skater_features, filter_implausible_frames


def main():
    rows = []
    for seg in load_labeled_segments():
        df = get_skater_features(seg["skater"], seg["video_path"], "scaled")
        if df is None:
            continue
        df = filter_implausible_frames(df)
        s = df[(df["frame"] >= int(seg["start_frame"])) & (df["frame"] <= int(seg["end_frame"]))]
        if s.empty:
            continue

        vertical_fixed = (s["norm_right_ankle_y"] - s["norm_right_hip_y"]).abs()
        # Undo the fixed scale (back to pixels), divide by this frame's own torso length
        vertical_per_frame = vertical_fixed * s["scale_used_px"] / s["frame_torso_length_px"]
        # Segment-level scale: 90th-percentile torso length = the torso at its
        # least foreshortened, at this shot's zoom level
        segment_torso_p90 = s["frame_torso_length_px"].quantile(0.9)
        vertical_segment_p90 = vertical_fixed * s["scale_used_px"] / segment_torso_p90

        rows.append({
            "skater": seg["skater"],
            "phase": seg["phase"],
            "frames": len(s),
            "fixed_scale_px": s["scale_used_px"].median(),
            "torso_px_median": s["frame_torso_length_px"].median(),
            "scale_over_torso": (s["scale_used_px"] / s["frame_torso_length_px"]).median(),
            "vertical_fixed_scale": vertical_fixed.mean(),
            "vertical_per_frame_torso": vertical_per_frame.mean(),
            "vertical_segment_p90": vertical_segment_p90.mean(),
        })

    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)

    print("\nPer segment:")
    print(out.round(3).to_string(index=False))

    print("\nPer skater (averaged over their segments):")
    variants = ["vertical_fixed_scale", "vertical_per_frame_torso", "vertical_segment_p90"]
    by_skater = out.groupby("skater")[["scale_over_torso"] + variants].mean()
    print(by_skater.round(3).to_string())

    print("\nCoefficient of variation ACROSS skaters (lower = more comparable):")
    for col in variants:
        v = by_skater[col]
        print(f"  {col:26s}: {v.std(ddof=1) / v.mean():.2f}")

    print("\nREADING THIS: if start_candidate_3's scale_over_torso is far from the")
    print("others AND its vertical_per_frame_torso moves back in line, the fixed")
    print("per-video calibration is the cause, not the skater's technique.")


if __name__ == "__main__":
    main()
