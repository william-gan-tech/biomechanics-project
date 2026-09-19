"""
Phase 4a: real start-vs-cruise comparison for start_candidate_3.mp4
(Milano Cortina 2026 short track, visually confirmed 9/18).

Unlike the earlier automated-detection attempts, this uses a MANUALLY
VISUALLY CONFIRMED segment structure, established by directly viewing
frame images rather than trusting the acceleration-spike algorithm
(which was shown to be unreliable on this chaotic broadcast footage --
see CAPABILITIES_PHASE4.md for the full debugging history):

  - REST segment (5.17s-5.64s, frames 155-169): skaters visually
    confirmed held in start position.
  - GAP (5.71s-6.37s): no landmark data -- a real broadcast camera cut
    around the gun obscures the literal launch instant. This is a
    genuine, honestly-documented data gap, not something papered over.
  - EARLY-ACCELERATION segment (6.44s onward, frame 193+): skaters
    visually confirmed immediately off the line, still accelerating.
  - CRUISE segment: a window later in the race, once skaters have
    settled into rhythmic stride (chosen well after the acceleration
    phase, same logic as Phase 3b's fresh/fatigued proxy).

Usage:
    python -m compare_start_vs_cruise
"""

import os
import numpy as np
import pandas as pd

from run_bone_scaling_ablation import get_skater_features, filter_implausible_frames
from start_phase_features import add_start_phase_features

VIDEO_NAME = "start_candidate_3"
VIDEO_PATH = "data/start_candidate_3.mp4"

REST_FRAME_RANGE = (155, 169)             # visually confirmed held start position
EARLY_ACCEL_FRAME_RANGE = (193, 245)      # visually confirmed immediately post-gun
CRUISE_FRAME_RANGE = (500, 600)           # NOT yet visually confirmed -- verify before trusting


def summarize_segment(df, frame_range, label):
    subset = df[(df["frame"] >= frame_range[0]) & (df["frame"] <= frame_range[1])]
    if subset.empty:
        print(f"{label}: NO DATA in frame range {frame_range} -- check this range against the real video.")
        return None

    summary = {
        "segment": label,
        "frame_range": f"{frame_range[0]}-{frame_range[1]}",
        "n_frames": len(subset),
        "torso_lean_mean": subset["torso_lean_angle_deg"].mean(),
        "torso_lean_std": subset["torso_lean_angle_deg"].std(),
        "velocity_mean": subset["hip_velocity"].mean(),
        "velocity_std": subset["hip_velocity"].std(),
        "acceleration_mean": subset["hip_acceleration"].abs().mean(),
    }
    return summary


def main():
    print(f"Loading features for {VIDEO_NAME}...\n")
    df = get_skater_features(VIDEO_NAME, VIDEO_PATH, "scaled")
    if df is None:
        print("Could not load features -- run test_start_candidate.py for this video first.")
        return

    df = filter_implausible_frames(df)
    df = add_start_phase_features(df)

    print(f"Total frames with data: {len(df)}, frame range: {df['frame'].min()}-{df['frame'].max()}\n")

    print("=" * 70)
    print("SEGMENT SUMMARY (real, visually-confirmed segments)")
    print("=" * 70)

    rest_summary = summarize_segment(df, REST_FRAME_RANGE, "REST (held start position)")
    accel_summary = summarize_segment(df, EARLY_ACCEL_FRAME_RANGE, "EARLY-ACCELERATION (post-gun)")
    cruise_summary = summarize_segment(df, CRUISE_FRAME_RANGE, "CRUISE (later steady-state)")

    results = [s for s in [rest_summary, accel_summary, cruise_summary] if s is not None]
    if not results:
        print("\nNo segments had data -- check the frame ranges above against your actual visual confirmation.")
        return

    results_df = pd.DataFrame(results)
    print("\n" + results_df.to_string(index=False))

    if rest_summary and accel_summary:
        vel_ratio = accel_summary["velocity_mean"] / max(rest_summary["velocity_mean"], 1e-6)
        print(f"\nVelocity ratio (early-acceleration / rest): {vel_ratio:.1f}x")
        print("(A real start should show a large ratio here -- rest should be near-zero,")
        print(" acceleration should be substantially higher.)")

    if accel_summary and cruise_summary:
        torso_diff = accel_summary["torso_lean_mean"] - cruise_summary["torso_lean_mean"]
        print(f"\nTorso lean difference (early-acceleration - cruise): {torso_diff:+.1f} degrees")
        print("(This is the actual Phase 4a signal -- does start-phase torso lean")
        print(" differ meaningfully from steady-state cruising lean?)")

    output_path = "phase4a_start_vs_cruise_summary.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved -> {output_path}")

    print("\nIMPORTANT: the CRUISE_FRAME_RANGE (500-600) has NOT been visually confirmed")
    print("yet -- verify it shows real steady-state cruising before trusting this comparison.")
    print("Check with: python -m browse_frames_manually --video data/start_candidate_3.mp4 --start 16 --end 20")


if __name__ == "__main__":
    main()
