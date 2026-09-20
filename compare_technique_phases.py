"""
Phase 4a (and groundwork for future Phase 5/6): compares bone-scaled
kinematic features across manually-categorized technique phases (start,
corner, straightaway) within the SAME skater's video -- a real
within-subject comparison, fixing the earlier problem where no clean
single "cruise" segment could be found in short-track footage.

Frame ranges below should be filled in based on YOUR visual review of
the images in manual_frame_browse/ -- this script does not guess at
categories, it uses your confirmed labels.

Usage:
    python -m compare_technique_phases
"""

import pandas as pd
from run_bone_scaling_ablation import get_skater_features, filter_implausible_frames
from start_phase_features import add_start_phase_features

VIDEO_NAME = "start_candidate_3"
VIDEO_PATH = "data/start_candidate_3.mp4"

# Fill these in based on what you visually confirmed in manual_frame_browse/.
# Each category can have MULTIPLE frame ranges (e.g. several straightaway
# segments across the race) -- list of (start_frame, end_frame) tuples.
PHASE_FRAME_RANGES = {
    "start": [
        (155, 169),   # confirmed REST (held start position)
        (197, 269),   # confirmed acceleration/launch phase, visually reviewed 9/18
    ],
    "corner": [
        (950, 1049),  # visually confirmed cornering, 9/18
    ],
    "straightaway": [
        (785, 884),   # visually confirmed straightaway strides, 9/18
    ],
}


def label_frames(df, phase_ranges):
    df = df.copy()
    df["phase"] = None
    for phase_name, ranges in phase_ranges.items():
        for start_f, end_f in ranges:
            mask = (df["frame"] >= start_f) & (df["frame"] <= end_f)
            df.loc[mask, "phase"] = phase_name
    return df


def summarize_by_phase(df):
    labeled = df[df["phase"].notna()]
    if labeled.empty:
        print("No frames matched any labeled phase range -- check PHASE_FRAME_RANGES.")
        return None

    summary = labeled.groupby("phase").agg(
        n_frames=("frame", "count"),
        torso_lean_mean=("torso_lean_angle_deg", "mean"),
        torso_lean_std=("torso_lean_angle_deg", "std"),
        velocity_mean=("hip_velocity", "mean"),
        velocity_std=("hip_velocity", "std"),
        acceleration_mean_abs=("hip_acceleration", lambda x: x.abs().mean()),
        right_knee_mean=("right_knee_filtered", "mean"),
        left_knee_mean=("left_knee_filtered", "mean"),
    ).reset_index()

    return summary


def main():
    print(f"Loading features for {VIDEO_NAME}...\n")
    df = get_skater_features(VIDEO_NAME, VIDEO_PATH, "scaled")
    if df is None:
        print("Could not load features.")
        return
    df = filter_implausible_frames(df)
    df = add_start_phase_features(df)

    total_labeled_ranges = sum(len(r) for r in PHASE_FRAME_RANGES.values())
    print(f"Configured {total_labeled_ranges} labeled frame range(s) across "
          f"{len(PHASE_FRAME_RANGES)} categories.\n")

    empty_categories = [name for name, ranges in PHASE_FRAME_RANGES.items() if not ranges]
    if empty_categories:
        print(f"NOTE: these categories have no frame ranges filled in yet: {empty_categories}")
        print("Edit PHASE_FRAME_RANGES in this script based on your visual review of")
        print("manual_frame_browse/ images, then re-run.\n")

    df = label_frames(df, PHASE_FRAME_RANGES)
    summary = summarize_by_phase(df)

    if summary is None:
        return

    print("=" * 70)
    print("PHASE COMPARISON (within-subject, same video)")
    print("=" * 70)
    print(summary.to_string(index=False))

    phases_present = summary["phase"].tolist()
    if len(phases_present) >= 2:
        print("\n--- Pairwise torso-lean std comparisons ---")
        print("(Lower std = more consistent/stable body position for that phase)")
        for i, phase_a in enumerate(phases_present):
            for phase_b in phases_present[i+1:]:
                std_a = summary[summary["phase"] == phase_a]["torso_lean_std"].values[0]
                std_b = summary[summary["phase"] == phase_b]["torso_lean_std"].values[0]
                print(f"  {phase_a} (std={std_a:.1f}) vs {phase_b} (std={std_b:.1f})")

    output_path = "phase4a_technique_phase_comparison.csv"
    summary.to_csv(output_path, index=False)
    print(f"\nSaved -> {output_path}")


if __name__ == "__main__":
    main()
