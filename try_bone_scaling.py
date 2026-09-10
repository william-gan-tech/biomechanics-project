"""
Quick standalone test for bone-length scaling.

Run this directly from your project root (no Streamlit needed) to see:
  1. The calibrated reference scale for one or more skater videos
  2. A cross-skater comparison of body-proportion ratios

Usage:
    python try_bone_scaling.py

Edit VIDEO_PATHS below to point at your actual video files.
"""

import os
from pipeline_engine import compute_video_reference_scale, extract_ensemble_reference_scale

# Point these at real files in your data/ folder (see the file list you
# already have -- e.g. data/sven_kramer_ref.mp4, data/patrick_meek_3000m.mp4)
VIDEO_PATHS = {
    "Sven Kramer": "data/sven_kramer_ref.mp4",
    "Patrick Meek": "data/patrick_meek_3000m.mp4",
    # Add more skaters here, e.g.:
    # "Haralds Silovs": "data/silovs.mp4",
}


def main():
    results = {}

    for skater_name, rel_path in VIDEO_PATHS.items():
        if not os.path.exists(rel_path):
            print(f"⚠️  Skipping {skater_name}: file not found at {rel_path}")
            continue

        print(f"\nProcessing {skater_name} ({rel_path}) ...")
        scale = compute_video_reference_scale(rel_path, max_frames=90)
        results[skater_name] = scale
        print(f"  -> Calibrated bone-length reference scale: {scale:.5f}")

    if len(results) >= 2:
        print("\n=== Cross-Skater Scale Comparison ===")
        names = list(results.keys())
        base_name = names[0]
        base_scale = results[base_name]
        for name, scale in results.items():
            ratio = scale / base_scale if base_scale else float("nan")
            print(f"  {name:20s} scale={scale:.5f}   ratio vs {base_name}: {ratio:.3f}")
        print(
            "\nA ratio near 1.0 means similar body proportions relative to the "
            "camera framing; consistently >1 or <1 means one skater is "
            "systematically larger/smaller in frame, which is exactly what "
            "bone-length normalization corrects for downstream."
        )
    elif len(results) == 1:
        print(
            "\nOnly one video processed. Add a second skater to VIDEO_PATHS "
            "above to see a cross-skater scale comparison."
        )
    else:
        print("\nNo videos were found. Check the paths in VIDEO_PATHS.")


if __name__ == "__main__":
    main()
