"""
Phase 4b prerequisite: audits existing footage for GENUINE simultaneous
multi-person frames (two people detected in the SAME frame), as opposed
to the sequential single-person-per-frame pattern most of your existing
clips have. This tells us whether Phase 4b's tracking test can start
immediately on footage you already have, or whether new start-line
footage genuinely needs to be sourced first.

Usage:
    python -m audit_footage_for_starts
"""

import os
import cv2
from pipeline_engine import _resolve_pose_model_path
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import mediapipe as mp

# Reuse the same skater video map from the Phase 3 ablation
from run_bone_scaling_ablation import SKATER_VIDEOS

SAMPLE_STRIDE = 5   # check every 5th frame (speed vs thoroughness tradeoff)
MAX_FRAMES_TO_SCAN = 900  # ~30 seconds at 30fps, per video


def scan_video_for_simultaneous_people(video_path):
    model_path = _resolve_pose_model_path()
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=2,
    )

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    frames_checked = 0
    frames_with_two_people = 0
    first_simultaneous_frame = None

    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened() and frame_idx < MAX_FRAMES_TO_SCAN:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % SAMPLE_STRIDE == 0:
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                candidates = [p for p in result.pose_landmarks if len(p) > 28] if result.pose_landmarks else []
                frames_checked += 1
                if len(candidates) >= 2:
                    frames_with_two_people += 1
                    if first_simultaneous_frame is None:
                        first_simultaneous_frame = frame_idx
            frame_idx += 1

    cap.release()
    return frames_checked, frames_with_two_people, first_simultaneous_frame


def main():
    print("=" * 70)
    print("Phase 4b prerequisite: auditing existing footage for genuine")
    print("simultaneous multi-person frames (not just sequential passes)")
    print("=" * 70)

    results = []
    for skater_name, rel_path in SKATER_VIDEOS.items():
        full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), rel_path)
        if not os.path.exists(full_path):
            print(f"\n{skater_name}: video not found, skipping")
            continue

        print(f"\n{skater_name} ({rel_path})...")
        checked, with_two, first_frame = scan_video_for_simultaneous_people(full_path)
        pct = (with_two / checked * 100) if checked else 0
        print(f"  Checked {checked} sampled frames, {with_two} ({pct:.1f}%) had 2+ people simultaneously")
        if first_frame is not None:
            print(f"  First simultaneous-people frame: {first_frame} (~{first_frame/30:.1f}s in, assuming 30fps)")

        results.append({
            "skater": skater_name,
            "frames_checked": checked,
            "frames_with_two_people": with_two,
            "pct_simultaneous": pct,
            "first_simultaneous_frame": first_frame,
        })

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    usable = [r for r in results if r["pct_simultaneous"] > 5]  # arbitrary "meaningfully present" threshold
    if usable:
        print(f"{len(usable)} skater video(s) have meaningful simultaneous multi-person content:")
        for r in usable:
            print(f"  - {r['skater']}: {r['pct_simultaneous']:.1f}% of sampled frames, "
                  f"starting around frame {r['first_simultaneous_frame']}")
        print("\nPhase 4b tracking test can start on this footage NOW -- no new video needed yet.")
    else:
        print("None of the existing 7 skater videos have meaningful simultaneous multi-person content.")
        print("Phase 4b genuinely needs new start-line footage with 2+ simultaneous skaters before")
        print("the tracking test can be run -- this is a real prerequisite, not something to skip.")


if __name__ == "__main__":
    main()
