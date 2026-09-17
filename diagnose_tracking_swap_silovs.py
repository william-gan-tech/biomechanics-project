"""
Diagnostic: traces the person-tracking decision on every frame of a video,
so we can see EXACTLY when and why a skater-swap happens, instead of
guessing at the cause.

Usage:
    python -m diagnose_tracking_swap
"""

import cv2
from pipeline_engine import (
    _resolve_pose_model_path, _hip_centroid, _landmark_bbox_area,
    _appearance_histogram,
)
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import mediapipe as mp

VIDEO_PATH = "data/silovs.mp4"
MAX_FRAMES = 999999  # adjust if the swap happens later in the clip


def main():
    model_path = _resolve_pose_model_path()
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=2,
    )

    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_idx = 0
    previous_centroid = None
    target_histogram = None

    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened() and frame_idx < MAX_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            candidates = [p for p in result.pose_landmarks if len(p) > 28] if result.pose_landmarks else []

            print(f"\nFrame {frame_idx}: {len(candidates)} candidate(s) detected")

            if not candidates:
                print("  -> NO candidates at all this frame")
                previous_centroid = None
                target_histogram = None
                frame_idx += 1
                continue

            for i, c in enumerate(candidates):
                cx, cy = _hip_centroid(c)
                area = _landmark_bbox_area(c)
                dist_str = ""
                if previous_centroid is not None:
                    dist = ((cx - previous_centroid[0]) ** 2 + (cy - previous_centroid[1]) ** 2) ** 0.5
                    dist_str = f" dist_from_prev={dist:.4f}"
                print(f"  Candidate {i}: hip_centroid=({cx:.3f}, {cy:.3f}) bbox_area={area:.4f}{dist_str}")

            if previous_centroid is None:
                selected = max(candidates, key=_landmark_bbox_area)
                cx, cy = _hip_centroid(selected)
                print(f"  -> No previous track. ANCHORING to largest candidate at ({cx:.3f}, {cy:.3f})")
                previous_centroid = (cx, cy)
                target_histogram = _appearance_histogram(frame, selected)
            else:
                scored = []
                for c in candidates:
                    cx, cy = _hip_centroid(c)
                    dist = ((cx - previous_centroid[0]) ** 2 + (cy - previous_centroid[1]) ** 2) ** 0.5
                    scored.append((dist, c))
                scored.sort(key=lambda t: t[0])
                best_dist, best_c = scored[0]

                if best_dist > 0.15:
                    print(f"  -> best_dist={best_dist:.4f} EXCEEDS max_jump=0.15 -> LOST TRACK (correctly rejected)")
                    previous_centroid = None
                    target_histogram = None
                else:
                    ambiguous = len(scored) > 1 and (scored[1][0] - best_dist) < 0.06
                    print(f"  -> best_dist={best_dist:.4f} (within max_jump). Ambiguous tiebreak triggered: {ambiguous}")
                    if ambiguous and target_histogram is not None:
                        for dist, c in scored:
                            if dist > 0.15:
                                continue
                            hist = _appearance_histogram(frame, c)
                            if hist is not None:
                                sim = cv2.compareHist(target_histogram, hist, cv2.HISTCMP_CORREL)
                                print(f"     appearance similarity for candidate at dist={dist:.4f}: {sim:.4f}")
                    cx, cy = _hip_centroid(best_c)
                    print(f"  -> SELECTED candidate at ({cx:.3f}, {cy:.3f})")
                    previous_centroid = (cx, cy)
                    new_hist = _appearance_histogram(frame, best_c)
                    if new_hist is not None:
                        target_histogram = new_hist

            frame_idx += 1

    cap.release()
    print("\nDone. Look for the frame where 'dist_from_prev' for the WRONG candidate was small")
    print("enough to pass max_jump, or where only one (wrong) candidate was detected at all.")


if __name__ == "__main__":
    main()
