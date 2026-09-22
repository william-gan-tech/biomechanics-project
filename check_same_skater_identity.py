"""
Checks whether the tracked skater stays the SAME throughout a single,
continuous frame range (863-1049 in start_candidate_3.mp4) where two
skaters are known to be visible -- straightaway for one, corner for the
other. Compares the EARLY portion of the range against the LATE portion:
if it's genuinely one consistently-tracked skater, these should be
similar in appearance; if the tracker swapped identity partway through,
they won't be.

Usage:
    python -m check_same_skater_identity
"""

import cv2
import numpy as np
from pipeline_engine import _resolve_pose_model_path, _appearance_histogram
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import mediapipe as mp

VIDEO_PATH = "data/patrick_meek_3000m.mp4"
EARLY_RANGE = (746, 764)
LATE_RANGE = (767, 790)
SAMPLE_STRIDE = 3


def get_landmarks_and_histogram(video_path, frame_num, landmarker):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return None

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    cap.release()

    candidates = [p for p in result.pose_landmarks if len(p) > 28] if result.pose_landmarks else []
    if not candidates:
        return None

    landmarks = candidates[0]
    return _appearance_histogram(frame, landmarks)


def average_histogram_over_range(video_path, frame_range, stride, landmarker):
    hists = []
    for frame_num in range(frame_range[0], frame_range[1] + 1, stride):
        hist = get_landmarks_and_histogram(video_path, frame_num, landmarker)
        if hist is not None:
            hists.append(hist)

    if not hists:
        return None, 0

    avg_hist = np.mean(hists, axis=0).astype(np.float32)
    return avg_hist, len(hists)


def main():
    model_path = _resolve_pose_model_path()
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=2,
    )

    print(f"Checking identity consistency WITHIN the ambiguous range 863-1049.")
    print(f"Comparing EARLY portion {EARLY_RANGE} vs LATE portion {LATE_RANGE}...\n")

    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        early_hist, early_n = average_histogram_over_range(VIDEO_PATH, EARLY_RANGE, SAMPLE_STRIDE, landmarker)
        late_hist, late_n = average_histogram_over_range(VIDEO_PATH, LATE_RANGE, SAMPLE_STRIDE, landmarker)

    print(f"Early portion: averaged over {early_n} frames")
    print(f"Late portion: averaged over {late_n} frames\n")

    if early_hist is None or late_hist is None:
        print("Could not build averaged histogram for one or both portions -- can't compare.")
        return

    similarity = cv2.compareHist(early_hist, late_hist, cv2.HISTCMP_CORREL)

    print(f"Appearance similarity (correlation), early vs late: {similarity:.4f}")
    print("(1.0 = identical appearance, 0.0 = no correlation, negative = anti-correlated)\n")

    if similarity > 0.7:
        print("HIGH similarity -- the SAME skater was likely tracked throughout the whole")
        print("863-1049 range. Safe to treat this as one continuous, clean segment for")
        print("whichever phase (straightaway/corner) it actually shows.")
    elif similarity > 0.4:
        print("MODERATE similarity -- inconclusive. Recommend NOT using this full range")
        print("as a single labeled segment; consider splitting it further or discarding it.")
    else:
        print("LOW similarity -- the tracker likely SWAPPED between the two skaters")
        print("somewhere in this range. This range should NOT be logged as a single")
        print("clean segment -- it mixes two different skaters' kinematics together.")
        print("Document this as a real labeling limitation rather than forcing a guess.")


if __name__ == "__main__":
    main()