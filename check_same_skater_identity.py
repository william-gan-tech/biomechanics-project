"""
Resolves the torso-lean sign-flip found in the 9/18 start-vs-acceleration
comparison: is it a real body-mechanics signal, or did the tracker lock
onto a DIFFERENT skater between the REST and EARLY-ACCELERATION segments?

Uses the same appearance-histogram comparison technique built for the
9/15 multi-person tracking fix.

Usage:
    python -m check_same_skater_identity
"""

import cv2
import numpy as np
from pipeline_engine import _resolve_pose_model_path, _appearance_histogram
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import mediapipe as mp

VIDEO_PATH = "data/start_candidate_3.mp4"
REST_RANGE = (155, 169)    # full confirmed rest segment
ACCEL_RANGE = (193, 245)   # full confirmed early-acceleration segment
SAMPLE_STRIDE = 2          # check every 2nd frame within each range (speed vs thoroughness)


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

    print(f"Averaging appearance across the FULL rest segment {REST_RANGE} "
          f"and the FULL early-acceleration segment {ACCEL_RANGE}...\n")

    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        rest_hist, rest_n = average_histogram_over_range(VIDEO_PATH, REST_RANGE, SAMPLE_STRIDE, landmarker)
        accel_hist, accel_n = average_histogram_over_range(VIDEO_PATH, ACCEL_RANGE, SAMPLE_STRIDE, landmarker)

    print(f"Rest segment: averaged over {rest_n} frames")
    print(f"Accel segment: averaged over {accel_n} frames\n")

    if rest_hist is None or accel_hist is None:
        print("Could not build averaged histogram for one or both segments -- can't compare.")
        return

    similarity = cv2.compareHist(rest_hist, accel_hist, cv2.HISTCMP_CORREL)

    print(f"Appearance similarity (correlation), AVERAGED across full segments: {similarity:.4f}")
    print("(1.0 = identical appearance, 0.0 = no correlation, negative = anti-correlated)\n")

    if similarity > 0.7:
        print("HIGH similarity -- likely the SAME skater across both segments.")
        print("The torso-lean sign flip is probably a REAL body-mechanics signal.")
    elif similarity > 0.4:
        print("MODERATE similarity -- still somewhat inconclusive, but averaging over")
        print(f"{rest_n} and {accel_n} frames respectively is far more reliable than a single")
        print("frame comparison. If this is still ambiguous, the honest conclusion is that")
        print("appearance-based identity checking has reached its limit for this footage,")
        print("and position/camera-framing evidence should be weighed instead.")
    else:
        print("LOW similarity -- likely a DIFFERENT skater was tracked in each segment.")
        print("The torso-lean sign flip should be treated as a tracking identity swap,")
        print("not a real finding, until fixed.")


if __name__ == "__main__":
    main()
