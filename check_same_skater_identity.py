"""
Checks whether the tracked skater stays the SAME throughout a single,
continuous frame range, by comparing an EARLY portion against a LATE
portion. If it's genuinely one consistently-tracked skater, these should
be similar in appearance; if the tracker swapped identity partway
through, they won't be.

FIXED 9/24: the original version reopened a new VideoCapture and seeked
directly to each isolated target frame on every call. MediaPipe's VIDEO
mode `detect_for_video()` relies on temporal/motion continuity between
calls -- feeding it disconnected, randomly-jumped-to frames (rather than
a real sequential stream) caused it to sometimes detect ZERO candidates
even on frames independently confirmed to have a real, visible skater
(confirmed directly: frame 1065 returned 0 candidates when queried in
isolation here, but 1 candidate when reached as part of a normal
sequential scan). This version reads sequentially through ONE continuous
pass, starting well before the target ranges to let the tracker warm up,
and only collects histograms for frames that fall within EARLY_RANGE or
LATE_RANGE as they're naturally reached.

Usage:
    python -m check_same_skater_identity
"""

import cv2
import numpy as np
from pipeline_engine import _resolve_pose_model_path, _appearance_histogram
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import mediapipe as mp

VIDEO_PATH = "data/ragne_wiklund.mp4"
EARLY_RANGE = (1065, 1070)
LATE_RANGE = (1075, 1080)
SAMPLE_STRIDE = 1  # sequential read -- check every frame, not every Nth,
                    # since we're only paying the cost of one continuous
                    # pass instead of many isolated re-opens
WARMUP_FRAMES = 30  # start reading this many frames before EARLY_RANGE,
                     # to give the tracker real temporal context before
                     # we start trusting its detections


def collect_histograms_sequentially(video_path, ranges_to_collect):
    """Reads the video ONCE, sequentially, from just before the earliest
    requested range through the end of the latest one. Returns a dict
    mapping each range (as a tuple key) to a list of histograms collected
    for frames that fell within it."""
    model_path = _resolve_pose_model_path()
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=2,
    )

    start_frame = max(0, min(r[0] for r in ranges_to_collect) - WARMUP_FRAMES)
    end_frame = max(r[1] for r in ranges_to_collect)

    results = {r: [] for r in ranges_to_collect}

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_num = start_frame

    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened() and frame_num <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            candidates = [p for p in result.pose_landmarks if len(p) > 28] if result.pose_landmarks else []

            for r in ranges_to_collect:
                if r[0] <= frame_num <= r[1] and candidates:
                    hist = _appearance_histogram(frame, candidates[0])
                    if hist is not None:
                        results[r].append(hist)

            frame_num += 1

    cap.release()
    return results


def main():
    print(f"Checking identity consistency (sequential read, warmed up {WARMUP_FRAMES} frames early).")
    print(f"Comparing EARLY portion {EARLY_RANGE} vs LATE portion {LATE_RANGE}...\n")

    collected = collect_histograms_sequentially(VIDEO_PATH, [EARLY_RANGE, LATE_RANGE])
    early_hists = collected[EARLY_RANGE]
    late_hists = collected[LATE_RANGE]

    print(f"Early portion: collected {len(early_hists)} frame(s)")
    print(f"Late portion: collected {len(late_hists)} frame(s)\n")

    if not early_hists or not late_hists:
        print("Could not build a histogram for one or both portions -- can't compare.")
        print("If this still happens after the sequential-read fix, the frames genuinely")
        print("have no detectable person, not a MediaPipe context issue.")
        return

    early_hist = np.mean(early_hists, axis=0).astype(np.float32)
    late_hist = np.mean(late_hists, axis=0).astype(np.float32)

    similarity = cv2.compareHist(early_hist, late_hist, cv2.HISTCMP_CORREL)

    print(f"Appearance similarity (correlation), early vs late: {similarity:.4f}")
    print("(1.0 = identical appearance, 0.0 = no correlation, negative = anti-correlated)\n")

    if similarity > 0.7:
        print("HIGH similarity -- likely the SAME skater throughout. Safe to treat this as")
        print("one continuous, clean segment.")
    elif similarity > 0.4:
        print("MODERATE similarity -- inconclusive. Recommend NOT using this full range")
        print("as a single labeled segment; consider narrowing further or discarding it.")
    else:
        print("LOW similarity -- the tracker likely tracked DIFFERENT people in the early")
        print("vs late portions. Do NOT log this range as a single clean segment.")


if __name__ == "__main__":
    main()