"""
Dual-skater extraction: for a video segment with TWO genuinely simultaneous
skaters (common in Olympic time-trial footage), tracks BOTH identities
consistently across the range and extracts real features for each,
instead of discarding the second skater as tracking noise.

Uses the same appearance-histogram approach as pipeline_engine.py's
single-target tracker, but maintains two running identity anchors
instead of one.

Usage:
    python -m extract_dual_skater_data --video "data/ragne_wiklund.mp4" --start 662 --end 695 --name-a "Ragne_outer" --name-b "Ragne_inner"
"""

import os
import argparse
import cv2
import numpy as np
import pandas as pd

from pipeline_engine import _resolve_pose_model_path, _hip_centroid, _appearance_histogram
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import mediapipe as mp

DUPLICATE_IOU_THRESHOLD = 0.5  # candidates with bbox IoU above this are
                                 # treated as ONE person (duplicate
                                 # detection), not two -- confirmed as a
                                 # real, if partial, cause of A/B mislabeling
                                 # via diagnose_dual_detection_overlap.py


def _landmark_bbox(landmarks):
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    return min(xs), min(ys), max(xs), max(ys)


def _bbox_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def _filter_duplicate_candidates(candidates):
    """If exactly 2 candidates are given and their bounding boxes overlap
    heavily, treat them as a duplicate detection of ONE person -- keep
    only one (arbitrarily the first) instead of passing both through to
    A/B assignment, which would otherwise force them into two different
    identity slots despite being the same physical person."""
    if len(candidates) != 2:
        return candidates
    box1 = _landmark_bbox(candidates[0])
    box2 = _landmark_bbox(candidates[1])
    iou = _bbox_iou(box1, box2)
    if iou > DUPLICATE_IOU_THRESHOLD:
        return [candidates[0]]
    return candidates


HISTORY_LENGTH = 6  # rolling window for appearance averaging -- matches the
                     # multi-frame averaging technique proven reliable
                     # earlier today (0.85-0.97 similarity vs noisy single-frame)


class TrackAnchor:
    """Maintains a rolling history of positions (for velocity-based
    prediction, robust to camera panning) and appearance histograms (for
    rolling-average comparison, robust to single-frame noise/motion blur)."""

    def __init__(self):
        self.positions = []
        self.histograms = []

    def predicted_position(self):
        """Linear extrapolation from the last 2 known positions instead of
        the static last position. Tracks WITH camera panning: if both
        skaters shift together frame-to-frame due to a pan, extrapolating
        each identity's own recent motion trend stays valid, whereas
        comparing to a stale static anchor lags behind the pan."""
        if len(self.positions) == 0:
            return None
        if len(self.positions) == 1:
            return self.positions[-1]
        (x0, y0), (x1, y1) = self.positions[-2], self.positions[-1]
        return (x1 + (x1 - x0), y1 + (y1 - y0))

    def averaged_histogram(self):
        if not self.histograms:
            return None
        return np.mean(self.histograms, axis=0).astype(np.float32)

    def update(self, position, histogram):
        self.positions.append(position)
        self.positions = self.positions[-HISTORY_LENGTH:]
        if histogram is not None:
            self.histograms.append(histogram)
            self.histograms = self.histograms[-HISTORY_LENGTH:]


def track_dual_skaters(video_path, start_frame, end_frame):
    model_path = _resolve_pose_model_path()
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=2,
    )

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    track_a = []
    track_b = []
    anchor_a = TrackAnchor()
    anchor_b = TrackAnchor()
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
            candidates = _filter_duplicate_candidates(candidates)

            if len(candidates) == 0:
                frame_num += 1
                continue

            if not anchor_a.positions and not anchor_b.positions:
                c0 = candidates[0]
                anchor_a.update(_hip_centroid(c0), _appearance_histogram(frame, c0))
                track_a.append((frame_num, c0))
                if len(candidates) > 1:
                    c1 = candidates[1]
                    anchor_b.update(_hip_centroid(c1), _appearance_histogram(frame, c1))
                    track_b.append((frame_num, c1))
            else:
                assigned = assign_candidates_to_anchors(candidates, frame, anchor_a, anchor_b)
                if assigned.get("a") is not None:
                    c = assigned["a"]
                    anchor_a.update(_hip_centroid(c), _appearance_histogram(frame, c))
                    track_a.append((frame_num, c))
                if assigned.get("b") is not None:
                    c = assigned["b"]
                    anchor_b.update(_hip_centroid(c), _appearance_histogram(frame, c))
                    track_b.append((frame_num, c))

            frame_num += 1

    cap.release()
    return track_a, track_b


def assign_candidates_to_anchors(candidates, frame, anchor_a, anchor_b):
    """Combines VELOCITY-PREDICTED position (not static last position) with
    ROLLING-AVERAGED appearance (not single-frame) for assignment."""

    def score(candidate, anchor):
        predicted_pos = anchor.predicted_position()
        if predicted_pos is None:
            return float("inf")
        cx, cy = _hip_centroid(candidate)
        dist = ((cx - predicted_pos[0]) ** 2 + (cy - predicted_pos[1]) ** 2) ** 0.5

        appearance_penalty = 0.0
        avg_hist = anchor.averaged_histogram()
        if avg_hist is not None:
            cand_hist = _appearance_histogram(frame, candidate)
            if cand_hist is not None:
                similarity = cv2.compareHist(avg_hist, cand_hist, cv2.HISTCMP_CORREL)
                appearance_penalty = (1.0 - similarity) * 0.15

        return dist + appearance_penalty

    if len(candidates) == 1:
        c = candidates[0]
        dist_a = score(c, anchor_a)
        dist_b = score(c, anchor_b)
        if dist_a <= dist_b:
            return {"a": c, "b": None}
        else:
            return {"a": None, "b": c}

    c0, c1 = candidates[0], candidates[1]
    pairing_1_cost = score(c0, anchor_a) + score(c1, anchor_b)
    pairing_2_cost = score(c1, anchor_a) + score(c0, anchor_b)

    if pairing_1_cost <= pairing_2_cost:
        return {"a": c0, "b": c1}
    else:
        return {"a": c1, "b": c0}


def landmarks_to_feature_row(frame_num, landmarks):
    sh_mid_x = (landmarks[11].x + landmarks[12].x) / 2.0
    hip_mid_x = (landmarks[23].x + landmarks[24].x) / 2.0
    hip_mid_y = (landmarks[23].y + landmarks[24].y) / 2.0
    sh_mid_y = (landmarks[11].y + landmarks[12].y) / 2.0

    return {
        "frame": frame_num,
        "norm_right_hip_x": landmarks[24].x - hip_mid_x,
        "norm_right_hip_y": landmarks[24].y - hip_mid_y,
        "norm_right_shoulder_x": landmarks[12].x - hip_mid_x,
        "norm_right_shoulder_y": landmarks[12].y - hip_mid_y,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--name-a", required=True)
    parser.add_argument("--name-b", required=True)
    args = parser.parse_args()

    print(f"Tracking both skaters in {args.video}, frames {args.start}-{args.end}...\n")
    track_a, track_b = track_dual_skaters(args.video, args.start, args.end)

    print(f"Skater A ('{args.name_a}'): {len(track_a)} frames tracked")
    print(f"Skater B ('{args.name_b}'): {len(track_b)} frames tracked\n")

    if len(track_a) < 5 or len(track_b) < 5:
        print("WARNING: one or both tracks have very few frames -- dual tracking may")
        print("not be reliable for this range.")

    os.makedirs("dual_skater_extracts", exist_ok=True)

    for label, track in [(args.name_a, track_a), (args.name_b, track_b)]:
        rows = [landmarks_to_feature_row(fn, lm) for fn, lm in track]
        df = pd.DataFrame(rows)
        safe_label = "".join(c if c.isalnum() else "_" for c in label)
        out_path = f"dual_skater_extracts/{safe_label}_{args.start}_{args.end}.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved {label} -> {out_path} ({len(df)} rows)")

    print("\nNEXT STEP: visually verify each track is actually consistent (same person")
    print("throughout) before treating either as confirmed data.")


if __name__ == "__main__":
    main()
