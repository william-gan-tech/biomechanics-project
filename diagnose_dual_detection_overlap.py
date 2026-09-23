"""
Diagnoses the root cause of "A and B sometimes land on the same physical
skater": checks whether the two candidates MediaPipe returns per frame
have significantly OVERLAPPING bounding boxes (suggesting a duplicate
detection of one person) versus genuinely separated boxes.

Usage:
    python -m diagnose_dual_detection_overlap --video "data/ragne_wiklund.mp4" --start 662 --end 695
"""

import argparse
import cv2

from pipeline_engine import _resolve_pose_model_path
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import mediapipe as mp


def landmark_bbox(landmarks):
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    return min(xs), min(ys), max(xs), max(ys)


def bbox_iou(box1, box2):
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

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    args = parser.parse_args()

    model_path = _resolve_pose_model_path()
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=2,
    )

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    frame_num = args.start

    high_overlap_count = 0
    total_two_candidate_frames = 0

    print(f"Checking bounding-box overlap between the 2 candidates at each frame, "
          f"{args.start}-{args.end}...\n")

    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened() and frame_num <= args.end:
            ret, frame = cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            candidates = [p for p in result.pose_landmarks if len(p) > 28] if result.pose_landmarks else []

            if len(candidates) == 2:
                total_two_candidate_frames += 1
                box1 = landmark_bbox(candidates[0])
                box2 = landmark_bbox(candidates[1])
                iou = bbox_iou(box1, box2)
                flag = " <-- HIGH OVERLAP, likely duplicate detection" if iou > 0.5 else ""
                print(f"Frame {frame_num}: IoU={iou:.3f}{flag}")
                if iou > 0.5:
                    high_overlap_count += 1

            frame_num += 1

    cap.release()

    print(f"\n{'='*60}")
    print(f"Frames with 2 candidates: {total_two_candidate_frames}")
    print(f"Frames with high overlap (IoU > 0.5, likely duplicate detection): {high_overlap_count}")
    print(f"{'='*60}")

    if total_two_candidate_frames == 0:
        print("\nNo frames had 2 simultaneous candidates in this range -- can't diagnose.")
    elif high_overlap_count / total_two_candidate_frames > 0.2:
        print("\nSIGNIFICANT duplicate-detection issue found: a meaningful fraction of")
        print("'2 candidate' frames are actually MediaPipe detecting ONE person twice,")
        print("not two real people. This explains 'A and B on the same skater' -- it's")
        print("a detection-layer issue, not a tracker assignment bug.")
    else:
        print("\nLow duplicate-detection rate -- the two candidates are generally genuinely")
        print("separated people. The 'same skater' labeling issue is more likely a real")
        print("tracker assignment failure, not a detection artifact.")


if __name__ == "__main__":
    main()
