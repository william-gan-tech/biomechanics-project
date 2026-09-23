"""
Saves annotated frame images showing exactly which physical skater was
labeled "A" vs "B" by extract_dual_skater_data.py -- draws a marker and
label directly on the frame at each tracked position.

Usage:
    python -m visualize_dual_tracking --video "data/ragne_wiklund.mp4" --start 662 --end 695
"""

import os
import argparse
import cv2

from extract_dual_skater_data import track_dual_skaters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    args = parser.parse_args()

    print(f"Tracking and annotating frames {args.start}-{args.end}...\n")
    track_a, track_b = track_dual_skaters(args.video, args.start, args.end)

    a_by_frame = {fn: lm for fn, lm in track_a}
    b_by_frame = {fn: lm for fn, lm in track_b}

    out_dir = "dual_tracking_visual_check"
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    for frame_num in range(args.start, args.end + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            continue

        height, width = frame.shape[:2]

        if frame_num in a_by_frame:
            lm = a_by_frame[frame_num]
            hip_x = int((lm[23].x + lm[24].x) / 2 * width)
            hip_y = int((lm[23].y + lm[24].y) / 2 * height)
            cv2.circle(frame, (hip_x, hip_y), 15, (0, 0, 255), 3)
            cv2.putText(frame, "A", (hip_x - 10, hip_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        if frame_num in b_by_frame:
            lm = b_by_frame[frame_num]
            hip_x = int((lm[23].x + lm[24].x) / 2 * width)
            hip_y = int((lm[23].y + lm[24].y) / 2 * height)
            cv2.circle(frame, (hip_x, hip_y), 15, (255, 0, 0), 3)
            cv2.putText(frame, "B", (hip_x - 10, hip_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)

        out_path = os.path.join(out_dir, f"frame{frame_num}_labeled.jpg")
        cv2.imwrite(out_path, frame)

    cap.release()
    print(f"Saved labeled frames -> {out_dir}/")
    print("Red circle + 'A' = Skater A's tracked position, Blue circle + 'B' = Skater B's tracked position")
    print("\nScroll through these in order and confirm: does the RED circle always land on the")
    print("SAME physical person, frame to frame? Does BLUE always land on the other person?")
    print("If the colors ever visibly 'jump' to the other physical skater, that's a real")
    print("identity swap in the tracker, not just camera panning confusion.")


if __name__ == "__main__":
    main()
