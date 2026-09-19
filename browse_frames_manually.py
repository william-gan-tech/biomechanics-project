"""
Manual frame browser for when automated acceleration/velocity detection
isn't reliable (e.g. chaotic broadcast footage with fast camera pans and
tight multi-person framing, where landmark jitter prevents hip_velocity
from ever reading genuinely near-zero -- confirmed as the case for
start_candidate_1.mp4 on 9/18).

Saves every Nth frame in a time window as an image, and prints the
computed features (torso lean, velocity, acceleration) at each -- for
YOU to pick the real start moment by eye, rather than relying on the
automated filter for this clip specifically.

Usage:
    python -m browse_frames_manually --video data/start_candidate_1.mp4 --start 10 --end 18 --step 3
"""

import os
import argparse
import cv2

from run_bone_scaling_ablation import get_skater_features, filter_implausible_frames
from start_phase_features import add_start_phase_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--start", type=float, required=True, help="Window start, seconds")
    parser.add_argument("--end", type=float, required=True, help="Window end, seconds")
    parser.add_argument("--step", type=int, default=3, help="Save every Nth frame (default 3)")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"{args.video} not found.")
        return

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_frame = int(args.start * fps)
    end_frame = int(args.end * fps)

    video_name = os.path.splitext(os.path.basename(args.video))[0]
    video_rel_path = os.path.relpath(args.video, os.path.dirname(os.path.abspath(__file__)))
    df = get_skater_features(video_name, video_rel_path, "scaled")
    features_by_frame = {}
    if df is not None:
        df = filter_implausible_frames(df)
        df = add_start_phase_features(df)
        features_by_frame = df.set_index("frame").to_dict("index")

    out_dir = "manual_frame_browse"
    os.makedirs(out_dir, exist_ok=True)

    print(f"Saving every {args.step} frame(s) from {args.start}s to {args.end}s "
          f"(frames {start_frame}-{end_frame})...\n")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame
    saved = 0

    while frame_idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if (frame_idx - start_frame) % args.step == 0:
            img_path = os.path.join(out_dir, f"{video_name}_frame{frame_idx}.jpg")
            cv2.imwrite(img_path, frame)
            saved += 1

            feat = features_by_frame.get(frame_idx)
            if feat:
                print(f"Frame {frame_idx} ({frame_idx/fps:.2f}s): "
                      f"torso_lean={feat['torso_lean_angle_deg']:.1f}°, "
                      f"velocity={feat['hip_velocity']:.5f}, "
                      f"accel={feat['hip_acceleration']:.5f}  -> {img_path}")
            else:
                print(f"Frame {frame_idx} ({frame_idx/fps:.2f}s): (no feature data) -> {img_path}")

        frame_idx += 1

    cap.release()
    print(f"\nSaved {saved} frames to {out_dir}/")
    print(f"Open the folder and scroll through to find the true rest-to-explosion moment by eye:")
    print(f"  Invoke-Item .\\{out_dir}\\")


if __name__ == "__main__":
    main()
