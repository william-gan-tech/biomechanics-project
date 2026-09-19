"""
General-purpose start-candidate tool: downloads a video (if not already
present), extracts a check frame, and runs the acceleration-spike
candidate finder in a specific time window you provide -- instead of
needing a new hardcoded script for every new video you find.

Usage:
    python -m test_start_candidate --url "https://youtube.com/watch?v=XXXX" --name candidate_2 --start 45 --end 55

Arguments:
    --url    YouTube URL to download (skipped if the video already exists)
    --name   Short identifier used for filenames (e.g. "candidate_2")
    --start  Search window start, in seconds
    --end    Search window end, in seconds
    --check-sec  Optional: specific second to save a quick visual-check frame for (defaults to the window midpoint)
"""

import os
import argparse
import cv2

from pipeline_engine import download_video_from_url
from find_candidate_start_moments import find_candidates_for_skater, save_candidate_frame_image


def download_and_check(url, name, check_sec):
    output_path = f"data/start_{name}.mp4"
    os.makedirs("data", exist_ok=True)

    if os.path.exists(output_path):
        print(f"{output_path} already exists, skipping download.")
    else:
        print(f"Downloading {url} -> {output_path} ...")
        success, result = download_video_from_url(url, output_path=os.path.abspath(output_path))
        if not success:
            print(f"Download failed: {result}")
            return None
        print("Downloaded successfully.")

    cap = cv2.VideoCapture(output_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video info: {fps} fps, {frame_count} frames, {frame_count/fps:.1f}s duration")

    target_frame = int(check_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    cap.release()

    if ret:
        img_path = f"start_{name}_check.jpg"
        cv2.imwrite(img_path, frame)
        print(f"Saved check frame at ~{check_sec}s -> {img_path}")
    else:
        print(f"Could not read frame at {check_sec}s.")

    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="YouTube URL")
    parser.add_argument("--name", required=True, help="Short identifier for filenames, e.g. candidate_2")
    parser.add_argument("--start", type=float, required=True, help="Search window start, seconds")
    parser.add_argument("--end", type=float, required=True, help="Search window end, seconds")
    parser.add_argument("--check-sec", type=float, default=None, help="Second to save a visual-check frame for (default: window midpoint)")
    args = parser.parse_args()

    check_sec = args.check_sec if args.check_sec is not None else (args.start + args.end) / 2

    video_path = download_and_check(args.url, args.name, check_sec)
    if video_path is None:
        return

    print(f"\nSearching {args.start}-{args.end}s for acceleration-spike candidates...\n")

    padded_start = max(0, args.start - 2)
    padded_end = args.end + 3

    candidates = find_candidates_for_skater(
        f"start_{args.name}", video_path,
        search_start_sec=padded_start, search_end_sec=padded_end
    )

    if not candidates:
        print("\nNo candidates found. The velocity-before-spike filter may have rejected")
        print("everything in this window -- check the console output above for rejected spikes,")
        print("or the true spike may be slightly outside the searched range.")
        return

    os.makedirs("start_moment_candidates", exist_ok=True)
    for c in candidates:
        print(f"\nCandidate: frame {c['frame']}, torso_lean={c['torso_lean_angle_deg']:.1f}°, "
              f"accel={c['hip_acceleration']:.4f}, velocity_before={c['velocity_before']:.5f}")
        img_path = f"start_moment_candidates/start_{args.name}_frame{c['frame']}.jpg"
        if save_candidate_frame_image(video_path, c["frame"], img_path):
            print(f"  Saved -> {img_path}")


if __name__ == "__main__":
    main()
