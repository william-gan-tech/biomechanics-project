"""
Diagnoses why compute_video_reference_scale() might be returning the
fallback 0.45 instead of a real calibrated value.

Usage:
    python -m diagnose_bone_scaling
"""

import os
import cv2

VIDEO_PATH = "data/sven_kramer_ref.mp4"  # change if needed

def main():
    print(f"Checking: {VIDEO_PATH}")
    print(f"File exists: {os.path.exists(VIDEO_PATH)}")

    cap = cv2.VideoCapture(VIDEO_PATH)
    print(f"cv2.VideoCapture opened successfully: {cap.isOpened()}")

    if not cap.isOpened():
        print("PROBLEM FOUND: OpenCV could not open this video file at all.")
        print("This usually means a codec issue or a corrupted/incomplete download.")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Frame count: {frame_count}, Resolution: {width}x{height}, FPS: {fps}")

    try:
        import mediapipe as mp
    except ImportError:
        print("PROBLEM FOUND: mediapipe is not installed in this environment.")
        cap.release()
        return

    print(f"mediapipe version: {mp.__version__}")

    mp_pose = mp.solutions.pose
    detected_count = 0
    total_checked = 0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened() and total_checked < 30:
            ret, frame = cap.read()
            if not ret:
                print(f"cap.read() returned False at frame {total_checked}")
                break
            total_checked += 1
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            if results.pose_landmarks:
                detected_count += 1

    cap.release()

    print(f"\nFrames checked: {total_checked}")
    print(f"Frames with detected pose landmarks: {detected_count}")

    if total_checked == 0:
        print("PROBLEM FOUND: Could not read any frames from the video (cap.read() failed immediately).")
    elif detected_count == 0:
        print("PROBLEM FOUND: MediaPipe never detected a person/pose in any sampled frame.")
        print("Possible causes: skater too small/far in frame, poor lighting, unusual camera angle,")
        print("or the first ~30 frames are pre-roll/black frames/intro before the skater appears.")
    else:
        print(f"MediaPipe IS detecting poses ({detected_count}/{total_checked} frames) -- ")
        print("the scale computation itself should be working. If try_bone_scaling.py still")
        print("shows 0.45, the issue may be in the visibility-threshold filtering inside")
        print("extract_ensemble_reference_scale (landmarks detected but below 0.5 visibility).")


if __name__ == "__main__":
    main()