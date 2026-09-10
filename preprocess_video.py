import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils.biomechanics_utils import calculate_angle, butter_lowpass_filter
import os


def process_skating_video_multivariate(video_path, fps=30.0, reference_scale=None):
    """Extracts per-frame joint kinematics from a skating video.

    If `reference_scale` is provided (a single calibrated bone-length value,
    e.g. from extract_ensemble_reference_scale), every frame is normalized
    against that FIXED value. This makes normalized coordinates comparable
    across frames of the same video AND across different skaters/videos,
    since the scale doesn't wobble frame-to-frame with occlusion or the
    skater's momentary pose.

    If `reference_scale` is None, falls back to the old behavior: each
    frame is normalized by its own torso length (self-normalizing, but not
    comparable across skaters or robust to per-frame occlusion noise).
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    model_path = os.path.join(project_root, 'pose_landmarker_lite.task')

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    data_records = []
    frame_count = 0
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    if fps_video > 0:
        fps = fps_video

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            h_img, w_img, _ = frame.shape
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

            detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
                landmarks = detection_result.pose_landmarks[0]

                def get_px(idx):
                    pt = landmarks[idx]
                    return np.array([pt.x * w_img, pt.y * h_img], dtype=np.float32)

                l_shoulder = get_px(11)
                r_shoulder = get_px(12)
                l_hip = get_px(23)
                r_hip = get_px(24)
                l_knee = get_px(25)
                r_knee = get_px(26)
                l_ankle = get_px(27)
                r_ankle = get_px(28)

                mid_hip = (l_hip + r_hip) / 2.0
                mid_shoulder = (l_shoulder + r_shoulder) / 2.0
                frame_torso_length = np.linalg.norm(mid_shoulder - mid_hip) + 1e-6

                if frame_torso_length < 10.0:  # Filter out anomalous frames
                    continue

                # Use the fixed calibrated scale when available, otherwise
                # fall back to this frame's own torso length.
                # NOTE: reference_scale from extract_ensemble_reference_scale()
                # is computed in NORMALIZED (0-1) landmark space, while
                # frame_torso_length here is in PIXEL space. Convert the
                # calibrated scale to pixels using image height so the two
                # are on the same footing.
                if reference_scale is not None and reference_scale > 0:
                    scale = reference_scale * h_img
                else:
                    scale = frame_torso_length

                r_hip_norm = (r_hip - mid_hip) / scale
                r_shoulder_norm = (r_shoulder - mid_hip) / scale

                r_knee_angle = calculate_angle(r_hip.tolist(), r_knee.tolist(), r_ankle.tolist())
                l_knee_angle = calculate_angle(l_hip.tolist(), l_knee.tolist(), l_ankle.tolist())

                data_records.append({
                    "frame": frame_count,
                    "right_knee_filtered": r_knee_angle,
                    "left_knee_filtered": l_knee_angle,
                    "norm_right_hip_x": r_hip_norm[0],
                    "norm_right_hip_y": r_hip_norm[1],
                    "norm_right_shoulder_x": r_shoulder_norm[0],
                    "norm_right_shoulder_y": r_shoulder_norm[1],
                    "frame_torso_length_px": frame_torso_length,
                    "scale_used_px": scale,
                })

    cap.release()

    if len(data_records) > 0:
        df = pd.DataFrame(data_records)
        df["right_knee_filtered"] = butter_lowpass_filter(df["right_knee_filtered"].values, cutoff_freq=5.0, sample_rate=fps)
        df["left_knee_filtered"] = butter_lowpass_filter(df["left_knee_filtered"].values, cutoff_freq=5.0, sample_rate=fps)
        return df
    return None


def extract_calibration_landmark_sequence(video_path, max_frames=90, min_detection_confidence=0.5, min_tracking_confidence=0.5):
    """Runs the lightweight (non-task-based) MediaPipe Pose solution over the
    first `max_frames` of a video and returns the raw landmark sequence, for
    feeding into extract_ensemble_reference_scale(). Kept separate from the
    main VIDEO-mode PoseLandmarker task above since calibration only needs a
    quick standing-frame sample, not the full task pipeline.
    """
    mp_pose = mp.solutions.pose
    landmark_sequence = []
    cap = cv2.VideoCapture(video_path)
    frames_read = 0
    with mp_pose.Pose(min_detection_confidence=min_detection_confidence,
                       min_tracking_confidence=min_tracking_confidence) as pose:
        while cap.isOpened() and frames_read < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            landmark_sequence.append(results.pose_landmarks.landmark if results.pose_landmarks else None)
            frames_read += 1
    cap.release()
    return landmark_sequence


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    source_video = "data/skater_time_trial.mp4"

    print("Processing full video for normalized multivariate features...")
    df_full = process_skating_video_multivariate(source_video)

    if df_full is not None:
        df_full.to_csv("data/extracted_multivariate_angles.csv", index=False)
        print(f"✅ Success! Saved {len(df_full)} normalized rows -> data/extracted_multivariate_angles.csv")
    else:
        print("⚠️ Warning: Could not process video. Check if skater_time_trial.mp4 exists in data/.")
