import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils.biomechanics_utils import calculate_angle, butter_lowpass_filter
import os

def process_skating_video_multivariate(video_path, fps=30.0):
    # Get the absolute path of the 'src' directory, then go up one level to 'biomechanics-project'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    model_path = os.path.join(project_root, 'pose_landmarker_lite.task')
    
    # Setup MediaPipe Pose Landmarker using the correct absolute path
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO
    )
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}.")
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
            
            # Convert OpenCV frame to MediaPipe Image format
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            
            # Calculate timestamp in milliseconds for the frame
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            
            # Detect pose landmarks
            detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            if detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
                landmarks = detection_result.pose_landmarks[0]
                
                # Helper function to get pixel coordinates [x, y]
                def get_px(idx):
                    pt = landmarks[idx]
                    return [pt.x * w_img, pt.y * h_img]
                
                # MediaPipe indices mapping
                l_shoulder = get_px(11)
                r_shoulder = get_px(12)
                l_elbow = get_px(13)
                r_elbow = get_px(14)
                l_hip = get_px(23)
                r_hip = get_px(24)
                l_knee = get_px(25)
                r_knee = get_px(26)
                l_ankle = get_px(27)
                r_ankle = get_px(28)
                
                # Calculate angles for joints
                r_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
                l_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
                
                data_records.append({
                    "frame": frame_count,
                    "right_knee_angle": r_knee_angle,
                    "left_knee_angle": l_knee_angle,
                    "right_hip_x": r_hip[0],
                    "right_hip_y": r_hip[1],
                    "right_shoulder_x": r_shoulder[0],
                    "right_shoulder_y": r_shoulder[1]
                })

    cap.release()
    
    if len(data_records) > 0:
        df = pd.DataFrame(data_records)
        
        # Apply Butterworth low-pass filter to angle columns to reduce noise
        df["right_knee_filtered"] = butter_lowpass_filter(df["right_knee_angle"].values, cutoff_freq=5.0, sample_rate=fps)
        df["left_knee_filtered"] = butter_lowpass_filter(df["left_knee_angle"].values, cutoff_freq=5.0, sample_rate=fps)
        
        return df
    return None

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    
    # Point to your 6-minute time trial video
    source_video = "data/skater_time_trial.mp4"
    
    print("Processing full video for time trial analysis...")
    df_full = process_skating_video_multivariate(source_video)
    
    if df_full is not None:
        # 1. Fresh state segment (20s to 50s at 25 fps -> frames 500 to 1250)
        df_fresh = df_full[(df_full['frame'] >= 500) & (df_full['frame'] <= 1250)]
        df_fresh.to_csv("data/angles_20_to_50.csv", index=False)
        print(f"✅ Success! Saved {len(df_fresh)} fresh frames -> data/angles_20_to_50.csv")

        # 2. Fatigued state segment (3:45 to 4:14 at 25 fps -> frames 5625 to 6350)
        df_fatigued = df_full[(df_full['frame'] >= 5625) & (df_full['frame'] <= 6350)]
        df_fatigued.to_csv("data/angles_345_to_414.csv", index=False)
        print(f"✅ Success! Saved {len(df_fatigued)} fatigued frames -> data/angles_345_to_414.csv")
    else:
        print("⚠️ Warning: Could not process skater_time_trial.mp4. Check if the file exists in your data folder.")