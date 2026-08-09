import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from biomechanics_utils import calculate_angle, butter_lowpass_filter

def process_skating_video_multivariate(video_path, fps=30.0):
    # Setup MediaPipe Pose Landmarker using the modern tasks API
    base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO
    )
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
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
                
                # MediaPipe indices mapping:
                # Shoulders: 11 (L), 12 (R)
                # Elbows: 13 (L), 14 (R)
                # Hips: 23 (L), 24 (R)
                # Knees: 25 (L), 26 (R)
                # Ankles: 27 (L), 28 (R)
                
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
                
                # Calculate angles for multiple joints
                r_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
                l_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
                
                # You can add more joints here as you expand your math functions in biomechanics_utils.py
                
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
    video_file = "sample_skating.mp4"
    df_results = process_skating_video_multivariate(video_file)
    
    if df_results is not None:
        # Save multivariate structured data to a new CSV file
        df_results.to_csv("extracted_multivariate_angles.csv", index=False)
        print(f"Success! Extracted and filtered {len(df_results)} frames of multi-joint data.")
        print("Data successfully saved to extracted_multivariate_angles.csv!")