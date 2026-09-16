import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Define paths dynamically
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)
video_path = os.path.join(base_dir, 'outputs', 'meek_segments', 'fatigued_state.mp4')
output_csv_path = os.path.join(base_dir, 'data', 'subject_meek_fatigued.csv')
# Path to your downloaded MediaPipe pose landmarker model file
# (Make sure pose_landmarker_full.task is in your project directory or data folder)
model_path = os.path.join(base_dir, 'data', 'pose_landmarker_full.task')

def calculate_angle(a, b, c):
    """Calculates 3D joint angle given three points."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360.0 - angle
    return angle

print(f"Processing reference video: {video_path}")

# Initialize PoseLandmarker via MediaPipe Tasks API
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE
)

frame_data = []
cap = cv2.VideoCapture(video_path)
frame_idx = 0

with vision.PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # OpenCV reads BGR, MediaPipe expects RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # Detect pose landmarks
        detection_result = landmarker.detect(mp_image)
        
        if detection_result.pose_landmarks:
            landmarks = detection_result.pose_landmarks[0]
            
            try:
                # MediaPipe landmark indices:
                # 24: right hip, 26: right knee, 28: right ankle
                r_hip = [landmarks[24].x, landmarks[24].y]
                r_knee = [landmarks[26].x, landmarks[26].y]
                r_ankle = [landmarks[28].x, landmarks[28].y]
                
                r_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
                
                frame_data.append({
                    'frame': frame_idx,
                    'right_knee_angle': r_knee_angle
                })
            except Exception as e:
                pass
                
        frame_idx += 1

cap.release()

# Save the extracted baseline angles to a CSV file
df = pd.DataFrame(frame_data)
df.to_csv(output_csv_path, index=False)

print(f"Success! Elite kinematic baseline saved to: {output_csv_path}")
print(f"Total valid frames recorded: {len(df)}")