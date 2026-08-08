import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from biomechanics_utils import calculate_angle, butter_lowpass_filter

def process_skating_video_with_mediapipe(video_path, fps=30.0):
    # Setup MediaPipe Pose Landmarker using the modern tasks API
    # (Make sure you have a model file or let it use the default configuration)
    base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO
    )
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return None

    knee_angles_stream = []
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
            
            # Convert OpenCV frame to MediaPipe Image format
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            
            # Calculate timestamp in milliseconds for the frame
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            
            # Detect pose landmarks
            detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            if detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
                landmarks = detection_result.pose_landmarks[0]
                
                # MediaPipe Pose Landmarker indices:
                # Right Hip: 24, Right Knee: 26, Right Ankle: 28
                h = landmarks[24]
                k = landmarks[26]
                a = landmarks[28]
                
                h_coords = [h.x * frame.shape[1], h.y * frame.shape[0]]
                k_coords = [k.x * frame.shape[1], k.y * frame.shape[0]]
                a_coords = [a.x * frame.shape[1], a.y * frame.shape[0]]
                
                angle = calculate_angle(h_coords, k_coords, a_coords)
                knee_angles_stream.append(angle)

    cap.release()
    
    if len(knee_angles_stream) > 0:
        smoothed_angles = butter_lowpass_filter(np.array(knee_angles_stream), cutoff_freq=5.0, sample_rate=fps)
        return smoothed_angles
    return None

if __name__ == "__main__":
    video_file = "sample_skating.mp4"
    smoothed_angles = process_skating_video_with_mediapipe(video_file)
    
    if smoothed_angles is not None:
        print(f"Success! Extracted and filtered {len(smoothed_angles)} frames of knee angles.")
        print(f"Sample data preview: {smoothed_angles[:5]}")