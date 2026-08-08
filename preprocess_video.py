import cv2
import numpy as np
import mediapipe as mp
from biomechanics_utils import calculate_angle, butter_lowpass_filter

def process_skating_video_with_mediapipe(video_path, fps=30.0):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return None

    knee_angles_stream = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Convert the BGR frame to RGB for MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Extract standard MediaPipe indices for the right leg:
            # Hip: 24, Knee: 26, Ankle: 28 (Left side would be 23, 25, 27)
            h = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            k = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
            a = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            
            # Convert normalized coordinates to pixel values
            h_coords = [h.x * frame.shape[1], h.y * frame.shape[0]]
            k_coords = [k.x * frame.shape[1], k.y * frame.shape[0]]
            a_coords = [a.x * frame.shape[1], a.y * frame.shape[0]]
            
            # Calculate angle and append
            angle = calculate_angle(h_coords, k_coords, a_coords)
            knee_angles_stream.append(angle)

    cap.release()
    pose.close()
    
    # Apply Butterworth low-pass filter
    if len(knee_angles_stream) > 0:
        smoothed_angles = butter_lowpass_filter(np.array(knee_angles_stream), cutoff_freq=5.0, sample_rate=fps)
        return smoothed_angles
    return None

if __name__ == "__main__":
    video_file = "sample_skating.mp4" # Make sure your video name matches
    smoothed_angles = process_skating_video_with_mediapipe(video_file)
    
    if smoothed_angles is not None:
        print(f"Success! Extracted and filtered {len(smoothed_angles)} frames of knee angles.")
        print(f"Sample data preview: {smoothed_angles[:5]}")