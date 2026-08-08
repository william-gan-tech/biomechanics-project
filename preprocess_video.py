import cv2
import numpy as np
from biomechanics_utils import calculate_angle, butter_lowpass_filter

def process_skating_video(video_path, fps=30.0):
    """
    Reads a speed skating video, extracts joint coordinates,
    calculates angles, and applies a low-pass filter.
    """
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return None

    frame_count = 0
    knee_angles_stream = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # --- FUTURE STEP: MediaPipe pose estimation goes here ---
        # For now, we simulate extracting hip, knee, and ankle coordinates from the frame
        # e.g., hip = [x, y], knee = [x, y], ankle = [x, y]
        
        # Simulated dummy coordinates for testing the pipeline flow
        hip = [100, 50 + (frame_count % 10)]
        knee = [100, 150]
        ankle = [50, 200]
        
        # 1. Calculate the knee angle using your utility function
        angle = calculate_angle(hip, knee, ankle)
        knee_angles_stream.append(angle)

    cap.release()
    print(f"Processed {frame_count} frames successfully.")
    
    # 2. Apply your Butterworth low-pass filter to the entire time series
    smoothed_angles = butter_lowpass_filter(
        np.array(knee_angles_stream), 
        cutoff_freq=5.0, 
        sample_rate=fps
    )
    
    return smoothed_angles

if __name__ == "__main__":
    # Test with a placeholder path (replace with your actual video file later)
    # processed_data = process_skating_video("sample_skating.mp4")
    print("Preprocessing script template ready.")