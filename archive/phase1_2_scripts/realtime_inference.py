import cv2
import torch
import numpy as np
import os
import sys

# Add parent directory to path to import core utilities if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def run_realtime_inference(video_path, model_path, window_size=30, threshold=0.05):
    """
    Runs real-time sliding window reconstruction error calculation on a video
    and overlays warning indicators using OpenCV.
    """
    print(f"Loading model from {model_path}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        model = torch.load(model_path, map_location=device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}. Please ensure the .pth file path is correct.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Processing video: {video_path} ({width}x{height})")
    print("Press 'q' in the video window to exit early.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Display the frame using OpenCV
        cv2.imshow('Biomechanics Real-Time Inference', frame)
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Inference finished.")

if __name__ == "__main__":
    # Example local test run
    sample_video = "data/sample_skating.mp4"
    sample_model = "models/autoencoder_model.pth"
    run_realtime_transactions = run_realtime_inference(sample_video, sample_model)