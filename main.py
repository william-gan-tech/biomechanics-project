import cv2
import torch
import mediapipe as mp

# 1. Verify PyTorch environment
print(f"PyTorch Version: {torch.__version__}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device.upper()}")

# 2. Initialize Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

print("Pose detector and PyTorch loaded successfully!")