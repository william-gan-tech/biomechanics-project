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

import numpy as np
from biomechanics_utils import create_sliding_windows

# Load your extracted angle data from the CSV
knee_angles = np.loadtxt("extracted_knee_angles.csv", delimiter=",", skiprows=1)

# Apply sliding window segmentation (30 frames per window, stepping by 5)
X_train_windows = create_sliding_windows(knee_angles, window_size=30, step_size=5)

print(f"Successfully generated training windows!")
print(f"Total windows: {X_train_windows.shape[0]}")
print(f"Window shape (frames per chunk): {X_train_windows.shape[1]}")