import torch
from torch.utils.data import Dataset
import numpy as np

def normalize_skeleton_sequence(keypoints, left_hip_idx=1, right_hip_idx=2, left_knee_idx=4, right_knee_idx=5):
    """
    Normalizes a sequence of skeletal keypoints for cross-subject invariance.
    Expects keypoints shape: (num_frames, num_joints, dimensions).
    """
    normalized_sequence = np.copy(keypoints)
    
    for i in range(len(normalized_sequence)):
        frame = normalized_sequence[i]
        
        # 1. Hip-Centric Translation
        mid_hip = (frame[left_hip_idx] + frame[right_hip_idx]) / 2.0
        frame -= mid_hip
        
        # 2. Bone-Length Scaling (Femur-based)
        left_femur = np.linalg.norm(frame[left_knee_idx] - frame[left_hip_idx])
        right_femur = np.linalg.norm(frame[right_knee_idx] - frame[right_hip_idx])
        avg_femur = (left_femur + right_femur) / 2.0
        
        if avg_femur > 0:
            frame /= avg_femur
            
        normalized_sequence[i] = frame
        
    return normalized_sequence

class SpeedSkatingDataset(Dataset):
    def __init__(self, data_array, labels, window_size=50, normalize=True):
        """
        Args:
            data_array (np.ndarray): Shape (num_frames, num_joints, 3) for spatial coordinates.
            labels (np.ndarray): Shape (num_frames,) with binary warning labels.
            window_size (int): Number of consecutive time frames per input sample.
            normalize (bool): Flag to toggle bone-length spatial normalization.
        """
        # Apply spatial normalization before tensor conversion if input is 3D keypoints
        if normalize and data_array.ndim == 3:
            data_array = normalize_skeleton_sequence(data_array)
            
        self.data = torch.tensor(data_array, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        # Extract temporal window of joint trajectories
        x_window = self.data[idx : idx + self.window_size]
        # Target label corresponding to the end of the window
        y_target = self.labels[idx + self.window_size - 1]
        
        return x_window, y_target