import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

def normalize_skeleton_sequence(keypoints, left_hip_idx=1, right_hip_idx=2, left_knee_idx=4, right_knee_idx=5, left_shoulder_idx=11, right_shoulder_idx=12):
    """
    Cross-subject skeleton normalization fallback for 3D/keypoint array inputs.
    """
    normalized_sequence = np.copy(keypoints)
    
    for i in range(len(normalized_sequence)):
        frame = normalized_sequence[i]
        mid_hip = (frame[left_hip_idx] + frame[right_hip_idx]) / 2.0
        frame -= mid_hip
        
        left_femur = np.linalg.norm(frame[left_knee_idx] - frame[left_hip_idx])
        right_femur = np.linalg.norm(frame[right_knee_idx] - frame[right_hip_idx])
        avg_femur = (left_femur + right_femur) / 2.0
        
        if avg_femur < 1e-5:
            mid_shoulder = (frame[left_shoulder_idx] + frame[right_shoulder_idx]) / 2.0
            avg_femur = np.linalg.norm(mid_shoulder - mid_hip)
            
        if avg_femur > 0:
            frame /= avg_femur
            
        normalized_sequence[i] = frame
        
    return normalized_sequence

class SpeedSkatingDataset(Dataset):
    def __init__(self, data_array, labels, window_size=50, normalize=True):
        if normalize:
            if data_array.ndim == 3:
                data_array = normalize_skeleton_sequence(data_array)
            else:
                scaler = StandardScaler()
                data_array = scaler.fit_transform(data_array)
            
        self.data = torch.tensor(data_array, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.window_size = window_size

    def __len__(self):
        return max(0, len(self.data) - self.window_size)

    def __getitem__(self, idx):
        x_window = self.data[idx : idx + self.window_size]
        y_target = self.labels[idx + self.window_size - 1]
        return x_window, y_target