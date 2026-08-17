import torch
from torch.utils.data import Dataset

class SpeedSkatingDataset(Dataset):
    def __init__(self, data_array, labels, window_size=50):
        """
        Args:
            data_array (np.ndarray): Shape (num_frames, num_features) containing 
                                     smoothed angles, velocities, and accelerations.
            labels (np.ndarray): Shape (num_frames,) with binary warning labels.
            window_size (int): Number of consecutive time frames per input sample.
        """
        self.data = torch.tensor(data_array, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        # Extract a temporal window of joint trajectories
        x_window = self.data[idx : idx + self.window_size]
        # Target label corresponding to the end of the window (or anticipation window)
        y_target = self.labels[idx + self.window_size - 1]
        
        return x_window, y_target