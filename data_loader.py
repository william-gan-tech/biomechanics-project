import numpy as np
import torch
from torch.utils.data import Dataset

class SpeedSkatingDataset(Dataset):
    def __init__(self, data_path):
        # Load your joint-angle trajectory data here
        # e.g., data shape: (num_samples, time_steps, num_features)
        self.data = np.load(data_path) 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return a specific window of joint angles and its label
        sample = self.data[idx]
        return torch.tensor(sample, dtype=torch.float32)

if __name__ == "__main__":
    print("Data loader script initialized successfully.")