import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class SpeedSkatingKinematicDataset(Dataset):
    """
    Dataset for loading temporal joint-angle trajectories 
    to anticipate fatigue-induced performance degradation.
    """
    def __init__(self, data_array, labels_array):
        # data_array shape: (num_samples, time_steps, num_features)
        # num_features could be your joint angles (hip, knee, ankle, etc.)
        self.data = torch.tensor(data_array, dtype=torch.float32)
        self.labels = torch.tensor(labels_array, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Example test setup
if __name__ == "__main__":
    # Dummy data: 100 skating strides, 50 time steps per window, 6 joint features
    dummy_data = np.random.rand(100, 50, 6) 
    dummy_labels = np.zeros(100) # 0 for stable, 1 for impending degradation
    
    dataset = SpeedSkatingKinematicDataset(dummy_data, dummy_labels)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    print(f"Dataset successfully created with {len(dataset)} samples.")
    for batch_data, batch_labels in loader:
        print(f"Batch data shape: {batch_data.shape}")
        print(f"Batch labels shape: {batch_labels.shape}")
        break