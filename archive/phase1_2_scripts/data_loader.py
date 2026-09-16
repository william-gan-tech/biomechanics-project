import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# Import your pre-deceleration labeling function
from label_pre_deceleration import label_pre_deceleration_windows

class SpeedSkatingKinematicDataset(Dataset):
    """
    Dataset for loading temporal joint-angle trajectories 
    to anticipate fatigue-induced performance degradation.
    """
    def __init__(self, data_array, labels_array):
        # data_array shape: (num_samples, time_steps, num_features)
        self.data = torch.tensor(data_array, dtype=torch.float32)
        self.labels = torch.tensor(labels_array, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def create_dataset_from_csv(csv_path, velocity_col='com_velocity', joint_angle_cols=None, window_size=10, drop_threshold=0.03):
    """
    Loads a raw skating CSV file and converts it into PyTorch-ready features and labels 
    using the pre-deceleration labeling function.
    """
    # 1. Load your raw CSV data
    df = pd.read_csv(csv_path)
    
    # Default joint columns if none provided
    if joint_angle_cols is None:
        joint_angle_cols = [col for col in df.columns if col != velocity_col]
        
    # 2. Generate sliding windows and labels using your labeler
    X, y = label_pre_deceleration_windows(
        data=df,
        velocity_col=velocity_col,
        joint_angle_cols=joint_angle_cols,
        window_size=window_size,
        drop_threshold=drop_threshold
    )
    
    # 3. Return initialized PyTorch Dataset
    return SpeedSkatingKinematicDataset(X, y)

# Example usage when running this script directly
if __name__ == "__main__":
    # If you have a real file, replace 'path/to/skater.csv' with your actual path:
    # dataset = create_dataset_from_csv('data/skater_trial_1.csv')
    
    # Fallback to dummy data test setup if no file path is ready yet
    dummy_data = np.random.rand(100, 10, 2) 
    dummy_labels = np.zeros(100)
    
    dataset = SpeedSkatingKinematicDataset(dummy_data, dummy_labels)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    print(f"Dataset successfully created with {len(dataset)} samples.")
    for batch_data, batch_labels in loader:
        print(f"Batch data shape: {batch_data.shape}")
        print(f"Batch labels shape: {batch_labels.shape}")
        break