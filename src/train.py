import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from .model import SkatingLSTMAutoencoder

def train_model():
    # 1. Load baseline reference data (e.g., Sven Kramer fresh baseline)
    csv_path = "data/extracted_multivariate_angles.csv"
    
    try:
        df = pd.read_csv(csv_path)
        feature_cols = ['left_knee_filtered', 'right_knee_filtered', 'right_hip_x', 'right_hip_y']
        data_array = df[feature_cols].values
        
        # Normalize features based on elite baseline mean/std
        data_array = (data_array - np.mean(data_array, axis=0)) / (np.std(data_array, axis=0) + 1e-8)
        
    except FileNotFoundError:
        print(f"Error: Could not find {csv_path}. Ensure preprocessed data exists.")
        return
    
    # 2. Build sliding window sequences for temporal dynamics
    window_size = 30
    sequences = []
    for i in range(len(data_array) - window_size):
        sequences.append(data_array[i:i+window_size])
        
    tensor_data = torch.tensor(np.array(sequences), dtype=torch.float32)
    dataset = TensorDataset(tensor_data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 3. Initialize Autoencoder, Loss (MSE for reconstruction tracking), and Optimizer
    n_features = len(feature_cols)
    model = SkatingLSTMAutoencoder(seq_len=window_size, n_features=n_features, embedding_dim=64)
    
    criterion = nn.MSELoss()  # Directly optimizes reconstruction error
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 4. Training Loop (Unsupervised: Model learns to perfectly reconstruct elite form)
    num_epochs = 10
    model.train()
    
    print("Training LSTM Autoencoder on elite baseline kinematics...")
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            batch_x = batch[0]
            optimizer.zero_grad()
            reconstructed = model(batch_x)
            
            # Loss measures how badly the model fails to reconstruct the motion sequence
            loss = criterion(reconstructed, batch_x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{num_epochs}], Reconstruction MSE Loss: {epoch_loss / len(dataloader):.5f}")
        
    print("Training complete! Autoencoder weights ready for early-degradation anomaly scoring.")
    torch.save(model.state_dict(), "skating_degradation_model.pth")
    print("Model weights saved to skating_degradation_model.pth")

if __name__ == "__main__":
    train_model()