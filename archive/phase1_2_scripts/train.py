import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

try:
    from src.dataset import SpeedSkatingDataset
    from src.model import SkatingLSTMAutoencoder
except ImportError:
    from dataset import SpeedSkatingDataset
    from model import SkatingLSTMAutoencoder

def train_model():
    csv_path = "data/extracted_multivariate_angles.csv"
    
    try:
        df = pd.read_csv(csv_path)
        feature_cols = [
            'left_knee_filtered', 
            'right_knee_filtered', 
            'norm_right_hip_x', 
            'norm_right_hip_y',
            'norm_right_shoulder_x',
            'norm_right_shoulder_y'
        ]
        data_array = df[feature_cols].values
        labels = np.zeros(len(data_array))
        
    except FileNotFoundError:
        print(f"Error: Could not find {csv_path}. Ensure preprocessed data exists.")
        return
    
    window_size = 30
    
    dataset = SpeedSkatingDataset(
        data_array=data_array, 
        labels=labels, 
        window_size=window_size, 
        normalize=False 
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    n_features = len(feature_cols)
    model = SkatingLSTMAutoencoder(seq_len=window_size, n_features=n_features, embedding_dim=64)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 10
    model.train()
    
    print("Training LSTM Autoencoder on cross-subject normalized features...")
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            
            # Unpack the tuple (reconstructed tensor and auxiliary phase logits)
            reconstructed, _ = model(batch_x)
            
            loss = criterion(reconstructed, batch_x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{num_epochs}], Reconstruction MSE Loss: {epoch_loss / len(dataloader):.5f}")
        
    print("Training complete! Autoencoder weights ready for multi-subject evaluation.")
    torch.save(model.state_dict(), "skating_degradation_model.pth")
    print("Model weights saved to skating_degradation_model.pth")

if __name__ == "__main__":
    train_model()