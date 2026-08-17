import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from .dataset import SpeedSkatingDataset
from .model import SkatingDegradationLSTM

def train_model():
    # 1. Load your real preprocessed CSV data
    csv_path = "path_to_your_preprocessed_kinematics.csv"  # Update with your actual filename/path
    
    # Update these with your exact CSV column names
    feature_cols = ['knee_angle_l', 'knee_angle_r', 'hip_angle_l', 'hip_angle_r'] 
    label_col = 'degradation_label'
    
    try:
        df = pd.read_csv(csv_path)
        data_array = df[feature_cols].values
        labels_array = df[label_col].values
        
        # Optional: Normalize features for stable training
        data_array = (data_array - np.mean(data_array, axis=0)) / (np.std(data_array, axis=0) + 1e-8)
        
    except FileNotFoundError:
        print(f"Error: Could not find {csv_path}. Please update the path to your CSV file.")
        return
    
    # 2. Initialize Dataset and DataLoader using your existing dataset.py
    window_size = 50
    dataset = SpeedSkatingDataset(data_array, labels_array, window_size=window_size)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 3. Initialize Model, Loss Function, and Optimizer
    input_dim = len(feature_cols)
    model = SkatingDegradationLSTM(input_dim=input_dim, hidden_dim=64, num_layers=2, output_dim=1)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 4. Training Loop
    num_epochs = 5
    model.train()
    
    print("Starting training loop on real skating data...")
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            predictions = model(batch_x).squeeze()
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(dataloader):.4f}")
        
    print("Training complete! Model successfully trained on real kinematic sequences.")
    torch.save(model.state_dict(), "skating_degradation_model.pth")

if __name__ == "__main__":
    train_model()