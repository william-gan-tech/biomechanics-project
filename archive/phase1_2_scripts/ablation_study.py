import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from .dataset import SpeedSkatingDataset
from .model import SkatingDegradationLSTM

def run_ablation_experiment(feature_sets, csv_path="data/extracted_multivariate_angles.csv", num_epochs=5):
    df = pd.read_csv(csv_path)
    
    # Ensure dummy label exists for training demonstration
    if 'degradation_label' not in df.columns:
        df['degradation_label'] = np.linspace(0, 1, len(df))
        
    results = {}
    
    for name, cols in feature_sets.items():
        print(f"\n--- Running Ablation: {name} ({cols}) ---")
        
        data_array = df[cols].values
        labels_array = df['degradation_label'].values
        
        # Normalize
        data_array = (data_array - np.mean(data_array, axis=0)) / (np.std(data_array, axis=0) + 1e-8)
        
        window_size = 30
        dataset = SpeedSkatingDataset(data_array, labels_array, window_size=window_size)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        model = SkatingDegradationLSTM(input_dim=len(cols), hidden_dim=64, num_layers=2, output_dim=1)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        final_epoch_loss = 0.0
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                predictions = model(batch_x).squeeze()
                loss = criterion(predictions, batch_y.float())
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            final_epoch_loss = epoch_loss / len(dataloader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {final_epoch_loss:.4f}")
            
        results[name] = final_epoch_loss
        
    print("\n=== Ablation Study Summary ===")
    for name, loss in results.items():
        print(f"Feature Set [{name}] -> Final Training Loss: {loss:.4f}")

if __name__ == "__main__":
    # Define the configurations to test
    configurations = {
        "Knees_Only": ['left_knee_filtered', 'right_knee_filtered'],
        "Hips_Only": ['right_hip_x', 'right_hip_y'],
        "All_Features": ['left_knee_filtered', 'right_knee_filtered', 'right_hip_x', 'right_hip_y']
    }
    run_ablation_experiment(configurations)