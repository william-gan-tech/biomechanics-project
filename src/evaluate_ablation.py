import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.model import SkatingLSTMAutoencoder
from src.dataset import SpeedSkatingDataset

def run_ablation_experiment(normalize_flag=False):
    csv_path = "data/extracted_multivariate_angles.csv"
    df = pd.read_csv(csv_path)
    
    if 'subject_id' not in df.columns:
        num_chunks = 4
        chunk_size = len(df) // num_chunks
        df['subject_id'] = [i // chunk_size for i in range(len(df))]
        df.loc[df['subject_id'] >= num_chunks, 'subject_id'] = num_chunks - 1

    feature_cols = [
        'left_knee_filtered', 'right_knee_filtered', 
        'norm_right_hip_x', 'norm_right_hip_y',
        'norm_right_shoulder_x', 'norm_right_shoulder_y'
    ]

    subjects = df['subject_id'].unique()
    results = []
    
    for test_subject in subjects:
        train_df = df[df['subject_id'] != test_subject]
        test_df = df[df['subject_id'] == test_subject]
        
        train_dataset = SpeedSkatingDataset(
            data_array=train_df[feature_cols].values, 
            labels=np.zeros(len(train_df)), 
            window_size=30, 
            normalize=normalize_flag
        )
        test_dataset = SpeedSkatingDataset(
            data_array=test_df[feature_cols].values, 
            labels=np.zeros(len(test_df)), 
            window_size=30, 
            normalize=normalize_flag
        )
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        model = SkatingLSTMAutoencoder(seq_len=30, n_features=len(feature_cols), embedding_dim=64)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        
        model.train()
        for epoch in range(3):
            for batch_x, _ in train_loader:
                optimizer.zero_grad()
                reconstructed, _ = model(batch_x)
                loss = criterion(reconstructed, batch_x)
                loss.backward()
                optimizer.step()
                
        model.eval()
        subject_losses = []
        with torch.no_grad():  # Fixed typo here
            for batch_x, _ in test_loader:
                reconstructed, _ = model(batch_x)
                batch_loss = torch.mean((reconstructed - batch_x) ** 2, dim=(1, 2))
                subject_losses.extend(batch_loss.tolist())
                
        results.append({
            "subject": test_subject, 
            "mode": "Bone-Normalized" if normalize_flag else "Raw",
            "mean_mse": np.mean(subject_losses)
        })
        
    return pd.DataFrame(results)

if __name__ == "__main__":
    print("Running Raw Evaluation...")
    raw_res = run_ablation_experiment(normalize_flag=False)
    
    print("Running Bone-Normalized Evaluation...")
    norm_res = run_ablation_experiment(normalize_flag=True)
    
    combined = pd.concat([raw_res, norm_res])
    combined.to_csv("data/ablation_comparison_metrics.csv", index=False)
    print("\nAblation complete. Summary saved to data/ablation_comparison_metrics.csv")
    print(combined.groupby("mode")["mean_mse"].mean())