import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.model import SkatingLSTMAutoencoder
from src.dataset import SpeedSkatingDataset

def evaluate_pipeline(df, feature_cols, normalize_flag, epochs=15):
    subjects = df['subject_id'].unique()
    results = []
    
    for test_subject in subjects:
        print(f"Evaluating LOSO (Normalization Enabled={normalize_flag}) for: Subject {test_subject}")
        
        train_df = df[df['subject_id'] != test_subject]
        test_df = df[df['subject_id'] == test_subject]
        
        train_dataset = SpeedSkatingDataset(
            train_df[feature_cols].values, 
            np.zeros(len(train_df)), 
            window_size=30, 
            normalize=normalize_flag
        )
        test_dataset = SpeedSkatingDataset(
            test_df[feature_cols].values, 
            np.zeros(len(test_df)), 
            window_size=30, 
            normalize=normalize_flag
        )
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        model = SkatingLSTMAutoencoder(seq_len=30, n_features=len(feature_cols), embedding_dim=64)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        
        model.train()
        for epoch in range(epochs):
            for batch_x, _ in train_loader:
                optimizer.zero_grad()
                reconstructed, _ = model(batch_x)
                loss = criterion(reconstructed, batch_x)
                loss.backward()
                optimizer.step()
                
        model.eval()
        subject_losses = []
        with torch.no_grad():
            for batch_x, _ in test_loader:
                reconstructed, _ = model(batch_x)
                batch_loss = torch.mean((reconstructed - batch_x) ** 2, dim=(1, 2))
                subject_losses.extend(batch_loss.tolist())
                
        mean_mse = np.mean(subject_losses)
        results.append({"subject": test_subject, "mean_reconstruction_mse": mean_mse})
        print(f"Subject {test_subject} Unseen MSE: {mean_mse:.4f}")
        
    return pd.DataFrame(results)

def main():
    csv_path = "data/extracted_multivariate_angles.csv"
    df = pd.read_csv(csv_path)
    
    if 'subject_id' not in df.columns:
        print("'subject_id' column not found. Splitting dataset into 4 pseudo-subjects for evaluation...")
        num_chunks = 4
        chunk_size = len(df) // num_chunks
        df['subject_id'] = [i // chunk_size for i in range(len(df))]
        df.loc[df['subject_id'] >= num_chunks, 'subject_id'] = num_chunks - 1

    feature_cols = [
        'left_knee_filtered', 'right_knee_filtered', 
        'norm_right_hip_x', 'norm_right_hip_y',
        'norm_right_shoulder_x', 'norm_right_shoulder_y'
    ]
    
    print("\n--- Running Evaluation: RAW Features ---")
    df_raw = evaluate_pipeline(df, feature_cols, normalize_flag=False, epochs=15)
    df_raw.to_csv("data/generalization_raw.csv", index=False)
    
    print("\n--- Running Evaluation: BONE-NORMALIZED Features ---")
    df_norm = evaluate_pipeline(df, feature_cols, normalize_flag=True, epochs=15)
    df_norm.to_csv("data/generalization_normalized.csv", index=False)
    
    print("\nLOSO Evaluation Complete. Results saved to:")
    print(" - data/generalization_raw.csv")
    print(" - data/generalization_normalized.csv")

if __name__ == "__main__":
    main()