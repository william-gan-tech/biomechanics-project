import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# 1. Setup paths
base_dir = r'C:\Users\qgan2\OneDrive\Desktop\Research - biomechanics_project\biomechanics-project'
data_dir = os.path.join(base_dir, 'data')
output_dir = os.path.join(base_dir, 'outputs')
os.makedirs(output_dir, exist_ok=True)

# 2. Define Autoencoder Architecture (Matches Training)
class BiomechanicsAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features):
        super(BiomechanicsAutoencoder, self).__init__()
        flat_dim = seq_len * n_features
        self.encoder = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(flat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, flat_dim),
            nn.Unflatten(1, (seq_len, n_features))
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

seq_len = 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model weights path once
model_path = os.path.join(base_dir, 'models', 'autoencoder_model.pth')

def create_windows(data, seq_len):
    windows = []
    for i in range(len(data) - seq_len):
        windows.append(data[i:i + seq_len])
    return np.array(windows)

# 3. Find all CSV files in the data directory
csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
summary_records = []

print(f"Found {len(csv_files)} dataset(s) to process.")

for file_name in csv_files:
    file_path = os.path.join(data_dir, file_name)
    print(f"\nProcessing: {file_name}...")
    
    # Load dataset
    df = pd.read_csv(file_path)
    data_matrix = df.values.astype(np.float32)
    n_features = data_matrix.shape[1]
    
    # Initialize model for this feature dimension
    model = BiomechanicsAutoencoder(seq_len, n_features).to(device)
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path)
            expected_features = checkpoint['encoder.1.weight'].shape[1] // seq_len
            if expected_features == n_features:
                model.load_state_dict(checkpoint)
                print(f"Loaded pre-trained weights successfully for {file_name}.")
            else:
                print(f"⚠️ Warning: Model expects {expected_features} features, but {file_name} has {n_features}. Running with uninitialized weights for this file.")
        except Exception as e:
            print(f"⚠️ Could not load weights for {file_name}: {e}")
    model.eval()
    
    # Run Sliding Windows & Inference
    windows = create_windows(data_matrix, seq_len)
    with torch.no_grad():
        tensor_data = torch.tensor(windows, dtype=torch.float32).to(device)
        reconstructed = model(tensor_data).cpu().numpy()
        
    mse_scores = np.mean(np.power(windows - reconstructed, 2), axis=(1, 2))
    
    # Dynamic Threshold (mu + 2sigma)
    baseline_count = min(10, len(mse_scores))
    b_mean = np.mean(mse_scores[:baseline_count])
    b_std = np.std(mse_scores[:baseline_count])
    threshold = b_mean + (2 * b_std)
    
    # Find Anomaly Frame
    anomaly_indices = np.where(mse_scores > threshold)[0]
    first_anomaly = int(anomaly_indices[0]) if len(anomaly_indices) > 0 else -1
    
    # Robust velocity proxy calculation matching main script
    velocity = np.linalg.norm(np.diff(data_matrix[seq_len:], axis=0), axis=1)
    smoothed_vel = pd.Series(velocity).rolling(window=10, min_periods=1).mean().values
    
    baseline_start = 30
    baseline_end = min(60, len(smoothed_vel))
    if len(smoothed_vel) > baseline_end:
        peak_baseline = smoothed_vel[baseline_start:baseline_end].max()
        dec_threshold = peak_baseline * 0.85
        dec_indices = np.where(smoothed_vel[baseline_end:] < dec_threshold)[0]
        first_dec = int(dec_indices[0] + baseline_end) if len(dec_indices) > 0 else -1
    else:
        first_dec = -1
    
    # Lead time calculation
    lead_frames = (first_dec - first_anomaly) if (first_dec != -1 and first_anomaly != -1) else 0
    lead_seconds = lead_frames / 30.0
    
    # Append record inside the loop correctly
    summary_records.append({
        'Dataset_Name': file_name,
        'Total_Frames': len(df),
        'AI_Anomaly_Frame': first_anomaly,
        'Deceleration_Frame': first_dec,
        'Lead_Time_Frames': lead_frames,
        'Lead_Time_Seconds': round(lead_seconds, 2),
        'Peak_MSE': round(float(np.max(mse_scores)), 4)
    })

# 4. Export Master Summary Report (Outside the loop)
summary_df = pd.DataFrame(summary_records)
summary_path = os.path.join(output_dir, 'summary_report.csv')
summary_df.to_csv(summary_path, index=False)

print(f"\n--- 📊 Batch Processing Complete ---")
print(summary_df)
print(f"\nMaster summary report saved successfully to '{summary_path}'!")