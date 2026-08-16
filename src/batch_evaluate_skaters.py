import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# 1. Setup paths
base_dir = r'C:\Users\qgan2\OneDrive\Desktop\Research - biomechanics_project\biomechanics-project'
data_dir = os.path.join(base_dir, 'data')
output_dir = os.path.join(base_dir, 'outputs')
os.makedirs(output_dir, exist_ok=True)

# Define Autoencoder Architecture
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

def evaluate_skater(file_name, skater_name):
    file_path = os.path.join(data_dir, file_name)
    if not os.path.exists(file_path):
        print(f"Skipping {skater_name}: File not found at {file_path}")
        return None

    df = pd.read_csv(file_path)
    data_matrix = df.values.astype(np.float32)
    seq_len = 30
    n_features = data_matrix.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiomechanicsAutoencoder(seq_len, n_features).to(device)
    
    model_path = os.path.join(base_dir, 'models', 'autoencoder_model.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    model.eval()

    # Sliding windows
    windows = []
    for i in range(len(data_matrix) - seq_len):
        windows.append(data_matrix[i:i + seq_len])
    windows = np.array(windows)

    # Inference
    criterion_none = nn.MSELoss(reduction='none')
    with torch.no_grad():
        tensor_data = torch.tensor(windows, dtype=torch.float32).to(device)
        reconstructed = model(tensor_data)
        loss_matrix = criterion_none(reconstructed, tensor_data)
        feature_errors = torch.mean(loss_matrix, dim=1).cpu().numpy()

    mse = np.mean(feature_errors, axis=1)
    velocity = np.linalg.norm(np.diff(data_matrix[seq_len:], axis=0), axis=1)

    # Thresholding
    baseline_count = min(10, len(mse))
    threshold = np.mean(mse[:baseline_count]) + (2 * np.std(mse[:baseline_count]))
    anomaly_indices = np.where(mse > threshold)[0]
    first_anomaly = anomaly_indices[0] if len(anomaly_indices) > 0 else 0

    # Deceleration detection
    smoothed_vel = pd.Series(velocity).rolling(window=10, min_periods=1).mean().values
    baseline_vel = smoothed_vel[30:min(60, len(smoothed_vel))].max()
    dec_indices = np.where(smoothed_vel[min(60, len(smoothed_vel)):] < (baseline_vel * 0.85))[0]
    
    if len(dec_indices) > 0 and len(anomaly_indices) > 0:
        first_dec = dec_indices[0] + min(60, len(smoothed_vel))
        lead_frames = first_dec - first_anomaly
        lead_sec = lead_frames / 30.0
    else:
        lead_frames, lead_sec = 0, 0.0

    return {
        "Skater": skater_name,
        "Anomaly_Frame": first_anomaly,
        "Deceleration_Frame": first_dec if 'first_dec' in locals() else 0,
        "Lead_Time_Frames": lead_frames,
        "Lead_Time_Seconds": lead_sec
    }

# Run for both skaters
results = []
for skater_info in [("skater_a_multivariate_angles.csv", "Skater A"), ("skater_b_multivariate_angles.csv", "Skater B")]:
    res = evaluate_skater(skater_info[0], skater_info[1])
    if res:
        results.append(res)

# Output summary table
summary_df = pd.DataFrame(results)
print("\n=== 📊 MULTI-SUBJECT COMPARISON SUMMARY ===")
print(summary_df.to_string(index=False))

summary_path = os.path.join(output_dir, "multi_subject_lead_time_summary.csv")
summary_df.to_csv(summary_path, index=False)
print(f"\nSummary table saved to: {summary_path}")