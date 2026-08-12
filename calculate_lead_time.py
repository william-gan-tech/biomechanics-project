import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# 1. Setup the absolute path safely and directly
base_dir = r'C:\Users\qgan2\OneDrive\Desktop\Research - biomechanics_project\biomechanics-project'
skater_b_path = os.path.join(base_dir, 'data', 'skater_b_multivariate_angles.csv')

print(f"Attempting to load data from: {skater_b_path}")

# 2. Load Skater B data safely
skater_b_data = pd.read_csv(skater_b_path)

# Ensure outputs directory exists
os.makedirs('outputs', exist_ok=True)

# 1. Define Autoencoder Architecture (Matches Training)
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

# 2. Configuration & Model Loading
seq_len = 30
data_matrix_b = skater_b_data.values.astype(np.float32)
n_features = data_matrix_b.shape[1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BiomechanicsAutoencoder(seq_len, n_features).to(device)

# Load your trained model weights (make sure autoencoder_model.pth is in your 'models/' folder)
model_path = os.path.join(base_dir, 'models', 'autoencoder_model.pth')
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    print("Loaded trained model weights successfully.")
else:
    print("Warning: Model weights not found in 'models/'. Running with uninitialized weights for testing.")

model.eval()

# 3. Helper for Sliding Windows
def create_windows(data, seq_len):
    windows = []
    for i in range(len(data) - seq_len):
        windows.append(data[i:i + seq_len])
    return np.array(windows)

windows_b = create_windows(data_matrix_b, seq_len)

# 4. Inference & Reconstruction Error (MSE) Calculation
with torch.no_grad():
    tensor_b = torch.tensor(windows_b, dtype=torch.float32).to(device)
    reconstructed_b = model(tensor_b).cpu().numpy()

# Global MSE per sliding window (Anomaly Score)
mse_b = np.mean(np.power(windows_b - reconstructed_b, 2), axis=(1, 2))

# 5. Velocity Proxy Calculation (Frame-to-frame displacement)
velocity_b = np.linalg.norm(np.diff(data_matrix_b[seq_len:], axis=0), axis=1)

# 6. Quantitative Lead-Time Calculation
threshold = 0.05

# Find where anomalies cross the threshold
anomalies = mse_b > threshold
anomaly_indices = np.where(anomalies)[0]

if len(anomaly_indices) > 0:
    first_anomaly_frame = anomaly_indices[0]
    print(f"First fatigue anomaly detected at frame: {first_anomaly_frame}")
else:
    print("No anomalies detected above threshold.")

# Plot and save results
plt.figure(figsize=(10, 4))
plt.plot(mse_b, label='Reconstruction Error (MSE)')
plt.axhline(y=threshold, color='r', linestyle='--', label='Fatigue Threshold')
plt.title('Skater B Biomechanical Anomaly Detection')
plt.xlabel('Window Index')
plt.ylabel('MSE Loss')
plt.legend()
plt.savefig(os.path.join(base_dir, 'outputs', 'lead_time_plot.png'))
plt.close()
print("Lead-time plot saved successfully to 'outputs/'!") 