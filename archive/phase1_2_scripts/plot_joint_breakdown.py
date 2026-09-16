import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# 1. Setup paths
base_dir = r'C:\Users\qgan2\OneDrive\Desktop\Research - biomechanics_project\biomechanics-project'
skater_b_path = os.path.join(base_dir, 'data', 'skater_b_multivariate_angles.csv')
output_dir = os.path.join(base_dir, 'outputs')

# Load data & model setup
df = pd.read_csv(skater_b_path)
data_matrix = df.values.astype(np.float32)
n_features = data_matrix.shape[1]

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
model = BiomechanicsAutoencoder(seq_len, n_features).to(device)
model_path = os.path.join(base_dir, 'models', 'autoencoder_model.pth')
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
model.eval()

# Sliding windows & feature error extraction
def create_windows(data, seq_len):
    return np.array([data[i:i + seq_len] for i in range(len(data) - seq_len)])

windows = create_windows(data_matrix, seq_len)
criterion_none = nn.MSELoss(reduction='none')

with torch.no_grad():
    tensor_data = torch.tensor(windows, dtype=torch.float32).to(device)
    reconstructed = model(tensor_data)
    loss_matrix = criterion_none(reconstructed, tensor_data)
    feature_errors_per_window = torch.mean(loss_matrix, dim=1).cpu().numpy()

# 2. Plot Individual Joint Error Traces
plt.figure(figsize=(12, 6))
joint_labels = [f"Joint Feature {i+1}" for i in range(n_features)]

for idx in range(n_features):
    plt.plot(feature_errors_per_window[:, idx], label=joint_labels[idx], alpha=0.7)

plt.axvline(x=10, color='r', linestyle='--', label='Global Anomaly Trigger (Frame 10)')
plt.axvline(x=63, color='g', linestyle='--', label='Physical Deceleration (Frame 63)')
plt.title('Joint-Specific Reconstruction Error Trajectories over Time')
plt.xlabel('Window Index')
plt.ylabel('Feature MSE Loss')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

joint_plot_path = os.path.join(output_dir, 'joint_breakdown_plot.png')
plt.savefig(joint_plot_path)
plt.close()

print(f"✅ Joint breakdown visualization successfully saved to '{joint_plot_path}'!")