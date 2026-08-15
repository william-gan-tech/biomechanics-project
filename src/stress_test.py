import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# 1. Paths
base_dir = r'C:\Users\qgan2\OneDrive\Desktop\Research - biomechanics_project\biomechanics-project'
data_path = os.path.join(base_dir, 'data', 'skater_b_multivariate_angles.csv')
model_path = os.path.join(base_dir, 'models', 'autoencoder_model.pth')

# 2. Re-define Autoencoder Architecture
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

# 3. Load Data & Model
df = pd.read_csv(data_path)
feature_names = df.columns.tolist()
data_matrix = df.values.astype(np.float32)
n_features = data_matrix.shape[1]
seq_len = 30

# Create normal window baseline
windows = []
for i in range(len(data_matrix) - seq_len):
    windows.append(data_matrix[i:i + seq_len])
tensor_data = torch.tensor(np.array(windows), dtype=torch.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BiomechanicsAutoencoder(seq_len, n_features).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# 4. Perturbation Analysis: Inject Synthetic Failure
# Pick a specific window (e.g., middle of the dataset) and corrupt a specific joint (e.g., index 0, 'right_knee_angle')
target_window_idx = 50
perturbed_tensor = tensor_data.clone()

# Inject an extreme artificial spike into the right knee angle for this window
perturbed_tensor[target_window_idx, :, 0] += 50.0  

# 5. Evaluate Reconstruction Error on Perturbed Data
criterion = nn.MSELoss(reduction='none')
with torch.no_grad():
    normal_outputs = model(tensor_data.to(device))
    perturbed_outputs = model(perturbed_tensor.to(device))
    
    # Calculate feature-level errors
    normal_feature_loss = criterion(normal_outputs, tensor_data.to(device)).mean(dim=(0, 1)).cpu().numpy()
    perturbed_feature_loss = criterion(perturbed_outputs, perturbed_tensor.to(device)).mean(dim=(0, 1)).cpu().numpy()

print("\n--- Stress Test Results (Perturbed vs Normal) ---")
for name, norm_err, pert_err in zip(feature_names, normal_feature_loss, perturbed_feature_loss):
    print(f"Feature: {name}")
    print(f"  Normal Error:   {norm_err:.4f}")
    print(f"  Perturbed Error: {pert_err:.4f} (Difference: +{(pert_err - norm_err):.4f})")