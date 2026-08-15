import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# 1. Paths
base_dir = r'C:\Users\qgan2\OneDrive\Desktop\Research - biomechanics_project\biomechanics-project'
data_path = os.path.join(base_dir, 'data', 'skater_b_multivariate_angles.csv')
model_path = os.path.join(base_dir, 'models', 'autoencoder_model.pth')

# 2. Re-define the Autoencoder Architecture (must match training)
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
data_matrix = df.values.astype(np.float32)
n_features = data_matrix.shape[1]
seq_len = 30

# Create windows
windows = []
for i in range(len(data_matrix) - seq_len):
    windows.append(data_matrix[i:i + seq_len])
tensor_data = torch.tensor(np.array(windows), dtype=torch.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BiomechanicsAutoencoder(seq_len, n_features).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# 4. Evaluate Reconstruction Error (Simulating Multi-Trial Generalization)
criterion = nn.MSELoss(reduction='none')
with torch.no_grad():
    outputs = model(tensor_data.to(device))
    # Calculate MSE per window to check for anomalies/variations
    loss_per_window = criterion(outputs, tensor_data.to(device)).mean(dim=(1, 2))

print(f"Evaluation complete!")
print(f"Average Reconstruction Error across data segments: {loss_per_window.mean().item():.4f}")
print(f"Max Error (Potential Anomaly/Technique Breakdown): {loss_per_window.max().item():.4f}")