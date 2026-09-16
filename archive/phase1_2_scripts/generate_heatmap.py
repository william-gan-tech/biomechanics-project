import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 1. Paths
base_dir = r'C:\Users\qgan2\OneDrive\Desktop\Research - biomechanics_project\biomechanics-project'
data_path = os.path.join(base_dir, 'data', 'skater_b_multivariate_angles.csv')
model_path = os.path.join(base_dir, 'models', 'autoencoder_model.pth')
output_plot_path = os.path.join(base_dir, 'Docs', 'joint_error_heatmap.png')

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

windows = []
for i in range(len(data_matrix) - seq_len):
    windows.append(data_matrix[i:i + seq_len])
tensor_data = torch.tensor(np.array(windows), dtype=torch.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BiomechanicsAutoencoder(seq_len, n_features).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# 4. Calculate Feature-Level Error Per Frame/Window
criterion = nn.MSELoss(reduction='none')
with torch.no_grad():
    outputs = model(tensor_data.to(device))
    # Error shape: [num_windows, seq_len, n_features]
    error_tensor = criterion(outputs, tensor_data.to(device)).cpu().numpy()

# Average across the sliding windows to map errors back to timeline frames
# Shape becomes [total_frames, n_features]
total_frames = len(data_matrix)
frame_errors = np.zeros((total_frames, n_features))
frame_counts = np.zeros(total_frames)

for w_idx, window in enumerate(error_tensor):
    for s_idx in range(seq_len):
        global_frame = w_idx + s_idx
        frame_errors[global_frame] += window[s_idx]
        frame_counts[global_frame] += 1

# Average out overlapping windows
frame_errors = frame_errors / np.maximum(frame_counts[:, np.newaxis], 1)

# 5. Generate and Save Heatmap Plot
plt.figure(figsize=(12, 6))
# Transpose so features are on the y-axis and time frames are on the x-axis
plt.imshow(frame_errors.T, aspect='auto', cmap='hot', interpolation='nearest')
plt.colorbar(label='Reconstruction Error (MSE)')
plt.yticks(ticks=range(len(feature_names)), labels=feature_names)
plt.xlabel('Time (Frames)')
plt.ylabel('Joint Features')
plt.title('Biomechanics Joint Error Timeline Heatmap')
plt.tight_layout()

os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
plt.savefig(output_plot_path, dpi=300)
print(f"Heatmap successfully generated and saved to:\n{output_plot_path}")