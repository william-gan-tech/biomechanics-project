import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# 1. Setup paths
base_dir = r'C:\Users\qgan2\OneDrive\Desktop\Research - biomechanics_project\biomechanics-project'
skater_b_path = os.path.join(base_dir, 'data', 'skater_b_multivariate_angles.csv')

# Load clean data
df_clean = pd.read_csv(skater_b_path)
data_matrix = df_clean.values.astype(np.float32)

# 2. Inject Synthetic Failure (Target a stable middle section)
perturbed_data = data_matrix.copy()
start_fault_frame = 150
end_fault_frame = 170

print(f"Injecting heavy synthetic failure between frames {start_fault_frame} and {end_fault_frame}...")

# Increase noise multiplier to ensure it completely dominates the error score
noise_multiplier = 20.0
perturbed_data[start_fault_frame:end_fault_frame, :] += noise_multiplier * np.random.randn(
    end_fault_frame - start_fault_frame, data_matrix.shape[1]
)
# 3. Define Autoencoder (matching your architecture)
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
model = BiomechanicsAutoencoder(seq_len, data_matrix.shape[1]).to(device)

# Load trained weights
model_path = os.path.join(base_dir, 'models', 'autoencoder_model.pth')
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
model.eval()

# 4. Helper for Sliding Windows
def create_windows(data, seq_len):
    windows = []
    for i in range(len(data) - seq_len):
        windows.append(data[i:i + seq_len])
    return np.array(windows)

windows_perturbed = create_windows(perturbed_data, seq_len)

# 5. Run Inference on Perturbed Data
with torch.no_grad():
    tensor_p = torch.tensor(windows_perturbed, dtype=torch.float32).to(device)
    reconstructed_p = model(tensor_p).cpu().numpy()

# Calculate MSE error per window
mse_perturbed = np.mean(np.power(windows_perturbed - reconstructed_p, 2), axis=(1, 2))

# 6. Evaluate if the model successfully flagged the injected fault window
peak_error_window = np.argmax(mse_perturbed)
print(f"\n--- 🧪 Stress-Test Results ---")
print(f"Injected Fault Zone: Frames {start_fault_frame} to {end_fault_frame}")
print(f"Model Peak Error Spike Detected at Window Index: {peak_error_window}")

if start_fault_frame <= peak_error_window <= end_fault_frame:
    print("✅ SUCCESS: Model successfully caught the synthetic failure right inside the injected window!")
else:
    print("⚠️ Model peak error occurred outside the fault zone. Adjust threshold or noise intensity.")