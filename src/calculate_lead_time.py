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

# Ensure outputs directory exists safely using absolute paths
output_dir = os.path.join(base_dir, 'outputs')
os.makedirs(output_dir, exist_ok=True)

# 3. Define Autoencoder Architecture (Matches Training)
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

# 4. Configuration & Model Loading
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

# 5. Helper for Sliding Windows
def create_windows(data, seq_len):
    windows = []
    for i in range(len(data) - seq_len):
        windows.append(data[i:i + seq_len])
    return np.array(windows)

windows_b = create_windows(data_matrix_b, seq_len)

# 6. Inference & Feature-Level Error Decomposition (reduction='none')
criterion_none = nn.MSELoss(reduction='none')

with torch.no_grad():
    tensor_b = torch.tensor(windows_b, dtype=torch.float32).to(device)
    reconstructed_b = model(tensor_b)
    # Compute error matrix across all dimensions: (Batch, Seq_Len, Features)
    loss_matrix = criterion_none(reconstructed_b, tensor_b)
    feature_errors_per_window = torch.mean(loss_matrix, dim=1).cpu().numpy()

# Global MSE per sliding window (Anomaly Score)
mse_b = np.mean(feature_errors_per_window, axis=1)

# 7. Velocity Proxy Calculation (Frame-to-frame displacement)
velocity_b = np.linalg.norm(np.diff(data_matrix_b[seq_len:], axis=0), axis=1)

# 8. Dynamic Statistical Thresholding for Lead-Time (mu + 2sigma)
baseline_count = min(10, len(mse_b))
baseline_mean = np.mean(mse_b[:baseline_count])
baseline_std = np.std(mse_b[:baseline_count])
threshold = baseline_mean + (2 * baseline_std)
print(f"Calculated Dynamic Threshold (mu + 2sigma): {threshold:.6f}")

# Find where global anomalies cross the threshold
anomalies = mse_b > threshold
anomaly_indices = np.where(anomalies)[0]

if len(anomaly_indices) > 0:
    first_anomaly_frame = anomaly_indices[0]
    print(f"First global fatigue anomaly detected at frame: {first_anomaly_frame}")
else:
    print("No global anomalies detected above threshold.")

# Plot and save results safely using absolute paths
plt.figure(figsize=(10, 4))
plt.plot(mse_b, label='Reconstruction Error (MSE)')
plt.axhline(y=threshold, color='r', linestyle='--', label=f'Dynamic Threshold (mu + 2sigma)')
plt.title('Skater B Biomechanical Anomaly Detection & Lead-Time Analysis')
plt.xlabel('Window Index')
plt.ylabel('MSE Loss')
plt.legend()

output_plot_path = os.path.join(output_dir, 'lead_time_plot.png')
plt.savefig(output_plot_path)
plt.close()
print(f"Lead-time plot saved successfully to '{output_plot_path}'!")

# 9. Programmatic Physical Deceleration Detection (Timeline B)
window_smooth = 10
smoothed_velocity = pd.Series(velocity_b).rolling(window=window_smooth, min_periods=1).mean().values

baseline_start = 30
baseline_end = min(60, len(smoothed_velocity))
baseline_velocity = smoothed_velocity[baseline_start:baseline_end].max()
deceleration_threshold = baseline_velocity * 0.85 

search_start_frame = baseline_end
deceleration_indices = np.where(smoothed_velocity[search_start_frame:] < deceleration_threshold)[0]

if len(deceleration_indices) > 0 and len(anomaly_indices) > 0:
    first_deceleration_frame = deceleration_indices[0] + search_start_frame
    
    # Global Lead Time Advantage
    lead_time_frames = first_deceleration_frame - first_anomaly_frame
    fps = 30.0
    lead_time_seconds = lead_time_frames / fps
    
    print("\n--- ⏱️ Quantitative Global Lead-Time Results ---")
    print(f"AI Anomaly Flagged at Frame: {first_anomaly_frame}")
    print(f"Physical Deceleration Started at Frame: {first_deceleration_frame}")
    print(f"Global Lead-Time Advantage: {lead_time_frames} frames ({lead_time_seconds:.2f} seconds)")
    
    # 10. Joint-Specific Lead-Time Breakdown
    print("\n--- 🔍 Joint-Specific Lead-Time Breakdown ---")
    # Map feature columns if available
    joint_names = ["Left_Knee", "Right_Knee", "Left_Hip", "Right_Hip"]
    if n_features >= len(joint_names):
        active_joints = joint_names
    else:
        active_joints = [f"Joint_{i+1}" for i in range(n_features)]

    for idx, joint_name in enumerate(active_joints):
        joint_signal = feature_errors_per_window[:, idx]
        j_mean = np.mean(joint_signal[:baseline_count])
        j_std = np.std(joint_signal[:baseline_count])
        j_threshold = j_mean + (2 * j_std)
        
        joint_exceeds = np.where(joint_signal > j_threshold)[0]
        if len(joint_exceeds) > 0:
            first_joint_flag = joint_exceeds[0]
            j_lead_frames = first_deceleration_frame - first_joint_flag
            j_lead_seconds = j_lead_frames / fps
            print(f"[{joint_name}] Flagged at Frame {first_joint_flag} | Lead-Time Advantage: {j_lead_frames} frames ({j_lead_seconds:.2f}s)")
        else:
            print(f"[{joint_name}] Threshold not crossed.")
else:
    print("\n⚠️ Could not compute precise lead time: Check threshold sensitivity or baseline data range.")