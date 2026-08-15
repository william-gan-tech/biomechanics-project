import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
from utils.biomechanics_utils import create_sliding_windows

print(f"PyTorch Version: {torch.__version__}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device.upper()}")

# 1. Load multi-joint or multi-channel angle data from your data folder
data_path = os.path.join('data', 'extracted_multivariate_angles.csv')

if not os.path.exists(data_path):
    data_path = os.path.join('data', 'extracted_knee_angles.csv')

try:
    angle_data = np.loadtxt(data_path, delimiter=",", skiprows=1)
    print(f"Successfully loaded angle data shape {angle_data.shape} from {data_path}.")
except Exception as e:
    print(f"Error loading CSV file: {e}")
    exit()

if len(angle_data.shape) == 1:
    angle_data = angle_data.reshape(-1, 1)

num_frames, num_features = angle_data.shape
print(f"Dataset dimensions -> Frames: {num_frames}, Features/Joints tracked: {num_features}")

# Normalize each feature channel independently between 0 and 1
data_min = np.min(angle_data, axis=0)
data_max = np.max(angle_data, axis=0)
data_range = data_max - data_min
data_range[data_range == 0] = 1.0  # Prevent division by zero

normalized_data = (angle_data - data_min) / data_range

# 2. Apply sliding window segmentation (30 frames per window, stepping by 5)
window_size = 30
windows = create_sliding_windows(normalized_data, window_size=window_size, step_size=5)

if len(windows) == 0:
    print("Error: Video or dataset is too short to create sliding windows of size 30.")
    exit()

X_train = torch.tensor(windows, dtype=torch.float32).to(device)

print(f"Successfully generated training windows!")
print(f"Tensor shape (Batch, Seq_Len, Features): {X_train.shape}")

# 3. Define a Multivariate Autoencoder Architecture
class MultiChannelAutoencoder(nn.Module):
    def __init__(self, seq_len, num_features):
        super(MultiChannelAutoencoder, self).__init__()
        input_dim = seq_len * num_features
        self.input_dim = input_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        encoded = self.encoder(x_flat)
        decoded = self.decoder(encoded)
        return decoded.view(batch_size, x.size(1), x.size(2))

model = MultiChannelAutoencoder(seq_len=window_size, num_features=num_features).to(device)
criterion = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Train the model on your sliding windows
epochs = 50
model.train()

print("\nTraining Multivariate Autoencoder on skating patterns...")
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, X_train)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss (Reconstruction Error): {loss.item():.4f}")

print("\nModel training complete!")

# 5. Evaluate the model with feature-level decomposition (reduction='none')
model.eval()
criterion_none = nn.MSELoss(reduction='none')

with torch.no_grad():
    reconstructed = model(X_train)
    loss_matrix = criterion_none(reconstructed, X_train)
    feature_errors_per_window = torch.mean(loss_matrix, dim=1).cpu().numpy()

# 6. Map Feature Errors to Specific Anatomical Joints & Overall Score
if num_features >= 4:
    results_df = pd.DataFrame({
        "Window_Index": range(len(feature_errors_per_window)),
        "Left_Knee_Error": feature_errors_per_window[:, 0],
        "Right_Knee_Error": feature_errors_per_window[:, 1],
        "Left_Hip_Error": feature_errors_per_window[:, 2],
        "Right_Hip_Error": feature_errors_per_window[:, 3],
        "Anomaly_Score": np.mean(feature_errors_per_window, axis=1)
    })
else:
    col_dict = {"Window_Index": range(len(feature_errors_per_window))}
    for i in range(num_features):
        col_dict[f"Joint_{i+1}_Error"] = feature_errors_per_window[:, i]
    col_dict["Anomaly_Score"] = np.mean(feature_errors_per_window, axis=1)
    results_df = pd.DataFrame(col_dict)

# 7. Compute Dynamic Statistical Baseline Metrics (Mu + 2Sigma)
baseline_count = min(5, len(results_df))
baseline_mean = results_df['Anomaly_Score'].iloc[:baseline_count].mean()
baseline_std = results_df['Anomaly_Score'].iloc[:baseline_count].std()
statistical_threshold = baseline_mean + (2 * baseline_std)

print("\n--- Biomechanical Fatigue & Statistical Thresholding Results ---")
print(f"Total analyzed stride windows: {len(results_df)}")
print(f"Baseline Mean (mu): {baseline_mean:.6f}")
print(f"Baseline Std Dev (sigma): {baseline_std:.6f}")
print(f"Calculated Statistical Threshold (mu + 2sigma): {statistical_threshold:.6f}")

highest_anomaly_index = results_df['Anomaly_Score'].idxmax()
print(f"\nHighest form deviation detected around window index: {highest_anomaly_index}")
print(f"Peak Anomaly Score (Error): {results_df.loc[highest_anomaly_index, 'Anomaly_Score']:.6f}")

# 8. Save results to CSV file
results_csv_path = "fatigue_results.csv"
results_df.to_csv(results_csv_path, index=False)
print(f"\nSuccessfully saved analysis results to {results_csv_path}!")