import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from biomechanics_utils import create_sliding_windows

print(f"PyTorch Version: {torch.__version__}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device.upper()}")

# 1. Load your extracted angle data from the CSV file
# --- NEW CODE (Correctly looks inside the 'data' folder) ---
import os

knee_path = os.path.join('data', 'extracted_knee_angles.csv')

try:
    knee_angles = np.loadtxt(knee_path, delimiter=",", skiprows=1)
    print(f"Successfully loaded {len(knee_angles)} frames of knee angle data from {knee_path}.")
except Exception as e:
    print(f"Error loading CSV file: {e}")
    exit()

# Normalize data between 0 and 1 (crucial for neural networks to train properly)
angle_min, angle_max = np.min(knee_angles), np.max(knee_angles)
if angle_max - angle_min == 0:
    print("Error: Angle data has no variation.")
    exit()
    
normalized_angles = (knee_angles - angle_min) / (angle_max - angle_min)

# 2. Apply sliding window segmentation (30 frames per window, stepping by 5)
window_size = 30
windows = create_sliding_windows(normalized_angles, window_size=window_size, step_size=5)

if len(windows) == 0:
    print("Error: Video is too short to create sliding windows of size 30.")
    exit()

X_train = torch.tensor(windows, dtype=torch.float32)

print(f"Successfully generated training windows!")
print(f"Total windows: {X_train.shape[0]}")
print(f"Window shape (frames per chunk): {X_train.shape[1]}")

# 3. Define the Autoencoder Neural Network Architecture
class KneeAutoencoder(nn.Module):
    def __init__(self, seq_len):
        super(KneeAutoencoder, self).__init__()
        # Encoder: compresses 30 frames down to 8 dimensions
        self.encoder = nn.Sequential(
            nn.Linear(seq_len, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        # Decoder: expands 8 dimensions back out to 30 frames
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, seq_len)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Initialize model, loss function, and optimizer
model = KneeAutoencoder(seq_len=window_size)
criterion = nn.MSELoss()  # Standard Mean Squared Error for training
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Train the model on your sliding windows
epochs = 50
model.train()

print("\nTraining Autoencoder on skating stride patterns...")
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
criterion_none = nn.MSELoss(reduction='none') # Preserves individual error elements per feature dimension

with torch.no_grad():
    reconstructed = model(X_train)
    
    # Calculate MSE matrix: shape matches sliding windows tensor dimensions
    loss_matrix = criterion_none(reconstructed, X_train)
    
    # Average across the sequence length dimension to get feature/error profile per window
    feature_errors_per_window = torch.mean(loss_matrix, dim=1).numpy()

# If your dataset features multiple channels, you can isolate them here. 
# For single or overall arrays, we take the average across features:
mse_per_window = np.mean(feature_errors_per_window, axis=1) if len(feature_errors_per_window.shape) > 1 else feature_errors_per_window

print("\n--- Biomechanical Fatigue Analysis Results ---")
print(f"Total analyzed stride windows: {len(mse_per_window)}")
print(f"Baseline (Start of video) Error: {mse_per_window[0]:.6f}")
print(f"Later (End of video) Error: {mse_per_window[-1]:.6f}")

# Show the windows with the highest anomaly scores (potential form breakdown points)
highest_anomaly_index = np.argmax(mse_per_window)
print(f"\nHighest form deviation detected around window index: {highest_anomaly_index}")
print(f"Peak Anomaly Score (Error): {mse_per_window[highest_anomaly_index]:.6f}")

# 6. Save results to a CSV file so they can be tracked and viewed on GitHub
results_df = pd.DataFrame({
    "Window_Index": range(len(mse_per_window)),
    "Anomaly_Score": mse_per_window
})

results_csv_path = "fatigue_results.csv"
results_df.to_csv(results_csv_path, index=False)
print(f"\nSuccessfully saved fatigue analysis results to {results_csv_path}!")