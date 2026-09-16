import os
import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
from scipy.signal import correlate

def run_advanced_lead_time_analysis():
    print("=" * 70)
    print(" ⏱️ ADVANCED PREDICTIVE LEAD-TIME & PHASE-LAG ANALYSIS")
    print("=" * 70)
    
    # 1. Setup paths
    base_dir = r'C:\Users\qgan2\OneDrive\Desktop\Research - biomechanics_project\biomechanics-project'
    skater_path = os.path.join(base_dir, 'data', 'skater_b_multivariate_angles.csv')
    output_dir = os.path.join(base_dir, 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(skater_path):
        print(f"Error: Could not find dataset at {skater_path}")
        return

    df = pd.read_csv(skater_path)
    data_matrix = df.values.astype(np.float32)
    seq_len = 30
    n_features = data_matrix.shape[1]
    
    # 2. Load Model & Compute Reconstruction Error (MSE)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiomechanicsAutoencoder(seq_len, n_features).to(device)
    
    model_path = os.path.join(base_dir, 'models', 'autoencoder_model.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("✅ Loaded trained model weights successfully.")
    else:
        print("⚠️ Warning: Model weights not found. Using uninitialized weights.")
    model.eval()

    # Create sliding windows
    windows = np.array([data_matrix[i:i + seq_len] for i in range(len(data_matrix) - seq_len)])
    
    criterion_none = nn.MSELoss(reduction='none')
    with torch.no_grad():
        tensor_data = torch.tensor(windows, dtype=torch.float32).to(device)
        reconstructed = model(tensor_data)
        loss_matrix = criterion_none(reconstructed, tensor_data)
        feature_errors_per_window = torch.mean(loss_matrix, dim=1).cpu().numpy()

    mse_scores = np.mean(feature_errors_per_window, axis=1)
    velocity = np.linalg.norm(np.diff(data_matrix[seq_len:], axis=0), axis=1)

    # 3. Dynamic Statistical Thresholding (Multi-Tiered: mu + 2sigma & mu + 3sigma)
    baseline_count = min(30, len(mse_scores))
    b_mean = np.mean(mse_scores[:baseline_count])
    b_std = np.std(mse_scores[:baseline_count])
    
    warning_threshold = b_mean + (2.0 * b_std)   # Early Warning
    critical_threshold = b_mean + (3.0 * b_std)  # Severe Breakdown
    
    print(f"\n📊 Statistical Thresholds Calculated:")
    print(f"   - Warning Level (mu + 2sigma):  {warning_threshold:.6f}")
    print(f"   - Critical Level (mu + 3sigma): {critical_threshold:.6f}")

    # 4. Persistence Window Filtering (Eliminates single-frame jitter false positives)
    def find_persistent_trigger(signal, threshold, min_consecutive=3):
        exceeds = signal > threshold
        count = 0
        for idx, val in enumerate(exceeds):
            if val:
                count += 1
                if count >= min_consecutive:
                    return idx - min_consecutive + 1  # Return start of persistence
            else:
                count = 0
        return None

    first_warning_frame = find_persistent_trigger(mse_scores, warning_threshold, min_consecutive=3)
    first_critical_frame = find_persistent_trigger(mse_scores, critical_threshold, min_consecutive=3)

    # 5. Physical Deceleration Point
    smoothed_velocity = pd.Series(velocity).rolling(window=10, min_periods=1).mean().values
    baseline_velocity = smoothed_velocity[30:60].max()
    decel_limit = baseline_velocity * 0.85
    
    decel_indices = np.where(smoothed_velocity[60:] < decel_limit)[0]
    first_decel_frame = decel_indices[0] + 60 if len(decel_indices) > 0 else None

    # 6. Cross-Correlation Phase Lag Analysis
    # Normalizing signals for cross-correlation
    norm_mse = (mse_scores - np.mean(mse_scores)) / (np.std(mse_scores) + 1e-8)
    norm_vel = (smoothed_velocity[:len(mse_scores)] - np.mean(smoothed_velocity[:len(mse_scores)])) / (np.std(smoothed_velocity[:len(mse_scores)]) + 1e-8)
    
    correlation = correlate(norm_vel, norm_mse, mode='full')
    lags = np.arange(-len(mse_scores) + 1, len(mse_scores))
    optimal_lag = lags[np.argmax(correlation)]

    # 7. Compute Final Lead-Time Metrics
    fps = 30.0
    results_summary = {}

    print("\n" + "=" * 30 + " RESULTS " + "=" * 30)
    if first_decel_frame and first_warning_frame:
        warning_lead_frames = first_decel_frame - first_warning_frame
        warning_lead_sec = warning_lead_frames / fps
        print(f"🟢 Warning Lead-Time (mu + 2sigma):  {warning_lead_frames} frames ({warning_lead_sec:.2f} seconds)")
        results_summary['warning_lead_time_seconds'] = warning_lead_sec
    else:
        print("⚠️ Warning threshold not persistently breached before deceleration.")

    if first_decel_frame and first_critical_frame:
        critical_lead_frames = first_decel_frame - first_critical_frame
        critical_lead_sec = critical_lead_frames / fps
        print(f"🔴 Critical Lead-Time (mu + 3sigma): {critical_lead_frames} frames ({critical_lead_sec:.2f} seconds)")
        results_summary['critical_lead_time_seconds'] = critical_lead_sec

    print(f"🔄 Optimal Phase-Lag (Cross-Correlation): {optimal_lag} windows")
    results_summary['optimal_lag_windows'] = int(optimal_lag)

    # 8. Export Structured Report
    report_path = os.path.join(output_dir, 'lead_time_metrics_report.json')
    with open(report_path, 'w') as f:
        json.dump(results_summary, f, indent=4)
    print(f"\n💾 Advanced lead-time report successfully exported to '{report_path}'!")
    print("=" * 70)

if __name__ == "__main__":
    run_advanced_lead_time_analysis()