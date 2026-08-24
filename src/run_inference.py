import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add root and src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)
sys.path.append(current_dir)

from preprocess_video import process_skating_video_multivariate
from model import SkatingLSTMAutoencoder

def find_model_file(preferred_path):
    if os.path.exists(preferred_path):
        return preferred_path
    search_dirs = [root_dir, os.path.join(root_dir, "models"), os.path.join(root_dir, "checkpoints")]
    for d in search_dirs:
        if os.path.exists(d):
            for file in os.listdir(d):
                if file.endswith(".pth"):
                    found_path = os.path.join(d, file)
                    print(f"ℹ️ Automatically located model weights at: {found_path}")
                    return found_path
    return preferred_path

def save_results_to_csv(results, output_path="fatigue_detection_report.csv"):
    if not results:
        print("ℹ️ No results to save.")
        return
    df_results = pd.DataFrame(results, columns=["frame", "timestamp_sec", "mse_loss"])
    df_results.to_csv(output_path, index=False)
    print(f"💾 Results successfully saved to {output_path}")

def print_summary(results, total_frames, fps=30.0):
    if not results:
        print("\n--- Biomechanical Summary ---")
        print("✅ Summary: Skater maintained stable form throughout the trial.")
        return
    
    first_spike_time = results[0][1]
    fatigue_percentage = (len(results) / total_frames) * 100
    
    print("\n--- Biomechanical Summary ---")
    print(f"⏱️ First Fatigue Onset: {first_spike_time:.2f}s")
    print(f"📊 Total Spikes Detected: {len(results)}")
    print(f"📉 Estimated Time in Fatigue: {fatigue_percentage:.1f}% of video")

def run_pipeline(video_path, model_path):
    resolved_model_path = find_model_file(model_path)
    if not os.path.exists(resolved_model_path):
        print(f"❌ Error: Model weights file not found.")
        return [], [], 0.0, 0

    print(f"Processing video frames: {video_path}...")
    df_features = process_skating_video_multivariate(video_path)
    if df_features is None or df_features.empty:
        print("❌ Error: No features extracted.")
        return [], [], 0.0, 0

    window_size = 30
    n_features = 4  
    model = SkatingLSTMAutoencoder(seq_len=window_size, n_features=n_features, embedding_dim=64)

    checkpoint = torch.load(resolved_model_path, map_location=torch.device('cpu'))
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model = checkpoint
    
    model.eval()
    
    feature_cols = ["right_knee_angle", "left_knee_angle", "right_knee_filtered", "left_knee_filtered"]
    buffer = []
    fps = 30.0
    
    all_losses = []
    frame_loss_pairs = []

    # Pass 1: Compute reconstruction loss for all windows
    for idx, row in df_features.iterrows():
        frame_idx = int(row["frame"])
        
        available_cols = [c for c in feature_cols if c in df_features.columns]
        if len(available_cols) < n_features:
            print(f"❌ Error: Expected {n_features} features, but only found {len(available_cols)} in DataFrame.")
            return [], [], 0.0, len(df_features)
            
        current_features = row[feature_cols].values
        buffer.append(current_features)
        
        if len(buffer) == window_size:
            window_data = np.array(buffer)
            tensor_input = torch.tensor(window_data, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                reconstruction = model(tensor_input)
                loss = torch.mean((tensor_input - reconstruction) ** 2).item()
            
            all_losses.append(loss)
            frame_loss_pairs.append((frame_idx, loss))
            buffer.pop(0)

    if not all_losses:
        print("❌ Error: Not enough frames to compute windows.")
        return [], [], 0.0, len(df_features)

    # --- AUTOMATED BASELINE CALIBRATION ---
    baseline_window_count = min(150, len(all_losses))
    baseline_losses = all_losses[:baseline_window_count]
    
    mean_loss = np.mean(baseline_losses)
    std_loss = np.std(baseline_losses)
    dynamic_threshold = mean_loss + (2.0 * std_loss)
    
    print(f"\n📊 Baseline Calibration Complete:")
    print(f"    - Baseline Mean Loss: {mean_loss:.4f}")
    print(f"    - Baseline Std Dev:   {std_loss:.4f}")
    print(f"    - Dynamic Threshold:  {dynamic_threshold:.4f}\n")

    # Pass 2: Detect fatigue spikes using the dynamic threshold
    fatigue_timestamps = []
    for frame_idx, loss in frame_loss_pairs:
        if loss > dynamic_threshold:
            timestamp = frame_idx / fps
            fatigue_timestamps.append((frame_idx, timestamp, loss))
            
    return fatigue_timestamps, frame_loss_pairs, dynamic_threshold, len(df_features)

def plot_fatigue_timeline(frame_loss_pairs, threshold):
    """Generates, displays, and saves a matplotlib timeline of the reconstruction loss."""
    if not frame_loss_pairs:
        return
        
    frames, losses = zip(*frame_loss_pairs)
    fps = 30.0
    timestamps = [f / fps for f in frames]

    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, losses, label='Reconstruction Loss (MSE)', color='blue', alpha=0.7)
    plt.axhline(y=threshold, color='red', linestyle='--', label=f'Dynamic Threshold ({threshold:.4f})')
    
    # Highlight points above threshold
    spike_times = [t for f, t, l in [(fr, fr/fps, ls) for fr, ls in frame_loss_pairs] if l > threshold]
    spike_losses = [l for fr, t, l in [(fr, fr/fps, ls) for fr, ls in frame_loss_pairs] if l > threshold]
    if spike_times:
        plt.scatter(spike_times, spike_losses, color='orange', zorder=5, label='Fatigue Spike')

    plt.title('Skater Biomechanical Fatigue Timeline')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Reconstruction Loss (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot automatically
    plot_output_path = "skating_fatigue_timeline.png"
    plt.savefig(plot_output_path, dpi=300)
    print(f"💾 Plot saved as {plot_output_path}")

    print("📈 Generating rolling fatigue timeline plot...")
    plt.show()

if __name__ == "__main__":
    video_file = "data/skater_time_trial.mp4"
    model_weight = "skating_degradation_model.pth"
    
    print("Running continuous inference pipeline with automated baseline calibration...")
    results, loss_pairs, threshold_val, total_frame_count = run_pipeline(video_file, model_weight)
    
    print("\n--- Fatigue Detection Results ---")
    if results:
        for frame, timestamp, loss in results:
            print(f"⚠️ Fatigue Spike detected at Frame {frame} | Time: {timestamp:.2f}s | MSE Loss: {loss:.4f}")
        
        # Save detection results to CSV
        save_results_to_csv(results)
    else:
        print("✅ No fatigue spikes detected above threshold.")
        
    # Print biomechanical summary statistics
    print_summary(results, total_frame_count)
        
    # Plot and save the timeline chart
    if loss_pairs:
        plot_fatigue_timeline(loss_pairs, threshold_val)