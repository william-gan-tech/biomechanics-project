import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def analyze_stride_degradation():
    print("=" * 70)
    print(" 📊 ADVANCED STRIDE-BY-STRIDE DEGRADATION & FATIGUE ANALYSIS")
    print("=" * 70)
    
    # 1. Setup clean relative paths matching your root workspace
    output_dir = "outputs"
    results_csv_path = os.path.join(output_dir, "fatigue_results.csv")
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(results_csv_path):
        print(f"Error: Could not find {results_csv_path}. Run your model inference pipeline first.")
        return

    df = pd.read_csv(results_csv_path)
    
    if 'Anomaly_Score' not in df.columns:
        print("Dataset missing 'Anomaly_Score' column.")
        return

    # Use Hip_Velocity or Anomaly_Score proxy for peak stride segmentation
    target_signal = df['Hip_Velocity'].values if 'Hip_Velocity' in df.columns else df['Anomaly_Score'].values
    
    # 2. Detect individual stride peaks
    peaks, _ = find_peaks(target_signal, distance=20, prominence=0.5)

    if len(peaks) < 3:
        print("⚠️ Warning: Not enough stride peaks detected to analyze cycles.")
        return

    print(f"✅ Successfully segmented {len(peaks)} stride cycles.")

    # 3. Calculate baseline threshold from fresh period (first 30 frames)
    fresh_data = df['Anomaly_Score'].iloc[:min(30, len(df))]
    baseline_threshold = fresh_data.mean() + (2 * fresh_data.std())

    # 4. Iterate through each stride and compute metrics
    stride_records = []
    for i in range(len(peaks) - 1):
        start_frame = peaks[i]
        end_frame = peaks[i+1]
        
        stride_data = df.iloc[start_frame:end_frame]
        avg_anomaly = stride_data['Anomaly_Score'].mean()
        max_anomaly = stride_data['Anomaly_Score'].max()
        stride_duration = end_frame - start_frame
        
        is_fatigued = avg_anomaly > baseline_threshold
        
        stride_records.append({
            'Stride_Index': i + 1,
            'Start_Frame': start_frame,
            'End_Frame': end_frame,
            'Duration': stride_duration,
            'Avg_Anomaly_Score': avg_anomaly,
            'Max_Anomaly_Score': max_anomaly,
            'Is_Fatigued': is_fatigued
        })

    metrics_df = pd.DataFrame(stride_records)
    
    # Calculate rolling trend line for smoothing
    metrics_df['Anomaly_Trend'] = metrics_df['Avg_Anomaly_Score'].rolling(window=3, min_periods=1).mean()

    # 5. Plot Stride Degradation Trajectory
    plt.figure(figsize=(10, 5))
    plt.plot(metrics_df['Stride_Index'], metrics_df['Avg_Anomaly_Score'], marker='o', alpha=0.4, label='Stride Avg Anomaly Score', color='blue')
    plt.plot(metrics_df['Stride_Index'], metrics_df['Anomaly_Trend'], linewidth=2, label='Degradation Trend (Moving Avg)', color='darkblue')
    plt.axhline(y=baseline_threshold, color='r', linestyle='--', label='Fresh Baseline Threshold (mu + 2sigma)')
    
    plt.title('Stride-by-Stride Biomechanical Degradation Progression')
    plt.xlabel('Chronological Stride Index')
    plt.ylabel('Reconstruction Error (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(output_dir, 'stride_degradation_trajectory.png')
    plt.savefig(plot_path)
    plt.close()
    
    # Save CSV report
    csv_out_path = os.path.join(output_dir, 'stride_degradation_report.csv')
    metrics_df.to_csv(csv_out_path, index=False)
    
    print(f"\n📊 Stride Degradation Analysis Complete!")
    print(f"💾 Progression plot saved to '{plot_path}'")
    print(f"💾 Structured CSV metrics saved to '{csv_out_path}'")
    print("=" * 70)

if __name__ == "__main__":
    analyze_stride_degradation()