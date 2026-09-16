import os
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

def generate_stride_report(results_csv_path):
    """
    Computes the average reconstruction error for each individual stride cycle 
    to track fatigue progression stride-by-stride.
    """
    if not os.path.exists(results_csv_path):
        print(f"Error: {results_csv_path} not found. Run your demo script first.")
        return

    df = pd.read_csv(results_csv_path)
    
    if 'Anomaly_Score' not in df.columns:
        print("Dataset missing 'Anomaly_Score' column.")
        return

    signal = df['Hip_Velocity'] if 'Hip_Velocity' in df.columns else df['Anomaly_Score']
    
    # 1. Detect stride peaks (same logic as before)
    peaks, _ = find_peaks(signal, distance=20, prominence=0.5)

    if len(peaks) < 2:
        print("Not enough stride peaks detected to analyze cycles.")
        return

    print("=== 📊 Stride-by-Stride Fatigue Report ===")
    print(f"{'Stride #':<10} | {'Frame Range':<15} | {'Avg Anomaly Score':<18} | {'Status'}")
    print("-" * 65)

    # 2. Iterate through each stride cycle and calculate metrics
    # Compute threshold strictly from the initial warm-up/fresh period (first 30 frames)
    fresh_data = df['Anomaly_Score'].iloc[:30]
    baseline_threshold = fresh_data.mean() + (2 * fresh_data.std())
    
    for i in range(len(peaks) - 1):
        start_frame = peaks[i]
        end_frame = peaks[i+1]
        
        # Slice the dataframe for this specific stride
        stride_data = df.iloc[start_frame:end_frame]
        avg_score = stride_data['Anomaly_Score'].mean()
        
        status = "⚠️ FATIGUE / DRIFT" if avg_score > baseline_threshold else "🟢 Normal Form"
        
        print(f"Stride {i+1:<4} | [{start_frame:3d} - {end_frame:3d}]     | {avg_score:.4f}           | {status}")

    # Handle the final stride up to the end of the video
    if peaks[-1] < len(df) - 1:
        final_stride = df.iloc[peaks[-1]:]
        avg_score = final_stride['Anomaly_Score'].mean()
        status = "⚠️ FATIGUE / DRIFT" if avg_score > baseline_threshold else "🟢 Normal Form"
        print(f"Stride {len(peaks):<4} | [{peaks[-1]:3d} - {len(df)-1:3d}]     | {avg_score:.4f}           | {status}")

if __name__ == "__main__":
    csv_path = os.path.join("outputs", "fatigue_results.csv")
    generate_stride_report(csv_path)