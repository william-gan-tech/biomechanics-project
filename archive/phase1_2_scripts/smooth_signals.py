import os
import pandas as pd
import numpy as np

def apply_smoothing():
    """
    Applies temporal sliding-window smoothing (Rolling Average and Exponential 
    Moving Average) to anomaly scores to eliminate single-frame noise.
    """
    results_path = os.path.join("outputs", "fatigue_results.csv")
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found. Run your pipeline first.")
        return

    df = pd.read_csv(results_path)
    
    if 'Anomaly_Score' not in df.columns:
        print("Dataset missing 'Anomaly_Score' column.")
        return

    raw_scores = df['Anomaly_Score'].values

    # 1. Apply Rolling Window Average (Moving Average with window size = 5)
    window_size = 5
    rolling_smoothed = pd.Series(raw_scores).rolling(window=window_size, min_periods=1).mean().values

    # 2. Apply Exponential Moving Average (EMA) for smoother long-term trend tracking
    ema_smoothed = pd.Series(raw_scores).ewm(span=5, adjust=False).mean().values

    df['Smoothed_Rolling'] = rolling_smoothed
    df['Smoothed_EMA'] = ema_smoothed

    print("=== 📈 Temporal Sliding-Window Smoothing ===")
    print(f"Original Signal Variance: {np.var(raw_scores):.4f}")
    print(f"Rolling Average Variance (Window={window_size}): {np.var(rolling_smoothed):.4f}")
    print(f"EMA Smoothed Variance: {np.var(ema_smoothed):.4f}")
    
    # Check threshold flag stability after smoothing
    fresh_baseline = rolling_smoothed[:30]
    threshold = fresh_baseline.mean() + (2 * fresh_baseline.std())
    
    smoothed_flags = np.where(rolling_smoothed > threshold)[0]
    first_smooth_flag = smoothed_flags[0] if len(smoothed_flags) > 0 else -1
    
    print(f"Stable Breakdown Detected at Frame: {first_smooth_flag} (Using Smoothed Signal)")

    # Save smoothed output
    smoothed_output_path = os.path.join("outputs", "smoothed_results.csv")
    df.to_csv(smoothed_output_path, index=False)
    print(f"Smoothed telemetry report saved to: {smoothed_output_path}")

if __name__ == "__main__":
    apply_smoothing()
    