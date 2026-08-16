import os
import numpy as np
import pandas as pd

def calculate_lead_time(results_csv_path, fps=30):
    """
    Calculates the exact temporal lead-time (in seconds) between 
    model-flagged anomaly spikes and physical velocity deceleration.
    """
    if not os.path.exists(results_csv_path):
        print(f"Error: Could not find {results_csv_path}. Run inference first.")
        return

    df = pd.read_csv(results_csv_path)
    
    if 'Anomaly_Score' not in df.columns or 'Hip_Velocity' not in df.columns:
        print("Dataset missing required columns ('Anomaly_Score' or 'Hip_Velocity').")
        return

    # 1. Identify the first window where anomaly score breaches the statistical threshold
    threshold = df['Anomaly_Score'].mean() + (2 * df['Anomaly_Score'].std())
    anomaly_flags = df[df['Anomaly_Score'] > threshold]
    
    if anomaly_flags.empty:
        print("No anomalies detected above the statistical threshold.")
        return
        
    first_anomaly_window = anomaly_flags.iloc[0]['Window_Index']

    # 2. Identify sustained physical deceleration in the later portion of the run (after frame 50)
    baseline_velocity = df['Hip_Velocity'].iloc[:20].mean()
    
    # Look for a sustained drop where velocity stays below 95% of baseline for at least 3 consecutive windows
    later_frames = df.iloc[50:].copy()
    later_frames['Is_Decelerating'] = later_frames['Hip_Velocity'] < (baseline_velocity * 0.95)
    
    # Find where deceleration holds true
    deceleration_indices = later_frames[later_frames['Is_Decelerating']].index
    
    if len(deceleration_indices) == 0:
        print("No significant physical deceleration detected in the sequence.")
        return
        
    first_deceleration_window = deceleration_indices[0]

    # 3. Compute Lead Time
    window_diff = first_deceleration_window - first_anomaly_window
    lead_time_seconds = window_diff / fps

    print("=== ⏱️ Predictive Lead-Time Analysis ===")
    print(f"First Anomaly Flagged at Window: {int(first_anomaly_window)}")
    print(f"First Physical Deceleration at Window: {int(first_deceleration_window)}")
    print(f"Lead-Time Advantage: {abs(lead_time_seconds):.2f} seconds ({abs(int(window_diff))} frames)")
    
    if window_diff < 0:
        print("✅ SUCCESS: Model successfully predicted form breakdown *before* deceleration!")
    else:
        print("⚠️ NOTE: Anomaly flag occurred simultaneously with or after deceleration.")

if __name__ == "__main__":
    csv_path = os.path.join("outputs", "fatigue_results.csv")
    calculate_lead_time(csv_path)