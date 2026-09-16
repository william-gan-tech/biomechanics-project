import os
import pandas as pd
import numpy as np

def calculate_deviation(test_skater_csv):
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    baseline_path = os.path.join(base_dir, 'data', 'sven_kramer_baseline.csv')
    
    # Load data
    test_df = pd.read_csv(test_skater_csv)
    baseline_df = pd.read_csv(baseline_path)
    
    # Ensure they have the same length for comparison (simple truncation for demo)
    min_len = min(len(test_df), len(baseline_df))
    test_angles = test_df['right_knee_angle'].iloc[:min_len].values
    base_angles = baseline_df['right_knee_angle'].iloc[:min_len].values
    
    # Calculate Mean Squared Error (MSE)
    mse = np.mean((test_angles - base_angles) ** 2)
    return mse

# Example usage:
# skater_file = 'data/fatigue_results.csv'
# score = calculate_deviation(skater_file)
# print(f"Kinematic Deviation Score: {score:.4f}")