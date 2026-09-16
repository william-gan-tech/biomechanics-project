import os
import pandas as pd
import numpy as np

def run_sensitivity_analysis():
    """
    Performs a sensitivity analysis by testing multiple threshold standard deviation 
    multipliers to evaluate robustness and lead-time stability.
    """
    results_path = os.path.join("outputs", "fatigue_results.csv")
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found. Run your pipeline first.")
        return

    df = pd.read_csv(results_path)
    
    if 'Anomaly_Score' not in df.columns:
        print("Dataset missing 'Anomaly_Score' column.")
        return

    scores = df['Anomaly_Score'].values
    fresh_baseline = scores[:30]
    mean_val = fresh_baseline.mean()
    std_val = fresh_baseline.std()

    # Test different standard deviation multipliers (sensitivity parameters)
    multipliers = [1.0, 1.5, 2.0, 2.5, 3.0]
    
    print("=== 🔬 Sensitivity Analysis & Ablation Study ===")
    print(f"{'Multiplier':<12} | {'First Flag Frame':<18} | {'Total Flags Triggered'}")
    print("-" * 55)

    summary_data = []

    for m in multipliers:
        threshold = mean_val + (m * std_val)
        flagged_indices = np.where(scores > threshold)[0]
        
        first_flag = flagged_indices[0] if len(flagged_indices) > 0 else -1
        total_flags = len(flagged_indices)
        
        print(f"mu + {m}*sigma    | Frame {first_flag:<12} | {total_flags} frames")
        
        summary_data.append({
            'Multiplier': m,
            'Threshold_Value': threshold,
            'First_Flag_Frame': first_flag,
            'Total_Flags': total_flags
        })

    # Save summary report
    summary_df = pd.DataFrame(summary_data)
    summary_output_path = os.path.join("outputs", "sensitivity_summary.csv")
    summary_df.to_csv(summary_output_path, index=False)
    print("-" * 55)
    print(f"Sensitivity report saved to: {summary_output_path}")

if __name__ == "__main__":
    run_sensitivity_analysis()