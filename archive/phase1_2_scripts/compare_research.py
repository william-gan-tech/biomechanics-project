import pandas as pd
import matplotlib.pyplot as plt

# 1. Load both datasets
start_df = pd.read_csv("data/angles_20_to_50.csv")
end_df = pd.read_csv("data/angles_345_to_414.csv")

# Normalize the frame index so both start at 0 for an easy side-by-side comparison
start_df['normalized_frame'] = start_df['frame'] - start_df['frame'].min()
end_df['normalized_frame'] = end_df['frame'] - end_df['frame'].min()

# 2. Define columns to plot
columns_to_plot = [
    'right_knee_angle', 'left_knee_angle', 
    'right_hip_x', 'right_shoulder_x', 
    'right_knee_filtered'
]

# Create a multi-row figure to compare all variables cleanly
fig, axes = plt.subplots(len(columns_to_plot), 2, figsize=(14, 12), sharex='col')

for i, col in enumerate(columns_to_plot):
    # Start Segment (Fresh)
    if col in start_df.columns:
        axes[i, 0].plot(start_df['normalized_frame'], start_df[col], color='blue', label='Fresh')
    axes[i, 0].set_ylabel(col, fontsize=9)
    axes[i, 0].grid(True, linestyle='--', alpha=0.5)
    if i == 0:
        axes[i, 0].set_title("Start Segment: Fresh State (Normalized)")

    # End Segment (Fatigued)
    if col in end_df.columns:
        axes[i, 1].plot(end_df['normalized_frame'], end_df[col], color='red', label='Fatigued')
    axes[i, 1].grid(True, linestyle='--', alpha=0.5)
    if i == 0:
        axes[i, 1].set_title("End Segment: Fatigued State (Normalized)")

axes[-1, 0].set_xlabel("Frames from Start of Clip")
axes[-1, 1].set_xlabel("Frames from Start of Clip")

plt.suptitle("Comprehensive Biomechanical Feature Comparison: Fresh vs. Fatigued", fontsize=14)
plt.tight_layout()

output_path = "data/comprehensive_comparison_plot.png"
plt.savefig(output_path)
print(f"✅ Success! Multi-panel comparison chart saved to '{output_path}'")