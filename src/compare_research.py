import pandas as pd
import matplotlib.pyplot as plt

# 1. Load both datasets correctly
try:
    start_df = pd.read_csv("data/angles_20_to_50.csv")
    print("✅ Loaded start segment data successfully.")
except Exception as e:
    print(f"❌ Error loading start segment data: {e}")

try:
    end_df = pd.read_csv("data/angles_345_to_414.csv")
    print("✅ Loaded end segment data successfully.")
except Exception as e:
    print(f"❌ Error loading end segment data: {e}")

# 2. Set up side-by-side comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

# Plot Start Segment (Fresh) - Right and Left Knee
ax1.plot(start_df['frame'], start_df['right_knee_angle'], label='Right Knee (Fresh)', color='blue')
ax1.plot(start_df['frame'], start_df['left_knee_angle'], label='Left Knee (Fresh)', color='deepskyblue', linestyle='--')
ax1.set_title("Start Segment: Fresh State (20s - 50s)")
ax1.set_xlabel("Frame Index")
ax1.set_ylabel("Knee Angle (Degrees)")
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend()

# Plot End Segment (Fatigued) - Right and Left Knee
ax2.plot(end_df['frame'], end_df['right_knee_angle'], label='Right Knee (Fatigued)', color='red')
ax2.plot(end_df['frame'], end_df['left_knee_angle'], label='Left Knee (Fatigued)', color='orange', linestyle='--')
ax2.set_title("End Segment: Fatigued State (3:45 - 4:14)")
ax2.set_xlabel("Frame Index")
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend()

plt.suptitle("Research Comparison: Fresh vs. Fatigued Joint Trajectories", fontsize=14)
plt.tight_layout()

# 3. Save the new plot
output_path = "data/research_comparison_plot.png"
plt.savefig(output_path)
print(f"✅ Success! Updated comparison chart saved to '{output_path}'")