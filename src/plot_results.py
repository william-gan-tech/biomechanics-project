import pandas as pd
import matplotlib.pyplot as plt

# Load the fatigue results CSV
df = pd.read_csv("fatigue_results.csv")

# Plot the anomaly scores over time (window indices)
plt.figure(figsize=(10, 5))
plt.plot(df["Window_Index"], df["Anomaly_Score"], label="Reconstruction Error (Anomaly Score)", color="crimson", linewidth=2)
plt.axhline(y=df["Anomaly_Score"].iloc[0], color="blue", linestyle="--", label="Fresh Baseline Form")

plt.title("Biomechanics Fatigue Analysis: Stride Form Deviation Over Time", fontsize=12, fontweight="bold")
plt.xlabel("Sliding Window Index (Time Progression)", fontsize=10)
plt.ylabel("Reconstruction Error (MSE)", fontsize=10)
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)

# Save the plot as an image file
plot_path = "fatigue_trend_plot.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
print(f"Successfully generated and saved plot to {plot_path}!")

# Display the plot window
plt.show()