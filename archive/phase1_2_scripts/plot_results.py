import os
import pandas as pd
import matplotlib.pyplot as plt

# Load the fatigue results CSV from the outputs folder
output_path = os.path.join("outputs", "fatigue_results.csv")
if not os.path.exists(output_path):
    print(f"Error: {output_path} not found. Please run your pipeline first.")
else:
    df = pd.read_csv(output_path)

    # Plot the anomaly scores over time (window indices)
    plt.figure(figsize=(10, 5))
    
    # Check if 'Window_Index' exists; otherwise fallback to DataFrame index or 'Frame'
    x_col = "Window_Index" if "Window_Index" in df.columns else df.index
    
    plt.plot(df[x_col], df["Anomaly_Score"], label="Reconstruction Error (Anomaly Score)", color="crimson", linewidth=2)
    plt.axhline(y=df["Anomaly_Score"].iloc[0], color="blue", linestyle="--", label="Fresh Baseline Form")

    plt.title("Biomechanics Fatigue Analysis: Stride Form Deviation Over Time", fontsize=12, fontweight="bold")
    plt.xlabel("Time Progression", fontsize=10)
    plt.ylabel("Reconstruction Error (MSE)", fontsize=10)
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)

    # Save the plot as an image file inside outputs/
    plot_path = os.path.join("outputs", "fatigue_trend_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Successfully generated and saved plot to {plot_path}!")

    # Display the plot window
    plt.show()