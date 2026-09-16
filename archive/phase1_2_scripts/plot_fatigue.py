import os
import pandas as pd
import matplotlib.pyplot as plt

# Define paths
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)
test_path = os.path.join(base_dir, 'data', 'fatigue_results.csv')

# Load fatigue results
df = pd.read_csv(test_path)

# Clear any existing plots to prevent overlay/caching issues
plt.clf()
plt.close('all')

# Plotting the fatigue error progression
plt.figure(figsize=(10, 5))
plt.plot(df['Window_Index'], df['Anomaly_Score'], label='Anomaly Score', color='red', marker='o', linewidth=2)
plt.plot(df['Window_Index'], df['Right_Knee_Error'], label='Right Knee Error', color='orange', marker='s', linewidth=2)

plt.title('Skater Fatigue & Kinematic Error Progression Over Time', fontsize=14)
plt.xlabel('Window Index (Time Progression)', fontsize=12)
plt.ylabel('Error / Deviation Metric', fontsize=12)
plt.legend()
plt.grid(True)

# Save to a new unique image filename
output_plot_path = os.path.join(base_dir, 'data', 'new_fatigue_analysis_plot.png')
plt.savefig(output_plot_path, bbox_inches='tight')
print(f"Fresh fatigue plot successfully saved to: {output_plot_path}")

# Display the plot
plt.show()