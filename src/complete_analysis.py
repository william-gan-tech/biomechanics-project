import os
import pandas as pd
import matplotlib.pyplot as plt

# Define paths
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)
test_path = os.path.join(base_dir, 'data', 'fatigue_results.csv')

# Load data
df = pd.read_csv(test_path)

# Print Summary Report for Joint-Specific MSE Decomposition
print('=== BIOMECHANICAL FATIGUE & JOINT ERROR SUMMARY ===')
for col in ['Left_Knee_Error', 'Right_Knee_Error', 'Left_Hip_Error', 'Right_Hip_Error', 'Anomaly_Score']:
    max_val = df[col].max()
    max_window = df.loc[df[col].idxmax(), 'Window_Index']
    print(f'Peak {col}: {max_val:.4f} at Window Index {int(max_window)}')
print('==================================================')

# Plotting Joint-Specific Breakdown
plt.figure(figsize=(12, 6))
plt.plot(df['Window_Index'], df['Left_Knee_Error'], label='Left Knee Error', marker='o')
plt.plot(df['Window_Index'], df['Right_Knee_Error'], label='Right Knee Error', marker='s')
plt.plot(df['Window_Index'], df['Left_Hip_Error'], label='Left Hip Error', marker='^')
plt.plot(df['Window_Index'], df['Right_Hip_Error'], label='Right Hip Error', marker='d')
plt.plot(df['Window_Index'], df['Anomaly_Score'], label='Anomaly Score', color='black', linewidth=2, linestyle='--')

plt.title('Joint-Specific MSE Decomposition & Fatigue Progression', fontsize=14)
plt.xlabel('Window Index (Time Progression)', fontsize=12)
plt.ylabel('Reconstruction Error / MSE', fontsize=12)
plt.legend()
plt.grid(True)

output_path = os.path.join(base_dir, 'data', 'joint_specific_decomposition_plot.png')
plt.savefig(output_path, bbox_inches='tight')
print(f'Detailed joint breakdown plot successfully saved to: {output_path}')
plt.show()
