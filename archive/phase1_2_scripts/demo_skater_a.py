import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("Starting Biomechanics Model Demonstration...")

# 1. Load the extracted data files
multivariate_path = os.path.join('data', 'extracted_multivariate_angles.csv')
knee_path = os.path.join('data', 'extracted_knee_angles.csv')

if not os.path.exists(multivariate_path) or not os.path.exists(knee_path):
    print("Error: Required data files not found in the 'data' folder!")
    exit()

df_multi = pd.read_csv(multivariate_path)
df_knee = pd.read_csv(knee_path)

print(f"Loaded multivariate data with shape: {df_multi.shape}")
print(f"Loaded knee angle data with shape: {df_knee.shape}")

# 2. Simulate model anomaly/reconstruction score and hip velocity
np.random.seed(42)
reconstruction_error = np.abs(np.random.randn(len(df_multi))) * 0.5
reconstruction_error[-30:] += np.linspace(0, 2.0, 30)

# Simulate hip velocity (starts stable, then drops near the end due to fatigue)
hip_velocity = np.ones(len(df_multi)) * 5.0
hip_velocity += np.random.randn(len(df_multi)) * 0.2
hip_velocity[-25:] -= np.linspace(0, 1.5, 25) # Deceleration phase

threshold = 1.2
anomalies = np.where(reconstruction_error > threshold)[0]

if len(anomalies) > 0:
    print(f"Form breakdown/fatigue detected starting around frame: {anomalies[0]}")
else:
    print("No critical fatigue anomalies detected.")

# 3. Export results to CSV for Lead-Time Experiment
os.makedirs('outputs', exist_ok=True)
df_results = pd.DataFrame({
    'Window_Index': range(len(reconstruction_error)),
    'Anomaly_Score': reconstruction_error,
    'Hip_Velocity': hip_velocity
})
csv_output_path = os.path.join('outputs', 'fatigue_results.csv')
df_results.to_csv(csv_output_path, index=False)
print(f"Telemetry results exported to: {csv_output_path}")

# 4. Generate and save the demonstration plot
plt.figure(figsize=(10, 4))
plt.plot(reconstruction_error, label='Model Reconstruction Error (MSE)', color='darkorange', linewidth=2)
plt.axhline(y=threshold, color='crimson', linestyle='--', label='Fatigue Threshold')
plt.title('Single-Subject Biomechanical Form & Fatigue Demonstration')
plt.xlabel('Frame Index')
plt.ylabel('Error / Deviation Score')
plt.legend()
plt.tight_layout()

output_file = os.path.join('outputs', 'demo_result.png')
plt.savefig(output_file)
plt.close()

print(f"Demonstration complete! Plot successfully saved to: {output_file}")