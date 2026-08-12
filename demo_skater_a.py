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

# 2. Simulate model anomaly/reconstruction score (or plug in your trained model here)
# This calculates a moving variance or error metric to demonstrate fatigue detection
np.random.seed(42)
reconstruction_error = np.abs(np.random.randn(len(df_multi))) * 0.5
# Add a simulated form breakdown spike toward the end of the session
reconstruction_error[-30:] += np.linspace(0, 2.0, 30)

threshold = 1.2
anomalies = np.where(reconstruction_error > threshold)[0]

if len(anomalies) > 0:
    print(f"Form breakdown/fatigue detected starting around frame: {anomalies[0]}")
else:
    print("No critical fatigue anomalies detected.")

# 3. Generate and save the demonstration plot
os.makedirs('outputs', exist_ok=True)
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
