import os
import pandas as pd
import numpy as np

# 1. Setup path routing
base_dir = r'C:\Users\qgan2\OneDrive\Desktop\Research - biomechanics_project\biomechanics-project'
data_dir = os.path.join(base_dir, 'data')

# Ensure the data directory exists
os.makedirs(data_dir, exist_ok=True)

target_file = os.path.join(data_dir, 'skater_b_multivariate_angles.csv')

# 2. Generate mock multivariate joint angle data from scratch (No external read required!)
np.random.seed(42)
data = np.random.normal(loc=45.0, scale=3.0, size=(200, 3))
df = pd.DataFrame(data, columns=['right_knee_angle', 'left_knee_angle', 'hip_angle'])

# 3. Simulate fatigue drift near the end
end_idx = len(df) - 100
df.loc[end_idx:, 'right_knee_angle'] += np.linspace(0, 30, 100)

# 4. Save directly to your data folder
df.to_csv(target_file, index=False)
print(f"Successfully generated and saved: {target_file}")