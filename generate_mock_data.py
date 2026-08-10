import os
import pandas as pd
import numpy as np

# Use absolute paths so Python never creates phantom folders
base_dir = r'C:\Users\qgan2\OneDrive\Desktop\Research - biomechanics_project'
data_dir = os.path.join(base_dir, 'data')

source_file = os.path.join(data_dir, 'extracted_multivariate_angles.csv')
target_file = os.path.join(data_dir, 'skater_b_multivariate_angles.csv')

# Load and process using absolute paths
df = pd.read_csv(source_file)
end_idx = len(df) - 100
df.loc[end_idx:, 'right_knee_angle'] += np.linspace(0, 30, 100)
df.to_csv(target_file, index=False)

print("Successfully generated synthetic fatigue data for Skater B!")