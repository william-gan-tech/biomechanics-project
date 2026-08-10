import pandas as pd
import numpy as np

# Use absolute path to point directly to your file
df = pd.read_csv(r'C:\Users\qgan2\OneDrive\Desktop\Research - biomechanics_project\data\extracted_multivariate_angles.csv')

# Simulate form breakdown in the last 100 frames by adding noise/deviation
end_idx = len(df) - 100
df.loc[end_idx:, 'right_knee_angle'] += np.linspace(0, 30, 100)

# Save it back
df.to_csv('data/skater_b_multivariate_angles.csv', index=False)
print("Successfully generated synthetic fatigue data for Skater B!")