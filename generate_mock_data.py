import pandas as pd
import numpy as np

# Load your duplicated file (make sure you manually copied and renamed it first!)
df = pd.read_csv('data/skater_b_multivariate_angles.csv')

# Simulate form breakdown in the last 100 frames by adding noise/deviation
end_idx = len(df) - 100
df.loc[end_idx:, 'right_knee_angle'] += np.linspace(0, 30, 100)

# Save it back
df.to_csv('data/skater_b_multivariate_angles.csv', index=False)
print("Successfully generated synthetic fatigue data for Skater B!")