import pandas as pd
import numpy as np

# Load your original working file
df = pd.read_csv('data/extracted_multivariate_angles.csv')

# Simulate minor changes
df['right_knee_angle'] += np.random.normal(0, 1, len(df))

# Save it directly into the data folder
df.to_csv('data/skater_b_multivariate_angles.csv', index=False)
print("Mock skater_b file generated successfully!")