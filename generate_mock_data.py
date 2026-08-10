import os
import pandas as pd
import numpy as np

# Define the absolute path to your data directory and files
base_dir = r'C:\Users\qgan2\OneDrive\Desktop\Research - biomechanics_project'
data_dir = os.path.join(base_dir, 'data')

# Automatically create the 'data' folder if it doesn't exist on disk
os.makedirs(data_dir, exist_ok=True)

source_file = os.path.join(data_dir, 'extracted_multivariate_angles.csv')
target_file = os.path.join(data_dir, 'skater_b_multivariate_angles.csv')

try:
    # Load your original working file
    df = pd.read_csv(source_file)
    
    # Simulate form breakdown in the last 100 frames by adding noise/deviation
    end_idx = len(df) - 100
    df.loc[end_idx:, 'right_knee_angle'] += np.linspace(0, 30, 100)

    # Save it back as skater_b
    df.to_csv(target_file, index=False)
    print("Successfully generated synthetic fatigue data for Skater B!")

except FileNotFoundError:
    print(f"CRITICAL: Could not find baseline file at: {source_file}")
    print("Please make sure 'extracted_multivariate_angles.csv' is placed inside your 'data' folder.")