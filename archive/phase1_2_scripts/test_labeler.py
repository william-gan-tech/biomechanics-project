import sys
import os

# Ensure the 'src' directory is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from label_pre_deceleration import label_pre_deceleration_windows
import numpy as np
import pandas as pd

def test_labeler():
    # 1. Create dummy time-series data simulating speed skating strides
    n_steps = 200
    time = np.arange(n_steps)
    
    # Simulate velocity with a steady phase followed by a drop (deceleration)
    velocity = np.ones(n_steps) * 10.0
    velocity[150:] = 9.0  # Drops by 10% after step 150
    
    # Simulate joint angles (e.g., knee and ankle angles)
    knee_angle = np.sin(time * 0.1) * 45 + 90
    ankle_angle = np.cos(time * 0.1) * 20 + 30
    
    df = pd.DataFrame({
        'com_velocity': velocity,
        'knee_angle': knee_angle,
        'ankle_angle': ankle_angle
    })
    
    joint_cols = ['knee_angle', 'ankle_angle']
    
    # 2. Run the labeling function
    X, y = label_pre_deceleration_windows(
        data=df, 
        velocity_col='com_velocity', 
        joint_angle_cols=joint_cols, 
        window_size=10, 
        drop_threshold=0.03
    )
    
    # 3. Print verification results
    print("Test passed successfully!")
    print(f"X shape (Expected: (samples, window_size, features)): {X.shape}")
    print(f"y shape (Expected: (samples,)): {y.shape}")
    print(f"Number of positive labels (approaching deceleration): {np.sum(y)}")

if __name__ == "__main__":
    test_labeler()