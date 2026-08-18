import numpy as np
import pandas as pd

def label_pre_deceleration_windows(
    data: pd.DataFrame, 
    velocity_col: str, 
    joint_angle_cols: list, 
    window_size: int = 5, 
    drop_threshold: float = 0.03
):
    """
    Labels pre-deceleration windows in time-series biomechanics data.
    
    Parameters:
    - data: pandas DataFrame containing time-series tracking data.
    - velocity_col: Name of the column representing forward velocity or COM speed.
    - joint_angle_cols: List of column names representing joint angles.
    - window_size: Number of time steps/cycles to look back prior to deceleration.
    - drop_threshold: Percentage drop (e.g., 0.03 for 3%) defining measurable deceleration.
    
    Returns:
    - X: Array of shape (samples, window_size, num_features) containing joint trajectories.
    - y: Array of shape (samples,) containing binary labels (1 = approaching degradation, 0 = stable).
    """
    velocities = data[velocity_col].values
    joints = data[joint_angle_cols].values
    
    # Calculate rolling peak velocity to detect significant performance drops
    rolling_peak = pd.Series(velocities).rolling(window=30, min_periods=1, center=True).max()
    deceleration_mask = (rolling_peak - velocities) / rolling_peak > drop_threshold
    
    X = []
    y = []
    
    n_samples = len(data)
    for i in range(window_size, n_samples - window_size):
        # Check if deceleration occurs within the upcoming window horizon
        future_window = deceleration_mask[i : i + window_size]
        is_approaching_decel = int(any(future_window))
        
        # Extract past joint-angle trajectory window
        past_trajectory = joints[i - window_size : i]
        
        X.append(past_trajectory)
        y.append(is_approaching_decel)
        
    return np.array(X), np.array(y)

if __name__ == "__main__":
    # Example usage template:
    # df = pd.read_csv("data/processed_skating_data.csv")
    # joint_features = ['hip_angle', 'knee_angle', 'ankle_angle']
    # X, y = label_pre_deceleration_windows(df, velocity_col='com_velocity', joint_angle_cols=joint_features)
    # print(f"Prepared X shape: {X.shape}, y shape: {y.shape}")
    pass