import numpy as np
import pandas as pd

def interpolate_camera_stream(target_timestamps: np.ndarray, source_df: pd.DataFrame, time_col: str = 'timestamp') -> pd.DataFrame:
    """
    Interpolates source camera dataframe coordinates to align with target timestamp grid.
    
    Args:
        target_timestamps (np.ndarray): The reference timeline array (e.g., 60fps stream).
        source_df (pd.DataFrame): Secondary camera dataframe to be synchronized (e.g., 30fps stream).
        time_col (str): Name of the timestamp column in the dataframes.
        
    Returns:
        pd.DataFrame: Synchronized dataframe matched to target timestamps.
    """
    if time_col not in source_df.columns:
        raise ValueError(f"Column '{time_col}' missing from source dataframe.")
        
    # Set time column as index for pandas interpolation
    indexed_df = source_df.set_index(time_col)
    
    # Reindex to the target timestamp grid, introducing NaN for missing gaps
    reindexed_df = indexed_df.reindex(index=np.union1d(indexed_df.index, target_timestamps))
    
    # Apply linear interpolation across spatial coordinates
    interpolated_df = reindexed_df.interpolate(method='linear')
    
    # Filter back strictly to the target timestamp grid
    synchronized_df = interpolated_df.reindex(index=target_timestamps).reset_index()
    
    return synchronized_df

def fuse_multi_view_data(df_cam_primary: pd.DataFrame, df_cam_secondary: pd.DataFrame, time_col: str = 'timestamp') -> pd.DataFrame:
    """
    Synchronizes and merges two multi-angle camera streams using the primary camera's timeline.
    """
    primary_times = df_cam_primary[time_col].values
    synced_secondary = interpolate_camera_stream(primary_times, df_cam_secondary, time_col)
    
    # Merge side-by-side using suffixes to separate camera views
    fused_df = pd.merge(df_cam_primary, synced_secondary, on=time_col, suffixes=('_cam1', '_cam2'))
    
    # Forward/backward fill any remaining edge-case NaNs from boundary matching
    fused_df = fused_df.ffill().bfill()
    
    return fused_df

if __name__ == "__main__":
    import pandas as pd
    # Quick mock test with mismatched framerates
    df_primary = pd.DataFrame({'timestamp': [0.0, 0.016, 0.033, 0.050], 'right_knee_cam1': [45.0, 46.2, 47.1, 48.0]})
    df_secondary = pd.DataFrame({'timestamp': [0.0, 0.033, 0.066], 'right_knee_cam2': [44.8, 46.9, 49.2]})

    fused = fuse_multi_view_data(df_primary, df_secondary)
    print("Fused Multi-View Stream Preview:")
    print(fused.head())