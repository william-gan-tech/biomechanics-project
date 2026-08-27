import numpy as np
import pandas as pd
from scipy.signal import find_peaks

def segment_skating_strides(df_features, signal_col="right_knee_angle", distance_threshold=25, prominence=5.0):
    """
    Automatically segments a continuous skating feature dataframe into individual 
    stride cycles based on cyclic peaks in the specified joint signal.
    """
    if signal_col not in df_features.columns:
        return []
    
    signal_values = df_features[signal_col].values
    
    # Find peaks representing the apex of each stride cycle
    peaks, _ = find_peaks(signal_values, distance=distance_threshold, prominence=prominence)
    
    stride_cycles = []
    for i in range(len(peaks) - 1):
        start_idx = peaks[i]
        end_idx = peaks[i+1]
        
        # Slice the dataframe for this specific stride
        stride_df = df_features.iloc[start_idx:end_idx].copy()
        stride_cycles.append({
            "stride_id": i + 1,
            "start_frame": int(stride_df["frame"].iloc[0]),
            "end_frame": int(stride_df["frame"].iloc[-1]),
            "data": stride_df
        })
        
    return stride_cycles