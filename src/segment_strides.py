import os
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

def segment_strides(data_csv_path):
    """
    Automatically segments continuous joint-angle or coordinate data 
    into individual stride cycles using peak detection.
    """
    if not os.path.exists(data_csv_path):
        print(f"Error: {data_csv_path} not found.")
        return

    df = pd.read_csv(data_csv_path)
    
    # Assuming we track vertical hip or ankle position (e.g., 'Right_Ankle_Y' or similar)
    # Let's check for a suitable column or use multivariate error as a proxy if needed
    target_col = 'Hip_Velocity' if 'Hip_Velocity' in df.columns else df.columns[1]
    
    signal = df[target_col].values
    
    # Find peaks representing the apex of each stride cycle
    # distance=20 ensures peaks are spaced out realistically for a skating stride
    peaks, _ = find_peaks(signal, distance=20, prominence=0.5)

    print(f"=== ⛸️ Stride Segmentation Analysis ===")
    print(f"Total frames processed: {len(df)}")
    print(f"Detected {len(peaks)} individual stride cycles.")
    print(f"Stride boundary frame indices: {peaks[:5]}... (showing first 5)")

    # You can now slice your dataframe between peaks to evaluate each stride independently!
    return peaks

if __name__ == "__main__":
    csv_path = os.path.join("outputs", "fatigue_results.csv")
    segment_strides(csv_path)