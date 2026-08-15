import numpy as np
from scipy.signal import butter, lfilter

def calculate_angle(a, b, c):
    """
    Calculates the joint angle given three 2D or 3D points.
    
    Parameters:
    a (array-like): First point coordinates [x, y] (e.g., Hip)
    b (array-like): Joint point coordinates [x, y] (e.g., Knee - vertex)
    c (array-like): Third point coordinates [x, y] (e.g., Ankle)
    
    Returns:
    float: Angle in degrees
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b
    bc = c - b
    
    dot_product = np.dot(ba, bc)
    magnitude_ba = np.linalg.norm(ba)
    magnitude_bc = np.linalg.norm(bc)
    
    if magnitude_ba == 0 or magnitude_bc == 0:
        return 0.0
        
    cosine_angle = dot_product / (magnitude_ba * magnitude_bc)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    angle_radians = np.arccos(cosine_angle)
    return float(np.degrees(angle_radians))


def butter_lowpass_filter(data, cutoff_freq, sample_rate, order=4):
    """
    Applies a Butterworth low-pass filter to smooth joint-angle time series.
    
    Parameters:
    data (array-like): Raw time-series angle data
    cutoff_freq (float): The cutoff frequency in Hz (e.g., 6.0 Hz)
    sample_rate (float): The sampling rate of the video/data in Hz (e.g., 30.0 FPS)
    order (int): The order of the filter
    
    Returns:
    numpy.ndarray: Filtered, smoothed angle data
    """
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = lfilter(b, a, data)
    return filtered_data


# Example test run to verify both work together
if __name__ == "__main__":
    # Test angle calculation
    hip = [100, 50]
    knee = [100, 150]
    ankle = [50, 200]
    angle = calculate_angle(hip, knee, ankle)
    print(f"Calculated Single Joint Angle: {angle:.2f} degrees")
    
    # Test filter on a dummy time series of angles
    time = np.linspace(0, 2, 60)
    noisy_angles = angle + np.random.normal(0, 2.0, 60)
    smoothed_angles = butter_lowpass_filter(noisy_angles, cutoff_freq=5.0, sample_rate=30.0)
    print(f"Successfully filtered {len(smoothed_angles)} frames of angle data!")

def create_sliding_windows(data, window_size=30, step_size=5):
    """
    Splits a 1D array of time-series angles into 2D sliding windows.
    
    Parameters:
    - data: List or numpy array of continuous angles (e.g., your knee angles).
    - window_size: The number of frames in each individual window chunk.
    - step_size: How many frames the window shifts forward each step.
    
    Returns:
    - A 2D numpy array of shape (num_windows, window_size).
    """
    windows = []
    
    # Slide the window across the data array
    for i in range(0, len(data) - window_size + 1, step_size):
        window = data[i : i + window_size]
        windows.append(window)
        
    return np.array(windows)