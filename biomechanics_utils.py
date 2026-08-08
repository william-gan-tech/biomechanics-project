import numpy as np

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
    
    # Create vectors going away from the joint (vertex b)
    ba = a - b
    bc = c - b
    
    # Calculate dot product and magnitudes (lengths) of the vectors
    dot_product = np.dot(ba, bc)
    magnitude_ba = np.linalg.norm(ba)
    magnitude_bc = np.linalg.norm(bc)
    
    # Prevent division by zero errors
    if magnitude_ba == 0 or magnitude_bc == 0:
        return 0.0
        
    # Calculate cosine of the angle using the dot product formula
    cosine_angle = dot_product / (magnitude_ba * magnitude_bc)
    
    # Clip to handle any slight floating-point precision issues outside [-1.0, 1.0]
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    # Convert radian angle to degrees
    angle_radians = np.arccos(cosine_angle)
    angle_degrees = np.degrees(angle_radians)
    
    return float(angle_degrees)

# Example test run
if __name__ == "__main__":
    # Example coordinates for a bent knee
    hip = [100, 50]
    knee = [100, 150]
    ankle = [50, 200]
    
    angle = calculate_angle(hip, knee, ankle)
    print(f"Calculated Knee Angle: {angle:.2f} degrees")