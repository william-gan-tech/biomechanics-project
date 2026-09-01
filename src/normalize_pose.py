import numpy as np

def normalize_landmarks(landmarks):
    # Center landmarks relative to the hip center (landmark index example)
    hip_center = (landmarks[23] + landmarks[24]) / 2.0
    centered = landmarks - hip_center

    # Scale by torso length to achieve dimensionless proportions
    torso_length = np.linalg.norm(centered[11] - centered[23]) # Shoulder to hip
    if torso_length > 0:
        normalized = centered / torso_length
    else:
        normalized = centered
    return normalized