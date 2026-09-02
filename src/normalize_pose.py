import numpy as np

def normalize_landmarks(landmarks):
    """Applies style-invariant bone-length scaling to a 3D MediaPipe landmark array.
    Scales coordinates relative to torso length (hip-to-shoulder distance) to remove
    anatomical size bias across diverse athletes.
    """
    if landmarks is None or len(landmarks) < 33:
        return None
        
    # MediaPipe indices: Shoulders (11, 12), Hips (23, 24)
    left_shoulder = np.array(landmarks[11][:3])
    right_shoulder = np.array(landmarks[12][:3])
    left_hip = np.array(landmarks[23][:3])
    right_hip = np.array(landmarks[24][:3])
    
    shoulder_midpoint = (left_shoulder + right_shoulder) / 2.0
    hip_midpoint = (left_hip + right_hip) / 2.0
    
    # Compute torso reference scale factor
    torso_length = np.linalg.norm(shoulder_midpoint - hip_midpoint)
    scale_factor = torso_length if torso_length > 1e-6 else 1.0
        
    normalized = []
    for lm in landmarks:
        coords = np.array(lm[:3])
        centered = coords - hip_midpoint
        scaled = centered / scale_factor
        
        if len(lm) > 3:
            normalized.append([scaled[0], scaled[1], scaled[2], lm[3]])
        else:
            normalized.append([scaled[0], scaled[1], scaled[2]])
            
    return np.array(normalized)