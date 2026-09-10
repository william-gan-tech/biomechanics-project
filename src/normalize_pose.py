import numpy as np

def normalize_landmarks(landmarks):
    """Applies style-invariant bone-length scaling to a 3D MediaPipe landmark array 
    with safety checks to prevent distortion during occlusion.
    """
    if landmarks is None or len(landmarks) < 25:
        return None
        
    try:
        left_shoulder = np.array(landmarks[11][:3])
        right_shoulder = np.array(landmarks[12][:3])
        left_hip = np.array(landmarks[23][:3])
        right_hip = np.array(landmarks[24][:3])
        
        shoulder_midpoint = (left_shoulder + right_shoulder) / 2.0
        hip_midpoint = (left_hip + right_hip) / 2.0
        
        torso_length = np.linalg.norm(shoulder_midpoint - hip_midpoint)
        
        if torso_length < 0.05 or np.isnan(torso_length):
            return None
            
        scale_factor = torso_length
            
        normalized = []
        for lm in landmarks:
            coords = np.array(lm[:3])
            if np.any(np.isnan(coords)):
                return None
                
            centered = coords - hip_midpoint
            scaled = centered / scale_factor
            
            if len(lm) > 3:
                normalized.append([scaled[0], scaled[1], scaled[2], lm[3]])
            else:
                normalized.append([scaled[0], scaled[1], scaled[2]])
                
        return np.array(normalized, dtype=np.float32)
        
    except (IndexError, TypeError, ValueError):
        return None