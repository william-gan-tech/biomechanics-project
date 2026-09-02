import numpy as np

def normalize_skeleton_sequence(keypoints, left_hip_idx=1, right_hip_idx=2, left_knee_idx=4, right_knee_idx=5):
    """
    Normalizes a sequence of skeletal keypoints for cross-subject invariance.
    
    Args:
        keypoints (np.ndarray): Shape (Frames, Joints, Dimensions), e.g., (N, 17, 3)
        
    Returns:
        np.ndarray: Normalized keypoints of the same shape.
    """
    normalized_sequence = np.copy(keypoints)
    
    for i in range(len(normalized_sequence)):
        frame = normalized_sequence[i]
        
        # 1. Hip-Centric Translation (Set mid-hip as origin [0, 0, 0])
        mid_hip = (frame[left_hip_idx] + frame[right_hip_idx]) / 2.0
        frame -= mid_hip
        
        # 2. Bone-Length Scaling (Normalize limbs using average femur length)
        left_femur = np.linalg.norm(frame[left_knee_idx] - frame[left_hip_idx])
        right_femur = np.linalg.norm(frame[right_knee_idx] - frame[right_hip_idx])
        avg_femur = (left_femur + right_femur) / 2.0
        
        if avg_femur > 0:
            frame /= avg_femur
            
        normalized_sequence[i] = frame
        
    return normalized_sequence