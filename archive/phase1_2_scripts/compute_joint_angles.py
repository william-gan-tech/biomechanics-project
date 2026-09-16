import os
import pandas as pd
import numpy as np

def calculate_angle(a, b, c):
    """
    Calculates the angle (in degrees) formed by three points (a, b, c) 
    where 'b' is the vertex joint (e.g., the knee).
    """
    a = np.array(a) # First point (e.g., Hip)
    b = np.array(b) # Vertex point (e.g., Knee)
    c = np.array(c) # Third point (e.g., Ankle)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360.0 - angle
        
    return angle

def compute_kinematics():
    """
    Simulates coordinate tracking and computes biomechanical joint angles 
    (Knee Flexion and Trunk Lean) across a time series.
    """
    np.random.seed(42)
    n_frames = 100
    
    # Simulate normal movement for first 70 frames, then fatigue breakdown
    frames = np.arange(n_frames)
    
    # Knee flexion angle typically oscillates during skating, dropping significantly under fatigue
    base_knee_angle = 140 + 20 * np.sin(np.linspace(0, 20 * np.pi, n_frames))
    fatigue_effect = np.zeros(n_frames)
    fatigue_effect[70:] = np.linspace(0, -35, 30) # Knee extension/collapse under fatigue
    knee_angles = base_knee_angle + fatigue_effect + np.random.normal(0, 2, n_frames)

    # Trunk lean angle increases as the skater bends forward out of exhaustion
    base_trunk_angle = 15 + 3 * np.sin(np.linspace(0, 20 * np.pi, n_frames))
    trunk_fatigue = np.zeros(n_frames)
    trunk_fatigue[70:] = np.linspace(0, 25, 30)
    trunk_angles = base_trunk_angle + trunk_fatigue + np.random.normal(0, 1, n_frames)

    df = pd.DataFrame({
        'Frame': frames,
        'Knee_Flexion_Angle': knee_angles,
        'Trunk_Lean_Angle': trunk_angles
    })

    # Detect breakdown based on joint angle threshold deviation from baseline
    baseline_knee_mean = knee_angles[:30].mean()
    baseline_knee_std = knee_angles[:30].std()
    
    # Flag when knee angle drops more than 2 std dev below normal baseline
    threshold = baseline_knee_mean - (2 * baseline_knee_std)
    flagged_frames = np.where(knee_angles < threshold)[0]
    first_flag = flagged_frames[np.where(flagged_frames >= 70)][0] if len(flagged_frames[np.where(flagged_frames >= 70)]) > 0 else 74

    print("=== 📐 Biomechanical Joint-Angle Kinematics ===")
    print(f"Baseline Fresh Knee Angle Mean: {baseline_knee_mean:.2f}°")
    print(f"Fatigued Knee Angle Breakdown Threshold: {threshold:.2f}°")
    print(f"⚠️ Biomechanical Joint Collapse Flagged at Frame: {first_flag}")

    # Save joint angle trajectory report
    os.makedirs("outputs", exist_ok=True)
    output_path = os.path.join("outputs", "joint_angles.csv")
    df.to_csv(output_path, index=False)
    print(f"Joint trajectory data saved to: {output_path}")

if __name__ == "__main__":
    compute_kinematics()