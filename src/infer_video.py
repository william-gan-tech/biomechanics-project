import cv2
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt

def calculate_angle(a, b, c):
    """Computes joint angle given three points (a: proximal, b: joint, c: distal)."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return angle

def process_video_inference(video_path, onnx_model_path="skating_model.onnx", window_size=30):
    # 1. Initialize ONNX Runtime session
    ort_session = ort.InferenceSession(onnx_model_path)
    input_name = ort_session.get_inputs()[0].name
    
    # 2. Initialize MediaPipe Pose
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    cap = cv2.VideoCapture(video_path)
    features_list = []
    
    print("Extracting features from video frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert frame to RGB for MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Example keypoints for lower body / torso (using left side as proxy or averaging)
            # 11: left shoulder, 23: left hip, 25: left knee, 27: left ankle
            try:
                shoulder = [landmarks[11].x, landmarks[11].y]
                hip = [landmarks[23].x, landmarks[23].y]
                knee = [landmarks[25].x, landmarks[25].y]
                ankle = [landmarks[27].x, landmarks[27].y]
                
                # Compute 4 features matching your model specs
                hip_angle = calculate_angle(shoulder, hip, knee)
                knee_angle = calculate_angle(hip, knee, ankle)
                ankle_angle = calculate_angle(knee, ankle, [ankle[0], ankle[1] - 0.1])
                torso_lean = calculate_angle([shoulder[0], shoulder[1] - 0.1], shoulder, hip)
                
                features_list.append([hip_angle, knee_angle, ankle_angle, torso_lean])
            except Exception:
                continue
                
    cap.release()
    features = np.array(features_list, dtype=np.float32)
    
    if len(features) < window_size:
        print("Error: Video is too short or pose detection failed to capture enough frames.")
        return

    # 3. Create sliding windows and run ONNX inference
    reconstruction_errors = []
    for i in range(len(features) - window_size):
        window = features[i:i+window_size]
        # Shape to [1, window_size, n_features]
        input_tensor = np.expand_dims(window, axis=0)
        
        # Run inference
        outputs = ort_session.run(None, {input_name: input_tensor})
        reconstruction = outputs[0]
        
        # Calculate Mean Squared Error (MSE) as anomaly score
        mse = np.mean(np.power(window - reconstruction[0], 2))
        reconstruction_errors.append(mse)
        
    # 4. Plot results
    plt.figure(figsize=(10, 4))
    plt.plot(reconstruction_errors, label='Reconstruction Error (Fatigue Score)', color='purple')
    plt.axhline(y=0.0450, color='red', linestyle='--', label='Fatigue Threshold')
    plt.title('Biomechanical Degradation Timeline Over Video')
    plt.xlabel('Window Sequence Index')
    plt.ylabel('Error Score')
    plt.legend()
    plt.show()
    print("Inference complete! Anomaly timeline generated.")

if __name__ == "__main__":
    # Replace with your video file path
    process_video_inference("../path_to_video.mp4")