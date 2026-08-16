import os
import pandas as pd
import numpy as np

def create_sliding_sequences(sequence_length=30):
    """
    Takes joint-angle trajectories and slices them into sliding temporal 
    windows for sequence-to-sequence deep learning models (LSTM/TCN).
    """
    input_path = os.path.join("outputs", "joint_angles.csv")
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Run your joint angle script first.")
        return

    df = pd.read_csv(input_path)
    
    # Extract feature columns (Knee Flexion and Trunk Lean)
    features = ['Knee_Flexion_Angle', 'Trunk_Lean_Angle']
    data = df[features].values

    sequences = []
    for i in range(len(data) - sequence_length + 1):
        window = data[i:(i + sequence_length)]
        sequences.append(window)

    sequences = np.array(sequences)

    print("=== 🔄 Temporal Sequence Dataset Generator ===")
    print(f"Total frames loaded: {len(data)}")
    print(f"Sliding window length: {sequence_length} frames")
    print(f"Generated training tensor shape: {sequences.shape} -> (Batch, TimeSteps, Features)")

    # Save sequence metadata or numpy array
    os.makedirs("outputs", exist_ok=True)
    np.save(os.path.join("outputs", "joint_sequences.npy"), sequences)
    print("Sequence tensor successfully saved to: outputs/joint_sequences.npy")

if __name__ == "__main__":
    create_sliding_sequences()