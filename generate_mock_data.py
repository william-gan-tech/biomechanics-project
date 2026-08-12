from pathlib import Path
import numpy as np
import pandas as pd

# Define robust directory paths using pathlib
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

def generate_datasets():
    # Ensure the data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    baseline_path = DATA_DIR / "baseline_angles.csv"
    test_path = DATA_DIR / "test_skater_angles.csv"
    
    print("[INFO] Generating synthetic biomechanical datasets...")
    
    # 1. Generate stable baseline data (e.g., 200 frames, 3 joint angles)
    np.random.seed(42)
    baseline_data = pd.DataFrame(
        np.random.normal(loc=45.0, scale=3.0, size=(200, 3)),
        columns=["knee_angle", "hip_angle", "ankle_angle"]
    )
    baseline_data.to_csv(baseline_path, index=False)
    print(f"[SUCCESS] Saved baseline data to: {baseline_path}")
    
    # 2. Generate test data with a sudden anomaly/breakdown near the end
    test_values = np.random.normal(loc=45.0, scale=3.0, size=(200, 3))
    # Inject a sharp deviation (anomaly) starting at frame 150
    test_values[150:, :] += np.linspace(0, 25, 50)[:, None]
    
    test_data = pd.DataFrame(
        test_values,
        columns=["knee_angle", "hip_angle", "ankle_angle"]
    )
    test_data.to_csv(test_path, index=False)
    print(f"[SUCCESS] Saved test data to: {test_path}")

if __name__ == "__main__":
    generate_datasets()