import torch
import numpy as np
from .model import SkatingDegradationLSTM

def sweep_thresholds(model_path, test_data, true_labels, speed_data, decel_threshold, fps=60):
    # 1. Load trained model
    input_dim = test_data.shape[1]
    model = SkatingDegradationLSTM(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 2. Generate predictions across the sequence
    window_size = 50
    predictions = []
    
    with torch.no_grad():
        for i in range(len(test_data) - window_size):
            window = torch.tensor(test_data[i:i+window_size], dtype=torch.float32).unsqueeze(0)
            pred = model(window).item()
            predictions.append(pred)
            
    predictions = np.array(predictions)
    
    # 3. Find true deceleration start index
    is_decel = speed_data[window_size:] < decel_threshold
    decel_indices = np.where(is_decel)[0]
    
    if len(decel_indices) == 0:
        print("No deceleration event found in test data.")
        return
        
    true_decel_start = decel_indices[0]
    
    # 4. Sweep through different thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    print(f"\n--- Threshold Tuning Sweep ---")
    print(f"{'Threshold':<12} | {'Lead Time (Seconds)':<20} | {'False Alarms (Before Decel)':<25}")
    print("-" * 65)
    
    for thresh in thresholds:
        warning_indices = np.where(predictions > thresh)[0]
        
        if len(warning_indices) == 0:
            lead_time = "No Trigger"
            false_alarms = 0
        else:
            first_warning = warning_indices[0]
            anticipation_frames = true_decel_start - first_warning
            
            if anticipation_frames >= 0:
                lead_time = f"{anticipation_frames / fps:.2f}s ({anticipation_frames} frames)"
                # Count how many warnings fired BEFORE true deceleration started
                false_alarms = np.sum(warning_indices < true_decel_start)
            else:
                lead_time = "Triggered AFTER decel"
                false_alarms = len(warning_indices)
                
        print(f"{thresh:<12} | {lead_time:<20} | {false_alarms:<25}")

if __name__ == "__main__":
    num_frames = 1000
    num_features = 12
    dummy_data = np.random.randn(num_frames, num_features)
    dummy_labels = np.zeros(num_frames)
    dummy_labels[450:500] = 1
    
    dummy_speed = np.ones(num_frames) * 10.0
    dummy_speed[500:] = 7.0 
    
    sweep_thresholds("skating_degradation_model.pth", dummy_data, dummy_labels, dummy_speed, decel_threshold=8.0)