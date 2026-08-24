import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from preprocess_video import process_skating_video_multivariate
from model import SkatingLSTMAutoencoder

def run_full_fatigue_pipeline(video_path, model_path="skating_degradation_model.pth"):
    """
    Auto-digests a skating video, runs LSTM autoencoder inference,
    calibrates a dynamic threshold, and returns structured results for a UI.
    """
    if not os.path.exists(video_path):
        return {"success": False, "error": f"Video not found: {video_path}"}
    
    if not os.path.exists(model_path):
        return {"success": False, "error": f"Model weights not found: {model_path}"}

    # 1. Feature Extraction
    df_features = process_skating_video_multivariate(video_path)
    if df_features is None or df_features.empty:
        return {"success": False, "error": "Failed to extract features from video."}

    window_size = 30
    n_features = 4  
    model = SkatingLSTMAutoencoder(seq_len=window_size, n_features=n_features, embedding_dim=64)

    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    if isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint.get('state_dict', checkpoint))
    else:
        model = checkpoint
    model.eval()
    
    feature_cols = ["right_knee_angle", "left_knee_angle", "right_knee_filtered", "left_knee_filtered"]
    buffer = []
    fps = 30.0
    
    all_losses = []
    frame_loss_pairs = []

    # 2. Window Inference Pass
    for idx, row in df_features.iterrows():
        frame_idx = int(row["frame"])
        if not all(col in df_features.columns for col in feature_cols):
            continue
            
        current_features = row[feature_cols].values
        buffer.append(current_features)
        
        if len(buffer) == window_size:
            window_data = np.array(buffer)
            tensor_input = torch.tensor(window_data, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                reconstruction = model(tensor_input)
                loss = torch.mean((tensor_input - reconstruction) ** 2).item()
            
            all_losses.append(loss)
            frame_loss_pairs.append((frame_idx, loss))
            buffer.pop(0)

    if not all_losses:
        return {"success": False, "error": "Not enough frames to compute windows."}

    # 3. Dynamic Baseline Calibration
    baseline_window_count = min(150, len(all_losses))
    baseline_losses = all_losses[:baseline_window_count]
    mean_loss = np.mean(baseline_losses)
    std_loss = np.std(baseline_losses)
    dynamic_threshold = mean_loss + (2.0 * std_loss)

    # 4. Fatigue Spike Detection
    fatigue_records = []
    for frame_idx, loss in frame_loss_pairs:
        if loss > dynamic_threshold:
            timestamp = frame_idx / fps
            fatigue_records.append({
                "frame": frame_idx,
                "timestamp_sec": round(timestamp, 2),
                "mse_loss": round(loss, 4)
            })

    # Summary calculations
    total_frames = len(df_features)
    first_onset = fatigue_records[0]["timestamp_sec"] if fatigue_records else None
    fatigue_percentage = round((len(fatigue_records) / total_frames) * 100, 1) if total_frames > 0 else 0.0

    return {
        "success": True,
        "metrics": {
            "mean_loss": round(mean_loss, 4),
            "std_loss": round(std_loss, 4),
            "dynamic_threshold": round(dynamic_threshold, 4),
            "first_onset_sec": first_onset,
            "total_spikes": len(fatigue_records),
            "fatigue_percentage": fatigue_percentage
        },
        "fatigue_records": fatigue_records,
        "frame_loss_pairs": frame_loss_pairs
    }