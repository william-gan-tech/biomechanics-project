import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# 1. Paths
base_dir = r'C:\Users\qgan2\OneDrive\Desktop\Research - biomechanics_project\biomechanics-project'
data_dir = os.path.join(base_dir, 'data')
models_dir = os.path.join(base_dir, 'models')
model_path = os.path.join(models_dir, 'autoencoder_model.pth')
output_path = os.path.join(base_dir, 'summary_report.csv')

# 2. Re-define Autoencoder Architecture
class BiomechanicsAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features):
        super(BiomechanicsAutoencoder, self).__init__()
        flat_dim = seq_len * n_features
        self.encoder = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(flat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, flat_dim),
            nn.Unflatten(1, (seq_len, n_features))
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# 3. Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seq_len = 30

# Find all CSV files in the data directory
csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

if not csv_files:
    print("No CSV files found in the data directory!")
else:
    # We need to inspect one file to get feature dimensions
    sample_df = pd.read_csv(os.path.join(data_dir, csv_files[0]))
    n_features = sample_df.shape[1]
    
    model = BiomechanicsAutoencoder(seq_len, n_features).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    criterion = nn.MSELoss(reduction='none')
    summary_results = []

    # 4. Loop Through Each File and Process
    for file_name in csv_files:
        file_path = os.path.join(data_dir, file_name)
        df = pd.read_csv(file_path)
        data_matrix = df.values.astype(np.float32)
        
        if len(data_matrix) <= seq_len:
            print(f"Skipping {file_name}: too short for sequence length {seq_len}.")
            continue

        windows = []
        for i in range(len(data_matrix) - seq_len):
            windows.append(data_matrix[i:i + seq_len])
        tensor_data = torch.tensor(np.array(windows), dtype=torch.float32)

        with torch.no_grad():
            outputs = model(tensor_data.to(device))
            loss_per_window = criterion(outputs, tensor_data.to(device)).mean(dim=(1, 2)).cpu().numpy()

        mean_loss = np.mean(loss_per_window)
        max_loss = np.max(loss_per_window)
        anomaly_count = np.sum(loss_per_window > (mean_loss + 3 * np.std(loss_per_window)))

        # Append summary metrics for this file
        summary_results.append({
            'file_name': file_name,
            'total_windows': len(loss_per_window),
            'mean_reconstruction_error': mean_loss,
            'max_reconstruction_error': max_loss,
            'anomalies_flagged': anomaly_count
        })

    # 5. Export to CSV
    summary_df = pd.DataFrame(summary_results)
    summary_df.to_csv(output_path, index=False)
    print(f"\nBatch processing complete! Summary report saved to:\n{output_path}")
    print("\n--- Generated Summary Preview ---")
    print(summary_df)