import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# 1. Setup paths
base_dir = r'C:\Users\qgan2\OneDrive\Desktop\Research - biomechanics_project\biomechanics-project'
skater_b_path = os.path.join(base_dir, 'data', 'skater_b_multivariate_angles.csv')
model_path = os.path.join(base_dir, 'models', 'autoencoder_model.pth')

# 2. Define Autoencoder Architecture
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

def main():
    print("Loading dataset...")
    if not os.path.exists(skater_b_path):
        print(f"Error: Could not find dataset at {skater_b_path}")
        return

    df = pd.read_csv(skater_b_path)
    data_matrix = df.values.astype(np.float32)
    n_features = data_matrix.shape[1]
    seq_len = 30

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiomechanicsAutoencoder(seq_len, n_features).to(device)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Loaded pre-trained model weights successfully.")
    else:
        print("Warning: Model weights file not found. Running with untrained weights.")

    model.eval()
    print("App is ready and configured!")

if __name__ == '__main__':
    main()