import pandas as pd
import numpy as np
import torch
import torch.nn as nn

class BiomechanicsAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features):
        super(BiomechanicsAutoencoder, self).__init__()
        flat_dim = seq_len * n_features
        self.encoder = nn.Sequential(nn.Flatten(start_dim=1), nn.Linear(flat_dim, 64), nn.ReLU(), nn.Linear(64, 16), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, flat_dim), nn.Unflatten(1, (seq_len, n_features)))
    def forward(self, x):
        return self.decoder(self.encoder(x))

def compute_joint_decomposition(model, data_tensor, seq_len=30):
    model.eval()
    with torch.no_grad():
        reconstructed = model(data_tensor)
        sq_error = (data_tensor - reconstructed) ** 2
        feature_mse = torch.mean(sq_error, dim=1).cpu().numpy()
    return feature_mse

print('Cross-evaluation and joint decomposition module ready!')
