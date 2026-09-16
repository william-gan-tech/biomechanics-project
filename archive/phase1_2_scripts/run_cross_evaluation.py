import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from cross_evaluation import BiomechanicsAutoencoder, compute_joint_decomposition

print('--- Starting Cross-Skater Biomechanics Analysis ---')

# 1. Generate or load multi-skater data simulating joint angles (Knee, Hip, Ankle, Torso)
np.random.seed(42)
seq_len = 30
n_features = 4

# Skater A (Clean Baseline Data)
_skater_a_data = np.random.normal(loc=0.5, scale=0.05, size=(100, seq_len, n_features))
train_tensor = torch.tensor(_skater_a_data, dtype=torch.float32)

# Skater B (Fatigued / Anomaly Data with higher variance on specific joints)
_skater_b_data = np.random.normal(loc=0.5, scale=0.12, size=(50, seq_len, n_features))
test_tensor = torch.tensor(_skater_b_data, dtype=torch.float32)

# 2. Initialize and Train Model on Skater A
model = BiomechanicsAutoencoder(seq_len=seq_len, n_features=n_features)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print('Training autoencoder on Skater A baseline...')
model.train()
for epoch in range(25):
    optimizer.zero_grad()
    output = model(train_tensor)
    loss = criterion(output, train_tensor)
    loss.backward()
    optimizer.step()

print(f'Training complete! Final Baseline Loss: {loss.item():.5f}')

# 3. Test Cross-Generalization on Skater B using Joint-Specific Decomposition
print('Running cross-evaluation and joint-specific MSE decomposition on Skater B...')
feature_mse = compute_joint_decomposition(model, test_tensor, seq_len=seq_len)

joint_names = ['Knee Flexion', 'Hip Angle', 'Ankle Dorsiflexion', 'Torso Lean']
mean_joint_errors = np.mean(feature_mse, axis=0)

print('\n--- Cross-Evaluation Results ---')
for i, name in enumerate(joint_names):
    print(f'{name} Error: {mean_joint_errors[i]:.4f}')

overall_anomaly_score = np.mean(mean_joint_errors)
print(f'Overall Cross-Skater Anomaly Score: {overall_anomaly_score:.4f}')

if overall_anomaly_score > 0.045:
    print('STATUS: ?? Anomaly Flagged (Technique breakdown detected when cross-evaluating Skater B using Skater A model!)')
else:
    print('STATUS: ? Normal Form')

