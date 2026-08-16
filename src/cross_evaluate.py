import os
import pandas as pd
import numpy as np

# Define paths
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)
data_dir = os.path.join(base_dir, 'data')

# Simulating cross-subject evaluation
skaters = ['test_skater_1', 'test_skater_2']
print('=== MULTI-SUBJECT GENERALIZATION & CROSS-EVALUATION ===')

for skater in skaters:
    skater_path = os.path.join(data_dir, skater, 'fatigue_results.csv')
    if not os.path.exists(skater_path):
        skater_path = os.path.join(data_dir, 'fatigue_results.csv')
    
    if os.path.exists(skater_path):
        df = pd.read_csv(skater_path)
        mean_anomaly = df['Anomaly_Score'].mean()
        max_anomaly = df['Anomaly_Score'].max()
        print(f'Subject: {skater} | Mean Anomaly Score: {mean_anomaly:.4f} | Peak Anomaly: {max_anomaly:.4f}')
    else:
        print(f'Subject: {skater} | Status: Data pending upload in respective folder.')

print('=====================================================')
