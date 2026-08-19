import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix

# Import your pre-deceleration labeling function
from src.label_pre_deceleration import label_pre_deceleration_windows
# 1. Paths
base_dir = r'C:\Users\qgan2\OneDrive\Desktop\Research - biomechanics_project'
data_path = os.path.join(base_dir, 'data', 'skater_a_multivariate_angles.csv')
model_path = os.path.join(base_dir, 'models', 'skating_degradation_model.pth')

# 2. Load Data & Generate Test Labels
print("Loading data and generating labels for evaluation...")
df = pd.read_csv(data_path)

# Use 'knee' as the proxy tracking column instead of com_velocity
velocity_column = 'knee'  
joint_cols = [col for col in df.columns if col != velocity_column]
window_size = 10

X, y = label_pre_deceleration_windows(
    data=df,
    velocity_col=velocity_column,
    joint_angle_cols=joint_cols,
    window_size=window_size,
    drop_threshold=0.26  # <--- Updated to match train_model.py (0.10)
)

tensor_X = torch.tensor(X, dtype=torch.float32)
tensor_y = torch.tensor(y, dtype=torch.float32)

dataset = TensorDataset(tensor_X, tensor_y)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

# 3. Define the Supervised LSTM Classifier Architecture
class FatigueLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super(FatigueLSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

# 4. Load the Trained Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_features = X.shape[2]

model = FatigueLSTMClassifier(input_dim=n_features, hidden_dim=32).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# 5. Run Evaluation
print("\nEvaluating model performance...")
all_preds = []
all_targets = []

with torch.no_grad():
    for batch_X, batch_y in dataloader:
        inputs = batch_X.to(device)
        outputs = model(inputs).squeeze()
        
        preds = (outputs >= 0.5).float()
        
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(batch_y.numpy())

# 6. Print Metrics
print("\n--- Classification Report ---")
print(classification_report(all_targets, all_preds, zero_division=0))

print("\n--- Confusion Matrix ---")
print(confusion_matrix(all_targets, all_preds))