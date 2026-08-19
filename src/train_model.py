import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Import your pre-deceleration labeling function from src
from src.label_pre_deceleration import label_pre_deceleration_windows
# 1. Setup paths
base_dir = r'C:\Users\qgan2\OneDrive\Desktop\Research - biomechanics_project'
data_path = os.path.join(base_dir, 'data', 'skater_a_multivariate_angles.csv') 
models_dir = os.path.join(base_dir, 'models')
os.makedirs(models_dir, exist_ok=True)
model_save_path = os.path.join(models_dir, 'skating_degradation_model.pth')

# 2. Load data and apply pre-deceleration labeling
print("Loading data and generating pre-deceleration labels...")
df = pd.read_csv(data_path)

# Use 'knee' as our reference column
velocity_column = 'knee'  
joint_cols = [col for col in df.columns if col != velocity_column]

window_size = 10
X, y = label_pre_deceleration_windows(
    data=df,
    velocity_col=velocity_column,
    joint_angle_cols=joint_cols,
    window_size=window_size,
    drop_threshold=0.26  # <--- Increased threshold for better label balance
)

print(f"Generated X shape: {X.shape}, y shape: {y.shape}")
print(f"Positive labels (approaching deceleration): {int(y.sum())} out of {len(y)}")

# Convert to PyTorch tensors
tensor_X = torch.tensor(X, dtype=torch.float32)
tensor_y = torch.tensor(y, dtype=torch.float32)

# 3. Define Supervised LSTM Classifier Architecture
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_features = X.shape[2]
model = FatigueLSTMClassifier(input_dim=n_features, hidden_dim=32).to(device)

# 4. Training Configuration
criterion = nn.BCELoss() # Binary Cross-Entropy Loss for 0/1 classification
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 30
batch_size = 16

dataset = TensorDataset(tensor_X, tensor_y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 5. Train the Model
print("Training pre-deceleration prediction model...")
model.train()
for epoch in range(epochs):
    epoch_loss = 0
    for batch_X, batch_y in dataloader:
        inputs = batch_X.to(device)
        targets = batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss / len(dataloader):.4f}")

# 6. Save the trained weights
torch.save(model.state_dict(), model_save_path)
print(f"\n✅ Training complete! Model weights successfully saved to: '{model_save_path}'")