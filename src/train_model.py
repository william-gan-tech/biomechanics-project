import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Setup paths
base_dir = r'C:\Users\qgan2\OneDrive\Desktop\Research - biomechanics_project\biomechanics-project'
data_path = os.path.join(base_dir, 'data', 'skater_b_multivariate_angles.csv')
models_dir = os.path.join(base_dir, 'models')
os.makedirs(models_dir, exist_ok=True)
model_save_path = os.path.join(models_dir, 'autoencoder_model.pth')

# 2. Load and prepare data
print("Loading data for training...")
df = pd.read_csv(data_path)
data_matrix = df.values.astype(np.float32)
n_features = data_matrix.shape[1]
seq_len = 30

def create_windows(data, seq_len):
    windows = []
    for i in range(len(data) - seq_len):
        windows.append(data[i:i + seq_len])
    return np.array(windows)

windows = create_windows(data_matrix, seq_len)
tensor_data = torch.tensor(windows, dtype=torch.float32)

# 3. Define Autoencoder Architecture
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BiomechanicsAutoencoder(seq_len, n_features).to(device)

# 4. Training Configuration
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 50
batch_size = 16

dataset = torch.utils.data.TensorDataset(tensor_data)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 5. Train the Model
print("Training autoencoder model...")
model.train()
for epoch in range(epochs):
    epoch_loss = 0
    for batch in dataloader:
        inputs = batch[0].to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss / len(dataloader):.4f}")

# 6. Save the trained weights directly to the models/ folder
torch.save(model.state_dict(), model_save_path)
print(f"\n✅ Training complete! Model weights successfully saved to: '{model_save_path}'")