import torch
import torch.nn as nn

class SkatingDegradationLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=1):
        super(SkatingDegradationLSTM, self).__init__()
        
        # LSTM layer to process temporal joint-angle trajectories
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=0.2
        )
        
        # Fully connected layer to map LSTM hidden states to a binary warning probability
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, window_size, input_dim)
        
        # Pass through LSTM
        out, (hn, cn) = self.lstm(x)
        
        # Take the output from the last time step of the window
        last_time_step_out = out[:, -1, :]
        
        # Pass through linear layer and sigmoid for probability (0 to 1)
        out = self.fc(last_time_step_out)
        return self.sigmoid(out)