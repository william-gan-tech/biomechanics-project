import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(Encoder, self).__init__()
        self.lstm1 = nn.LSTM(
            input_size=n_features, 
            hidden_size=embedding_dim, 
            num_layers=2, 
            batch_first=True, 
            dropout=0.2
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, n_features)
        _, (hidden, _) = self.lstm1(x)
        return hidden[-1] # Extract final hidden state representing compressed latent space

class Decoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(Decoder, self).__init__()
        self.seq_len = seq_len
        self.lstm1 = nn.LSTM(
            input_size=embedding_dim, 
            hidden_size=embedding_dim, 
            num_layers=2, 
            batch_first=True, 
            dropout=0.2
        )
        self.output_layer = nn.Linear(embedding_dim, n_features)

    def forward(self, x):
        # Repeat the latent vector across the sequence length to reconstruct time-steps
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        x, _ = self.lstm1(x)
        return self.output_layer(x)

class SkatingLSTMAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(SkatingLSTMAutoencoder, self).__init__()
        self.encoder = Encoder(seq_len, n_features, embedding_dim)
        self.decoder = Decoder(seq_len, n_features, embedding_dim)

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed