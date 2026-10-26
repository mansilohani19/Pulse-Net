# src/model.py
import torch.nn as nn


class Autoencoder(nn.Module):
    """Symmetric autoencoder: input -> 64 -> 32 -> bottleneck -> 32 -> 64 -> input."""

    def __init__(self, input_dim, bottleneck=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, bottleneck), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, input_dim),      # linear output: data is scaled to ~[0, 1]
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))
