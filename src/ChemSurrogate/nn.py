import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim=337, output_dim=333, latent_dim=12, hidden_dims=(320,160), noise=0.0, dropout=0.0):
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0], bias=False),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.GELU(),
            
            nn.Linear(hidden_dims[0], hidden_dims[1], bias=False),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.GELU(),
            nn.Dropout(dropout),
        
            nn.Linear(hidden_dims[1], latent_dim),
            nn.RMSNorm(latent_dim),
            nn.GELU(),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[1], bias=False),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.GELU(),
            
            nn.Linear(hidden_dims[1], hidden_dims[0], bias=False),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dims[0], output_dim),
            nn.Sigmoid(),
        )
        
        self.noise = noise

    def encode(self, x):
        z = self.encoder(x)
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        if self.training and self.noise > 0:
            noise = torch.randn_like(z) * self.noise
            z += noise
        x_reconstructed = self.decode(z)
        return x_reconstructed, z


class Emulator(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layer, dropout=0.0):
        super(Emulator, self).__init__()
        self.layers = nn.Sequential(
            #Input Layer
            nn.Linear(input_dim, hidden_layer),
            nn.RMSNorm(hidden_layer),
            nn.GELU(),
            nn.Dropout(dropout),

            #Hidden Layer #1
            nn.Linear(hidden_layer, hidden_layer),
            nn.RMSNorm(hidden_layer),
            nn.GELU(),
            nn.Dropout(dropout),

            #Hidden Layer #2
            nn.Linear(hidden_layer, hidden_layer),
            nn.RMSNorm(hidden_layer),
            nn.GELU(),
            nn.Dropout(dropout),

            #Output Layer
            nn.Linear(hidden_layer, output_dim),
            nn.Sigmoid() # Data is normalized between 0 & 1. 
        )
    
    def forward(self, x):
        return self.layers(x)


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, output_dim),
        )
        
    def forward(self, x):
        return x[:, 5:] + self.net(x)


class RecursiveResNet(nn.Module):
    def __init__(self, input_dim=17, hidden_dim=64, output_dim=12, num_blocks=2, dropout=0.0):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            ResidualBlock(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                dropout=dropout
            ) for _ in range(num_blocks)
        ])
        
    def forward(self, x):
        params = x[:, :5]
        x = x[:, 5:]
        
        for block in self.blocks:
            x = torch.cat([params, x], dim=1)
            x = block(x)
        return x


