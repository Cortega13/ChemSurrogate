import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, input_dim=333, latent_dim=12, hidden_dims=(320,160), noise=0.1, dropout=0.0):
        super(Autoencoder, self).__init__()
        
        # Encoder weights
        self.encoder_fc1 = nn.Linear(input_dim, hidden_dims[0], bias=False)
        self.encoder_bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.encoder_fc2 = nn.Linear(hidden_dims[0], hidden_dims[1], bias=False)
        self.encoder_bn2 = nn.BatchNorm1d(hidden_dims[1])
        self.encoder_fc3 = nn.Linear(hidden_dims[1], latent_dim)
        self.encoder_norm3 = nn.BatchNorm1d(latent_dim)
        
        self.decoder_bn1 = nn.BatchNorm1d(hidden_dims[1])
        self.decoder_bn2 = nn.BatchNorm1d(hidden_dims[0])
        
        self.decoder_bias1 = nn.Parameter(torch.zeros(hidden_dims[1]))
        self.decoder_bias2 = nn.Parameter(torch.zeros(hidden_dims[0]))
        self.decoder_bias3 = nn.Parameter(torch.zeros(input_dim))
        
        self.activation = nn.GELU()
        self.final_activation = nn.Sigmoid()  # For 0-1 bounded output
        self.dropout = nn.Dropout(dropout)
        self.noise = noise

    def encode(self, x):
        x = self.activation(self.encoder_bn1(self.encoder_fc1(x)))
        x = self.activation(self.encoder_bn2(self.encoder_fc2(x)))
        x = self.dropout(x)
        z = self.activation(self.encoder_norm3(self.encoder_fc3(x)))
        return z

    def decode(self, z):
        z = F.linear(z, self.encoder_fc3.weight.t()) + self.decoder_bias1
        z = self.activation(self.decoder_bn1(z))
        
        z = F.linear(z, self.encoder_fc2.weight.t()) + self.decoder_bias2
        z = self.activation(self.decoder_bn2(z))
        z = self.dropout(z)
        
        x_reconstructed = F.linear(z, self.encoder_fc1.weight.t()) + self.decoder_bias3
        x_reconstructed = self.final_activation(x_reconstructed)
        return x_reconstructed

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


class ResidualMLP(nn.Module):
    def __init__(self, input_dim=17, hidden_dim=256, output_dim=12, num_blocks=2, dropout=0.0):
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
        params = x[:, :5] # Physical Parameters
        
        x = x[:, 5:]# Chemical Abundances
        
        for block in self.blocks:
            x = torch.cat([params, x], dim=1)
            x = block(x)
        return x