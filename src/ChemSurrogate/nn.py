import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Autoencoder(nn.Module):
    def __init__(self, input_dim=333, latent_dim=12, hidden_dims=(320,160), noise=0.1, dropout=0.0):
        super(Autoencoder, self).__init__()
        
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
        return x_reconstructed


class ResidualBlockSequential(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.RMSNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.RMSNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.RMSNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, output_dim),
        )
        
    def forward(self, x):
        return x[:, 4:] + self.net(x)


class ResNetSequential(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=64, output_dim=12, num_blocks=2, dropout=0.0):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            ResidualBlockSequential(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                dropout=dropout
            ) for _ in range(num_blocks)
        ])
                
    def forward(self, physical_parameters, components, timesteps):        
        outputs = []
        for step in range(timesteps):
            curr_physical_parameter = physical_parameters[:, step, :]
            for block in self.blocks:
                x = torch.cat([curr_physical_parameter, components], dim=1)
                x = block(x)
            outputs.append(x)
        
        return torch.stack(outputs, dim=1)


class ChemSeq2Seq(nn.Module):
    def __init__(self, num_phys, num_chem, d_model=64,
                 nhead=8, num_layers=2, ff_mult=4, max_len=256):
        super().__init__()
        self.phys_proj = nn.Linear(num_phys, d_model, bias=False)
        self.chem_proj = nn.Linear(num_chem, d_model, bias=False)
        self.pos = nn.Parameter(torch.randn(max_len, d_model) / math.sqrt(d_model))
        enc = nn.TransformerEncoderLayer(d_model, nhead, ff_mult * d_model,
                                         batch_first=True, norm_first=False)
        self.tr = nn.TransformerEncoder(enc, num_layers)
        self.out = nn.Linear(d_model, num_chem)
        
        mask = torch.triu(torch.full((max_len, max_len), float('-inf')), 1)
        self.register_buffer('causal_mask', mask)

    def forward(self, phys, comps):
        x = self.phys_proj(phys) + self.chem_proj(comps)
        x = x + self.pos[:phys.size(1)]
        
        seq_len = phys.size(1)
        mask = self.causal_mask[:seq_len, :seq_len].to(phys.device)
        h = self.tr(x, mask=mask)

        delta = self.out(h)
        return comps + delta
    

class MLP(nn.Module):
    def __init__(self, input_dim=18, output_dim=14, hidden_dim=128, dropout=0.0):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        x = self.net(x)
        return x