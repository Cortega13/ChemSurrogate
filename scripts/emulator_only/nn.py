import torch
import torch.nn as nn
import torch.nn.functional as F

class Emulator(nn.Module):
    def __init__(self, input_dim=337, output_dim=333, hidden_dim=32, dropout=0.0):
        super(Emulator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, phys, latents):
        B, T, P = phys.shape
        L = latents.shape[1]
        outputs = torch.empty(B, T, L, device=latents.device, dtype=latents.dtype)

        for t in range(T):
            current_phys = phys[:, t, :]  # [B, P]
            input = torch.cat([current_phys, latents], dim=1)  # [B, P+L]
            latents = latents + self.net(input)
            outputs[:, t, :] = latents

        return outputs