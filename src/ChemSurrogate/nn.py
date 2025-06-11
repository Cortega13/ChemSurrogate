import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, num_physical_features, latent_dim,
                 enc_hidden=128, dec_hidden=128, attn_dim=32,
                 teacher_force_prob=0.05):
        super().__init__()
        self.p_tf = teacher_force_prob
        self.enc  = nn.LSTM(num_physical_features, enc_hidden,
                            batch_first=True, bidirectional=False)
        self.dec  = nn.LSTMCell(latent_dim + num_physical_features + enc_hidden,
                                dec_hidden)
        self.W_h  = nn.Linear(enc_hidden, attn_dim, bias=False)
        self.W_s  = nn.Linear(dec_hidden, attn_dim, bias=False)
        self.v    = nn.Linear(attn_dim, 1, bias=False)
        self.norm = nn.RMSNorm(dec_hidden)
        self.proj = nn.Linear(dec_hidden + enc_hidden, latent_dim)
        self.init_h = nn.Linear(latent_dim, dec_hidden)
        self.init_c = nn.Linear(latent_dim, dec_hidden)

    def forward(self, phys, comps):
        B, T, _   = phys.shape
        enc_out,_ = self.enc(phys)
        enc_proj  = self.W_h(enc_out)
        h = torch.tanh(self.init_h(comps[:, 0]))
        c = torch.tanh(self.init_c(comps[:, 0]))
        prev = comps[:, 0]
        outs = []

        for t in range(T):
            e      = torch.tanh(enc_proj[:, :t+1] + self.W_s(h)[:, None])
            alpha  = torch.softmax(self.v(e).squeeze(-1), -1)
            ctx    = (alpha.unsqueeze(1) @ enc_out[:, :t+1]).squeeze(1)

            h, c   = self.dec(torch.cat((prev, phys[:, t], ctx), -1), (h, c))
            h      = self.norm(h)

            delta  = self.proj(torch.cat((h, ctx), -1))
            pred = prev + delta
            outs.append(pred)

            if self.training and t < T - 1:
                m    = (torch.rand(B, 1, device=phys.device) < self.p_tf).float()
                prev = m * comps[:, t + 1] + (1 - m) * pred.detach()
            else:
                prev = pred.detach()

        return torch.stack(outs, 1)