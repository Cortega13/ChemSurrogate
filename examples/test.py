from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_flat = z.view(-1, z.shape[-1])
        dist = (
            z_flat.pow(2).sum(1, keepdim=True)
            - 2 * torch.matmul(z_flat, self.embedding.weight.t())
            + self.embedding.weight.pow(2).sum(1)
        )
        indices = torch.argmin(dist, dim=1)
        z_q = self.embedding(indices)
        z_q_st = z_flat + (z_q - z_flat).detach()
        return z_q_st.view_as(z), z_q, z_flat

def mlp(dims: Tuple[int, ...], final_dim: int) -> nn.Sequential:
    layers = [layer for i in range(len(dims)-1) for layer in (nn.Linear(dims[i], dims[i+1]), nn.ReLU())]
    layers.append(nn.Linear(dims[-1], final_dim))
    return nn.Sequential(*layers)

class VQVAE(nn.Module):
    def __init__(
        self,
        input_dim: int = 333,
        hidden_dims: Tuple[int, int] = (256, 128),
        embedding_dim: int = 128,
        num_embeddings: int = 4096,
    ) -> None:
        super().__init__()
        self.encoder = mlp((input_dim,) + hidden_dims, embedding_dim)
        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim)
        self.decoder = mlp((embedding_dim,) + hidden_dims[::-1], input_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        z_q_st, z_q, z_flat = self.quantizer(z)
        x_hat = self.decoder(z_q_st)
        return x_hat, z_q, z_flat

def compute_vqvae_loss(x: torch.Tensor, x_hat: torch.Tensor, z_q: torch.Tensor, z_flat: torch.Tensor, beta: float) -> torch.Tensor:
    return F.mse_loss(x_hat, x) + F.mse_loss(z_q.detach(), z_flat) + beta * F.mse_loss(z_q, z_flat.detach())

def _step(model: VQVAE, x: torch.Tensor, optimizer: torch.optim.Optimizer, beta: float) -> float:
    optimizer.zero_grad()
    x_hat, z_q, z_flat = model(x)
    loss = compute_vqvae_loss(x, x_hat, z_q, z_flat, beta)
    loss.backward()
    optimizer.step()
    return loss.item()

def train_vqvae(
    model: VQVAE,
    dataloader: torch.utils.data.DataLoader,
    epochs: int = 50,
    lr: float = 3e-4,
    beta: float = 0.25,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        total = sum(_step(model, x.to(device), opt, beta) for (x,) in dataloader)
        print(f"Epoch {epoch:02d}/{epochs} — loss: {total / len(dataloader):.4f}")

if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(1024_000, 333)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x), batch_size=8192)
    model = VQVAE()
    train_vqvae(model, loader, epochs=3)
