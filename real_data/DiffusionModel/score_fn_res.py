import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Utility: sinusoidal timestep embeddings (DDPM / Transformer style)
# -----------------------------------------------------------------------------

def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Return fixed sinusoidal embeddings for a vector of timesteps.

    Args:
        timesteps: (B,) integer tensor of diffusion steps
        dim: embedding dimension (must be even)
    Returns:
        (B, dim) float32 tensor suitable for conditioning a network.
    """
    assert dim % 2 == 0, "dim must be even"
    device = timesteps.device
    half_dim = dim // 2
    exponent = -math.log(10000.0) / (half_dim - 1)
    freqs = torch.exp(torch.arange(half_dim, device=device) * exponent)
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return emb  # shape (B, dim)


# -----------------------------------------------------------------------------
# Residual dilated block with **separate** FiLM for (time, condition)
# -----------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Depth‑wise dilated Conv1d + two FiLM gates (timestep & condition).

    * Accepts **separate** embeddings for time and condition; they are applied
      one after the other instead of being summed first.  Works for any
      sequence length ≥ 1 and any #channels because the convolutions are
      depth‑wise (groups = channels).
    """

    def __init__(self, channels: int, dilation: int, groups: int = 8):
        super().__init__()
        self.dconv = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=channels,  # depth‑wise
        )
        self.norm = nn.GroupNorm(min(groups, channels), channels)

        # Separate FiLM affine generators
        self.film_t = nn.Linear(channels, 2 * channels)  # timestep gate
        self.film_c = nn.Linear(channels, 2 * channels)  # condition gate

        self.pw = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,      # (B, H, L)
        t_ctx: torch.Tensor,  # (B, H)
        c_ctx: torch.Tensor,  # (B, H)
    ) -> torch.Tensor:
        y = self.dconv(x)
        y = F.silu(self.norm(y))

        # --- FiLM from timestep ---
        scale_t, shift_t = self.film_t(t_ctx).chunk(2, dim=-1)
        y = y * (1 + scale_t.unsqueeze(-1)) + shift_t.unsqueeze(-1)

        # --- FiLM from condition ---
        scale_c, shift_c = self.film_c(c_ctx).chunk(2, dim=-1)
        y = y * (1 + scale_c.unsqueeze(-1)) + shift_c.unsqueeze(-1)

        y = self.pw(y)
        return x + y


# -----------------------------------------------------------------------------
# Dilated TCN ε-network (separate t & condition paths)
# -----------------------------------------------------------------------------

class EpsilonDilatedTCN(nn.Module):
    """ε‑network: depth‑wise dilated TCN + **two** FiLM conditioners."""

    def __init__(
        self,
        num_assets: int = 1,
        hidden_width: int = 128,
        cond_dim: int = 50,
        t_embed_dim: int = 64,
        dilations: List[int] | None = None,
    ) -> None:
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4, 8, 16, 32]

        # time‑step embedding → hidden_width
        self.t_embed_mlp = nn.Sequential(
            nn.Linear(t_embed_dim, 4 * hidden_width),
            nn.SiLU(),
            nn.Linear(4 * hidden_width, hidden_width),
        )

        # condition projection
        self.cond_proj = nn.Linear(cond_dim, hidden_width)

        self.stem = nn.Conv1d(num_assets, hidden_width, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_width, d) for d in dilations]
        )
        self.head = nn.Conv1d(hidden_width, num_assets, kernel_size=3, padding=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,          # (B, C, L)
        timesteps: torch.Tensor,  # (B,)
        cond: torch.Tensor,       # (B, cond_dim)
    ) -> torch.Tensor:
        t_emb = sinusoidal_embedding(timesteps, self.t_embed_mlp[0].in_features)
        t_ctx = self.t_embed_mlp(t_emb)          # (B, H)
                # allow condition to be (B, cond_dim) or (B, 1, cond_dim) or (B, L_c, cond_dim)
        if cond.dim() == 3:
            # average over the sequence dimension (length could be 1)
            cond_flat = cond.mean(1)
        else:
            cond_flat = cond  # already (B, cond_dim)
        c_ctx = self.cond_proj(cond_flat)        # (B, H)

        y = self.stem(x)
        for blk in self.blocks:
            y = blk(y, t_ctx, c_ctx)
        return self.head(y)


# -----------------------------------------------------------------------------
# Smoke test: C = 1, L = 1 edge‑case
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    B, C, L = 2, 1, 1
    cond_dim = 50
    device = "cuda" if torch.cuda.is_available() else "cpu"

    net = EpsilonDilatedTCN(num_assets=C, hidden_width=128, cond_dim=cond_dim).to(device)

    x_t = torch.randn(B, C, L, device=device)
    t   = torch.randint(0, 1000, (B,), device=device)
    h   = torch.randn(B, cond_dim, device=device)

    out = net(x_t, t, h)
    print("out shape:", out.shape)
    F.mse_loss(out, torch.randn_like(out)).backward()
    print("✓ separate‑FiLM variant works.")
