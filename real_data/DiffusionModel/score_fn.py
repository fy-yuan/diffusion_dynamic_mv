# simple_unet_1d_conditioned.py
"""
A **same‑resolution UNet** for 1‑D time‑series.

Updates in this version
-----------------------
* **No temporal interpolation.** When the conditioning sequence length ≠ input length we now
  pass it through a *learnable* `nn.Linear` projection along the time dimension instead of using
  `F.interpolate`.
* Projection layers are created lazily and **cached** in a `ModuleDict`, so each unique
  `(L_cond → L_input)` mapping learns its own parameters.
* Still supports `depth > 0` at `length = 1` because we never down/upsample spatially—only
  channels.

Quick demo
~~~~~~~~~~
```python
import torch
from simple_unet_1d_conditioned import UNet1DSameRes

B = 2; Lx, Lc = 1, 5              # main length 1, cond length 5
net = UNet1DSameRes(in_channels=1, cond_channels=1, depth=3, base_channels=16)

x   = torch.randn(B, 1, Lx)
cond= torch.randn(B, 1, Lc)
step= torch.randint(0, 1000, (B,))

out = net(x, step, cond)
print(out.shape)                   # torch.Size([2, 1, 1])
```
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

########################################
# Sinusoidal time embedding ###########
########################################

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:  # (B,)
        device   = timesteps.device
        half_dim = self.dim // 2
        exponent = torch.exp(
            torch.arange(half_dim, device=device) * -(math.log(10000.0) / (half_dim - 1))
        )
        emb = timesteps.float().unsqueeze(1) * exponent.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb  # (B, dim)

########################################
# FiLM ################################
########################################

class FiLM(nn.Module):
    def __init__(self, channels: int, cond_dim: int) -> None:
        super().__init__()
        self.to_scale_shift = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, channels * 2),
        )
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.to_scale_shift(cond)           # (B, 2C)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1)
        beta  = beta.unsqueeze(-1)
        return (1 + gamma) * x + beta

########################################
# Conv Blocks #########################
########################################

class ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, cond_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_c, out_c, 3, padding=1)
        self.bn1   = nn.BatchNorm1d(out_c)
        self.conv2 = nn.Conv1d(out_c, out_c, 3, padding=1)
        self.bn2   = nn.BatchNorm1d(out_c)
        self.film1 = FiLM(out_c, cond_dim)
        self.film2 = FiLM(out_c, cond_dim)
    def forward(self, x: torch.Tensor, cond_vec: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.film1(x, cond_vec)
        x = F.silu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.film2(x, cond_vec)
        x = F.silu(x)
        return x

########################################
# Same‑resolution Down/Up #############
########################################

class Down(nn.Module):
    def __init__(self, in_c: int, out_c: int, cond_dim: int):
        super().__init__()
        self.block = ConvBlock(in_c, out_c, cond_dim)
    def forward(self, x, cond):
        return self.block(x, cond)

class Up(nn.Module):
    def __init__(self, in_c: int, skip_c: int, out_c: int, cond_dim: int):
        super().__init__()
        self.block = ConvBlock(in_c + skip_c, out_c, cond_dim)
    def forward(self, x, skip, cond):
        x = torch.cat([skip, x], dim=1)
        return self.block(x, cond)

########################################
# Main model ##########################
########################################

class UNet1DSameRes(nn.Module):
    """UNet‑style 1‑D network that never changes sequence length.

    If the conditioning length differs from the input length, we use a **learnable linear
    projection** (per channel) to map it to the required size. Projection layers are stored in
    `self.len_proj` keyed by "Lc→Lx" so they are shared across calls with the same size pair.
    """

    def __init__(
        self,
        *,
        in_channels: int = 1,
        cond_channels: int = 0,
        time_embed_dim: int = 64,
        base_channels: int = 32,
        depth: int = 4,
    ) -> None:
        super().__init__()
        self.cond_channels = cond_channels

        # cache of learnable length projections
        self.len_proj: nn.ModuleDict = nn.ModuleDict()

        # ---- time embedding → FiLM vector ----
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 2, time_embed_dim),
        )

        in_c_total = in_channels + cond_channels
        self.inc = ConvBlock(in_c_total, base_channels, time_embed_dim)

        # Encoder: progressively double channels
        down_blocks: List[nn.Module] = []
        ch = base_channels
        enc_channels: List[int] = [ch]
        for _ in range(depth):
            down_blocks.append(Down(ch, ch * 2, time_embed_dim))
            ch *= 2
            enc_channels.append(ch)
        self.downs = nn.ModuleList(down_blocks)

        # Bottleneck keeps ch unchanged
        self.bottleneck = ConvBlock(ch, ch, time_embed_dim)

        # Decoder: halve channels each step
        up_blocks: List[nn.Module] = []
        for skip_c in reversed(enc_channels[:-1]):
            up_blocks.append(Up(ch, skip_c, skip_c, time_embed_dim))
            ch = skip_c
        self.ups = nn.ModuleList(up_blocks)

        self.out_conv = nn.Conv1d(base_channels, in_channels, kernel_size=1)

    # ------------------------------------------------------------
    # helper: project cond length with learnable Linear ----------
    # ------------------------------------------------------------
    def _project_cond_length(self, cond: torch.Tensor, target_len: int) -> torch.Tensor:
        """Project cond from (B, C, Lc) to (B, C, target_len) with channel‑wise Linear."""
        B, C, Lc = cond.shape
        key = f"{Lc}->{target_len}"
        if key not in self.len_proj:
            # one Linear shared across channels: (Lc → target_len)
            self.len_proj[key] = nn.Linear(Lc, target_len, bias=False).to(cond.device)
        proj = self.len_proj[key]
        cond = proj(cond.view(B * C, Lc)).view(B, C, target_len)
        return cond

    # ------------------------------------------------------------
    # forward
    # ------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,                 # (B, in_channels, L)
        t: torch.Tensor,                 # (B,) timesteps
        cond: Optional[torch.Tensor] = None,  # (B, cond_channels, Lc)
    ) -> torch.Tensor:
        if self.cond_channels > 0:
            if cond is None:
                raise ValueError("cond tensor required but None provided")
            if cond.shape[1] != self.cond_channels:
                raise ValueError(
                    f"cond_channels mismatch: expected {self.cond_channels}, got {cond.shape[1]}"
                )
            # match length via learnable linear projection if needed
            if cond.shape[-1] != x.shape[-1]:
                cond = self._project_cond_length(cond, x.shape[-1])
            x = x + cond
            x = torch.cat([x, cond], dim=1)
        else:
            if cond is not None:
                raise ValueError("Model was built with cond_channels=0 but cond tensor was provided")

        cond_vec = self.time_mlp(t)

        # Encoder
        skips: List[torch.Tensor] = []
        x_enc = self.inc(x, cond_vec)
        skips.append(x_enc)
        for down in self.downs:
            x_enc = down(x_enc, cond_vec)
            skips.append(x_enc)

        # Bottleneck
        x_bott = self.bottleneck(x_enc, cond_vec)

        # Decoder
        x_dec = x_bott
        for up, skip in zip(self.ups, reversed(skips[:-1])):
            x_dec = up(x_dec, skip, cond_vec)

        return self.out_conv(x_dec)

# -------------------- smoke tests --------------------
if __name__ == "__main__":
    B = 2
    # 1) length mismatch cond (Lc != Lx)
    net = UNet1DSameRes(in_channels=1, cond_channels=1, depth=3, base_channels=16)
    x   = torch.randn(B, 1, 1)
    c   = torch.randn(B, 1, 5)
    t   = torch.randint(0, 1000, (B,))
    y   = net(x, t, c)
    assert y.shape == x.shape

    # 2) same length cond
    x2  = torch.randn(B, 1, 17)
    c2  = torch.randn(B, 1, 17)
    y2  = net(x2, t, c2)
    assert y2.shape == x2.shape
    print("✓ All smoke tests passed.")
