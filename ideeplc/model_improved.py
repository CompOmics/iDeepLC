"""
Improved iDeepLC retention-time model.

This network is a drop-in replacement for ``model.MyNet``. It consumes the
exact same input tensor as the original iDeepLC model, i.e. the 41-channel
chemical-feature matrix of shape (batch, 41, 62), and returns a single
retention-time prediction per peptide of shape (batch, 1).

The architecture adopts the modern building blocks used in the Kaggle
retention-time model (rt_model_2.py), ported onto iDeepLC's hand-crafted
feature encoding:

  - a 1x1 + 3-kernel convolutional stem that projects the 41 input channels
    to a wider hidden width,
  - a learned positional encoding,
  - a stack of residual convolutional blocks with GroupNorm, GELU, dropout
    and Squeeze-and-Excitation channel attention,
  - one multi-head self-attention layer over the sequence positions,
  - global max + average pooling (replacing the original giant Flatten -> FC),
  - a small MLP regression head with LayerNorm.

The original MyNet has no normalization, no residual connections, no
attention, and flattens the full feature map into a very large fully
connected layer. The blocks below address each of those limitations while
keeping the input/output contract identical so the existing data pipeline,
training loop and evaluation code can be reused unchanged.
"""

from typing import Optional

import torch
from torch import nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=2)
        s = self.fc(s).unsqueeze(2)
        return x * s


class ResBlock(nn.Module):
    """Residual 1D-conv block: Conv-GN-GELU-Dropout-Conv-GN + SE, then skip."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 7,
        dropout: float = 0.1,
        groups: int = 8,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(groups, channels),
        )
        self.se = SEBlock(channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        out = self.se(out)
        return self.act(x + out)


class PositionalEncoding(nn.Module):
    """Learned additive positional encoding over the sequence dimension."""

    def __init__(self, n_channels: int, max_len: int) -> None:
        super().__init__()
        self.pos_emb = nn.Parameter(torch.zeros(1, n_channels, max_len))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_emb[:, :, : x.size(2)]


class ImprovedNet(nn.Module):
    """Improved iDeepLC RT regressor (input/output compatible with MyNet)."""

    def __init__(self, x_shape, config: Optional[dict] = None) -> None:
        super().__init__()
        config = config or {}

        in_channels = int(x_shape[1])
        max_len = int(x_shape[2])
        hidden = int(config.get("hidden", 128))
        n_heads = int(config.get("n_heads", 4))
        groups = int(config.get("groups", 8))
        proj_ch = int(config.get("proj_ch", 16))

        self.pos_enc = PositionalEncoding(in_channels, max_len)

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(groups, hidden),
            nn.GELU(),
        )

        self.res_blocks = nn.Sequential(
            ResBlock(hidden, kernel_size=7, dropout=0.10, groups=groups),
            ResBlock(hidden, kernel_size=7, dropout=0.10, groups=groups),
            ResBlock(hidden, kernel_size=5, dropout=0.10, groups=groups),
            ResBlock(hidden, kernel_size=5, dropout=0.15, groups=groups),
            ResBlock(hidden, kernel_size=3, dropout=0.15, groups=groups),
        )

        self.attn_norm = nn.LayerNorm(hidden)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden, num_heads=n_heads, dropout=0.1, batch_first=True
        )
        self.attn_drop = nn.Dropout(0.1)

        # Positional projection path: keep per-position detail (the strength of
        # the original Flatten-based head) at a small channel count, so it can
        # be combined with the global max/avg pooled summary.
        self.proj = nn.Sequential(
            nn.Conv1d(hidden, proj_ch, kernel_size=1),
            nn.GELU(),
        )
        head_in = 2 * hidden + proj_ch * max_len

        self.head = nn.Sequential(
            nn.Linear(head_in, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden // 2, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos_enc(x)
        x = self.stem(x)
        x = self.res_blocks(x)

        xt = x.transpose(1, 2)
        xt = self.attn_norm(xt)
        att, _ = self.self_attn(xt, xt, xt)
        x = x + self.attn_drop(att.transpose(1, 2))

        pooled = torch.cat([x.amax(dim=2), x.mean(dim=2)], dim=1)
        proj = self.proj(x).flatten(1)
        return self.head(torch.cat([pooled, proj], dim=1))
