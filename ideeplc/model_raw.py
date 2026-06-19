"""
Embedding-based improved iDeepLC retention-time model.

This is the full adaptation of the Kaggle retention-time model
(rt_model_2.py, ``RT_CNN``) to iDeepLC. It consumes raw-sequence encodings
produced by ``ideeplc.dataset_raw.RawRTDataset``:

  - ``aa_idx`` (B, L)            -> learned amino-acid embedding,
  - ``mod_x``  (B, n_mod, L)     -> modification channels concatenated to it,
  - ``feat``   (B, N_EXTRA)      -> global biochemical features fused at the head.

Compared with the original iDeepLC ``MyNet`` it adds learned embeddings,
GroupNorm, GELU, residual blocks, Squeeze-and-Excitation channel attention,
multi-head self-attention, a learned positional encoding, global max+avg
pooling, and a biochemical-feature fusion. It also keeps the positional
projection path introduced in ``model_improved`` so per-position detail is
preserved alongside the pooled summary.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from model_improved import SEBlock, ResBlock, PositionalEncoding


class RawRTNet(nn.Module):
    def __init__(
        self,
        n_aa_tokens: int = 21,
        n_mod_channels: int = 15,
        n_extra: int = 17,
        max_len: int = 66,
        config: Optional[dict] = None,
    ) -> None:
        super().__init__()
        config = config or {}
        emb_dim = int(config.get("emb_dim", 64))
        hidden = int(config.get("hidden", 128))
        n_heads = int(config.get("n_heads", 4))
        groups = int(config.get("groups", 8))
        proj_ch = int(config.get("proj_ch", 16))

        self.aa_emb = nn.Embedding(n_aa_tokens, emb_dim, padding_idx=0)
        stem_in = emb_dim + n_mod_channels
        self.pos_enc = PositionalEncoding(stem_in, max_len)

        self.stem = nn.Sequential(
            nn.Conv1d(stem_in, hidden, kernel_size=1),
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

        self.proj = nn.Sequential(nn.Conv1d(hidden, proj_ch, kernel_size=1), nn.GELU())
        head_in = 2 * hidden + proj_ch * max_len + n_extra

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
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, aa_idx: torch.Tensor, mod_x: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        emb = self.aa_emb(aa_idx).transpose(1, 2)        # (B, emb_dim, L)
        return self.forward_from_emb(emb, mod_x, feat)

    def forward_from_emb(self, emb: torch.Tensor, mod_x: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        """Forward starting from the (continuous) embedding tensor, so attribution
        methods (Integrated Gradients) can flow through the embedding."""
        x = torch.cat([emb, mod_x], dim=1)               # (B, emb_dim + n_mod, L)
        x = self.pos_enc(x)
        x = self.stem(x)
        x = self.res_blocks(x)

        xt = self.attn_norm(x.transpose(1, 2))
        att, _ = self.self_attn(xt, xt, xt)
        x = x + self.attn_drop(att.transpose(1, 2))

        pooled = torch.cat([x.amax(dim=2), x.mean(dim=2)], dim=1)
        proj = self.proj(x).flatten(1)
        return self.head(torch.cat([pooled, proj, feat], dim=1))
