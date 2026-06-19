"""
Combined (two-branch) improved iDeepLC retention-time model.

This model fuses both peptide representations iDeepLC has access to:

  Branch A (raw / learned):   amino-acid embedding + modification channels,
                              processed with residual conv blocks, SE channel
                              attention and self-attention.
  Branch B (chemical):        the original iDeepLC 41-channel hand-crafted
                              feature matrix (chemical features, atomic
                              composition, di-amino features, one-hot, sequence
                              metadata) produced by ideeplc.utilities, processed
                              with its own residual conv stack.

Each branch is summarized with global max+avg pooling plus a small positional
projection, then concatenated together with the global biochemical feature
vector and passed through an MLP head. The idea is that the learned embedding
branch and the hand-crafted chemical branch carry complementary information,
so combining them should not be worse than either alone and may be better.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from model_improved import SEBlock, ResBlock, PositionalEncoding


class _Branch(nn.Module):
    """Conv stem + residual blocks + (optional) self-attention, returns a
    pooled summary concatenated with a small positional projection."""

    def __init__(self, in_ch, max_len, hidden, groups, proj_ch, use_attn=True,
                 n_heads=4, kernels=(7, 7, 5, 5, 3)):
        super().__init__()
        self.pos_enc = PositionalEncoding(in_ch, max_len)
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, hidden, kernel_size=1), nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(groups, hidden), nn.GELU(),
        )
        drops = [0.10, 0.10, 0.10, 0.15, 0.15]
        self.res_blocks = nn.Sequential(*[
            ResBlock(hidden, kernel_size=k, dropout=d, groups=groups)
            for k, d in zip(kernels, drops)
        ])
        self.use_attn = use_attn
        if use_attn:
            self.attn_norm = nn.LayerNorm(hidden)
            self.self_attn = nn.MultiheadAttention(hidden, n_heads, dropout=0.1,
                                                   batch_first=True)
            self.attn_drop = nn.Dropout(0.1)
        self.proj = nn.Sequential(nn.Conv1d(hidden, proj_ch, kernel_size=1), nn.GELU())
        self.out_dim = 2 * hidden + proj_ch * max_len

    def forward(self, x):
        x = self.pos_enc(x)
        x = self.stem(x)
        x = self.res_blocks(x)
        if self.use_attn:
            xt = self.attn_norm(x.transpose(1, 2))
            att, _ = self.self_attn(xt, xt, xt)
            x = x + self.attn_drop(att.transpose(1, 2))
        pooled = torch.cat([x.amax(dim=2), x.mean(dim=2)], dim=1)
        proj = self.proj(x).flatten(1)
        return torch.cat([pooled, proj], dim=1)


class CombinedNet(nn.Module):
    def __init__(
        self,
        n_aa_tokens: int = 21,
        n_mod_channels: int = 15,
        n_extra: int = 17,
        raw_len: int = 66,
        chem_ch: int = 41,
        chem_len: int = 62,
        config: Optional[dict] = None,
    ) -> None:
        super().__init__()
        config = config or {}
        emb_dim = int(config.get("emb_dim", 64))
        hidden = int(config.get("hidden", 128))
        chem_hidden = int(config.get("chem_hidden", 96))
        groups = int(config.get("groups", 8))
        proj_ch = int(config.get("proj_ch", 16))
        n_heads = int(config.get("n_heads", 4))

        # Branch A: raw / learned embedding.
        self.aa_emb = nn.Embedding(n_aa_tokens, emb_dim, padding_idx=0)
        self.raw_branch = _Branch(emb_dim + n_mod_channels, raw_len, hidden,
                                  groups, proj_ch, use_attn=True, n_heads=n_heads)

        # Branch B: chemical 41-channel matrix.
        self.chem_branch = _Branch(chem_ch, chem_len, chem_hidden, groups, proj_ch,
                                   use_attn=False)

        head_in = self.raw_branch.out_dim + self.chem_branch.out_dim + n_extra
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden),
            nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.30),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(hidden // 2, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, aa_idx, mod_x, feat, mat):
        emb = self.aa_emb(aa_idx).transpose(1, 2)
        raw_in = torch.cat([emb, mod_x], dim=1)
        raw_vec = self.raw_branch(raw_in)
        chem_vec = self.chem_branch(mat)
        return self.head(torch.cat([raw_vec, chem_vec, feat], dim=1))
