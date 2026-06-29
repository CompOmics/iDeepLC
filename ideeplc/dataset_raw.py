"""
Raw-sequence dataset/encoder for the improved iDeepLC retention-time model.

Unlike the original iDeepLC pipeline, which consumes a fixed (41, 62)
chemical-feature matrix, this encoder works directly from raw peptide
sequences and modification strings, the way the Kaggle retention-time model
(rt_model_2.py) does. Each peptide is turned into three tensors:

  - ``aa_idx``  : integer amino-acid indices (for a learned embedding),
  - ``mod_x``   : multi-hot modification channels (one channel per known mod),
  - ``feat``    : a vector of global biochemical features (length, residue-class
                  fractions, GRAVY, bulkiness, terminal hydrophobicity, missed
                  cleavages, modification counts).

This removes the fixed input-size constraint: ``max_len`` is configurable and
the modification vocabulary is explicit, so new modifications can be added by
extending ``MOD_LIST``.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")

# Union of modifications seen across the 20-dataset and 14-PTM benchmarks.
MOD_LIST = [
    "Carbamidomethyl", "Oxidation", "Acetyl", "Acetylation", "Phospho",
    "Methyl", "Dimethyl", "Trimethyl", "Formyl", "Propionyl", "Succinyl",
    "Malonyl", "Crotonyl", "Deamidated", "Nitro",
]

GRAVY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

BULK = {
    "A": 11.50, "R": 14.28, "N": 12.82, "D": 11.68, "C": 13.46, "Q": 14.45,
    "E": 13.57, "G": 3.40, "H": 13.69, "I": 21.40, "L": 21.40, "K": 15.71,
    "M": 16.25, "F": 19.80, "P": 17.43, "S": 9.47, "T": 15.77, "W": 21.67,
    "Y": 18.03, "V": 21.57,
}

# Atomic composition (delta) of each modification, order = ATOMS.
# Derived from pyteomics proforma mod.composition; aa-independent. This is the
# physico-chemical signal the original iDeepLC encoding gives the model and that
# the one-hot modification channel lacks (see SHAP analysis).
ATOMS = ["C", "H", "O", "N", "P", "S"]
N_ATOMS = len(ATOMS)
# Per-atom scale (max |count| over the vocabulary) so the atomic-composition
# channels land in roughly [-1, 1], matching the 0/1 one-hot scale. Unscaled
# integer counts (Trimethyl H=6) destabilized training; zeros stay zero.
MOD_ATOM_SCALE = [4.0, 6.0, 3.0, 1.0, 1.0, 1.0]  # C, H, O, N, P, S
MOD_ATOM_COMP = {
    "Carbamidomethyl": [2, 3, 1, 1, 0, 0],
    "Oxidation": [0, 0, 1, 0, 0, 0],
    "Acetyl": [2, 2, 1, 0, 0, 0],
    "Acetylation": [2, 2, 1, 0, 0, 0],
    "Phospho": [0, 1, 3, 0, 1, 0],
    "Methyl": [1, 2, 0, 0, 0, 0],
    "Dimethyl": [2, 4, 0, 0, 0, 0],
    "Trimethyl": [3, 6, 0, 0, 0, 0],
    "Formyl": [1, 0, 1, 0, 0, 0],
    "Propionyl": [3, 4, 1, 0, 0, 0],
    "Succinyl": [4, 4, 3, 0, 0, 0],
    "Malonyl": [3, 2, 3, 0, 0, 0],
    "Crotonyl": [4, 4, 1, 0, 0, 0],
    "Deamidated": [0, -1, 1, -1, 0, 0],
    "Nitro": [0, -1, 2, 1, 0, 0],
}

HYDROPHOBIC = set("AILMFWVY")
ACIDIC = set("DE")
BASIC = set("KRH")
AROMATIC = set("FWY")
POLAR = set("STNQ")
SMALL = set("AGSTPV")

N_EXTRA = 17


def parse_mod_pairs(mod_string: Optional[str]) -> List[Tuple[int, str]]:
    """Parse a ``pos|name|pos|name|...`` modification string (1-based positions)."""
    if mod_string is None:
        return []
    s = str(mod_string)
    if s == "" or s.lower() == "nan":
        return []
    parts = s.split("|")
    out: List[Tuple[int, str]] = []
    for i in range(0, len(parts) - 1, 2):
        try:
            pos = int(parts[i])
        except ValueError:
            continue
        out.append((pos, parts[i + 1].strip()))
    return out


class RawRTDataset(Dataset):
    """Encode raw peptide sequences + modifications for the embedding-based model."""

    def __init__(
        self,
        seqs,
        mods,
        tr=None,
        max_len: int = 66,
        scaler_mu: float = 0.0,
        scaler_sd: float = 1.0,
    ) -> None:
        self.seqs = [str(s) for s in seqs]
        self.mods = ["" if m is None else str(m) for m in mods]
        self.max_len = max_len
        self.mu = scaler_mu
        self.sd = scaler_sd if scaler_sd else 1.0

        if tr is not None:
            tr = np.asarray(tr, dtype=np.float32).reshape(-1, 1)
            self.tr = (tr - self.mu) / self.sd
        else:
            self.tr = None

        self.aa_index = {aa: i + 1 for i, aa in enumerate(AA_LIST)}  # 0 = padding
        self.mod_index = {m: i for i, m in enumerate(MOD_LIST)}
        self.n_aa_tokens = len(AA_LIST) + 1
        self.n_mod_channels = len(MOD_LIST)

    def __len__(self) -> int:
        return len(self.seqs)

    def encode_aa(self, seq: str) -> np.ndarray:
        idx = np.zeros(self.max_len, dtype=np.int64)
        for j, aa in enumerate(seq[: self.max_len]):
            idx[j] = self.aa_index.get(aa, 0)
        return idx

    def encode_mods(self, mod_string: str) -> np.ndarray:
        x = np.zeros((self.n_mod_channels, self.max_len), dtype=np.float32)
        for pos, name in parse_mod_pairs(mod_string):
            if name not in self.mod_index:
                continue
            # 1-based residue position -> 0-based array index; N-term (pos 0) -> 0.
            enc = max(pos - 1, 0)
            if 0 <= enc < self.max_len:
                x[self.mod_index[name], enc] = 1.0
        return x

    def encode_mod_atoms(self, mod_string: str) -> np.ndarray:
        """Per-position atomic composition (C,H,O,N,P,S) of the modifications.
        Returns (N_ATOMS, max_len). aa-independent physico-chemical mod signal."""
        x = np.zeros((N_ATOMS, self.max_len), dtype=np.float32)
        for pos, name in parse_mod_pairs(mod_string):
            comp = MOD_ATOM_COMP.get(name)
            if comp is None:
                continue
            enc = max(pos - 1, 0)
            if 0 <= enc < self.max_len:
                x[:, enc] += np.asarray(comp, dtype=np.float32) / np.asarray(MOD_ATOM_SCALE, dtype=np.float32)
        return x

    def extra_features(self, seq: str, mod_string: str) -> np.ndarray:
        seq = seq[: self.max_len]
        n = len(seq)
        if n == 0:
            return np.zeros(N_EXTRA, dtype=np.float32)
        hyd = sum(a in HYDROPHOBIC for a in seq) / n
        aci = sum(a in ACIDIC for a in seq) / n
        bas = sum(a in BASIC for a in seq) / n
        aro = sum(a in AROMATIC for a in seq) / n
        pol = sum(a in POLAR for a in seq) / n
        sml = sum(a in SMALL for a in seq) / n
        pro = seq.count("P") / n
        gravy = sum(GRAVY.get(a, 0.0) for a in seq) / n
        bulk = sum(BULK.get(a, 15.0) for a in seq) / n
        nterm_h = float(seq[0] in HYDROPHOBIC)
        cterm_h = float(seq[-1] in HYDROPHOBIC)
        missed = float(sum(1 for a in seq[:-1] if a in "KR"))
        pairs = parse_mod_pairs(mod_string)
        names = [nm for _, nm in pairs]
        n_mods = len(names) / max(n, 1)
        acetyl = float(sum(nm.startswith("Acetyl") for nm in names))
        oxid = float(sum(nm == "Oxidation" for nm in names))
        phos = float(sum(nm == "Phospho" for nm in names))
        return np.array([
            n / self.max_len, hyd, aci, bas, aro, pol, sml, pro,
            gravy / 5.0, bulk / 25.0, nterm_h, cterm_h,
            missed / max(n, 1), n_mods, acetyl, oxid, phos,
        ], dtype=np.float32)

    def __getitem__(self, i: int):
        seq, mod = self.seqs[i], self.mods[i]
        aa = torch.from_numpy(self.encode_aa(seq))
        mx = torch.from_numpy(self.encode_mods(mod))
        ft = torch.from_numpy(self.extra_features(seq, mod))
        if self.tr is not None:
            y = torch.tensor(self.tr[i], dtype=torch.float32)
            return aa, mx, ft, y
        return aa, mx, ft
