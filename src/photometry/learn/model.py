"""Set-transformer spin-state proposal network.

A detection set is unordered and variable-length; the encoder is
permutation-invariant by construction (self-attention over tokens with a
padding mask, then mean+max pooling) so it cannot invent structure the
physics says is unobservable (observer ordering, facet position).

Heads: spin pole (unit vector, trained with an *axial* loss so p and -p
are the same answer — the antipode is an exact photometric degeneracy),
log spin period, and body spin-axis class (which principal axis lies
along the pole — flat spin vs propeller tumble).
"""

from __future__ import annotations

import torch
from torch import nn

from .data import N_FEATURES


class SpinNet(nn.Module):
    def __init__(self, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 3, d_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(N_FEATURES, d_model), nn.GELU(),
                                   nn.Linear(d_model, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, dropout=dropout, batch_first=True,
            norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(2 * d_model, 256), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(256, 3 + 1 + 3))

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor):
        """tokens (B,N,9), mask (B,N) True=real -> (pole (B,3) unit,
        log_period (B,), axis_logits (B,3))."""
        h = self.norm(self.encoder(self.embed(tokens),
                                   src_key_padding_mask=~mask))
        m = mask.unsqueeze(-1).float()
        mean = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        mx = h.masked_fill(~mask.unsqueeze(-1), -1e4).max(1).values
        out = self.head(torch.cat([mean, mx], dim=-1))
        pole = nn.functional.normalize(out[:, :3], dim=-1)
        return pole, out[:, 3], out[:, 4:]


def spin_loss(pred, pole_true, logp_true, axis_true, geom_mask=None,
              w_pole=1.0, w_period=2.0, w_axis=0.5):
    """Axial pole loss (1 - (p.p_true)^2), smooth-L1 log period, CE axis.

    geom_mask (B,) bool restricts the pole/axis terms to examples whose
    phase-fold features are meaningful (Tier-0 found the period)."""
    pole, logp, axis_logits = pred
    per_pole = 1.0 - (pole * pole_true).sum(-1) ** 2
    per_axis = nn.functional.cross_entropy(axis_logits, axis_true, reduction="none")
    if geom_mask is not None and geom_mask.any():
        m = geom_mask.float()
        l_pole = (per_pole * m).sum() / m.sum()
        l_axis = (per_axis * m).sum() / m.sum()
    else:
        l_pole, l_axis = per_pole.mean(), per_axis.mean()
    l_period = nn.functional.smooth_l1_loss(logp, logp_true, beta=0.1)
    total = w_pole * l_pole + w_period * l_period + w_axis * l_axis
    return total, dict(pole=float(l_pole.detach()), period=float(l_period.detach()),
                       axis=float(l_axis.detach()))
