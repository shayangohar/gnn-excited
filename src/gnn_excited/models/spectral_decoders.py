"""Spectral decoders on frozen per-atom features (bet A ladder, TASK-277 / TASK-310).

All decoders share the rich-readout trunk of scripts/train_uma_qm9gwbse.py (_PooledHead): a per-atom
MLP, then mean and standard deviation pooling per molecule. They differ only in how the pooled
molecule vector becomes the five singlet energies:

  mlp   the control: an MLP with standardized outputs (the production head, retrained on frozen features)
  r0    KNOW-156 on UMA features: a 5x5 symmetric tridiagonal latent Hamiltonian; energies are its
        eigenvalues (ascending by construction)
  r1    a latent configuration space larger than the output window: an N x N symmetric matrix
        (N = configs, default 16); energies are its lowest five eigenvalues, so states above S5 can
        mix into and cross into the window

Oscillator strengths (log1p f) come from a separate MLP in every variant. Matrix entries are in eV.
At initialization the off-diagonal couplings are zero and the diagonal sits at the train-set state
means (r1: higher configurations spread above S5), so every decoder starts at the target mean.
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class AtomTrunkPool(nn.Module):
    """Per-atom trunk, then mean+std pooling per molecule: [atoms, in_dim] -> [molecules, 2 * hidden]."""

    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.trunk = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.SiLU(),
                                   nn.Linear(hidden, hidden), nn.SiLU())

    def forward(self, per_atom: torch.Tensor, node_batch: torch.Tensor, num_molecules: int) -> torch.Tensor:
        h = self.trunk(per_atom)
        counts = torch.bincount(node_batch, minlength=num_molecules).clamp_min(1).unsqueeze(1).to(h.dtype)
        mean = h.new_zeros(num_molecules, h.size(1)).index_add_(0, node_batch, h) / counts
        second = h.new_zeros(num_molecules, h.size(1)).index_add_(0, node_batch, h * h) / counts
        std = (second - mean * mean).clamp_min(0.0).add(1e-6).sqrt()
        return torch.cat([mean, std], dim=-1)


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    net = nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, out_dim))
    nn.init.zeros_(net[-1].weight)
    nn.init.zeros_(net[-1].bias)
    return net


def symmetric_from_parameters(diagonal: torch.Tensor, offdiagonal: torch.Tensor, pattern: str) -> torch.Tensor:
    """[M, N] diagonal and off-diagonal entries -> [M, N, N] symmetric matrices.

    pattern "tridiagonal": offdiagonal is [M, N-1] (adjacent couplings). "full": [M, N(N-1)/2]
    (upper triangle, row-major)."""
    m, n = diagonal.shape
    matrix = diagonal.new_zeros(m, n, n)
    idx = torch.arange(n, device=diagonal.device)
    matrix[:, idx, idx] = diagonal
    if pattern == "tridiagonal":
        rows, cols = idx[:-1], idx[:-1] + 1
    elif pattern == "full":
        rows, cols = torch.triu_indices(n, n, offset=1, device=diagonal.device)
    else:
        raise ValueError(f"unknown pattern {pattern!r}")
    if offdiagonal.shape[1] != rows.numel():
        raise ValueError(f"expected {rows.numel()} off-diagonal entries, got {offdiagonal.shape[1]}")
    matrix[:, rows, cols] = offdiagonal
    matrix[:, cols, rows] = offdiagonal
    return matrix


class SpectralDecoder(nn.Module):
    """Frozen per-atom features -> (energies [M, states] in eV, log1p f [M, states])."""

    def __init__(self, in_dim: int, energy_mean: Sequence[float], energy_std: Sequence[float],
                 osc_mean: Sequence[float], osc_std: Sequence[float], kind: str = "mlp", hidden: int = 256,
                 configs: int = 16, offdiag_scale: float = 0.5, upper_spread_eV: float = 3.0):
        super().__init__()
        self.kind = kind
        self.states = len(energy_mean)
        self.pool = AtomTrunkPool(in_dim, hidden)
        self.register_buffer("energy_mean", torch.as_tensor(energy_mean, dtype=torch.float32))
        self.register_buffer("energy_std", torch.as_tensor(energy_std, dtype=torch.float32))
        self.register_buffer("osc_mean", torch.as_tensor(osc_mean, dtype=torch.float32))
        self.register_buffer("osc_std", torch.as_tensor(osc_std, dtype=torch.float32))
        self.osc_head = _mlp(2 * hidden, hidden, self.states)
        self.offdiag_scale = float(offdiag_scale)
        if kind == "mlp":
            self.size = self.states
            self.energy_head = _mlp(2 * hidden, hidden, self.states)
            return
        if kind == "r0":
            self.size, self.pattern = self.states, "tridiagonal"
            n_off = self.states - 1
        elif kind == "r1":
            if configs < self.states:
                raise ValueError("r1 needs at least as many configurations as output states")
            self.size, self.pattern = int(configs), "full"
            n_off = self.size * (self.size - 1) // 2
        else:
            raise ValueError(f"unknown decoder kind {kind!r}")
        self.energy_head = _mlp(2 * hidden, hidden, self.size + n_off)
        mean = torch.as_tensor(energy_mean, dtype=torch.float32)
        diag0 = mean
        if self.size > self.states:
            upper = torch.linspace(0.3, float(upper_spread_eV), self.size - self.states) + mean[-1]
            diag0 = torch.cat([mean, upper])
        with torch.no_grad():
            self.energy_head[-1].bias[: self.size] = diag0

    def forward(self, per_atom: torch.Tensor, node_batch: torch.Tensor, num_molecules: int):
        pooled = self.pool(per_atom, node_batch, num_molecules)
        osc = self.osc_head(pooled) * self.osc_std + self.osc_mean
        raw = self.energy_head(pooled)
        if self.kind == "mlp":
            return raw * self.energy_std + self.energy_mean, osc
        diagonal = raw[:, : self.size]
        offdiagonal = self.offdiag_scale * torch.tanh(raw[:, self.size:])
        matrix = symmetric_from_parameters(diagonal, offdiagonal, self.pattern)
        energies = torch.linalg.eigvalsh(matrix)[:, : self.states]
        return energies, osc
