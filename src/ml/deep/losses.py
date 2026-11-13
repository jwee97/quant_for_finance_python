"""Losses that compare the assets of one day with each other.

A squared error asks a network to get every asset's return right. A long-short book only needs the ORDER of the assets on each day, and the squared error spends most of its capacity on the market's common
move and on outliers. The two losses here are what cross-sectional equity models are trained with in the recent literature:

* ``daily_ic``: the Pearson correlation between score and target across the assets of a day, averaged over the days of the batch (the information coefficient, the quantity the research reports).
* ``pairwise_rank``: for every pair of assets of the same day, a penalty when the scores order them the other way round from the targets, in proportion to how far apart both are (Feng et al. 2019;
  the ranking term of StockMixer, Fan and Shen 2024).

Both need batches made of whole days (``trainer.batches`` with ``groups``). ``pred``, ``target`` and ``mask`` are ``[rows, heads]``; ``day`` is ``[rows]`` and names each row's day. A masked entry (a label
that does not exist) takes no part.
"""

from __future__ import annotations

import torch

LOSSES = ("mse", "ic", "mse_rank")
CROSS_SECTIONAL = ("ic", "mse_rank")
IC_ANCHOR = 0.05                # weight of the squared error beside the correlation: a correlation does not see the scale of the score, and this keeps the score in the target's units
MIN_ASSETS = 3                  # a correlation across fewer assets is not one


def mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (((pred - target) ** 2) * mask).sum() / mask.sum().clamp(min=1.0)


def daily_ic(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, day: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(mean correlation, days used)``: per head and day, the correlation of score and target across the day's assets; the mean over the days with at least ``MIN_ASSETS`` labels and a target that varies."""
    ids, inverse = torch.unique(day, return_inverse=True)
    onehot = torch.zeros(len(day), len(ids), dtype=pred.dtype, device=pred.device)
    onehot[torch.arange(len(day), device=pred.device), inverse] = 1.0
    n = onehot.T @ mask                                                                                    # [days, heads]
    safe = n.clamp(min=1.0)
    pc = (pred - ((onehot.T @ (pred * mask)) / safe)[inverse]) * mask
    yc = (target - ((onehot.T @ (target * mask)) / safe)[inverse]) * mask
    cov, vp, vy = onehot.T @ (pc * yc), onehot.T @ (pc * pc), onehot.T @ (yc * yc)
    ok = ((n >= MIN_ASSETS) & (vy > 1e-10)).to(pred.dtype)
    corr = cov / (torch.sqrt(vp + 1e-8) * torch.sqrt(vy + 1e-8))
    return (corr * ok).sum() / ok.sum().clamp(min=1.0), ok.sum()


def pairwise_rank(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, day: torch.Tensor) -> torch.Tensor:
    """Mean over the pairs (i, j) of one day of ``max(0, -(pred_i - pred_j) * (target_i - target_j))``: zero when the scores order the pair as the targets do."""
    same = day.unsqueeze(0) == day.unsqueeze(1)
    total, count = pred.new_zeros(()), pred.new_zeros(())
    for k in range(pred.shape[1]):
        valid = (same & (mask[:, k].unsqueeze(0) > 0) & (mask[:, k].unsqueeze(1) > 0)).to(pred.dtype)
        dp, dy = pred[:, k].unsqueeze(1) - pred[:, k].unsqueeze(0), target[:, k].unsqueeze(1) - target[:, k].unsqueeze(0)
        total = total + (torch.relu(-dp * dy) * valid).sum()
        count = count + valid.sum()
    return total / count.clamp(min=1.0)


def cross_sectional(kind: str, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, day: torch.Tensor, rank_weight: float = 1.0) -> torch.Tensor:
    """The training loss ``kind`` of a batch of whole days: ``ic`` is one minus the mean daily correlation plus a small squared error; ``mse_rank`` is the squared error plus ``rank_weight`` times the pairwise term."""
    if kind == "ic":
        return 1.0 - daily_ic(pred, target, mask, day)[0] + IC_ANCHOR * mse(pred, target, mask)
    if kind == "mse_rank":
        return mse(pred, target, mask) + float(rank_weight) * pairwise_rank(pred, target, mask, day)
    raise ValueError(f"unknown cross-sectional loss '{kind}'; choose from {list(CROSS_SECTIONAL)}")
