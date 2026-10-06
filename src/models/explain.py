"""Model explanation by three methods that each carry a checkable identity (Generation 5, Stage 35).

* ``permutation_importance`` - how much does the out-of-sample error rise when one feature is shuffled? Model-agnostic, global.
* ``shapley_exact`` - interventional Shapley values by enumerating every subset of features against a background sample. Exact, not sampled,
  so the *efficiency* identity must hold to rounding: sum(phi) = f(x) - mean(f(background)).
* ``integrated_gradients`` - Sundararajan et al. (2017): the path integral of the gradient from a baseline to x, by the trapezoid rule. The
  *completeness* identity is sum(attribution) = f(x) - f(baseline), up to the integration error, which ``completeness_gap`` reports.

Each function takes plain callables or torch modules and numpy arrays, so any fitted model can be explained. Nothing here is fitted to
returns; that is Stage 35's job.
"""

from __future__ import annotations

from math import factorial

import numpy as np


def permutation_importance(predict, X: np.ndarray, y: np.ndarray, n_repeats: int = 20, seed: int = 7) -> np.ndarray:
    """Mean rise in MSE when each column is shuffled (rows permuted independently per repeat). Returns shape (n_features,)."""
    rng = np.random.default_rng(seed)
    base = float(np.mean((predict(X) - y) ** 2))
    out = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        rises = []
        for _ in range(n_repeats):
            shuffled = X.copy()
            shuffled[:, j] = X[rng.permutation(len(X)), j]
            rises.append(float(np.mean((predict(shuffled) - y) ** 2)) - base)
        out[j] = np.mean(rises)
    return out


def _subset_masks(n: int) -> np.ndarray:
    """All 2^n subsets as a boolean matrix, row index = bitmask."""
    codes = np.arange(2 ** n)
    return ((codes[:, None] >> np.arange(n)) & 1).astype(bool)


def shapley_exact(predict, x: np.ndarray, background: np.ndarray) -> tuple[np.ndarray, float]:
    """Exact interventional Shapley values of one row ``x`` (shape (n,)) against ``background`` (m, n).

    The value of a feature subset S is the mean prediction over the background rows with the features in S set to ``x``'s values.
    Returns ``(phi, base)`` with ``base = mean(predict(background))`` and ``phi.sum() == predict(x) - base``.
    """
    n = len(x)
    masks = _subset_masks(n)                                                     # (2^n, n)
    m = len(background)
    mixed = np.where(masks[:, None, :], x[None, None, :], background[None, :, :]).reshape(-1, n)
    value = predict(mixed).reshape(len(masks), m).mean(axis=1)                    # v(S) for every subset
    size = masks.sum(axis=1)
    weight = np.array([factorial(k) * factorial(n - k - 1) / factorial(n) for k in range(n)])
    phi = np.zeros(n)
    for j in range(n):
        without = np.flatnonzero(~masks[:, j])
        with_j = without | (1 << j)
        phi[j] = float(np.sum(weight[size[without]] * (value[with_j] - value[without])))
    return phi, float(value[0])


def integrated_gradients(net, x: np.ndarray, baseline: np.ndarray, steps: int = 100) -> tuple[np.ndarray, float]:
    """Integrated gradients of a torch module ``net`` (maps (batch, n) to (batch,)) for one row, trapezoid rule.

    Returns ``(attribution, completeness_gap)`` where the gap is ``|sum(attribution) - (f(x) - f(baseline))|``.
    """
    import torch

    xt = torch.as_tensor(x, dtype=torch.float32)
    bt = torch.as_tensor(baseline, dtype=torch.float32)
    alphas = torch.linspace(0.0, 1.0, steps + 1)[:, None]
    path = (bt + alphas * (xt - bt)).clone().requires_grad_(True)
    was_training = net.training
    net.eval()
    out = net(path).sum()
    out.backward()
    grads = path.grad
    avg = (grads[0] / 2 + grads[1:-1].sum(dim=0) + grads[-1] / 2) / steps
    attribution = ((xt - bt) * avg).detach().numpy().astype(float)
    with torch.no_grad():
        delta = float(net(xt[None])[0] - net(bt[None])[0])
    net.train(was_training)
    return attribution, abs(float(attribution.sum()) - delta)
