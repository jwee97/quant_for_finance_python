"""Portfolio optimisation as a QUBO, the form a quantum annealer takes, solved here by simulated annealing and checked against exhaustive search.

A QUBO (quadratic unconstrained binary optimisation) is ``minimise x'Qx + offset`` over ``x`` in ``{0, 1}^N``. Quantum annealers (D-Wave) are built to find low-energy states of exactly this function,
and a portfolio problem becomes one in two standard ways, both implemented here. Because ``x^2 = x`` for a bit, every linear term sits on the diagonal of ``Q``.

**Discretised weights.** Give every asset ``b`` bits and let ``w_i = w_max * sum_k 2^k x_(i,k) / (2^b - 1)``, a grid of ``2^b`` weights from 0 to ``w_max``. The mean-variance objective
``(risk_aversion / 2) w'Sigma w - mu'w`` is quadratic in the bits, and the budget ``sum w = 1`` becomes the penalty ``P (sum w - 1)^2``, also quadratic.

**Cardinality.** Choose exactly ``K`` of ``n`` assets and hold them equally: with ``x_i`` in ``{0, 1}`` the objective is ``(risk_aversion / 2K^2) x'Sigma x - mu'x / K`` and the constraint ``sum x = K`` becomes
``P (sum x - K)^2``. Picking the best ``K`` of ``n`` is combinatorial (``C(n, K)`` subsets), which is why it is the textbook case for annealing.

No quantum hardware is used here. :func:`simulated_annealing` is a classical Metropolis search with a falling temperature, run from many random starts at once; :func:`brute_force` enumerates every state
of small problems, so a result can be shown to be the true optimum. The QUBO matrices are the same object either way and can be handed to an annealer. The penalty must be large enough that breaking
the constraint costs more than the objective can gain; too large and the landscape is dominated by the constraint and annealing finds the right subset less often, so it defaults to ten times the objective's scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class QUBO:
    Q: np.ndarray                  # symmetric: energy = x'Qx + offset, with the linear terms on the diagonal
    offset: float
    decode: object                 # function from a bit vector to portfolio weights
    n_assets: int
    bits: int
    penalty: float

    def energy(self, x) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=float))
        return np.einsum("ci,ij,cj->c", x, self.Q, x) + self.offset


def _symmetric(M) -> np.ndarray:
    M = np.asarray(M, dtype=float)
    return 0.5 * (M + M.T)


def weight_qubo(mu, cov, risk_aversion: float = 5.0, bits: int = 3, w_max: float = 0.5, budget: float = 1.0, penalty: float | None = None) -> QUBO:
    """The mean-variance problem with weights on a ``2^bits``-point grid in ``[0, w_max]`` and the budget as a penalty."""
    m = np.asarray(mu, dtype=float)
    S = _symmetric(cov)
    n = len(m)
    if S.shape != (n, n) or bits < 1 or w_max <= 0 or risk_aversion < 0:
        raise ValueError("cov must match mu, bits >= 1, w_max > 0, risk_aversion >= 0")
    levels = (2.0 ** np.arange(bits)) / (2.0 ** bits - 1.0) * w_max                      # the weight each bit adds
    C = np.zeros((n, n * bits))                                                          # w = C x
    for i in range(n):
        C[i, i * bits:(i + 1) * bits] = levels
    a = C.sum(axis=0)                                                                    # what each bit adds to the total weight
    scale = max(np.abs(m).max(), risk_aversion * np.abs(S).max(), 1e-12)
    pen = float(penalty) if penalty is not None else 10.0 * scale
    quadratic = 0.5 * risk_aversion * C.T @ S @ C + pen * np.outer(a, a)
    linear = -(C.T @ m) - 2.0 * pen * budget * a
    Q = quadratic + np.diag(linear)
    return QUBO(Q, pen * budget ** 2, lambda x: C @ np.asarray(x, dtype=float), n, bits, pen)


def cardinality_qubo(mu, cov, k: int, risk_aversion: float = 5.0, penalty: float | None = None) -> QUBO:
    """Choose ``k`` of the assets, equally weighted: minimise ``(risk_aversion / 2 k^2) x'Sigma x - mu'x / k + P (sum x - k)^2``."""
    m = np.asarray(mu, dtype=float)
    S = _symmetric(cov)
    n = len(m)
    if S.shape != (n, n) or not 1 <= k <= n or risk_aversion < 0:
        raise ValueError("cov must match mu and 1 <= k <= n")
    scale = max(np.abs(m).max() / k, risk_aversion * np.abs(S).max() / k ** 2, 1e-12)
    pen = float(penalty) if penalty is not None else 10.0 * scale
    quadratic = 0.5 * risk_aversion / k ** 2 * S + pen * np.ones((n, n))
    linear = -m / k - 2.0 * pen * k
    Q = quadratic + np.diag(linear)
    return QUBO(Q, pen * k * k, lambda x: np.asarray(x, dtype=float) / max(float(np.sum(x)), 1.0), n, 1, pen)


def _quench(x: np.ndarray, M: np.ndarray, max_rounds: int = 200) -> np.ndarray:
    """Zero-temperature descent for every chain: repeatedly apply the best single flip or pair flip (any two bits) until no move lowers the energy."""
    diag = np.diag(M)
    for _ in range(max_rounds):
        field = x @ M
        delta = 1.0 - 2.0 * x
        single = diag + 2.0 * delta * field                                                # (chains, N): the change if one bit flips
        pair = single[:, :, None] + single[:, None, :] + 2.0 * delta[:, :, None] * delta[:, None, :] * M[None]
        idx = np.arange(M.shape[0])
        pair[:, idx, idx] = np.inf                                                         # a bit cannot be flipped twice
        best_single, best_pair = single.min(axis=1), pair.reshape(len(x), -1).min(axis=1)
        improving = np.minimum(best_single, best_pair) < -1e-12
        if not improving.any():
            break
        for c in np.flatnonzero(improving):
            if best_single[c] <= best_pair[c]:
                x[c, int(np.argmin(single[c]))] = 1.0 - x[c, int(np.argmin(single[c]))]
            else:
                i, j = np.unravel_index(int(np.argmin(pair[c])), pair[c].shape)
                x[c, i], x[c, j] = 1.0 - x[c, i], 1.0 - x[c, j]
    return x


def simulated_annealing(Q, sweeps: int = 400, restarts: int = 32, t_start: float | None = None, t_end: float | None = None, seed: int = 0) -> tuple[np.ndarray, float, np.ndarray]:
    """Minimise ``x'Qx`` over binary ``x`` by Metropolis annealing: ``restarts`` independent chains advance together and the temperature falls geometrically from ``t_start`` to ``t_end``.

    Each sweep proposes every single-bit flip and the same number of *pair* flips (two random bits at once). Pair flips matter for problems with a penalised constraint: with a budget or a cardinality
    penalty, no single flip can move between feasible states without paying the penalty, so at the temperatures where the objective is resolved the chains would freeze on whatever they chose earlier.
    After the last sweep every chain is quenched (best single or pair flip until none helps). Returns the best state found, its energy and the final energies of all chains."""
    M = _symmetric(Q)
    N = M.shape[0]
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, (restarts, N)).astype(float)
    field = x @ M                                                                        # (Mx)_j for every chain
    diag = np.diag(M).copy()
    reach = np.abs(diag) + 2.0 * (np.abs(M).sum(axis=1) - np.abs(diag)) + 1e-12            # the largest energy change flipping bit j can cause
    t0 = float(t_start if t_start is not None else 0.5 * np.median(reach))
    t1 = float(t_end if t_end is not None else max(1e-4 * np.median(reach), 1e-12))
    rows = np.arange(restarts)
    for temperature in np.geomspace(t0, t1, sweeps):
        for j in rng.permutation(N):
            delta = 1.0 - 2.0 * x[:, j]                                                  # +1: 0 -> 1, -1: 1 -> 0
            change = diag[j] + 2.0 * delta * field[:, j]                                 # the exact change in x'Mx when bit j flips
            accept = (change <= 0.0) | (rng.random(restarts) < np.exp(-np.maximum(change, 0.0) / temperature))
            flip = np.flatnonzero(accept)
            if len(flip):
                x[flip, j] += delta[flip]
                field[flip] += delta[flip, None] * M[j][None, :]
        if N > 1:
            for _ in range(N):
                i = rng.integers(0, N, restarts)
                j = (i + rng.integers(1, N, restarts)) % N                               # a different bit in every chain
                di, dj = 1.0 - 2.0 * x[rows, i], 1.0 - 2.0 * x[rows, j]
                fi, fj = field[rows, i], field[rows, j]
                change = diag[i] + diag[j] + 2.0 * di * fi + 2.0 * dj * fj + 2.0 * di * dj * M[i, j]
                accept = (change <= 0.0) | (rng.random(restarts) < np.exp(-np.maximum(change, 0.0) / temperature))
                for c in np.flatnonzero(accept):
                    x[c, i[c]] += di[c]
                    x[c, j[c]] += dj[c]
                    field[c] += di[c] * M[i[c]] + dj[c] * M[j[c]]
    x = _quench(x, M)
    energies = np.einsum("ci,ij,cj->c", x, M, x)
    best = int(np.argmin(energies))
    return x[best].copy(), float(energies[best]), energies


def brute_force(Q, chunk: int = 1 << 16) -> tuple[np.ndarray, float]:
    """The exact minimiser of ``x'Qx`` by enumerating all ``2^N`` states (N up to 26)."""
    M = _symmetric(Q)
    N = M.shape[0]
    if N > 26:
        raise ValueError("brute force is for up to 26 bits")
    best_e, best_x = np.inf, None
    shifts = np.arange(N)
    for start in range(0, 1 << N, chunk):
        idx = np.arange(start, min(start + chunk, 1 << N))
        X = ((idx[:, None] >> shifts) & 1).astype(float)
        e = np.einsum("ci,ij,cj->c", X, M, X)
        j = int(np.argmin(e))
        if e[j] < best_e:
            best_e, best_x = float(e[j]), X[j].copy()
    return best_x, best_e


@dataclass
class AnnealedPortfolio:
    weights: np.ndarray
    energy: float                  # the QUBO energy including the offset: the penalised objective
    chain_energies: np.ndarray
    budget_error: float            # sum(w) - budget: the grid cannot always hit it exactly
    qubo: QUBO


def anneal_weights(mu, cov, risk_aversion: float = 5.0, bits: int = 3, w_max: float = 0.5, budget: float = 1.0, **kw) -> AnnealedPortfolio:
    """Mean-variance weights on a bit grid, found by annealing."""
    q = weight_qubo(mu, cov, risk_aversion, bits, w_max, budget)
    x, e, es = simulated_annealing(q.Q, **kw)
    w = q.decode(x)
    return AnnealedPortfolio(w, e + q.offset, es + q.offset, float(w.sum() - budget), q)


def selection_objective(mu, cov, mask, risk_aversion: float = 5.0) -> float:
    """``risk_aversion/2 w'Sigma w - mu'w`` for the equal-weighted portfolio of the assets in ``mask`` (infinite for an empty set)."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return float("inf")
    w = mask / mask.sum()
    return float(0.5 * risk_aversion * w @ np.asarray(cov, dtype=float) @ w - w @ np.asarray(mu, dtype=float))


def anneal_selection(mu, cov, k: int, risk_aversion: float = 5.0, **kw) -> tuple[np.ndarray, float]:
    """The ``k`` assets (a boolean mask) that minimise the equal-weighted mean-variance objective, found by annealing; returns the mask and its objective."""
    q = cardinality_qubo(mu, cov, k, risk_aversion)
    x, _, _ = simulated_annealing(q.Q, **kw)
    mask = x > 0.5
    return mask, selection_objective(mu, cov, mask, risk_aversion)
