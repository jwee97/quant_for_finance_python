"""Smart order routing: split an order across venues, learning from what fills how much hidden liquidity each of them has.

A dark pool or a displayed venue will trade against you only as much as the other side happens to have there. Send it ``s`` shares and you learn ``min(s, V)``, where ``V`` is the liquidity it held and
which you never see: a fill of everything you sent says only that there was *at least* that much (the observation is *censored*). A venue is characterised by ``P(V >= j)``, the chance that it can fill ``j``
shares. Since the value of the first ``s`` shares sent to a venue is ``sum_(j <= s) P(V >= j)``, which has ever smaller increments, the best way to spread ``S`` shares is to give the next one to whichever
venue is most likely to fill it, ``argmax_i value_i P_i(V >= s_i + 1)``: send to venues in order of execution probability, *at the margin*. Sending everything to the venue most likely to fill the first
share ignores that a venue's probability falls as the order grows (:func:`greedy_allocation`).

Learning the probabilities without being told them is Ganchev, Kearns, Nevmyvaka and Vaughan's "dark pool problem" (2010). Each round the Kaplan-Meier estimator (:func:`kaplan_meier_tail`) turns the history of
(sent, filled) pairs into an estimate of ``P(V >= j)`` for every venue, treating a full fill as censored, and is *optimistic* where there is no data (a venue that was never sent more than ``s`` shares is
assumed to fill any larger quantity as likely as ``s``), which makes the router try bigger sizes and unvisited venues until they have been shown to be bad. :func:`simulate_routing` plays this out against
simulated venues and compares the router with an even split, with the "best single venue" rule, and with an oracle that knows the true distributions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Venue:
    """A venue whose hidden liquidity ``V`` (in lots) has ``P(V >= j) = first * decay^(j-1)``, ``j = 1, 2, ...``: ``first`` is the chance it can fill anything, ``decay`` how fast that chance falls
    with the quantity. ``value`` is the net benefit of a share filled there (price improvement less fees) relative to the others."""

    name: str
    first: float
    decay: float
    value: float = 1.0

    def __post_init__(self):
        if not 0.0 <= self.first <= 1.0 or not 0.0 < self.decay <= 1.0 or self.value < 0:
            raise ValueError("0 <= first <= 1, 0 < decay <= 1, value >= 0")

    def tail(self, upto: int) -> np.ndarray:
        """``P(V >= j)`` for ``j = 1 ... upto``."""
        return self.first * self.decay ** np.arange(upto)

    def draw(self, rng: np.random.Generator, size: int | None = None):
        """Hidden liquidity: zero with probability ``1 - first``, else one lot plus a geometric number of further lots."""
        has = rng.random(size) < self.first
        extra = np.floor(np.log(np.clip(rng.random(size), 1e-300, 1.0)) / np.log(self.decay)) if self.decay < 1.0 else 1e12
        return np.where(has, 1.0 + extra, 0.0)


def example_venues() -> list:
    """Five venues of different character: a displayed venue that nearly always fills a little, three dark pools of increasing depth and decreasing probability, and a venue that almost never fills."""
    return [Venue("lit exchange", 0.95, 0.985), Venue("mid-size pool", 0.75, 0.996), Venue("deep pool", 0.45, 0.9985), Venue("deepest pool", 0.25, 0.9995), Venue("quiet pool", 0.05, 0.90)]


def expected_fill(venues, allocation) -> float:
    """The expected value-weighted number of lots filled by an allocation, under the venues' true distributions."""
    return float(sum(v.value * v.tail(int(s)).sum() for v, s in zip(venues, allocation) if s > 0))


def greedy_allocation(tails: np.ndarray, shares: int, values=None) -> np.ndarray:
    """Allocate ``shares`` lots across venues to maximise the sum of value times fill probability at the margin. ``tails[i, j-1]`` is the (estimated) ``P(V_i >= j)`` for ``j = 1 ... shares``; each
    row must be non-increasing, in which case taking the ``shares`` largest entries of ``value_i * tails[i, j]`` is optimal and gives a prefix of every row."""
    tails = np.asarray(tails, dtype=float)
    k, width = tails.shape
    if width < shares:
        raise ValueError("tails must cover the whole order")
    w = np.ones(k) if values is None else np.asarray(values, dtype=float)
    gain = (w[:, None] * tails[:, :shares] - 1e-9 * np.arange(shares)[None, :]).ravel()        # the tiny slope splits ties evenly: every venue's first lot before anyone's second
    order = np.argsort(-gain, kind="stable")[:shares]
    return np.bincount(order // shares, minlength=k).astype(int)


def kaplan_meier_tail(sent, filled, width: int, optimism: float = 0.0) -> np.ndarray:
    """Estimate ``P(V >= j)`` for ``j = 1 ... width`` from pairs (lots sent, lots filled) with censoring: a fill smaller than the amount sent reveals ``V`` exactly, a full fill says only ``V >= sent``.

    The discrete Kaplan-Meier estimator: the hazard ``h_k = P(V = k | V >= k)`` is the number of exact observations equal to ``k`` over the number still *at risk* at ``k`` (exact observations of at least
    ``k`` and censored ones sent more than ``k``), and ``P(V >= j+1) = P(V >= j)(1 - h_j)``. Where no observation is at risk the hazard is taken to be zero, so the tail stays flat: optimistic beyond the
    data. That alone does not make a router explore: a venue that happened to fill nothing in its first few visits has an estimate of zero and is never sent anything again. So ``optimism`` adds to
    each estimate ``z`` standard errors of a proportion measured on the ``n_j`` observations that bear on ``P(V >= j)`` (the upper end of a Wilson interval, which is 1 when there are none), kept
    non-increasing in ``j``: a venue or a size the data say little about keeps being tried until the data say it is poor. ``optimism = 0`` is the plain estimator."""
    sent = np.asarray(sent, dtype=int)
    filled = np.asarray(filled, dtype=int)
    if len(sent) != len(filled) or (filled > sent).any() or (filled < 0).any():
        raise ValueError("filled must be between 0 and sent, one per observation")
    exact = filled < sent
    beyond = width + 1                                                                                      # an observation past the range we estimate counts as at risk throughout it
    exact_count = np.bincount(np.minimum(filled[exact], beyond), minlength=width + 2).astype(float)          # exact_count[f]: V = f was seen
    censored_count = np.bincount(np.minimum(sent[~exact], beyond), minlength=width + 2).astype(float)        # censored_count[s]: V >= s was seen
    at_risk = np.cumsum(exact_count[::-1])[::-1][:width + 1] + np.cumsum(censored_count[::-1])[::-1][1:width + 2]      # exact with f >= k, plus censored with s > k
    hazard = np.divide(exact_count[:width + 1], at_risk, out=np.zeros(width + 1), where=at_risk > 0)
    survival = np.cumprod(1.0 - hazard)[:width]                                                             # survival[k] = P(V >= k + 1)
    if optimism > 0.0:
        n = at_risk[:width]                                                                                  # the observations that bear on P(V >= k + 1)
        z = float(optimism)
        with np.errstate(divide="ignore", invalid="ignore"):
            upper = (survival + z * z / (2 * n) + z * np.sqrt(survival * (1 - survival) / n + z * z / (4 * n * n))) / (1 + z * z / n)
        upper = np.where(n > 0, upper, 1.0)
        survival = np.maximum(survival, np.minimum.accumulate(np.clip(upper, 0.0, 1.0)))
    return survival


POLICIES = ("uniform", "best first-unit probability", "learned (Kaplan-Meier, optimistic)", "oracle (true distributions)")


def simulate_routing(venues, rounds: int = 300, shares: int = 500, policy: str = "learned (Kaplan-Meier, optimistic)", seed: int = 0, optimism: float = 2.0) -> dict:
    """Route ``shares`` lots a round for ``rounds`` rounds. ``policy``: ``uniform`` (an even split), ``best first-unit probability`` (everything to the venue the estimates say is most likely to fill
    anything), ``learned (Kaplan-Meier, optimistic)`` (the greedy marginal allocation on the estimates) or ``oracle (true distributions)`` (the greedy allocation on the true ones). Returns the realised
    fill rate (lots filled over lots sent) and the *expected* fill rate of the allocations chosen (under the true distributions, so with no sampling noise), over all rounds and over the last third, and the
    final allocation and estimates."""
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")
    rng = np.random.default_rng(seed)
    k = len(venues)
    values = np.array([v.value for v in venues])
    sent_hist = [[] for _ in range(k)]
    fill_hist = [[] for _ in range(k)]
    true_tails = np.vstack([v.tail(shares) for v in venues])
    realised, expected = np.zeros(rounds), np.zeros(rounds)
    estimate = np.ones((k, shares))
    allocation = np.zeros(k, dtype=int)
    for r in range(rounds):
        if policy == "oracle (true distributions)":
            estimate = true_tails
        elif policy != "uniform":
            estimate = np.vstack([kaplan_meier_tail(sent_hist[i], fill_hist[i], shares, optimism) if sent_hist[i] else np.ones(shares) for i in range(k)])
        if policy == "uniform":
            allocation = np.full(k, shares // k)
            allocation[: shares - allocation.sum()] += 1
        elif policy == "best first-unit probability":
            allocation = np.zeros(k, dtype=int)
            allocation[int(np.argmax(values * estimate[:, 0]))] = shares
        else:
            allocation = greedy_allocation(estimate, shares, values)
        V = np.array([v.draw(rng) for v in venues])
        filled = np.minimum(allocation, V).astype(int)
        for i in range(k):
            if allocation[i] > 0:
                sent_hist[i].append(int(allocation[i]))
                fill_hist[i].append(int(filled[i]))
        realised[r] = filled.sum() / shares
        expected[r] = expected_fill(venues, allocation) / shares
    third = max(rounds // 3, 1)
    return {"fill_rate": float(realised.mean()), "fill_rate_last_third": float(realised[-third:].mean()), "expected_fill_rate": float(expected.mean()),
            "expected_fill_rate_last_third": float(expected[-third:].mean()), "allocation": allocation, "estimate": estimate, "expected_by_round": expected}
