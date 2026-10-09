"""Working an order through a simulated day: the placement styles, the child-order engine and the transaction-cost analysis.

**What is simulated.** For each of ``paths`` days the engine draws the interval volumes (a profile times lognormal noise, one shock common to the whole day), the interval price moves (variance profile times
the stress window, optional drift and autocorrelation) and then, interval by interval, asks the algorithm how many shares it wants, splits that wish by the *style* and fills it:

* **marketable** shares cross the spread and walk the book: they pay half the spread and the temporary impact ``eta sigma (q / V)^beta``;
* **passive** shares rest at the near touch: they earn half the spread and a rebate when they fill, but fill only with some probability that FALLS when the price runs away from them (a limit buy on a rising
  stock is the one that does not fill), so the unfilled part is carried and later chased;
* **dark** shares trade at the midpoint with a random fill rate: no spread, no impact, little information leaked.

Every executed share also pays half of the interval's permanent impact (weighted by how much information the venue leaks), which moves the mid for the intervals after. The order must be complete by the end of its
window: whatever is left in the last interval or intervals is swept with marketable orders, impact included.

**What is reported.** The implementation shortfall against the arrival price ``side * (sum q P - X ref) / (X ref)`` in bps and its parts (price drift while waiting, spread, temporary and permanent impact, fees and
rebates), the slippage against the market's volume-weighted price, the participation, the share the deadline forced past the participation cap, and the distribution across paths. With noiseless volumes and a plain schedule the simulated mean and
variance match :func:`src.algo.impact.expected_cost` and :func:`~src.algo.impact.timing_variance`; ``tests/test_algo.py`` checks that, so the optimiser and the simulator describe the same market.

**What is not simulated.** Queue position, the order book's shape, latency, other participants reacting to us, hidden liquidity beyond the stylised dark fill, auctions, halts. The passive and dark parameters are
round numbers, not measurements: use the styles to compare *methods* and to see which way a parameter pushes, not to predict a venue's fill rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .impact import permanent_fraction, temporary_fraction
from .market import SCENARIOS, Market, Order, Scenario


# ------------------------------------------------------------------------------------------------------------------ placement styles
@dataclass(frozen=True)
class Style:
    """How a wish to trade is placed. ``market`` + ``dark`` + the passive remainder add up to one. ``chase`` is how strongly the marketable share rises when the order is behind its plan (a working order tops
    up with market orders); in the last ``sweep_last`` intervals everything left is marketable so the order finishes. ``leak_*`` is how much of its permanent impact each venue's executions have, relative to a
    marketable one."""

    name: str
    market: float
    dark: float = 0.0
    chase: float = 0.0
    sweep_last: int = 1
    passive_fill: float = 0.55
    passive_sensitivity: float = 0.15
    dark_fill: float = 0.30
    dark_concentration: float = 6.0
    fee_bps: float = 0.30
    rebate_bps: float = 0.20
    leak_passive: float = 0.8
    leak_dark: float = 0.25
    description: str = ""

    def __post_init__(self):
        if min(self.market, self.dark) < 0 or self.market + self.dark > 1 + 1e-12 or self.sweep_last < 1 or not 0 <= self.passive_fill <= 1 or not 0 <= self.dark_fill <= 1:
            raise ValueError("market, dark >= 0 with market + dark <= 1; sweep_last >= 1; fill rates in [0, 1]")

    @property
    def passive(self) -> float:
        return max(0.0, 1.0 - self.market - self.dark)


AGGRESSIVE = Style("aggressive", market=1.0, chase=0.0, sweep_last=1, description="takes liquidity across venues: every share is marketable, so the order follows its schedule exactly and pays the spread and the impact")
WORKING = Style("working", market=0.35, dark=0.10, chase=0.6, sweep_last=2, description="a mix of limit and market orders: rests most of each slice, tops up with market orders when it falls behind, sweeps in the last two intervals")
PASSIVE = Style("passive", market=0.0, dark=0.30, chase=0.15, sweep_last=1, passive_fill=0.45, description="mostly limit orders and dark pools: earns the spread and leaks little, but may not fill and is swept at the end")
STYLES: dict[str, Style] = {s.name: s for s in (AGGRESSIVE, WORKING, PASSIVE)}


# ------------------------------------------------------------------------------------------------------------------ algorithm protocol
@dataclass
class State:
    """What an algorithm may look at when it decides interval ``i``: the shares left, the mid price at the start of the interval, the interval's volume (others' trading, as it prints), and its own history. Arrays
    have one entry per simulated path."""

    i: int
    n: int
    side: int
    arrival: float
    remaining: np.ndarray
    mid: np.ndarray
    volume: np.ndarray
    expected_volume: float
    expected_total_volume: float
    spent: np.ndarray                      # signed dollars paid so far above the arrival value (side * sum q (P - ref))
    temp_spent: np.ndarray                 # dollars of temporary impact paid so far (what a cost target tracks: it excludes the price's own moves)
    executed: np.ndarray
    variance_elapsed: float                # the share of a day's variance that has passed (0 at the start)
    sigma: float
    spread_mult: float
    impact_mult: float
    vol_mult: float
    volume_mult: float

    def favourable(self) -> np.ndarray:
        """The price move since arrival, in the standard deviations of the elapsed time, positive when it favours us (for a buyer: the price fell)."""
        scale = self.arrival * self.sigma * np.sqrt(max(self.variance_elapsed, 1e-12))
        z = self.side * (self.arrival - self.mid) / scale
        return z if self.variance_elapsed > 0 else np.zeros_like(self.mid)


class Algo:
    """A trading algorithm: ``prepare`` once, then ``want`` each interval. ``plan`` (shares per interval, summing to the order) lets the engine finish an order that fell behind by spreading what is left in the
    plan's proportions; an algorithm without a plan (POV) decides from the volume it sees."""

    name = "algo"
    category = "algorithm"
    description = ""

    def prepare(self, order: Order, market: Market, scenario: Scenario) -> None:
        self.order, self.market, self.scenario = order, market, scenario

    def plan(self) -> np.ndarray | None:
        return None

    def want(self, state: State) -> np.ndarray:
        """The shares to trade in this interval (one per path). The default follows the plan in proportion to what is left: ``remaining * plan_i / sum_{j >= i} plan_j``."""
        plan = self.plan()
        if plan is None:
            raise NotImplementedError
        tail = plan[state.i:].sum()
        return state.remaining * (plan[state.i] / tail if tail > 0 else 1.0)


# ------------------------------------------------------------------------------------------------------------------ result
@dataclass
class ExecutionResult:
    algo: str
    style: str
    scenario: str
    order: Order
    market: Market
    shortfall_bps: np.ndarray
    parts_bps: dict
    vwap_slippage_bps: np.ndarray
    arrival_slippage_bps: np.ndarray
    executed: np.ndarray                  # paths x intervals
    mid: np.ndarray                       # paths x (intervals + 1)
    participation: np.ndarray             # paths: executed shares over the market volume in the window
    swept: np.ndarray                     # paths: share of the order pushed past the participation cap by the deadline
    price_improvement: np.ndarray         # paths: share of the shares executed at a price better than the prevailing mid
    target_bps: float | None = None
    info: dict = field(default_factory=dict)

    def summary(self) -> dict:
        s = self.shortfall_bps
        out = {"algo": self.algo, "style": self.style, "scenario": self.scenario, "paths": int(len(s)),
               "shortfall_bps": float(s.mean()), "std_bps": float(s.std(ddof=1)) if len(s) > 1 else 0.0, "median_bps": float(np.median(s)),
               "p05_bps": float(np.percentile(s, 5)), "p95_bps": float(np.percentile(s, 95)), "se_bps": float(s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 1 else 0.0,
               "vwap_slippage_bps": float(self.vwap_slippage_bps.mean()), "arrival_slippage_bps": float(self.arrival_slippage_bps.mean()),
               "participation": float(self.participation.mean()), "swept": float(self.swept.mean()), "price_improvement": float(self.price_improvement.mean())}
        out.update({f"{k}_bps": float(v.mean()) for k, v in self.parts_bps.items()})
        if self.target_bps is not None:
            out["target_bps"] = self.target_bps
            out["prob_beat_target"] = float((s < self.target_bps).mean())
        return out

    def mean_schedule(self) -> np.ndarray:
        """Shares executed in each interval, averaged over the paths."""
        return self.executed.mean(axis=0)


# ------------------------------------------------------------------------------------------------------------------ the engine
def simulate(order: Order, market: Market, algo: Algo, style: Style | str = AGGRESSIVE, scenario: Scenario | str = "normal", paths: int = 400, seed: int = 0,
             target_bps: float | None = None, noise: bool = True) -> ExecutionResult:
    """Run ``algo`` on ``paths`` simulated days and return the per-path shortfall and its decomposition. ``noise=False`` removes the volume noise (the day is exactly the profile), which is how the engine is
    checked against the closed forms. ``target_bps`` adds the probability of beating that cost to the summary."""
    style = STYLES[style] if isinstance(style, str) else style
    scenario = SCENARIOS[scenario] if isinstance(scenario, str) else scenario
    rng = np.random.default_rng(seed)
    window = order.window(market)
    n = window.stop - window.start
    ref, X, side = order.reference(market), order.shares, order.side
    exp_vol = market.expected_volume()[window]
    w = market.variance_weights()[window]
    mult = scenario.multipliers(n)
    algo.prepare(order, market, scenario)
    P = paths

    # random inputs, drawn up front so a path does not depend on what the algorithm does
    z = rng.standard_normal((P, n))
    vol_noise = (np.exp(market.volume_noise * rng.standard_normal((P, n)) - 0.5 * market.volume_noise ** 2) * np.exp(market.day_noise * rng.standard_normal((P, 1)) - 0.5 * market.day_noise ** 2)
                 if noise else np.ones((P, n)))
    a_d, b_d = style.dark_fill * style.dark_concentration, (1.0 - style.dark_fill) * style.dark_concentration
    dark_fill = rng.beta(max(a_d, 1e-6), max(b_d, 1e-6), (P, n)) if style.dark > 0 else np.zeros((P, n))
    drift = np.full(n, scenario.drift_bps * 1e-4 / n)
    sig_i = market.sigma * np.sqrt(w) * mult["vol"]
    phi = scenario.persistence

    exo = np.full(P, ref)                                                                                       # the price without us: noise and drift only
    own = np.zeros(P)                                                                                           # what our own earlier trades have added to it (in price units, in our direction)
    mid = exo + side * own
    mids = np.empty((P, n + 1))
    mids[:, 0] = mid
    R = np.full(P, float(X))
    executed = np.zeros((P, n))
    prev_noise = np.zeros(P)
    spent = np.zeros(P)
    temp_paid = np.zeros(P)
    parts = {k: np.zeros(P) for k in ("timing", "spread", "temporary", "permanent", "fees")}
    sweep_start = n - style.sweep_last
    v_total = np.zeros(P)
    vwap_num = np.zeros(P)
    vwap_den = np.zeros(P)
    price_improved = np.zeros(P)
    swept_shares = np.zeros(P)
    var_elapsed = 0.0

    for i in range(n):
        V = exp_vol[i] * vol_noise[:, i] * mult["volume"][i]
        state = State(i, n, side, ref, R.copy(), mid.copy(), V, float(exp_vol[i]), float(exp_vol.sum()), spent.copy(), temp_paid.copy(), X - R, var_elapsed, market.sigma, float(mult["spread"][i]),
                      float(mult["impact"][i]), float(mult["vol"][i]), float(mult["volume"][i]))
        want = np.clip(algo.want(state), 0.0, R)
        last = i == n - 1
        in_sweep = i >= sweep_start
        cap = order.max_participation / (1.0 - order.max_participation) * V
        want_capped = np.minimum(want, np.maximum(cap, 0.0))
        want = np.where(last, R, want_capped)                                                                  # the last interval finishes the order whatever the cap
        if last:
            swept_shares += np.maximum(R - want_capped, 0.0)                                                   # shares the deadline forced past the participation cap
        plan = algo.plan()
        lag = np.zeros(P) if plan is None else np.maximum(R - plan[i:].sum(), 0.0)                              # shares behind the plan
        mu = np.clip(style.market + style.chase * lag / np.maximum(want, 1.0), 0.0, 1.0)
        mu = np.where(in_sweep, 1.0, mu)
        rest = 1.0 - mu
        base_rest = max(style.dark + style.passive, 1e-12)
        dk = rest * style.dark / base_rest
        ps = rest - dk
        q_m, q_d_want, q_p_want = want * mu, want * dk, want * ps

        # the price move of this interval: unknown to the algorithm, but a passive fill depends on it
        noise_i = phi * prev_noise + np.sqrt(max(1.0 - phi ** 2, 1e-9)) * sig_i[i] * z[:, i] if phi else sig_i[i] * z[:, i]
        r = noise_i + drift[i]
        fav = -side * r / max(sig_i[i], 1e-12)
        fill_p = np.clip(style.passive_fill + style.passive_sensitivity * fav, 0.0, 1.0)
        q_p, q_d = q_p_want * fill_p, q_d_want * dark_fill[:, i]
        q_p = np.where(last | in_sweep, 0.0, q_p)                                                                # passive and dark orders stop resting when the sweep begins
        q_d = np.where(last | in_sweep, 0.0, q_d)
        q_m = np.where(last | in_sweep, want, q_m)
        q_all = q_m + q_p + q_d
        q_all = np.minimum(q_all, R)

        temp = temporary_fraction(q_m, V, market.sigma, market.eta, market.beta) * mult["impact"][i]
        spread_i = market.spread * mult["spread"][i]
        perm_shift = permanent_fraction(q_m + style.leak_passive * q_p + style.leak_dark * q_d, market.adv, market.sigma, market.gamma)
        half_perm = 0.5 * perm_shift
        px_m = mid + side * ref * (0.5 * spread_i + temp + half_perm)
        px_p = mid - side * ref * (0.5 * spread_i) + side * ref * half_perm
        px_d = mid + side * ref * half_perm
        cash_m, cash_p, cash_d = q_m * px_m, q_p * px_p, q_d * px_d
        # shortfall accounting: side * q * (price paid - ref), split into the parts
        parts["timing"] += side * q_all * (exo - ref) / (ref * X) * 1e4
        parts["spread"] += (q_m * 0.5 * spread_i - q_p * 0.5 * spread_i) / X * 1e4
        parts["temporary"] += q_m * temp / X * 1e4
        temp_paid += q_m * temp * ref
        parts["permanent"] += q_all * (own / ref + half_perm) / X * 1e4
        parts["fees"] += (q_m * style.fee_bps - q_p * style.rebate_bps) / X
        spent += side * (cash_m + cash_p + cash_d - q_all * ref)
        price_improved += q_p + q_d
        executed[:, i] = q_all
        R = R - q_all
        v_total += V
        vwap_num += mid * V
        vwap_den += V
        exo = exo + ref * r
        own = own + ref * perm_shift
        mid = exo + side * own
        mids[:, i + 1] = mid
        prev_noise = noise_i
        var_elapsed += float(w[i])

    if np.abs(R).max() > 1e-6 * X:                                                                              # the sweep finishes every path; anything else is a bug in an algorithm
        raise RuntimeError("an order was left unfinished")
    value = ref * X
    shortfall = spent / value * 1e4 + parts["fees"]
    mkt_vwap = vwap_num / np.maximum(vwap_den, 1e-9)
    avg_px = ref + side * spent / X
    result = ExecutionResult(algo=algo.name, style=style.name, scenario=scenario.name, order=order, market=market, shortfall_bps=shortfall, parts_bps=parts,
                             vwap_slippage_bps=side * (avg_px - mkt_vwap) / mkt_vwap * 1e4, arrival_slippage_bps=side * (avg_px - ref) / ref * 1e4, executed=executed, mid=mids,
                             participation=X / np.maximum(v_total + X, 1e-9), swept=swept_shares / X, price_improvement=price_improved / X, target_bps=target_bps)
    return result


def compare(order: Order, market: Market, algos, style: Style | str = AGGRESSIVE, scenario: Scenario | str = "normal", paths: int = 400, seed: int = 0, target_bps: float | None = None):
    """Run several algorithms on the SAME simulated days (same seed) and return one summary row each, so the differences are the algorithms' and not the luck of the draw."""
    import pandas as pd

    rows = [simulate(order, market, a, style, scenario, paths, seed, target_bps).summary() for a in algos]
    return pd.DataFrame(rows).set_index("algo")
