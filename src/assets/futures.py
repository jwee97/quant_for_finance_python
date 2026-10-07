"""Futures: contract calendars, term-structure panels, continuous series, carry and roll yield, and a synthetic two-factor commodity curve.

A futures price series is not a price series: every contract expires, so a long history has to be stitched from many contracts, and HOW it is stitched changes every
statistic computed from it.

* ``contract_calendar``: expiry dates for a monthly cycle under common exchange rules.
* ``term_structure``: a panel ``F1, F2, ...`` (nearest, next-nearest, ...) from a long table of contract prices, with each contract's time to expiry.
* ``continuous_series``: the continuous front contract rolled ``roll_days`` before expiry, ``adjust = 'ratio'`` (back-adjusted multiplicatively, so percentage returns are
  right), ``'difference'`` (Panama canal, additive, keeps point moves) or ``'none'`` (raw splice, which contains the roll gap as a fake return).
* ``excess_return_series``: the return of holding the front contract and rolling, measured contract by contract so the roll gap never appears: the real, tradable excess return,
  which contains the roll yield (positive in backwardation, negative in contango).
* ``carry`` / ``roll_yield``: annualised ``ln(F1 / F2) / (T2 - T1)``: what the position earns if the curve does not move (Koijen, Moskowitz, Pedersen & Vrugt 2018).
* ``basis_momentum`` (Boons & Prado 2019): trailing average of the return difference between the first and second contract.
* ``synthetic_term_structure``: the Schwartz-Smith (2000) two-factor model for a commodity: short-term deviation ``chi`` (mean reverting) and equilibrium level ``xi`` (random walk with
  drift), with a risk premium that bends the curve into contango or backwardation and an optional seasonal pattern; the generator returns the full contract panel and the true parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MONTH_CODES = "FGHJKMNQUVXZ"


def contract_calendar(start, end, months=tuple(range(1, 13)), rule: str = "third_friday") -> pd.DataFrame:
    """Expiry dates between ``start`` and ``end`` for the delivery ``months`` (1-12). ``rule``: ``third_friday`` (equity index), ``business_day_before_25th`` (CME energy: three
    business days before the 25th calendar day), ``15th`` (the 15th, rolled back to a business day) or ``last_business_day``. Returns ``contract`` (e.g. ``H22``) and ``expiry``."""
    rows = []
    for ts in pd.date_range(pd.Timestamp(start) - pd.offsets.MonthBegin(1), pd.Timestamp(end) + pd.DateOffset(months=18), freq="MS"):
        if ts.month not in months:
            continue
        if rule == "third_friday":
            first = ts + pd.Timedelta(days=(4 - ts.weekday()) % 7)
            exp = first + pd.Timedelta(days=14)
        elif rule == "business_day_before_25th":
            exp = pd.Timestamp(ts.year, ts.month, 25) - pd.offsets.BDay(3)
        elif rule == "15th":
            exp = pd.Timestamp(ts.year, ts.month, 15)
            exp = exp if exp.weekday() < 5 else exp - pd.offsets.BDay(1)
        elif rule == "last_business_day":
            exp = ts + pd.offsets.BMonthEnd(0)
        else:
            raise ValueError("rule must be third_friday, business_day_before_25th, 15th or last_business_day")
        rows.append({"contract": f"{MONTH_CODES[ts.month - 1]}{ts.year % 100:02d}", "delivery_month": ts, "expiry": pd.Timestamp(exp)})
    return pd.DataFrame(rows).sort_values("expiry").reset_index(drop=True)


def term_structure(long: pd.DataFrame, depth: int = 4) -> dict[str, pd.DataFrame]:
    """From a long table with columns ``date, contract, expiry, price`` build the panels ``price`` and ``ttm`` (years to expiry) for the ``depth`` nearest unexpired
    contracts, columns ``F1 .. F{depth}``, and ``contract`` (the name of each rank)."""
    long = long.copy()
    long["date"], long["expiry"] = pd.to_datetime(long["date"]), pd.to_datetime(long["expiry"])
    long = long[long["expiry"] >= long["date"]].dropna(subset=["price"])
    long["rank"] = long.groupby("date")["expiry"].rank(method="first").astype(int)
    long = long[long["rank"] <= depth]
    out = {}
    for name, col in (("price", "price"), ("contract", "contract")):
        out[name] = long.pivot(index="date", columns="rank", values=col).rename(columns=lambda r: f"F{r}").sort_index()
    ttm = long.assign(ttm=(long["expiry"] - long["date"]).dt.days / 365.0).pivot(index="date", columns="rank", values="ttm").rename(columns=lambda r: f"F{r}").sort_index()
    out["ttm"] = ttm
    return out


def carry(price: pd.DataFrame, ttm: pd.DataFrame, near: str = "F1", far: str = "F2") -> pd.Series:
    """Annualised carry (roll yield) ``ln(F_near / F_far) / (T_far - T_near)``: positive in backwardation (the curve slopes down, a long earns the roll)."""
    dt = (ttm[far] - ttm[near]).where(lambda x: x > 1e-6)
    return np.log(price[near] / price[far]) / dt


roll_yield = carry


@dataclass
class ContinuousSeries:
    price: pd.Series                  # the continuous (adjusted) series
    raw: pd.Series                    # the unadjusted splice
    contract: pd.Series               # the contract held on each date
    roll_dates: list
    returns: pd.Series                # returns of the continuous series (adjust-dependent)


def continuous_series(long: pd.DataFrame, roll_days: int = 5, adjust: str = "ratio") -> ContinuousSeries:
    """Stitch the front contract: hold the nearest contract until ``roll_days`` calendar days before its expiry, then switch to the next one. At a roll the new contract's
    price is compared with the old one's on the SAME day, and all earlier history is rescaled (``ratio``) or shifted (``difference``) by that gap, so the adjusted series has no
    jump at the roll. ``none`` leaves the gap in (an artificial return)."""
    if adjust not in ("ratio", "difference", "none"):
        raise ValueError("adjust must be ratio, difference or none")
    long = long.copy()
    long["date"], long["expiry"] = pd.to_datetime(long["date"]), pd.to_datetime(long["expiry"])
    by_contract = {c: g.set_index("date")["price"] for c, g in long.groupby("contract")}
    expiry = long.drop_duplicates("contract").set_index("contract")["expiry"]
    dates = sorted(long["date"].unique())
    held, raw, roll_dates = {}, {}, []
    adjusted_offsets = {}
    current = None
    for d in dates:
        live = expiry[(expiry - d).dt.days > roll_days].sort_values()
        live = live[[c for c in live.index if d in by_contract[c].index]]
        if live.empty:
            continue
        target = live.index[0]
        if current is None:
            current = target
        elif target != current:
            old_px, new_px = by_contract[current].get(d, np.nan), by_contract[target].get(d, np.nan)
            if np.isfinite(old_px) and np.isfinite(new_px):
                roll_dates.append(d)
                adjusted_offsets[d] = (old_px, new_px)
            current = target
        held[d] = current
        raw[d] = by_contract[current][d]
    raw = pd.Series(raw)
    contract = pd.Series(held)
    adj = raw.copy()
    if adjust != "none":
        for d in sorted(roll_dates, reverse=True):
            old_px, new_px = adjusted_offsets[d]
            mask = adj.index < d
            adj[mask] = adj[mask] * (new_px / old_px) if adjust == "ratio" else adj[mask] + (new_px - old_px)
    rets = adj.pct_change() if adjust != "difference" else adj.diff() / adj.shift(1)
    return ContinuousSeries(adj, raw, contract, roll_dates, rets)


def excess_return_series(long: pd.DataFrame, roll_days: int = 5) -> pd.Series:
    """The tradable excess return of a roll strategy: each day's return is the change in the price of the contract held at the PREVIOUS close, measured on that contract.
    A roll costs nothing in this series (the old contract is sold and the new one bought at the same day's prices); the roll yield shows up as the drift of the contract."""
    long = long.copy()
    long["date"], long["expiry"] = pd.to_datetime(long["date"]), pd.to_datetime(long["expiry"])
    pivot = long.pivot(index="date", columns="contract", values="price").sort_index()
    expiry = long.drop_duplicates("contract").set_index("contract")["expiry"]
    out, prev_contract = {}, None
    dates = pivot.index
    for i, d in enumerate(dates):
        live = expiry[(expiry - d).dt.days > roll_days].sort_values()
        live = [c for c in live.index if np.isfinite(pivot.at[d, c])]
        if not live:
            continue
        if prev_contract is not None and np.isfinite(pivot.at[d, prev_contract]) and i > 0 and np.isfinite(pivot.at[dates[i - 1], prev_contract]):
            out[d] = pivot.at[d, prev_contract] / pivot.at[dates[i - 1], prev_contract] - 1.0
        prev_contract = live[0]
    return pd.Series(out, name="excess_return")


def basis_momentum(contract_returns: pd.DataFrame, window: int = 252, near: str = "F1", far: str = "F2") -> pd.Series:
    """Boons & Prado (2019): the trailing mean of the daily return of the nearest contract minus that of the second; high basis momentum means the front of the curve is
    outperforming the back (tightening supply) and has predicted subsequent returns."""
    return (contract_returns[near] - contract_returns[far]).rolling(window, min_periods=window // 2).mean() * 252.0


def contract_returns(price_panel: pd.DataFrame, contract_panel: pd.DataFrame) -> pd.DataFrame:
    """Daily returns of the contract that occupied each rank on the PREVIOUS date (so ``F2``'s return is that of the contract that was second yesterday, a tradable series)."""
    out = pd.DataFrame(index=price_panel.index, columns=price_panel.columns, dtype=float)
    dates = price_panel.index
    for i in range(1, len(dates)):
        d0, d1 = dates[i - 1], dates[i]
        for col in price_panel.columns:
            c = contract_panel.at[d0, col]
            hit = contract_panel.loc[d1][contract_panel.loc[d1] == c]
            if len(hit) and np.isfinite(price_panel.at[d0, col]):
                out.at[d1, col] = price_panel.at[d1, hit.index[0]] / price_panel.at[d0, col] - 1.0
    return out


# ------------------------------------------------------------------------------------------------------ synthetic commodity curve
@dataclass
class SchwartzSmithParams:
    kappa: float = 1.5               # mean reversion of the short-term deviation chi
    sigma_chi: float = 0.35
    mu_xi: float = 0.02              # drift of the equilibrium log price (physical measure)
    sigma_xi: float = 0.15
    rho: float = 0.3
    lambda_chi: float = 0.10         # risk premium on chi: positive -> backwardation-leaning curve in this sign convention; see A(T)
    mu_xi_star: float = 0.01         # risk-neutral drift of xi
    seasonal_amp: float = 0.0        # amplitude of an annual cosine in log futures prices (delivery month)
    seasonal_peak_month: int = 1
    s0: float = 50.0


def schwartz_smith_A(tau, p: SchwartzSmithParams):
    """``A(T)`` of Schwartz & Smith (2000), eq. (9): ``ln F(T) = e^{-kappa T} chi + xi + A(T)``."""
    tau = np.asarray(tau, dtype=float)
    k = p.kappa
    return (p.mu_xi_star * tau - (1.0 - np.exp(-k * tau)) * p.lambda_chi / k
            + 0.5 * ((1.0 - np.exp(-2.0 * k * tau)) * p.sigma_chi ** 2 / (2.0 * k) + p.sigma_xi ** 2 * tau + 2.0 * (1.0 - np.exp(-k * tau)) * p.rho * p.sigma_chi * p.sigma_xi / k))


def synthetic_term_structure(n_days: int = 1000, start: str = "2018-01-02", params: SchwartzSmithParams | None = None, months=tuple(range(1, 13)), rule: str = "business_day_before_25th",
                             n_listed: int = 12, noise: float = 0.002, seed: int = 0) -> dict:
    """Simulate the two factors daily (exact Gaussian transitions) and price every listed contract with the model. Returns the long contract table (``date, contract, expiry,
    price``), the factors, and the parameters. ``noise`` is multiplicative pricing noise per quote."""
    p = params or SchwartzSmithParams()
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0
    a = np.exp(-p.kappa * dt)
    sd_chi = p.sigma_chi * np.sqrt((1 - a ** 2) / (2 * p.kappa))
    cov = np.array([[sd_chi ** 2, p.rho * sd_chi * p.sigma_xi * np.sqrt(dt)], [p.rho * sd_chi * p.sigma_xi * np.sqrt(dt), p.sigma_xi ** 2 * dt]])
    L = np.linalg.cholesky(cov)
    chi, xi = np.zeros(n_days), np.zeros(n_days)
    xi[0] = np.log(p.s0)
    for t in range(1, n_days):
        z = L @ rng.standard_normal(2)
        chi[t] = a * chi[t - 1] + z[0]
        xi[t] = xi[t - 1] + (p.mu_xi - 0.5 * p.sigma_xi ** 2) * dt + z[1]
    idx = pd.bdate_range(start, periods=n_days)
    cal = contract_calendar(idx[0], idx[-1], months, rule)
    rows = []
    for t, d in enumerate(idx):
        live = cal[cal["expiry"] >= d].head(n_listed)
        tau = ((live["expiry"] - d).dt.days.to_numpy() / 365.0)
        seasonal = p.seasonal_amp * np.cos(2 * np.pi * (live["delivery_month"].dt.month.to_numpy() - p.seasonal_peak_month) / 12.0)
        logf = np.exp(-p.kappa * tau) * chi[t] + xi[t] + schwartz_smith_A(tau, p) + seasonal + noise * rng.standard_normal(len(tau))
        rows.append(pd.DataFrame({"date": d, "contract": live["contract"].to_numpy(), "expiry": live["expiry"].to_numpy(), "price": np.exp(logf), "ttm": tau, "log_price": logf}))
    return {"long": pd.concat(rows, ignore_index=True), "chi": pd.Series(chi, index=idx), "xi": pd.Series(xi, index=idx), "params": p, "calendar": cal}
