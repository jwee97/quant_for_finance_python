"""A panel of factors and returns whose truth is known, to check the equity tools against.

Every factor is a table of standardised exposures (dates by assets). The return of the next period is ``exposure @ premium + noise`` where the premium of each factor is ``mean + shock`` on that date,
so the true average IC of a factor, the correlation between factors (cross-sectionally) and the correlation between their premia over time are all things the caller chose. A tool that estimates
them should recover them, and one that claims an edge on a factor with premium zero is wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FactorPanel:
    factors: dict            # name -> dates x assets, known at the date
    forward: pd.DataFrame    # forward[t] = the return from t to the next date
    realized: pd.DataFrame   # realized[t] = the return of the period that ENDS at t (forward shifted by one)
    premia: pd.DataFrame     # the true premium per unit of exposure that each date's exposures earned (dates x factors)
    mean_premia: pd.Series   # the long-run mean of the premium of each factor


def _chol(corr, k: int, what: str) -> np.ndarray:
    c = np.eye(k) if corr is None else np.asarray(corr, dtype=float)
    if c.shape != (k, k) or not np.allclose(c, c.T):
        raise ValueError(f"{what} must be a symmetric {k} x {k} matrix")
    return np.linalg.cholesky(c)


def simulate_factor_panel(n_assets: int = 200, n_periods: int = 120, premia=(0.004, 0.003, 0.0, -0.002), names: list[str] | None = None, corr=None, premium_vol: float = 0.004,
                          premium_corr=None, idio_vol: float = 0.06, persistence: float = 0.0, seed: int = 0, start: str = "2005-01-01") -> FactorPanel:
    """Simulate ``n_periods`` monthly cross-sections of ``n_assets`` stocks.

    ``premia`` is the mean premium of each factor (the return per unit of exposure, where an exposure has a cross-sectional standard deviation of one); ``corr`` the cross-sectional correlation of
    the factors' exposures; ``premium_vol`` and ``premium_corr`` the volatility and correlation across time of the premia; ``idio_vol`` the standard deviation of the unexplained return;
    ``persistence`` the autocorrelation of an asset's exposure from one date to the next (0 = redrawn every date)."""
    mean = np.asarray(premia, dtype=float)
    k = len(mean)
    names = list(names) if names is not None else [f"f{i + 1}" for i in range(k)]
    if len(names) != k or len(set(names)) != k:
        raise ValueError("names must be unique and match premia")
    if not 0.0 <= persistence < 1.0:
        raise ValueError("persistence must be in [0, 1)")
    L_x, L_p = _chol(corr, k, "corr"), _chol(premium_corr, k, "premium_corr")
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=n_periods, freq="MS")
    assets = [f"S{i:03d}" for i in range(n_assets)]
    exposure = np.empty((n_periods, n_assets, k))
    prev = rng.normal(size=(n_assets, k)) @ L_x.T
    for t in range(n_periods):
        fresh = rng.normal(size=(n_assets, k)) @ L_x.T
        prev = persistence * prev + np.sqrt(1.0 - persistence ** 2) * fresh if t else prev
        exposure[t] = prev
    premium = mean + premium_vol * (rng.normal(size=(n_periods, k)) @ L_p.T)
    forward = np.einsum("tak,tk->ta", exposure, premium) + idio_vol * rng.normal(size=(n_periods, n_assets))
    factors = {n: pd.DataFrame(exposure[:, :, j], index=index, columns=assets) for j, n in enumerate(names)}
    fwd = pd.DataFrame(forward, index=index, columns=assets)
    return FactorPanel(factors, fwd, fwd.shift(1), pd.DataFrame(premium, index=index, columns=names), pd.Series(mean, index=names))


# ------------------------------------------------------------------------------------------------------------------ a world with statements
DEFAULT_PREMIA = {"value": 0.005, "quality": 0.003, "investment": -0.003, "momentum": 0.003, "reversal": 0.003, "revision": 0.003}


@dataclass
class FundamentalWorld:
    prices: pd.DataFrame     # daily adjusted prices
    table: pd.DataFrame      # the contents of a fundamentals file: one row per company and quarter (statements) and per company and month (analyst fields)
    premia: dict             # the planted monthly premium per unit of z-score of each driver
    drivers: pd.DataFrame    # the drivers' z-scores that returns were drawn with (a column per driver and asset, rows are the month starts), long form: (date, ticker) index
    n_days: int = 0


def _z(x: np.ndarray, clip: float = 3.0) -> np.ndarray:
    """Cross-sectional z-score with missing and infinite values at zero and outliers clipped, which is how the planted effects see a driver."""
    ok = np.isfinite(x)
    if ok.sum() < 3:
        return np.zeros_like(x)
    mu, sd = x[ok].mean(), x[ok].std()
    return np.where(ok, np.clip((x - mu) / sd, -clip, clip), 0.0) if sd > 0 else np.zeros_like(x)


def _ar1(rng, phi: float, sd: float, shape) -> np.ndarray:
    out = np.empty(shape)
    out[0] = rng.normal(0, sd / np.sqrt(1 - phi ** 2), shape[1:])
    for t in range(1, shape[0]):
        out[t] = phi * out[t - 1] + rng.normal(0, sd, shape[1:])
    return out


def simulate_fundamental_world(n_assets: int = 80, n_years: int = 14, seed: int = 0, premia: dict | None = None, idio_vol: float = 0.06, start: str = "2006-01-02",
                               filing_lag: int = 45) -> FundamentalWorld:
    """Companies with consistent quarterly statements, monthly analyst fields and daily prices whose expected returns depend on known drivers.

    The drivers are the factors the library measures: ``value`` (book to price), ``quality`` (return on net operating assets), ``investment`` (an inverted U in abnormal capital expenditure: both
    ends are worse than the middle), ``momentum`` (the nine-month return ending a month ago), ``reversal`` (minus the last month's return) and ``revision`` (the nine-month change in the consensus EPS
    over price). On the first trading day of every month the drivers are computed from the statements filed by then (``filing_lag`` calendar days after each quarter end) and the prices so far; the
    month's expected return is ``sum(premium * z-score)``; daily returns add a market factor with betas, and idiosyncratic noise (``idio_vol`` monthly). Each statement is therefore consistent with
    the others (the balance sheet balances, net income follows from operating income, interest and tax) and everything the premia depend on is known at the time."""
    p = {**DEFAULT_PREMIA, **(premia or {})}
    rng = np.random.default_rng(seed)
    n = n_assets
    n_days = n_years * 252
    days = pd.bdate_range(start, periods=n_days)
    n_q = (n_years + 4) * 4 + 1
    period_end = pd.period_range(pd.Period(pd.Timestamp(start) - pd.DateOffset(years=4), "Q"), periods=n_q, freq="Q").to_timestamp(how="end").normalize()
    available = period_end + pd.Timedelta(days=int(filing_lag))
    avail_pos = days.searchsorted(available, side="left")
    tickers = [f"F{i:03d}" for i in range(n)]

    sales0 = np.exp(rng.normal(np.log(2e9), 0.8, n))
    gbar = rng.normal(0.05, 0.03, n)
    cogs_r, sga_r, dep_r = rng.uniform(0.45, 0.70, n), rng.uniform(0.08, 0.20, n), rng.uniform(0.03, 0.06, n)
    turnover, leverage, capex_r, payout = rng.uniform(0.6, 1.6, n), rng.uniform(0.10, 0.45, n), rng.uniform(0.03, 0.08, n), rng.uniform(0.0, 0.5, n)
    beta, plant_ratio = rng.uniform(0.7, 1.3, n), rng.uniform(1.6, 2.2, n)

    shape = (n_q, n)
    sales = sales0 * np.cumprod(1.0 + gbar / 4 + _ar1(rng, 0.6, 0.012, shape), axis=0)
    cogs = sales * (cogs_r + _ar1(rng, 0.8, 0.01, shape))
    sga, dep = sales * sga_r, sales * dep_r
    ebitda = sales - cogs - sga
    operating_income = ebitda - dep
    assets = sales / turnover * np.exp(_ar1(rng, 0.9, 0.02, shape))
    net_plant = 0.45 * assets
    current_assets, cash, long_term_investments = 0.35 * assets, 0.08 * assets, 0.03 * assets
    current_liabilities, short_debt, long_debt = 0.14 * assets, 0.03 * assets, (leverage - 0.03) * assets
    total_liabilities = current_liabilities + long_debt + 0.02 * assets
    equity = assets - total_liabilities
    interest = 0.05 * (short_debt + long_debt)
    net_income = (operating_income - interest) * 0.75
    capex = capex_r * sales * np.exp(_ar1(rng, 0.5, 0.25, shape))
    cfo = net_income + dep + _ar1(rng, 0.5, 0.01, shape) * sales
    dividends = payout * np.maximum(net_income, 0.0)
    issuance = sales * 0.01 * np.maximum(0.0, _ar1(rng, 0.7, 1.0, shape) + 0.2)
    buyback = sales * 0.01 * np.maximum(0.0, _ar1(rng, 0.7, 1.0, shape) + 0.3)
    debt = short_debt + long_debt
    change = np.vstack([np.zeros((1, n)), np.diff(debt, axis=0)])
    debt_issued, debt_repaid = np.maximum(change, 0.0) + 0.002 * sales, np.maximum(-change, 0.0) + 0.002 * sales
    shares0 = 1e8 * np.exp(rng.normal(0, 0.5, n))
    shares = shares0 * np.cumprod(1.0 + (issuance - buyback) / (1.5 * equity) / 4, axis=0)
    eps = net_income / shares

    asfiled = lambda pos: np.searchsorted(avail_pos, pos, side="right") - 1          # the latest quarter filed by trading day `pos` (the same rule the point-in-time panel uses)
    noa = (assets - cash - long_term_investments) - (total_liabilities - (short_debt + long_debt))

    month_pos = np.arange(0, n_days, 21)
    ltg = gbar + 0.01 * rng.normal(size=n)
    u = np.zeros((len(month_pos), n))
    u[0] = rng.normal(0, 0.05, n)
    for m in range(1, len(month_pos)):
        u[m] = 0.92 * u[m - 1] + 0.025 * rng.normal(size=n)
    latent_q = np.searchsorted(period_end.to_numpy(), days[month_pos].to_numpy(), side="right") - 1
    eps_fy1 = eps[latent_q] * np.sqrt(1.0 + ltg) * np.exp(u)                      # the analysts know the current quarter's earnings, so the latent quarter, not the filed one
    du = np.vstack([np.zeros((1, n)), np.diff(u, axis=0)])
    n_est = rng.integers(5, 21, (len(month_pos), n))
    n_up = rng.binomial(n_est, np.clip(0.25 + 4 * du, 0.02, 0.9))
    n_down = rng.binomial(n_est - n_up, np.clip(0.25 - 4 * du, 0.02, 0.9))
    eps_at = lambda m: eps_fy1[m]

    price = np.empty((n_days, n))
    q0 = asfiled(0)
    price[0] = equity[q0] * np.exp(rng.normal(0.35, 0.45, n)) / shares[q0]
    market = rng.normal(0.0003, 0.008, n_days)
    noise = rng.normal(0.0, idio_vol / np.sqrt(21), (n_days, n))
    driver_rows = {}
    for m, pos in enumerate(month_pos):
        q = asfiled(pos)
        z = {"value": _z(np.log(equity[q] / (price[pos] * shares[q])))}                       # value spreads are lognormal: the premium is paid on the log of book to price
        z["quality"] = _z((operating_income[q] * 0.75 / ((noa[q] + noa[asfiled(pos - 252)]) / 2.0)) if pos >= 252 else np.full(n, np.nan))
        if pos >= 756:
            history = (capex[asfiled(pos - 252)] + capex[asfiled(pos - 504)] + capex[asfiled(pos - 756)]) / 3.0
            zi = _z(capex[q] / history - 1.0)
            z["investment"] = zi ** 2 - np.mean(zi ** 2)
        else:
            z["investment"] = np.zeros(n)
        z["momentum"] = _z(price[pos - 21] / price[pos - 21 - 189] - 1.0) if pos >= 210 else np.zeros(n)
        z["reversal"] = _z(-(price[pos] / price[pos - 21] - 1.0)) if pos >= 21 else np.zeros(n)
        z["revision"] = _z((eps_at(m) - eps_at(m - 9)) / price[pos]) if m >= 9 else np.zeros(n)
        driver_rows[days[pos]] = z
        expected = sum(p[k] * z[k] for k in p)
        end = min(pos + 21, n_days - 1)
        if pos == 0:
            lo = 1
        else:
            lo = pos + 1
        for d in range(lo, end + 1):
            r = beta * market[d] + expected / 21.0 + noise[d]
            price[d] = price[d - 1] * (1.0 + np.maximum(r, -0.5))
    prices = pd.DataFrame(price, index=days, columns=tickers)

    rows = []
    for k, ticker in enumerate(tickers):
        rows.append(pd.DataFrame({
            "ticker": ticker, "period_end": period_end, "available": available, "sales": sales[:, k], "cogs": cogs[:, k], "sga": sga[:, k], "ebitda": ebitda[:, k], "operating_income": operating_income[:, k],
            "net_income": net_income[:, k], "cfo": cfo[:, k], "capex": capex[:, k], "depreciation": dep[:, k], "interest_expense": interest[:, k], "dividends": dividends[:, k],
            "equity_issuance": issuance[:, k], "equity_repurchase": buyback[:, k], "debt_issuance": debt_issued[:, k], "debt_repayment": debt_repaid[:, k], "tax_rate": 0.25,
            "total_assets": assets[:, k], "current_assets": current_assets[:, k], "current_liabilities": current_liabilities[:, k], "cash": cash[:, k], "short_term_debt": short_debt[:, k],
            "long_term_debt": long_debt[:, k], "total_liabilities": total_liabilities[:, k], "book_equity": equity[:, k], "long_term_investments": long_term_investments[:, k],
            "gross_plant": net_plant[:, k] * plant_ratio[k], "net_plant": net_plant[:, k], "shares_outstanding": shares[:, k]}))
        rows.append(pd.DataFrame({"ticker": ticker, "available": days[month_pos], "eps_fy1": eps_fy1[:, k], "ltg": ltg[k], "n_up": n_up[:, k], "n_down": n_down[:, k],
                                  "n_estimates": n_est[:, k]}))
    table = pd.concat(rows, ignore_index=True)
    drivers = pd.concat({d: pd.DataFrame(z, index=tickers) for d, z in driver_rows.items()}, names=["date", "ticker"])
    return FundamentalWorld(prices, table, p, drivers, n_days)


# ------------------------------------------------------------------------------------------------------------------ worlds for factor timing
def _month_ends(index: pd.DatetimeIndex) -> np.ndarray:
    m = pd.Series(index.to_period("M"), index=index)
    return np.flatnonzero((m != m.shift(-1)).to_numpy())                                            # positions of the last trading day of each month


def simulate_timing_world(seed: int = 0, years: int = 16, n: int = 50, premium=None, state_of=None, market_regimes: bool = False, sigma: float = 0.015, start: str = "2004-01-05"):
    """Daily returns in which momentum is paid a premium that depends on the calendar or on a state: ``premium(month, state)`` a month per cross-sectional standard deviation of the exposure, where
    ``month`` (1 to 12) is the month the return is earned in and ``state`` is ``state_of(position, returns)`` at the month-end before it (``position`` indexes the trading days; ``returns`` is the array of
    daily returns simulated so far, which may be inspected up to and including that position). The exposure is the 12-1 month return of the prices as simulated up to that month-end, which is what the
    price-based factors compute, so a timing model has a premium of known size to find. ``market_regimes`` adds a common component that swings between up and down years.

    Returns the trading dates, the prices (days by assets) and the state at each month-end after the first year."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=years * 252)
    ends = _month_ends(idx)
    ret = rng.normal(0.0002, sigma, (len(idx), n))
    if market_regimes:
        regime = 1
        for k in range(len(ends)):
            if rng.random() < 0.08:
                regime = -regime
            lo = 0 if k == 0 else ends[k - 1] + 1
            ret[lo:ends[k] + 1] += regime * 0.0012
    prices = np.zeros_like(ret)
    prices[:ends[0] + 1] = 100.0 * np.cumprod(1 + ret[:ends[0] + 1], axis=0)
    states = {}
    for k, s in enumerate(ends[:-1]):
        lo, hi = s + 1, ends[k + 1] + 1
        if s >= 252:
            mom = prices[s - 21] / prices[s - 252] - 1.0
            z = (mom - mom.mean()) / mom.std()
            st = state_of(s, ret) if state_of else 0
            states[s] = st
            ret[lo:hi] += (premium(idx[lo].month, st) / (hi - lo)) * z[None, :]
        prices[lo:hi] = prices[lo - 1] * np.cumprod(1 + ret[lo:hi], axis=0)
    return idx, prices, states


def simulate_earnings_world(seed: int = 0, years: int = 12, n: int = 60, premium: float = 0.02, start: str = "2005-01-03"):
    """Companies that report every quarter in the same month of the quarter and on the same business day, earning ``premium`` more in the months in which they report, with a spike in trading volume on the
    day. Returns the prices, the volume and the planted announcement days (each a days by companies table)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=years * 252)
    month = idx.month.to_numpy()
    bday = idx.to_series().groupby(idx.to_period("M")).cumcount().to_numpy()
    offset = rng.integers(0, 3, n)                                                                  # which month of the quarter the company reports in
    day = rng.integers(4, 16, n)
    reporting_month = ((month[:, None] - 1) % 3) == offset[None, :]
    events = (reporting_month & (bday[:, None] == day[None, :])).astype(float)
    ret = rng.normal(0.0003, 0.015, (len(idx), n)) + premium / 21.0 * reporting_month
    volume = np.exp(rng.normal(0.0, 0.3, (len(idx), n))) * 1e6 * (1 + 4 * events)
    cols = [f"S{i:02d}" for i in range(n)]
    return (pd.DataFrame(100 * np.cumprod(1 + ret, axis=0), index=idx, columns=cols), pd.DataFrame(volume, index=idx, columns=cols), pd.DataFrame(events, index=idx, columns=cols))


def simulate_nonlinear_world(seed: int = 0, years: int = 16, n: int = 60, effect=None, sigma: float = 0.015, start: str = "2004-01-05"):
    """Daily returns in which the expected return of next month is a function of two price characteristics: ``effect(z_mom, z_rev)`` (a vector over the assets, in return per month) with ``z_mom`` the
    standardised 12-1 month return and ``z_rev`` the standardised minus the last month's return, both computed from the prices simulated so far, exactly as the price-based factors compute them. A linear
    learner can find the part of ``effect`` that is a straight line in them; a square or a product needs a nonlinear one."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=years * 252)
    ends = _month_ends(idx)
    ret = rng.normal(0.0002, sigma, (len(idx), n))
    prices = np.zeros_like(ret)
    prices[:ends[0] + 1] = 100.0 * np.cumprod(1 + ret[:ends[0] + 1], axis=0)
    zs = lambda v: (v - v.mean()) / v.std()
    for k, s in enumerate(ends[:-1]):
        lo, hi = s + 1, ends[k + 1] + 1
        if s >= 252:
            z_mom = zs(prices[s - 21] / prices[s - 252] - 1.0)
            z_rev = zs(-(prices[s] / prices[s - 21] - 1.0))
            ret[lo:hi] += (effect(z_mom, z_rev) / (hi - lo))[None, :]
        prices[lo:hi] = prices[lo - 1] * np.cumprod(1 + ret[lo:hi], axis=0)
    return idx, prices
