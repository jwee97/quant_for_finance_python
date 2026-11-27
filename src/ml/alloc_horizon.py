"""Horizon arithmetic for the portfolio allocation: four quantities that are easy to confuse and are kept apart here.

``h``  the prediction horizon: the window the model's number is a forecast of (``dataset.horizon``); outcomes resolve ``lag + h`` sessions after the decision.
``L``  the intended holding window: how long a position the allocation takes is meant to be held before it is reconsidered; the window the optimisation is posed over (``allocation.holding_days``; ``h`` when not set).
``r``  the rebalancing frequency: how often a new decision is made (``allocation.opt_every``, in sessions), and, separately, how often the engine is allowed to trade (``backtest.rebalance``).
``lag`` the execution lag between a decision and the first return it earns (``dataset.lag`` + ``backtest.extra_lag``).

Expected return over L. A forecast is an expected return over h. Over a window L it is taken to accrue evenly within the horizon and not at all beyond it: ``mu_L = mu_h * min(L, h) / h``. This is an assumption (a model
trained on an h-day target says nothing about days after h), and it is conservative: it never extrapolates an edge past its horizon.

Covariance over L. Three estimators, recorded with every decision:

``linear``          the daily covariance times L (returns uncorrelated through time: the legacy rule when L = h).
``variance_ratio``  the daily correlation, each asset's variance multiplied by its Lo-MacKinlay variance ratio at L (Bartlett-weighted autocorrelations of its past daily returns, clipped to [0.25, 4]): a
                    positively autocorrelated asset is riskier over L days than the square-root rule says, a mean-reverting one less so.
``multiperiod``     the covariance of overlapping L-day compounded returns over the same past window, blended with the linear estimate by ``n_eff / (n_eff + n0)`` where ``n_eff = scenarios / L`` is the number of
                    independent windows the past holds (overlapping windows share days, so they carry far less information than their count): with few independent windows the answer is the linear estimate.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import risk_models as rm

COV_HORIZON_METHODS = ("linear", "variance_ratio", "multiperiod")
N0 = 12.0                                   # independent windows at which the direct estimate and the linear one carry equal weight


EDGE_MODELS = ("linear", "compound")
BEYOND = ("zero", "persist")
FORECAST_SOURCES = ("scaled", "direct")


def edge_scale(holding: int, horizon: int, beyond: str = "zero") -> float:
    """Share of the h-day expected return earned over an L-day holding window. ``zero`` (the default): evenly within the horizon and nothing beyond it, ``min(L, h) / h``. ``persist``: the same daily rate
    continues past the horizon, ``L / h`` (an assumption nothing in a model trained on an h-day target supports; offered, labelled, and never the default)."""
    frac = int(holding) / float(horizon) if beyond == "persist" else min(int(holding), int(horizon)) / float(horizon)
    return frac


def scale_expected_return(mu, holding: int, horizon: int, model: str = "linear", beyond: str = "zero"):
    """Expected simple return over the holding window from the h-day forecast ``mu``.

    Identity versus assumption, kept apart:
    * IDENTITY: if the daily log return has a constant mean, the expected log return over a window is proportional to its length, and for L = h the forecast needs no scaling at all.
    * ASSUMPTION ``linear``: the simple return scales the same way (``mu * L / h``): exact to first order, and a small overstatement for large returns (it ignores compounding).
    * ASSUMPTION ``compound``: ``(1 + mu) ** (L / h) - 1``: exact under a constant daily growth rate.
    * ASSUMPTION ``zero`` beyond the horizon: the model says nothing about days after h, so none is credited. ``persist`` assumes the rate continues.
    With a *direct* forecast (a model trained on an L-day target, ``forecast_source = direct``) none of these is used."""
    frac = edge_scale(holding, horizon, beyond)
    if model == "compound":
        return (1.0 + mu).clip(lower=1e-9) ** frac - 1.0
    return mu * frac


def assumptions(holding: int, horizon: int, model: str, beyond: str, cov_method: str, source: str) -> list[str]:
    """The modelling assumptions behind a decision's expected return and covariance, in words (what is an identity is not listed)."""
    out = []
    if source == "direct":
        out.append(f"the forecast is a direct {horizon}-day forecast of the holding window: no scaling is applied")
    elif int(holding) != int(horizon):
        out.append(f"the {horizon}-day forecast is scaled to the {holding}-day holding window ({model} scaling; "
                   + ("nothing is credited beyond the forecast horizon" if beyond == "zero" else "the same daily rate is assumed to continue beyond the horizon") + ")")
    out.append({"linear": "returns are uncorrelated through time, so the covariance over the window is the daily covariance times its length",
                "variance_ratio": "each asset's variance over the window follows its past serial correlation (variance ratio); the correlation between assets is the daily one",
                "multiperiod": "the covariance is estimated directly from overlapping windows of past returns and shrunk toward the linear estimate by the number of independent windows"}[cov_method])
    return out


def variance_ratio(window: pd.DataFrame, L: int) -> pd.Series:
    """Per-asset variance ratio at L: ``1 + 2 * sum_{k=1}^{L-1} (1 - k/L) rho_k`` from the window's daily returns (Bartlett weights keep it non-negative). L = 1 gives 1."""
    out = {}
    for c in window.columns:
        x = window[c].dropna().to_numpy(dtype=float)
        n = len(x)
        if L <= 1 or n < 4 * L or n < 40:
            out[c] = 1.0
            continue
        x = x - x.mean()
        v = float((x ** 2).sum() / n)
        if v <= 0:
            out[c] = 1.0
            continue
        s = 0.0
        for k in range(1, min(L, n // 4)):
            s += (1.0 - k / L) * float((x[k:] * x[:-k]).sum() / n) / v
        out[c] = float(np.clip(1.0 + 2.0 * s, 0.25, 4.0))
    return pd.Series(out)


def horizon_covariance(daily: pd.DataFrame, window: pd.DataFrame, L: int, method: str) -> tuple[pd.DataFrame, dict]:
    """The covariance over an L-day window from the daily one and the past daily returns ``window`` it was fitted on, and what was done."""
    L = int(L)
    if method not in COV_HORIZON_METHODS:
        raise ValueError(f"cov_horizon must be one of {list(COV_HORIZON_METHODS)}")
    lin = daily * float(L)
    info: dict = {"method": method, "window_days": L}
    if method == "linear" or L <= 1:
        info["method"] = "linear" if method == "linear" else f"{method} (L = 1 is the daily covariance)"
        return lin, info
    names = list(daily.index)
    if method == "variance_ratio":
        vr = variance_ratio(window[names], L)
        d = np.sqrt(vr.reindex(names).fillna(1.0).to_numpy())
        out = lin.to_numpy() * np.outer(d, d)
        info["variance_ratio"] = {str(k): float(v) for k, v in vr.items()}
        return pd.DataFrame(out, index=daily.index, columns=daily.columns), info
    w = window[names].dropna(how="any")
    n = len(w)
    n_scen = n - L + 1
    if n_scen < max(3 * L, 40):
        info.update(method="linear", fallback=f"{n_scen} overlapping {L}-day windows in the past window: fewer than {max(3 * L, 40)} needed for a direct estimate, so the linear estimate is used")
        return lin, info
    comp = np.expm1(np.log1p(w.clip(lower=-0.999999)).rolling(L).sum()).dropna().to_numpy(dtype=float)
    direct = np.cov(comp, rowvar=False, ddof=1)
    direct = np.atleast_2d(direct)
    n_eff = n_scen / float(L)
    lam = n_eff / (n_eff + N0)
    blend = lam * direct + (1.0 - lam) * lin.to_numpy()
    sym = (blend + blend.T) / 2.0
    fixed = rm.pc.nearest_positive_definite(pd.DataFrame(sym, index=daily.index, columns=daily.columns), 1e-10) if np.linalg.eigvalsh(sym).min() <= 1e-10 else pd.DataFrame(sym, index=daily.index, columns=daily.columns)
    info.update(direct_weight=float(lam), independent_windows=float(n_eff), overlapping_windows=int(n_scen), repaired=bool(np.linalg.eigvalsh(sym).min() <= 1e-10))
    return fixed, info


def tranche_count(holding: int, every: int) -> int:
    """Staggered tranches alive at once: a tranche is held for L sessions and a new one is opened every ``every`` sessions, so ``ceil(L / every)`` overlap."""
    return max(1, int(math.ceil(int(holding) / max(int(every), 1))))


def ledger_replay(targets: pd.DataFrame, returns: pd.DataFrame, frequency: str, lag: int, cost_bps, capital: float = 1.0, cost_on: str = "traded_notional") -> dict:
    """An independent units-and-cash accounting of a book, for checking an engine's weight-based accounting (cash conservation, trades, costs, P&L).

    The account holds ``units`` of each asset and ``cash``. On a day the (lagged, rebalance-scheduled) target is struck at the close: equity before trading is ``cash + units . price``, costs are
    ``rate * |units traded| * price`` taken out of equity, and the new units are ``target weight * equity after costs / price``; cash is the rest. Between trades units are fixed, so the book drifts with prices
    exactly as an account does. ``cost_bps`` is one rate or one per asset. ``cost_on`` is the base the proportional cost is charged on: ``traded_notional`` (physical: a fraction of the value traded at the close) or ``prior_equity`` (the
    engine's convention, a fraction of the equity at the previous close, subtracted from the day's return; it differs from the physical one by the cost times the day's return, about a hundredth of a cost). Returns the daily equity, net return, cash, costs, traded notional and the largest violation of
    ``cash + units . price = equity`` and of ``cash >= 0``."""
    from ..backtest.execution import apply_execution_lag, restrict_to_rebalances

    index = pd.DatetimeIndex(targets.index)
    cols = list(targets.columns)
    ret = returns.reindex(index=index, columns=cols).fillna(0.0).to_numpy(dtype=float)
    sched = restrict_to_rebalances(apply_execution_lag(targets, lag), index, frequency).to_numpy(dtype=float)
    rate = (pd.Series(float(cost_bps), index=cols) if np.isscalar(cost_bps) else pd.Series(cost_bps).reindex(cols).fillna(0.0)).to_numpy(dtype=float) / 1e4
    price = np.ones(len(cols))
    units = np.zeros(len(cols))
    cash, started, prev_equity = float(capital), False, float(capital)
    eq_out, cash_out, cost_out, traded_out, ident_err, min_cash = [], [], [], [], 0.0, 0.0
    for i in range(len(index)):
        price = price * (1.0 + ret[i])
        equity = cash + float(units @ price)
        target = sched[i]
        cost = 0.0
        traded = 0.0
        if np.isfinite(target).any():
            w = np.nan_to_num(target, nan=0.0)
            pre_w = (units * price) / equity if equity > 0 else np.zeros(len(cols))
            dw = np.abs(w - pre_w)
            cost = float((dw * rate).sum()) * (prev_equity if cost_on == "prior_equity" else equity)
            traded = float(dw.sum()) * equity
            equity -= cost
            units = w * equity / price
            cash = equity - float(units @ price)
            started = True
        elif not started:
            cash = float(capital)
        ident_err = max(ident_err, abs(cash + float(units @ price) - (equity if started else cash)))
        min_cash = min(min_cash, cash)
        prev_equity = cash + float(units @ price)
        eq_out.append(prev_equity)
        cash_out.append(cash)
        cost_out.append(cost)
        traded_out.append(traded)
    eq = pd.Series(eq_out, index=index)
    return {"equity": eq, "net_return": eq / eq.shift(1).fillna(float(capital)) - 1.0, "cash": pd.Series(cash_out, index=index), "costs": pd.Series(cost_out, index=index) / eq.shift(1).fillna(float(capital)),
            "traded_notional": pd.Series(traded_out, index=index), "identity_error": float(ident_err), "min_cash": float(min_cash)}
