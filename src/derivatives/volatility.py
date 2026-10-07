"""Volatility analytics on option chains: the VIX-style index, implied versus realised variance (the variance risk premium), term structure, skew, and dispersion.

* ``vix_series``: the CBOE methodology (a model-free variance-swap strike from a strip of out-of-the-money options) at a constant 30-day maturity, one value per date.
* ``variance_risk_premium``: ``implied variance - realised variance``. The TRAILING version (realised over the last ``window`` days) is known at the decision date and can be
  traded on; the FORWARD version (realised over the days that follow) is the premium a short-variance position actually earned and is only available in hindsight.
* ``term_structure`` / ``smile_summary``: at-the-money volatility by expiry, and an SVI-based summary of one expiry's smile (ATM, risk reversal, butterfly, skew slope).
* Dispersion trading (``implied_correlation``, ``simulate_dispersion_world``, ``dispersion_backtest``): short index variance against long single-stock variance, a bet that the
  correlation the index options imply exceeds the correlation that is realised. No single-stock option data exist in this repository, so the trade is simulated on variance-swap payoffs
  with the implied correlation premium as an explicit input.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .surface import fit_svi_slice, smile_metrics, svi_total_variance, vix_style_index


def _bracketing_expiries(chain: pd.DataFrame, target_days: float = 30.0):
    exp = chain.drop_duplicates("expiry")[["expiry", "T"]].sort_values("T")
    days = exp["T"].to_numpy() * 365.0
    below, above = exp[days <= target_days], exp[days > target_days]
    if below.empty or above.empty:
        return None
    return below.iloc[-1]["expiry"], above.iloc[0]["expiry"]


def vix_series(market_or_chains, target_days: float = 30.0) -> pd.Series:
    """The VIX-style index (volatility, in percent) on every date that has expiries on both sides of ``target_days``. Accepts a ``SyntheticMarket`` or ``{date: chain}``."""
    chains = {d: market_or_chains.chain(d) for d in market_or_chains.dates()} if hasattr(market_or_chains, "dates") else market_or_chains
    out = {}
    for d, ch in chains.items():
        br = _bracketing_expiries(ch, target_days)
        if br is None:
            continue
        front, nxt = ch[ch["expiry"] == br[0]], ch[ch["expiry"] == br[1]]
        try:
            out[pd.Timestamp(d)] = vix_style_index(front, nxt, target_days)
        except (ValueError, KeyError):
            continue
    return pd.Series(out, name="vix")


def realised_variance(spot: pd.Series, window: int = 21, forward: bool = False) -> pd.Series:
    """Annualised realised variance of daily log returns over ``window`` days: trailing (through the date) or forward (the ``window`` days after the date)."""
    r2 = np.log(spot).diff() ** 2
    trailing = r2.rolling(window).sum() * 252.0 / window
    return trailing.shift(-window) if forward else trailing


def variance_risk_premium(vix: pd.Series, spot: pd.Series, window: int = 21) -> pd.DataFrame:
    """``implied_var = (VIX / 100)^2`` against trailing and forward realised variance (both in annualised variance units) and their differences."""
    iv = (vix / 100.0) ** 2
    frame = pd.DataFrame({"implied_var": iv, "trailing_rv": realised_variance(spot, window).reindex(iv.index), "forward_rv": realised_variance(spot, window, forward=True).reindex(iv.index)})
    frame["vrp_trailing"] = frame["implied_var"] - frame["trailing_rv"]
    frame["vrp_forward"] = frame["implied_var"] - frame["forward_rv"]
    return frame


def term_structure(chain: pd.DataFrame) -> pd.Series:
    """At-the-money implied volatility by days to expiry (the strike nearest the forward, average of the call and the put)."""
    rows = {}
    for expiry, g in chain.groupby("expiry"):
        atm = g.iloc[(g["strike"] - g["F"]).abs().argsort().iloc[:2]]
        rows[int(round(g["T"].iloc[0] * 365))] = float(atm["iv"].mean())
    return pd.Series(rows, name="atm_vol").sort_index()


def smile_summary(chain: pd.DataFrame, expiry) -> pd.Series:
    """Fit an SVI slice to the OTM quotes of one expiry and report the ATM vol, 25-delta risk reversal and butterfly, and the ATM skew slope."""
    g = chain[chain["expiry"] == expiry]
    otm = g[((g["right"] == "P") & (g["strike"] <= g["F"])) | ((g["right"] == "C") & (g["strike"] > g["F"]))]
    otm = otm[(otm["bid"] > 0) & np.isfinite(otm["iv"])].sort_values("k")
    T = float(g["T"].iloc[0])
    sl = fit_svi_slice(otm["k"].to_numpy(), otm["iv"].to_numpy(), T, n_starts=6)
    out = smile_metrics(lambda k, t: svi_total_variance(k, *sl.params), T)
    out["svi_rmse"] = sl.rmse
    return out


# ------------------------------------------------------------------------------------------------------------------- dispersion
def implied_correlation(index_vol: float, component_vols, weights) -> float:
    """Average implied pairwise correlation that reconciles an index volatility with its components': ``rho = (s_I^2 - sum w_i^2 s_i^2) / (2 sum_{i<j} w_i w_j s_i s_j)``."""
    w, s = np.asarray(weights, float), np.asarray(component_vols, float)
    ws = w * s
    cross = (ws.sum() ** 2 - (ws ** 2).sum()) / 2.0
    return float((index_vol ** 2 - (ws ** 2).sum()) / (2.0 * cross))


def simulate_dispersion_world(n_assets: int = 10, n_days: int = 1500, vol: float = 0.25, realised_corr: float = 0.35, implied_corr_premium: float = 0.10, vol_dispersion: float = 0.3,
                              implied_vol_premium: float = 0.0, corr_vol: float = 0.0, seed: int = 0) -> dict:
    """Correlated components with an equicorrelation and stochastic individual volatilities. ``corr_vol > 0`` lets the correlation drift (AR(1) in Fisher-z space around
    ``realised_corr``); the default 0 keeps it constant, which makes ``implied_corr_premium = 0`` a clean null (stochastic correlation alone creates a small positive carry
    for long dispersion through the convexity of the correlation-to-variance map). Implied component vols are the model-fair expectations of the next month's volatility plus ``implied_vol_premium``; the INDEX
    implied vol uses the trailing realised correlation plus ``implied_corr_premium`` (the crash-insurance premium that makes index options rich relative to their components)."""
    rng = np.random.default_rng(seed)
    base_vol = vol * np.exp(vol_dispersion * rng.standard_normal(n_assets) * 0.5)
    z = np.zeros(n_days)
    z0 = np.arctanh(realised_corr)
    for t in range(1, n_days):
        z[t] = 0.98 * z[t - 1] + corr_vol * rng.standard_normal()
    rho = np.tanh(z0 + z)
    phi, sd_x = 0.985, 0.10
    lvol = np.zeros((n_days, n_assets))
    for t in range(1, n_days):
        lvol[t] = phi * lvol[t - 1] + sd_x * rng.standard_normal(n_assets)
    vols = base_vol * np.exp(lvol) / np.sqrt(252.0)
    common = rng.standard_normal(n_days)
    idio = rng.standard_normal((n_days, n_assets))
    shocks = np.sqrt(rho)[:, None] * common[:, None] + np.sqrt(1 - rho)[:, None] * idio
    returns = pd.DataFrame(shocks * vols, index=pd.bdate_range("2015-01-01", periods=n_days), columns=[f"S{i}" for i in range(n_assets)])
    weights = np.full(n_assets, 1.0 / n_assets)
    # model-fair implied variance: the exact conditional expectation of the average daily variance over the next 21 days given today's log-volatility state x_t
    # (log sigma is AR(1), so E[exp(2 x_{t+h}) | x_t] = exp(2 phi^h x_t + 2 Var_h)), plus the explicit premium
    h = np.arange(1, 22)
    var_h = sd_x ** 2 * (1.0 - phi ** (2 * h)) / (1.0 - phi ** 2)
    fair = np.stack([np.exp(2.0 * phi ** hh * lvol + 2.0 * v) for hh, v in zip(h, var_h)]).mean(axis=0)
    component_iv = pd.DataFrame(np.sqrt(base_vol ** 2 * fair) + implied_vol_premium, index=returns.index, columns=returns.columns)
    return {"returns": returns, "weights": weights, "component_iv": component_iv, "implied_corr_premium": implied_corr_premium, "true_rho": pd.Series(rho, index=returns.index),
            "index_returns": returns @ weights}


def dispersion_backtest(world: dict, period: int = 21, cost_vol_points: float = 0.5, ignore_premium: bool = False) -> pd.DataFrame:
    """Variance-swap dispersion: every ``period`` days, long single-stock variance and short index variance, vega-neutral in aggregate.

    At the start of each period the strikes are the implied variances: ``K_i = IV_i^2`` for each component and ``K_I = sum w_i^2 K_i + 2 sum_{i<j} w_i w_j IV_i IV_j rho_impl`` for the index,
    with ``rho_impl`` the (known) trailing realised correlation PLUS the index premium. The P&L of each leg is the notional times (realised variance - strike); component notionals are
    ``w_i IV_I / IV_i`` times the index notional so the vega adds up. ``cost_vol_points`` is the bid-ask in volatility points paid on the unit of vega of each side.
    Returns one row per period: the P&L per unit of index vega (in volatility units: 0.01 is one volatility point), the realised and implied correlation.
    """
    r, w, civ = world["returns"], world["weights"], world["component_iv"]
    premium = 0.0 if ignore_premium else world["implied_corr_premium"]
    rows = []
    for start in range(63, len(r) - period, period):
        d0 = r.index[start]
        iv = civ.iloc[start].to_numpy()
        if not np.isfinite(iv).all():
            continue
        past = r.iloc[start - 63:start]
        corr = past.corr().to_numpy()
        rho_past = float((corr.sum() - len(corr)) / (len(corr) * (len(corr) - 1)))
        rho_impl = min(rho_past + premium, 0.99)
        ws = w * iv
        k_index = float((ws ** 2).sum() + 2.0 * rho_impl * ((ws.sum() ** 2 - (ws ** 2).sum()) / 2.0))
        fwd = r.iloc[start + 1:start + 1 + period]
        rv_comp = (fwd ** 2).mean().to_numpy() * 252.0
        rv_index = float(((fwd @ w) ** 2).mean() * 252.0)
        idx_vol = np.sqrt(k_index)
        notional_i = 1.0 / (2.0 * idx_vol)                              # variance notional per unit of vega: vega = 2 N sigma
        notional_c = w * idx_vol / iv * notional_i
        pnl_comp = float((notional_c * (rv_comp - iv ** 2)).sum())                 # a variance swap pays notional x (realised variance - strike), annualised variance units
        pnl_index = float(notional_i * (rv_index - k_index))
        cost = 2.0 * cost_vol_points / 100.0                                        # the bid-ask on the index leg and on the component legs, each of unit total vega
        corr_f = fwd.corr().to_numpy()
        rows.append({"date": d0, "pnl": pnl_comp - pnl_index - cost, "pnl_before_cost": pnl_comp - pnl_index, "implied_corr": rho_impl, "realised_corr": float((corr_f.sum() - len(corr_f)) / (len(corr_f) * (len(corr_f) - 1)))})
    return pd.DataFrame(rows).set_index("date")
