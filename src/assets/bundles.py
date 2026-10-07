"""Market bundles for the other asset classes, so every model, allocator and validator in the framework runs on futures, FX, rates and mixed books unchanged.

A bundle holds one TOTAL-RETURN INDEX per instrument (``prices``, started at 100) so that the engine's simple returns are the instruments' excess returns, plus a macro panel carrying each
instrument's ``CARRY_<name>`` (the expected return if nothing moves, known at each close) and other signals, so a model such as ``carry_xs`` reads them like any macro series.

* ``futures_bundle``: continuous excess-return indices from contract tables (``excess_return_series``) with the annualised roll yield, curve slope and basis momentum as signals.
* ``fx_bundle``: currencies against USD funded in USD, carry = the interest differential.
* ``multi_asset_demo_bundle``: a fully synthetic cross-asset universe (commodity futures from Schwartz-Smith curves, equity-index futures, constant-maturity bond indices from a dynamic
  Nelson-Siegel curve, FX), with a common risk factor linking them. It exists to exercise the multi-asset machinery end to end offline; its statistics say nothing about real markets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..framework.data import MarketBundle, bundle_from_prices
from . import fx as fxmod
from . import rates as ratesmod
from .futures import SchwartzSmithParams, basis_momentum, carry, contract_returns, excess_return_series, synthetic_term_structure, term_structure


def _index(returns: pd.Series, start: float = 100.0) -> pd.Series:
    return start * (1.0 + returns.fillna(0.0)).cumprod()


def futures_bundle(curves: dict[str, pd.DataFrame], asset_class: dict | None = None, roll_days: int = 5, name: str = "futures") -> MarketBundle:
    """From ``{root: long contract table}`` (columns ``date, contract, expiry, price``) build the excess-return index and the carry signals of each root."""
    prices, macro = {}, {}
    for root, long in curves.items():
        ex = excess_return_series(long, roll_days)
        prices[root] = _index(ex)
        ts = term_structure(long, 3)
        macro[f"CARRY_{root}"] = carry(ts["price"], ts["ttm"])
        macro[f"SLOPE_{root}"] = np.log(ts["price"]["F3"] / ts["price"]["F1"]) if "F3" in ts["price"] else np.nan
        cr = contract_returns(ts["price"], ts["contract"])
        macro[f"BASISMOM_{root}"] = basis_momentum(cr)
    price_frame = pd.DataFrame(prices).dropna(how="all")
    macro_frame = pd.DataFrame(macro).reindex(price_frame.index)
    return bundle_from_prices(price_frame, asset_class=asset_class or {r: "commodity" for r in curves}, macro=macro_frame, name=name)


def fx_bundle(market: dict, name: str = "fx") -> MarketBundle:
    """``market`` as returned by ``synthetic_fx_market`` (or built the same way from real spot and rate series)."""
    ex = market["excess_returns"]
    prices = pd.DataFrame({c: _index(ex[c]) for c in ex.columns})
    rates, usd = market["rates"], market["usd_rate"]
    macro = pd.DataFrame({f"CARRY_{c}": rates[c] - usd for c in ex.columns})
    return bundle_from_prices(prices, asset_class={c: "fx" for c in ex.columns}, macro=macro, name=name)


def multi_asset_demo_bundle(n_days: int = 2500, seed: int = 0) -> tuple[MarketBundle, dict]:
    """A synthetic cross-asset universe. Returns the bundle and a dict of the underlying simulated components (curves, rates, FX market, true parameters)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2012-01-02", periods=n_days)
    risk = rng.standard_normal(n_days)                                                          # a common "global risk" factor linking the asset classes
    prices, macro, classes, parts = {}, {}, {}, {}

    # commodity futures: different Schwartz-Smith worlds, one seasonal
    specs = {"CRUDE": SchwartzSmithParams(kappa=1.2, sigma_chi=0.30, sigma_xi=0.22, lambda_chi=0.08, s0=60.0),
             "GAS": SchwartzSmithParams(kappa=2.5, sigma_chi=0.55, sigma_xi=0.18, lambda_chi=-0.10, seasonal_amp=0.08, seasonal_peak_month=1, s0=3.0),
             "GOLD": SchwartzSmithParams(kappa=0.6, sigma_chi=0.06, sigma_xi=0.14, mu_xi=0.04, lambda_chi=0.0, s0=1300.0),
             "CORN": SchwartzSmithParams(kappa=1.8, sigma_chi=0.28, sigma_xi=0.14, lambda_chi=0.05, seasonal_amp=0.05, seasonal_peak_month=7, s0=4.0)}
    curves = {}
    for i, (root, p) in enumerate(specs.items()):
        sim = synthetic_term_structure(n_days, str(idx[0].date()), p, seed=seed * 100 + i)
        curves[root] = sim["long"]
        parts[root] = sim
    fb = futures_bundle(curves)
    for root in curves:
        prices[root] = fb.prices[root].reindex(idx)
        for k in ("CARRY", "SLOPE", "BASISMOM"):
            macro[f"{k}_{root}"] = fb.macro[f"{k}_{root}"].reindex(idx)
        classes[root] = "commodity"

    # equity index futures: price index with a dividend yield; excess return = return - cash; carry = dividend yield - cash rate
    cash = pd.Series(0.015 + 0.01 * np.sin(np.arange(n_days) / 400.0), index=idx)
    for j, (name, beta, vol, dy) in enumerate((("EQ_US", 1.0, 0.16, 0.018), ("EQ_EU", 1.1, 0.19, 0.030), ("EQ_JP", 0.9, 0.18, 0.022))):
        shock = 0.7 * beta * risk * vol / np.sqrt(252) + 0.71 * rng.standard_normal(n_days) * vol / np.sqrt(252)
        ret = 0.065 / 252 + shock - 0.5 * vol ** 2 / 252 - cash.to_numpy() / 252
        prices[name] = _index(pd.Series(ret, index=idx))
        macro[f"CARRY_{name}"] = dy - cash
        classes[name] = "equity"

    # bond indices from a dynamic Nelson-Siegel curve: carry + roll-down minus duration x yield change
    for j, (name, seedshift, maturity) in enumerate((("UST10", 1, 10), ("BUND10", 2, 10), ("UST2", 3, 2))):
        panel = ratesmod.synthetic_curve_panel(n_days, seed=seed * 100 + 50 + seedshift)
        panel.index = idx
        mats = np.asarray(panel.columns, float)
        y = pd.Series([np.interp(maturity, mats, row) for row in panel.to_numpy()], index=idx)
        dur = pd.Series([ratesmod.duration(maturity, yy, yy) for yy in y.to_numpy()], index=idx)
        dy_ = y.diff()
        ret = (y.shift(1) - cash.shift(1)) / 252.0 - dur.shift(1) * dy_ + 0.5 * 80.0 * dy_ ** 2 - 0.0
        ret = ret + 0.0006 * risk * (-1.0 if name != "UST2" else -0.5) / np.sqrt(252)
        prices[name] = _index(ret)
        roll = np.array([ratesmod.carry_rolldown(maturity, yy, lambda t, row=row: float(np.interp(t, mats, row)), 0.25)["rolldown"] * 4.0 for yy, row in zip(y.to_numpy(), panel.to_numpy())])
        macro[f"CARRY_{name}"] = (y - cash) + pd.Series(roll, index=idx)
        classes[name] = "bond"
        parts[name] = panel

    # FX
    fxm = fxmod.synthetic_fx_market(n_days, 6, uip_beta=0.3, seed=seed + 7, start=str(idx[0].date()))
    fxb = fx_bundle(fxm)
    for c in fxb.assets:
        prices[f"FX_{c}"] = fxb.prices[c].reindex(idx)
        macro[f"CARRY_FX_{c}"] = fxb.macro[f"CARRY_{c}"].reindex(idx)
        classes[f"FX_{c}"] = "fx"
    parts["fx"] = fxm
    price_frame = pd.DataFrame(prices)
    bundle = bundle_from_prices(price_frame, asset_class=classes, macro=pd.DataFrame(macro, index=idx), name="synthetic-multi-asset", min_history=300)
    return bundle, parts
