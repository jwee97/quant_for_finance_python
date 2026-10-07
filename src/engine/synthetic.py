"""A synthetic multi-asset market with a known data-generating process, as a registry plus normalised market events.

It is the offline test bed for the engine, the strategies and the mixed-asset example: ONE market in which a single portfolio can hold an equity-index future chain, a commodity future
chain, currencies, crypto spot and perpetuals, listed options and (optionally) swaps. The processes are deliberately plain, with just enough structure for each strategy family to have
something to find (and to lose money on when the structure is absent):

* **equity index** ``SPX`` (and the ETF ``SPY`` = SPX / 10): a regime-switching volatility process (calm / stress) with a slow trend component; the ES futures are
  ``SPX x exp((r - q) T)`` with a noisy basis; ``SPY`` options are quoted at an implied volatility that exceeds the latent volatility by a premium (so selling variance has positive
  expectation before costs) plus a skew;
* **commodity** ``CL``: a mean-reverting convenience yield makes the curve alternate between backwardation and contango (roll yield is the carry);
* **currencies** ``EURUSD, GBPUSD, AUDUSD, USDJPY``: spot drifts by a fraction of the interest differential (the forward-premium puzzle: carry survives partly), interest rates are published as
  ``RATE-{ccy}`` reference series;
* **crypto** ``BTC``, ``ETH``: spot and linear perpetuals on a venue, with a mean-reverting premium of perpetual over spot and a funding rate that follows it, paid every 8 hours;
* **rates** (optional ``with_swaps``): a Nelson-Siegel market from ``swaps.synthetic`` for USD (and EUR).

All data are stamped at the 16:00 close and are available at once (``lag``). Nothing here is vendor data: results on it show that the machinery works, not that a strategy has an edge in
real markets.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..derivatives.pricing import bsm_price
from ..instruments import Instrument, InstrumentRegistry, RollSpec, build_chain, crypto_perp, crypto_spot, fx_spot, make_option
from ..marketdata import concat_events, events_from_funding, events_from_prices
from ..marketdata.schema import normalise_events

RATES = {"USD": 0.045, "EUR": 0.030, "GBP": 0.047, "AUD": 0.040, "JPY": 0.002}
FX_PAIRS = {"EURUSD": ("EUR", "USD", 1.10, 0.060), "GBPUSD": ("GBP", "USD", 1.27, 0.065), "AUDUSD": ("AUD", "USD", 0.66, 0.075), "USDJPY": ("USD", "JPY", 148.0, 0.080)}


@dataclass
class SyntheticMarket:
    registry: InstrumentRegistry
    events: pd.DataFrame
    dates: pd.DatetimeIndex
    truth: dict = field(default_factory=dict)       # latent paths and parameters (for tests: never visible to strategies)
    ids: dict = field(default_factory=dict)         # named ids: ``chains``, ``fx``, ``crypto_spot``, ``crypto_perp``, ``options``, ``equity``

    def config_kwargs(self, **overrides) -> dict:
        """Arguments for ``EngineConfig`` covering the whole simulated span."""
        out = dict(start=self.dates[0], end=self.dates[-1] + pd.Timedelta(hours=23, minutes=59), pegs={"USDT": 1.0})
        out.update(overrides)
        return out


def _regime_vol(n: int, rng, calm=0.11, stress=0.28, p_enter=0.012, p_exit=0.06) -> tuple[np.ndarray, np.ndarray]:
    state = np.zeros(n, dtype=int)
    for t in range(1, n):
        if state[t - 1] == 0:
            state[t] = int(rng.random() < p_enter)
        else:
            state[t] = int(rng.random() >= p_exit)
    vol = np.where(state == 1, stress, calm) * np.exp(rng.normal(0, 0.05, n))
    return vol, state


def _ou(n: int, rng, mean: float, phi: float, sigma: float, x0: float | None = None) -> np.ndarray:
    x = np.empty(n)
    x[0] = mean if x0 is None else x0
    for t in range(1, n):
        x[t] = mean + phi * (x[t - 1] - mean) + sigma * rng.standard_normal()
    return x


def synthetic_multi_asset_market(start: str = "2022-01-03", n_days: int = 500, seed: int = 0, lag: str = "0s", with_options: bool = True, with_swaps: bool = False,
                                 option_band: float = 0.15, option_months: int = 3, spread_bps: dict | None = None) -> SyntheticMarket:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    n = len(dates)
    end = dates[-1]
    spreads = {"future": 1.0, "fx": 1.0, "crypto": 3.0, "equity": 1.0, "option": 150.0, **(spread_bps or {})}
    reg = InstrumentRegistry()
    parts, truth, ids = [], {}, {"chains": [], "fx": [], "crypto_spot": [], "crypto_perp": [], "options": [], "equity": []}

    # ------------------------------------------------------------------------------------------------------------------ interest rates
    rates = {c: np.clip(_ou(n, rng, r, 0.995, 0.0004), 0.0, None) for c, r in RATES.items()}
    ref = pd.DataFrame({"timestamp": np.repeat((dates + pd.Timedelta(hours=16)).to_numpy(), len(rates)), "instrument_id": [f"RATE-{c}" for _ in range(n) for c in rates],
                        "event_type": "reference", "value": [rates[c][t] for t in range(n) for c in rates], "source": "synthetic"})
    parts.append(normalise_events(ref, lag=lag))
    truth["rates"] = pd.DataFrame(rates, index=dates)

    # ------------------------------------------------------------------------------------------------------------------------- equity
    vol, state = _regime_vol(n, rng)
    trend = _ou(n, rng, 0.0, 0.985, 0.00008)                                         # slowly varying drift: what trend followers can harvest
    ret = trend + 0.0002 + vol / np.sqrt(252.0) * rng.standard_normal(n) - 0.5 * (vol / np.sqrt(252.0)) ** 2
    spx = 4500.0 * np.exp(np.cumsum(ret))
    truth.update(spx=pd.Series(spx, index=dates), vol=pd.Series(vol, index=dates), regime=pd.Series(state, index=dates), trend=pd.Series(trend, index=dates))
    reg.add(Instrument(instrument_id="SPY", asset_class="equity", instrument_type="equity", currency="USD", calendar="US", tick_size=0.01, lot_size=1.0))
    ids["equity"].append("SPY")
    parts.append(events_from_prices(pd.DataFrame({"SPY": spx / 10.0}, index=dates), "bar", "16:00", lag, spreads["equity"]))
    div = 0.015

    es = build_chain("ES", dates[0] - pd.Timedelta(days=5), end, months=(3, 6, 9, 12), expiry_rule="third_friday", multiplier=50.0, tick_size=0.25, initial_margin=12000.0,
                     maintenance_margin=10000.0, roll=RollSpec("calendar", 5), calendar="US")
    reg.add_chain(es)
    ids["chains"].append("ES")
    basis_noise = _ou(n, rng, 0.0, 0.9, 0.0004)
    fut = {}
    for c in es.contracts:
        T = np.maximum((c.expiry - dates).days, 0) / 365.0
        px = spx * np.exp((rates["USD"] - div) * T + basis_noise * np.sqrt(np.minimum(T, 0.5) * 2))
        alive = (dates <= c.expiry) & (dates >= c.first_trade)
        if alive.any():
            fut[c.instrument_id] = pd.Series(np.where(alive, px, np.nan), index=dates)
    parts.append(events_from_prices(pd.DataFrame(fut), "settlement", "16:00", lag, spreads["future"]))

    # --------------------------------------------------------------------------------------------------------------------- commodity
    conv = _ou(n, rng, 0.0, 0.985, 0.012)                                           # convenience yield minus storage: > 0 backwardation
    cl_ret = 0.0001 + 0.30 / np.sqrt(252.0) * rng.standard_normal(n)
    cl = 70.0 * np.exp(np.cumsum(cl_ret))
    clc = build_chain("CL", dates[0] - pd.Timedelta(days=5), end, months=tuple(range(1, 13)), expiry_rule="business_day_before_25th", multiplier=1000.0, tick_size=0.01,
                      initial_margin=6000.0, maintenance_margin=5500.0, roll=RollSpec("calendar", 5), calendar="US", asset_class="commodity")
    reg.add_chain(clc)
    ids["chains"].append("CL")
    fut = {}
    for c in clc.contracts:
        T = np.maximum((c.expiry - dates).days, 0) / 365.0
        px = cl * np.exp((rates["USD"] * 0.0 - conv * 4.0) * T + 0.0) * np.exp(rng.normal(0, 0.0015, n))
        alive = (dates <= c.expiry) & (dates >= c.first_trade)
        if alive.any():
            fut[c.instrument_id] = pd.Series(np.where(alive, px, np.nan), index=dates)
    parts.append(events_from_prices(pd.DataFrame(fut), "settlement", "16:00", lag, spreads["future"]))
    truth["cl"] = pd.Series(cl, index=dates)
    truth["convenience"] = pd.Series(conv, index=dates)

    # -------------------------------------------------------------------------------------------------------------------------- FX
    common = rng.standard_normal(n)
    fxp = {}
    for pid, (b, q, s0, v) in FX_PAIRS.items():
        diff = rates[b] - rates[q]
        drift = -0.7 * diff / 252.0                                                  # uncovered parity holds for 70%: the rest is carry
        shock = (0.5 * common + np.sqrt(0.75) * rng.standard_normal(n)) * v / np.sqrt(252.0)
        spot = s0 * np.exp(np.cumsum(drift + shock - 0.5 * (v / np.sqrt(252.0)) ** 2))
        fxp[pid] = spot
        reg.add(fx_spot(b, q))
        ids["fx"].append(pid)
    parts.append(events_from_prices(pd.DataFrame(fxp, index=dates), "bar", "16:00", lag, spreads["fx"]))
    truth["fx"] = pd.DataFrame(fxp, index=dates)

    # ------------------------------------------------------------------------------------------------------------------------ crypto
    funding_rows = {}
    crypto = {}
    btc_common = rng.standard_normal(n)
    for coin, s0, v, beta in (("BTC", 40000.0, 0.55, 1.0), ("ETH", 2500.0, 0.70, 1.2)):
        r = 0.0004 + (0.5 * btc_common + np.sqrt(0.75) * rng.standard_normal(n)) * v / np.sqrt(365.0) * beta + 0.3 * trend
        spot = s0 * np.exp(np.cumsum(r))
        prem = _ou(n, rng, 0.0004, 0.92, 0.0004)
        sp, pp = crypto_spot("BINANCE", coin, "USDT"), crypto_perp("BINANCE", coin, "USDT", margin_mode="cross")
        reg.add(sp)
        reg.add(pp)
        ids["crypto_spot"].append(sp.instrument_id)
        ids["crypto_perp"].append(pp.instrument_id)
        crypto[sp.instrument_id] = spot
        crypto[pp.instrument_id] = spot * (1.0 + prem)
        ft = pd.date_range(dates[0].normalize() + pd.Timedelta(hours=8), end + pd.Timedelta(hours=23), freq="8h")
        daily_prem = pd.Series(prem, index=dates + pd.Timedelta(hours=16)).reindex(ft, method="ffill").bfill().to_numpy()
        funding_rows[pp.instrument_id] = pd.Series(np.clip(0.0001 + 0.20 * (daily_prem - 0.0004) + rng.normal(0, 0.00003, len(ft)), -0.0075, 0.0075), index=ft)
    reg.add(fx_spot("USDT", "USD", instrument_id="USDTUSD", lot_size=1.0))
    crypto["USDTUSD"] = np.ones(n)
    parts.append(events_from_prices(pd.DataFrame(crypto, index=dates), "bar", "16:00", lag, spreads["crypto"]))
    parts.append(events_from_funding(pd.DataFrame(funding_rows), lag=lag))
    truth["crypto"] = pd.DataFrame(crypto, index=dates)

    # ------------------------------------------------------------------------------------------------------------------------ options
    if with_options:
        spy = spx / 10.0
        rows = {}
        step, band = 5.0, option_band
        for ms in pd.date_range(dates[0], end + pd.Timedelta(days=45), freq="MS"):
            third = ms + pd.Timedelta(days=(4 - ms.weekday()) % 7 + 14)
            alive = (dates >= third - pd.DateOffset(months=option_months)) & (dates <= third)
            if not alive.any():
                continue
            lo, hi = spy[alive].min() * (1 - band), spy[alive].max() * (1 + band)
            T = np.asarray(np.maximum((third - dates).days, 0), float) / 365.0
            for K in np.arange(np.floor(lo / step) * step, hi + step, step):
                near = alive & (np.abs(spy / K - 1.0) <= band)               # strikes are quoted while they are within the band of the spot (new strikes appear as the market moves)
                if not near.any():
                    continue
                level = 0.17 + (vol - 0.17) * np.exp(-2.0 * T)                                  # volatility mean-reverts: the term structure inverts in stress
                iv = np.clip(level * 1.18 + 0.01 - 0.10 * np.log(K / spy), 0.05, 1.5)      # premium over the expected volatility plus a downward skew
                for right in ("call", "put"):
                    o = make_option("SPY", third, right, float(K))
                    reg.add(o)
                    ids["options"].append(o.instrument_id)
                    live = T > 0
                    px = np.where(live, np.asarray(bsm_price(spy, K, np.where(live, T, 1.0), rates["USD"], div, iv, right == "call"), float),
                                  np.maximum(spy - K, 0.0) if right == "call" else np.maximum(K - spy, 0.0))
                    rows[o.instrument_id] = pd.Series(np.where(near, np.maximum(px, 0.01), np.nan), index=dates)
        if rows:
            parts.append(events_from_prices(pd.DataFrame(rows), "bar", "16:00", lag, spreads["option"]))

    # ----------------------------------------------------------------------------------------------------------------------------- swaps
    if with_swaps:
        from ..swaps.synthetic import synthetic_rates_market

        sw = synthetic_rates_market(start=dates[0], n_days=n, currencies=("USD", "EUR"), seed=seed + 1, lag=lag, fx_pairs={})
        parts.append(sw["events"])
        truth["swap_curves"] = sw["curves"]
        truth["swap_factors"] = sw["factors"]

    events = concat_events(*parts)
    return SyntheticMarket(reg, events, dates, truth, ids)
