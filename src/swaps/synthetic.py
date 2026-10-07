"""A synthetic rates market: curves, fixings and FX with a known data-generating process, as normalised market events.

``synthetic_rates_market`` simulates a dynamic Nelson-Siegel term structure for each currency (level, slope and curvature follow mean-reverting AR(1) processes, the Diebold-Li
construction), turns the zero rates into daily ``curve`` events (``{ccy}-DISCOUNT``) and projection curves (``{ccy}-PROJ-3M`` and ``{ccy}-PROJ-6M``: the discount curve plus tenor-basis spreads, the 6M one mean-reverting around ``basis_6m_bp``), publishes
a daily floating-rate ``fixing`` for each currency's index (``{ccy}-3M``: the 3-month forward from the projection curve, available at the end of the day), and an FX spot series
with a cross-currency basis. It is the offline test bed for swap strategies and for the mixed-asset example; as with every synthetic market here, results on it test the machinery, not
the market.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..assets.rates import nelson_siegel
from ..marketdata.schema import normalise_events
from .curves import DiscountCurve

TENORS = (0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0)


def _factor_paths(n: int, rng, start, mean, phi, vol):
    out = np.zeros((n, 3))
    out[0] = start
    for t in range(1, n):
        out[t] = mean + phi * (out[t - 1] - mean) + vol * rng.standard_normal(3)
    return out


def synthetic_rates_market(start="2023-01-02", n_days: int = 500, currencies=("USD",), seed: int = 0, close_time: str = "16:00", lag: str = "0s", basis_bp: float = 3.0, basis_6m_bp: float = 8.0,
                           curve_noise_bp: float = 0.3, fx_pairs: dict | None = None) -> dict:
    """Events for the market described above. Returns ``{"events", "curves", "dates", "factors"}`` where ``curves`` maps ``(currency, date)`` to the true projection ``DiscountCurve``.

    ``fx_pairs`` maps a pair id to ``(base, quote, spot0)`` (default: ``EURUSD`` when both EUR and USD are simulated); spot follows a random walk with drift equal to the rate
    differential, so forwards are unbiased."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    h, m = (int(x) for x in close_time.split(":"))
    stamps = dates + pd.Timedelta(hours=h, minutes=m)
    levels = {"USD": (0.045, -0.012, 0.005), "EUR": (0.030, -0.010, 0.004), "GBP": (0.047, -0.010, 0.005)}
    rows, curves, factors = [], {}, {}
    b6 = np.zeros(n_days)
    b6[0] = basis_6m_bp
    for i in range(1, n_days):
        b6[i] = basis_6m_bp + 0.97 * (b6[i - 1] - basis_6m_bp) + 0.6 * rng.standard_normal()
    for ccy in currencies:
        mean = np.array(levels.get(ccy, (0.04, -0.01, 0.004)))
        f = _factor_paths(n_days, rng, mean, mean, np.array([0.995, 0.99, 0.98]), np.array([0.0004, 0.0004, 0.0008]))
        factors[ccy] = pd.DataFrame(f, index=dates, columns=["level", "slope", "curvature"])
        for i, d in enumerate(dates):
            zeros = np.asarray(nelson_siegel(np.asarray(TENORS), f[i, 0], f[i, 1], f[i, 2], 0.5), float) + rng.normal(0, curve_noise_bp * 1e-4, len(TENORS))
            disc = {float(t): float(z) for t, z in zip(TENORS, zeros)}
            proj = {float(t): float(z) + basis_bp * 1e-4 for t, z in zip(TENORS, zeros)}
            rows.append({"timestamp": stamps[i], "instrument_id": f"{ccy}-DISCOUNT", "event_type": "curve", "curve_values": disc, "source": "synthetic_rates"})
            rows.append({"timestamp": stamps[i], "instrument_id": f"{ccy}-PROJ-3M", "event_type": "curve", "curve_values": proj, "source": "synthetic_rates"})
            proj6 = {float(t): float(z) + (basis_bp + b6[i]) * 1e-4 for t, z in zip(TENORS, zeros)}
            rows.append({"timestamp": stamps[i], "instrument_id": f"{ccy}-PROJ-6M", "event_type": "curve", "curve_values": proj6, "source": "synthetic_rates"})
            rows.append({"timestamp": d + pd.Timedelta(hours=9), "instrument_id": f"{ccy}-6M", "event_type": "fixing",
                         "value": float(DiscountCurve.from_values(d, proj6).forward_rate(0.0, 0.5, 0.5 * 365 / 360)), "source": "synthetic_rates"})
            pc = DiscountCurve.from_values(d, proj)
            curves[(ccy, d)] = pc
            fix = pc.forward_rate(0.0, 0.25, 0.25 * 365 / 360)
            rows.append({"timestamp": d + pd.Timedelta(hours=9), "instrument_id": f"{ccy}-3M", "event_type": "fixing", "value": float(fix), "source": "synthetic_rates"})
    pairs = fx_pairs or ({"EURUSD": ("EUR", "USD", 1.10)} if {"EUR", "USD"} <= set(currencies) else {})
    for pid, (base, quote, spot0) in pairs.items():
        spot = np.empty(n_days)
        spot[0] = spot0
        for i in range(1, n_days):
            drift = (factors[quote].iloc[i - 1, 0] - factors[base].iloc[i - 1, 0] + factors[quote].iloc[i - 1, 1] - factors[base].iloc[i - 1, 1]) / 252.0
            spot[i] = spot[i - 1] * np.exp(drift - 0.5 * 0.006 ** 2 + 0.006 * rng.standard_normal())
        for i in range(n_days):
            half = spot[i] * 0.5e-4
            rows.append({"timestamp": stamps[i], "instrument_id": pid, "event_type": "quote", "bid": spot[i] - half, "ask": spot[i] + half, "source": "synthetic_rates"})
    events = normalise_events(pd.DataFrame(rows), lag=lag)
    return {"events": events, "curves": curves, "dates": dates, "factors": factors}
