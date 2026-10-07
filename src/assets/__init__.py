"""Other asset classes: futures, FX, commodities and fixed income, with bundles that plug them into the research framework.

    futures       contract calendars, term structures, continuous series (ratio / difference), excess returns, carry, basis momentum, synthetic Schwartz-Smith curves
    fx            covered interest parity, forward points, Fama regression, a synthetic G10-style market, the carry portfolio
    commodities   Schwartz-Smith Kalman estimation, seasonality, curve factors
    rates         Nelson-Siegel / Svensson / Diebold-Li, bond analytics, key-rate durations, zero-curve bootstrap, carry and roll-down
    bundles       MarketBundle builders (futures, FX) and a synthetic multi-asset universe
"""

from . import bundles, commodities, futures, fx, rates
