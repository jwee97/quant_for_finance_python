"""Other asset classes: futures, FX, commodities and fixed income, with bundles that plug them into the research framework.

    futures       contract calendars, term structures, continuous series (ratio / difference), excess returns, carry, basis momentum, synthetic Schwartz-Smith curves
    fx            covered interest parity, forward points, Fama regression, a synthetic G10-style market, the carry portfolio
    commodities   Schwartz-Smith Kalman estimation, seasonality, curve factors
    rates         Nelson-Siegel / Svensson / Diebold-Li, bond analytics, key-rate durations, zero-curve bootstrap, carry and roll-down
    bundles       MarketBundle builders (futures, FX) and a synthetic multi-asset universe

The dashboard's three workspaces (the shapes of the payloads, with the provenance of every number) build on these:

    common             the AssetSnapshot every instrument reduces to, data provenance (platform / user file / downloaded / simulated) and shared statistics
    options_analytics  option-chain viewer, smile / surface / skew / term structure, Greeks and stress, strategy payoff and margin, strategy backtests
    futures_analytics  curve, roll, carry, basis and margin dashboards over simulated, downloaded or user-supplied contract tables
    crypto_analytics   spot / perpetual, funding by venue, basis, cross-venue spreads, order-book summary and crypto strategy backtests
    strategy_runs      run any registered strategy on a bundle through the platform pipeline and report it compactly
    free_data          the Real data (free) downloads behind those tabs: option chains (Cboe, Yahoo, Deribit), futures curves (Yahoo, Deribit), coin histories, books and funding (OKX, Coinbase, Kraken, Bitstamp, Deribit)
"""

from . import bundles, commodities, futures, fx, rates
