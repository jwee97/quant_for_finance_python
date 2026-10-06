"""The strategy library: every model registers itself with the framework on import.

    time_series       momentum, dual momentum, volatility breakout, Donchian, moving-average crossover, trend with an ATR filter
    cross_sectional   12-1 momentum, relative strength, low volatility, defensive beta, value and quality proxies, carry
    statarb           PCA residual, Kalman pairs, rolling-cointegration pairs, sparse-basket mean reversion
    volatility        variance risk premium, variance carry, implied versus realised
    fixed_income      curve steepener, butterfly, carry and roll-down, duration timing
    macro             inflation rotation, yield-curve regimes, dollar strength, commodity supercycle, risk-on/off
    ml                walk-forward ridge on price and macro features
    crypto            funding carry, basis reversion, stablecoin flow (optional branch; needs the crypto bundle)

Run ``quant list models`` to see them with their hypotheses.
"""

from . import crypto, cross_sectional, fixed_income, macro, ml, statarb, time_series, volatility  # noqa: F401
