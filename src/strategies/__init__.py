"""The strategy library: every model registers itself with the framework on import.

    time_series       momentum, dual momentum, volatility breakout, Donchian, moving-average crossover, trend with an ATR filter
    cross_sectional   12-1 momentum, relative strength, low volatility, defensive beta, value and quality proxies, carry
    statarb           PCA residual, Kalman pairs, rolling-cointegration pairs, sparse-basket mean reversion
    volatility        variance risk premium, variance carry, implied versus realised
    fixed_income      curve steepener, butterfly, carry and roll-down, duration timing
    macro             inflation rotation, yield-curve regimes, dollar strength, commodity supercycle, risk-on/off
    ml                walk-forward ridge on price and macro features; online (monthly-updated) ridge / NLMS / Kalman
    deep              window networks (patch transformer, mixer, N-BEATS, N-HiTS, TimeMixer-style) refit yearly; zero-shot Chronos-Bolt (optional)
    custom            `expression`: a strategy written as a one-line formula; your own files in user_strategies/ (see user.py)
    crypto            funding carry, basis reversion, stablecoin flow (optional branch; needs the crypto bundle)

Run ``quant list models`` to see them with their hypotheses.
"""

from . import crypto, cross_sectional, deep, expression, fixed_income, macro, ml, online, statarb, time_series, volatility  # noqa: F401
