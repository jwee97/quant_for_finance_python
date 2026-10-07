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
    trend             time-series momentum, MA ensemble, breakouts, KAMA, ADX, SuperTrend, Keltner, MACD, Ichimoku; 52-week high, residual and smooth momentum, 13612W
    reversion         RSI(2), IBS, Bollinger, stochastic, down-streaks, Ornstein-Uhlenbeck, range-bound reversion, short-term reversal
    seasonal          turn of the month, sell in May, same-month seasonality
    factors           betting against beta, low idiosyncratic volatility, MAX, illiquidity, value with momentum
    risk_timing       volatility-managed long, buy-the-VIX-spike, credit risk appetite
    taa               Faber GTAA, Protective / Vigilant / Defensive / Adaptive asset allocation, static model portfolios
    crypto            funding carry, basis reversion, stablecoin flow (optional branch; needs the crypto bundle)

Run ``quant list models`` to see them with their hypotheses.
"""

from . import (crypto, cross_sectional, deep, expression, factors, fixed_income, macro, ml, online, reversion, risk_timing, seasonal, statarb, taa,  # noqa: F401
               time_series, trend, volatility)
