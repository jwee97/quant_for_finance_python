"""The strategy library: every model registers itself with the framework on import.

    time_series       momentum, dual momentum, volatility breakout, Donchian, moving-average crossover, trend with an ATR filter
    cross_sectional   12-1 momentum, relative strength, low volatility, defensive beta, value and quality proxies, carry
    statarb           PCA residual, Kalman pairs, rolling-cointegration pairs, sparse-basket mean reversion
    volatility        variance risk premium, variance carry, implied versus realised
    fixed_income      curve steepener, butterfly, carry and roll-down, duration timing
    macro             inflation rotation, yield-curve regimes, dollar strength, commodity supercycle, risk-on/off
    ml                walk-forward ridge on price and macro features, tree ensembles (gradient boosting, forests, LightGBM, XGBoost); online (monthly-updated) ridge / NLMS / Kalman
    deep              window networks (patch transformer, mixer, N-BEATS, N-HiTS, TimeMixer-style, Temporal Fusion Transformer) refit yearly; zero-shot Chronos-Bolt (optional)
    representation    autoencoder / contrastive encoders trained without labels, then a ridge on the embedding
    custom            `expression`: a strategy written as a one-line formula; your own files in user_strategies/ (see user.py)
    trend             time-series momentum, MA ensemble, breakouts, KAMA, ADX, SuperTrend, Keltner, MACD, Ichimoku; 52-week high, residual and smooth momentum, 13612W
    reversion         RSI(2), IBS, Bollinger, stochastic, down-streaks, Ornstein-Uhlenbeck, range-bound reversion, short-term reversal
    seasonal          turn of the month, sell in May, same-month seasonality
    factors           betting against beta, low idiosyncratic volatility, MAX, illiquidity, value with momentum
    risk_timing       volatility-managed long, buy-the-VIX-spike, credit risk appetite
    taa               Faber GTAA, Protective / Vigilant / Defensive / Adaptive asset allocation, static model portfolios
    multi_asset       cross-asset carry (cross-sectional and time-series), basis momentum, long-term reversal (need CARRY_<asset> signals: src.assets.bundles)
    econometric       Kalman trend, GARCH- and EVT-managed exposure, shrunk-VAR lead-lag
    crypto            funding carry, basis reversion, stablecoin flow (optional branch; needs the crypto bundle)
    alpha_styles      appraisal-ratio alpha, adaptive autocorrelation, volatility-squeeze breakout, abnormal-volume drift, a walk-forward event study; news sentiment and panel signals read files in data/user/
    rebalance_styles  policy portfolio (calendar and band), the month-end rebalancing flow, flight to quality, market outlook
    economic_outlook  yield-curve quadrants and credit-cycle rotation, with states learned from earlier data only
    factor_models     statistical APT (PCA residual alpha), macroeconomic factor model, cross-sectional characteristic regression, machine learning on the characteristics (lasso, elastic net, PCR, PLS, CART, forest, network)
    fundamental_factors   value, quality, momentum and estimate-revision factors, discounted cash flow (point and multipath), the optimal alpha model, contextual and nonlinear versions; read data/user/fundamentals.csv

Run ``quant list models`` to see them with their hypotheses.
"""

from . import (alpha_styles, crypto, cross_sectional, deep, econometric, economic_outlook, expression, factor_models, factors, fixed_income, fundamental_factors, macro, ml, multi_asset,  # noqa: F401
               online, rebalance_styles, representation, reversion, risk_timing, seasonal, statarb, taa, time_series, trend, volatility)
