"""The Risk Engine: factor models, risk decomposition and hedging, and whether the risk forecast was right.

Paleologo, *Advanced Portfolio Management* (2021): returns split into systematic factor returns and stock-specific ones, ``r = alpha + B f + e``, giving the covariance ``Sigma = B F B' + D`` with ``D`` diagonal;
risk reported by factor group; factor-neutral portfolios and hedges; the bias statistic as the test of a risk model. Cajas, *Advanced Portfolio Optimization* (2025): denoising, detoning and graphical
estimators of the covariance matrix.

    factor_model   time-series, cross-sectional (fundamental) and statistical factor models -> exposures, factor returns, factor covariance, specific variance
    decompose      risk by factor, by group (market, sector, style, specific) and by asset; a factor risk budget; active risk against a benchmark
    hedge          factor-neutral holdings, a minimum-variance hedge with named instruments
    monitor        the bias statistic (was the forecast risk right?), exposure and factor-volatility drift
    covariance     detoned correlation and the graphical-lasso covariance (shrinkage, Marchenko-Pastur denoising and PCA factor covariance already exist in :mod:`src.portfolio.covariance` and :mod:`src.ml.risk_models`)
"""
