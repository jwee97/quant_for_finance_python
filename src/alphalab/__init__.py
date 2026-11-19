"""The Alpha Lab: from a raw signal to a calibrated forecast of residual return, and from a forecast to what a constrained portfolio keeps of it.

Grinold and Kahn, *Active Portfolio Management* (2nd ed.), and *Advances in Active Portfolio Management* (Grinold and Kahn, 2019): the basic forecasting formula, the fundamental law, the information horizon, and the transfer
coefficient (Clarke, de Silva and Thorley 2002). Paleologo, *Advanced Portfolio Management* (2021): attribution into selection, sizing and timing.

    ic          information coefficients: summary, rolling, by regime, stability, the lagged-IC decay curve and the information horizon (:func:`ic.fit_decay`)
    calibrate   ``alpha = IC x omega x z``: a signal made into an expected residual return, walk-forward, with a reliability check
    breadth     effective breadth, the fundamental law, and the transfer coefficient (the cosine of risk-adjusted alpha and risk-adjusted holdings)
    registry    versioned signals with the information lag each one carries, and a point-in-time check
    dynamic     alpha decay in the multi-period optimiser, trading rules compared, and where the paper alpha goes (implementation loss)
    attribution selection, sizing and timing; stop-loss and leverage policies simulated
    capacity    capacity of a book against the market impact it causes, crowding, optimal gearing

What already existed and is reused rather than rewritten: the IC-weighted combination of factors (:mod:`src.equity.alpha_model`), the multi-period optimiser and the no-trade region
(:mod:`src.equity.multiperiod`), the impact-cost capacity curve (:mod:`src.backtest.impact`), Brinson attribution (:mod:`src.backtest.attribution`).
"""
