"""Stock Universe Research Lab: which stocks to hold, and whether a portfolio built from them survives honest out-of-sample testing.

* ``membership``: point-in-time eligibility (who could have been held on a date) from a listings table, with the hindsight of today's constituents labelled when no such table exists;
* ``screens``: universe screens (liquidity, size proxy, volatility, momentum, sector, manual) computed only from data up to the decision date;
* ``walk``: a walk-forward portfolio on a universe that changes over time, with drift, costs and the one-day information lag;
* ``discover``: a nested chronological search in which the screen, the construction method and the lookback are ONE selection process, chosen on past data only and judged on data it never saw;
* ``robust``: what makes a result believable or not: drawdown and tail, turnover and cost, beta, concentration, constituent stability, stock-removal sensitivity, a block bootstrap, regimes.
"""
