# Strategy survey: every rule-based strategy, once, on the same 15 ETFs

79 strategies, the platform's 15 ETFs, 2006-01-03 to 2026-09-21 (20.7 years), net of costs. **This is a survey, not a study.** Nothing was tuned: every strategy uses its default parameters,
rules are traded as written (no calibration against history) at the rebalance frequency each rule declares (daily, weekly or monthly): per-asset rules such as a pullback entry or a trend filter run as independent sleeves with an equal slice of capital each and cash when flat, selection and cross-sectional rules as a ranked book; costs are the platform's 10 bps per unit traded,
and the deflated Sharpe probability counts all 79 strategies as trials. Equal weight over the same ETFs earns a net Sharpe of about 0.67 (the median over the windows below). **Each strategy starts on a different day** (it needs history before its first signal), so compare it with the *Equal weight, same dates* column, not with a single number.

How to read it: with this many strategies, some at the top are there by luck. Look for a rule that beats equal weight **and** has a deflated probability near or above 0.95, then ask whether its mechanism is plausible and whether it survives your own tickers, a different period and a higher cost. Several rules (calendar effects, single-stock factors) are weaker on a 15-ETF universe than in the papers that made them famous.

| Strategy | Family | Net Sharpe | Equal weight, same dates | CAGR | Volatility | Max drawdown | Turnover (x/yr) | Deflated Sharpe | First day |
|---|---|---|---|---|---|---|---|---|---|
| [dual_momentum](strategies/dual_momentum.md) | time-series | +0.88 | +0.88 | 9.8% | 11.3% | -20.1% | 5.3 | 0.89 | 2009-02-03 |
| [model_portfolio](strategies/model_portfolio.md) | allocation | +0.83 | +0.65 | 9.1% | 11.4% | -26.4% | 1.4 | 0.86 | 2008-03-03 |
| [vol_managed_long](strategies/vol_managed_long.md) | time-series | +0.81 | +0.94 | 13.5% | 17.7% | -30.6% | 10.2 | 0.82 | 2009-03-04 |
| [daa](strategies/daa.md) | allocation | +0.76 | +0.92 | 8.1% | 11.0% | -26.0% | 13.0 | 0.77 | 2009-03-02 |
| [ibs_reversion](strategies/ibs_reversion.md) | time-series | +0.76 | +0.86 | 6.9% | 9.3% | -18.4% | 6.3 | 0.78 | 2008-11-14 |
| [stochastic_reversion](strategies/stochastic_reversion.md) | time-series | +0.76 | +0.86 | 6.6% | 9.0% | -19.8% | 2.9 | 0.77 | 2008-11-14 |
| [faber_gtaa](strategies/faber_gtaa.md) | allocation | +0.75 | +0.86 | 6.9% | 9.6% | -27.8% | 3.1 | 0.75 | 2008-12-01 |
| [paa](strategies/paa.md) | allocation | +0.73 | +0.92 | 7.9% | 11.3% | -31.2% | 6.9 | 0.73 | 2009-03-02 |
| [evt_risk_managed](strategies/evt_risk_managed.md) | volatility | +0.68 | +0.76 | 6.8% | 10.5% | -20.3% | 2.0 | 0.56 | 2012-01-25 |
| [garch_vol_managed](strategies/garch_vol_managed.md) | volatility | +0.56 | +0.76 | 7.0% | 13.7% | -26.4% | 13.1 | 0.39 | 2012-01-24 |
| [adaptive_asset_allocation](strategies/adaptive_asset_allocation.md) | allocation | +0.55 | +0.68 | 6.0% | 12.0% | -30.5% | 9.3 | 0.46 | 2008-08-28 |
| [accelerating_dual_momentum](strategies/accelerating_dual_momentum.md) | time-series | +0.46 | +0.67 | 4.2% | 10.2% | -25.0% | 16.7 | 0.31 | 2008-08-04 |
| [yield_curve_regime](strategies/yield_curve_regime.md) | macro | +0.45 | +0.65 | 3.9% | 9.4% | -27.7% | 2.6 | 0.31 | 2008-02-04 |
| [sell_in_may](strategies/sell_in_may.md) | time-series | +0.45 | +0.65 | 2.1% | 5.0% | -12.2% | 0.7 | 0.30 | 2008-02-04 |
| [tsmom](strategies/tsmom.md) | time-series | +0.41 | +0.89 | 5.3% | 15.5% | -34.9% | 13.4 | 0.24 | 2009-02-03 |
| [vaa](strategies/vaa.md) | allocation | +0.39 | +0.92 | 3.9% | 11.3% | -21.4% | 16.5 | 0.21 | 2009-03-02 |
| [ma_crossover](strategies/ma_crossover.md) | time-series | +0.37 | +0.85 | 2.1% | 6.1% | -13.7% | 1.9 | 0.19 | 2008-11-14 |
| [rsi2](strategies/rsi2.md) | time-series | +0.35 | +0.86 | 0.9% | 2.7% | -6.1% | 13.9 | 0.17 | 2008-11-14 |
| [momentum](strategies/momentum.md) | time-series/cross-sectional | +0.35 | +0.67 | 2.1% | 6.7% | -14.8% | 9.3 | 0.17 | 2008-08-05 |
| [commodity_supercycle](strategies/commodity_supercycle.md) | macro | +0.34 | +0.74 | 3.3% | 11.5% | -34.9% | 1.8 | 0.12 | 2013-02-05 |
| [high_52w](strategies/high_52w.md) | cross-sectional | +0.34 | +0.88 | 1.7% | 5.5% | -12.1% | 5.6 | 0.15 | 2009-02-02 |
| [residual_momentum](strategies/residual_momentum.md) | cross-sectional | +0.33 | +0.84 | 2.0% | 6.6% | -16.4% | 5.7 | 0.14 | 2009-08-04 |
| [dollar_strength](strategies/dollar_strength.md) | macro | +0.32 | +0.67 | 3.1% | 11.6% | -41.0% | 3.8 | 0.14 | 2008-08-04 |
| [xs_momentum](strategies/xs_momentum.md) | cross-sectional | +0.32 | +0.88 | 2.0% | 7.2% | -19.3% | 6.5 | 0.13 | 2009-02-03 |
| [smooth_momentum](strategies/smooth_momentum.md) | cross-sectional | +0.32 | +0.88 | 2.0% | 7.2% | -19.2% | 6.4 | 0.13 | 2009-02-03 |
| [bollinger_reversion](strategies/bollinger_reversion.md) | time-series | +0.27 | +0.67 | 1.7% | 6.9% | -22.9% | 8.7 | 0.10 | 2008-03-03 |
| [ma_ensemble](strategies/ma_ensemble.md) | time-series | +0.26 | +0.87 | 1.1% | 4.8% | -13.6% | 4.0 | 0.09 | 2009-06-18 |
| [relative_strength](strategies/relative_strength.md) | cross-sectional | +0.24 | +0.67 | 1.5% | 7.5% | -20.6% | 9.3 | 0.08 | 2008-08-04 |
| [vix_spike_reversion](strategies/vix_spike_reversion.md) | time-series | +0.24 | +0.66 | 1.1% | 5.1% | -16.5% | 2.2 | 0.08 | 2008-05-02 |
| [trend_atr](strategies/trend_atr.md) | time-series | +0.24 | +0.85 | 1.2% | 5.8% | -14.6% | 1.1 | 0.08 | 2008-11-14 |
| [carry](strategies/carry.md) | cross-sectional | +0.20 | +0.65 | 1.0% | 6.2% | -17.5% | 0.9 | 0.06 | 2008-02-04 |
| [cftc_positioning](strategies/cftc_positioning.md) | macro | +0.19 | +0.87 | 1.4% | 9.9% | -28.7% | 7.4 | 0.05 | 2009-07-20 |
| [momentum_13612w](strategies/momentum_13612w.md) | cross-sectional | +0.18 | +0.88 | 1.1% | 7.8% | -21.7% | 15.4 | 0.04 | 2009-02-03 |
| [vrp_timing](strategies/vrp_timing.md) | volatility | +0.18 | +0.65 | 1.4% | 12.0% | -49.4% | 8.5 | 0.05 | 2008-03-05 |
| [carry_rolldown](strategies/carry_rolldown.md) | fixed income | +0.17 | +0.65 | 0.9% | 6.1% | -18.1% | 0.9 | 0.04 | 2008-02-04 |
| [quality_proxy](strategies/quality_proxy.md) | cross-sectional | +0.17 | +0.88 | 0.8% | 5.9% | -25.8% | 5.1 | 0.04 | 2009-02-03 |
| [breakout_ensemble](strategies/breakout_ensemble.md) | time-series | +0.16 | +0.66 | 0.7% | 5.2% | -10.4% | 9.5 | 0.04 | 2008-04-01 |
| [bvar_lead_lag](strategies/bvar_lead_lag.md) | cross-sectional | +0.10 | +0.76 | 0.4% | 6.0% | -17.4% | 18.8 | 0.02 | 2014-06-02 |
| [duration_timing](strategies/duration_timing.md) | fixed income | +0.09 | +0.67 | 0.4% | 8.8% | -28.1% | 3.9 | 0.02 | 2008-08-04 |
| [risk_on_off](strategies/risk_on_off.md) | macro | +0.07 | +0.65 | 0.2% | 9.0% | -32.2% | 6.0 | 0.02 | 2008-02-04 |
| [kalman_trend](strategies/kalman_trend.md) | time-series | +0.07 | +0.66 | 0.1% | 2.6% | -6.8% | 1.9 | 0.02 | 2008-03-05 |
| [low_volatility](strategies/low_volatility.md) | cross-sectional | +0.06 | +0.67 | 0.2% | 5.9% | -23.8% | 1.3 | 0.01 | 2008-08-04 |
| [bab](strategies/bab.md) | cross-sectional | +0.06 | +0.68 | 0.1% | 10.7% | -44.4% | 3.7 | 0.01 | 2008-08-28 |
| [kalman_pairs](strategies/kalman_pairs.md) | stat-arb | +0.03 | +0.65 | 0.0% | 4.7% | -17.0% | 25.0 | 0.01 | 2008-02-04 |
| [max_effect](strategies/max_effect.md) | cross-sectional | +0.03 | +0.65 | -0.0% | 6.0% | -26.8% | 5.9 | 0.01 | 2008-03-05 |
| [value_momentum](strategies/value_momentum.md) | cross-sectional | +0.03 | +0.74 | -0.0% | 6.5% | -27.4% | 9.1 | 0.01 | 2013-02-05 |
| [low_idio_vol](strategies/low_idio_vol.md) | cross-sectional | +0.02 | +0.88 | 0.0% | 4.6% | -19.0% | 1.3 | 0.01 | 2009-02-03 |
| [turn_of_month](strategies/turn_of_month.md) | time-series | +0.01 | +0.67 | -0.0% | 3.1% | -12.2% | 8.1 | 0.01 | 2008-02-04 |
| [tsmom_multi](strategies/tsmom_multi.md) | time-series | -0.01 | +0.89 | -0.2% | 6.5% | -26.2% | 12.5 | 0.01 | 2009-02-03 |
| [defensive_beta](strategies/defensive_beta.md) | cross-sectional | -0.01 | +0.67 | -0.3% | 7.0% | -29.1% | 1.4 | 0.01 | 2008-08-04 |
| [butterfly](strategies/butterfly.md) | fixed income | -0.03 | +0.65 | -0.5% | 7.7% | -24.5% | 10.3 | 0.01 | 2008-02-04 |
| [cointegration_pairs](strategies/cointegration_pairs.md) | stat-arb | -0.04 | +0.65 | -0.2% | 3.4% | -10.6% | 8.0 | 0.00 | 2008-02-04 |
| [vol_breakout](strategies/vol_breakout.md) | time-series | -0.05 | +0.65 | -0.2% | 2.7% | -10.8% | 1.6 | 0.00 | 2008-03-03 |
| [keltner_breakout](strategies/keltner_breakout.md) | time-series | -0.08 | +0.67 | -0.2% | 2.7% | -11.2% | 2.4 | 0.00 | 2008-03-03 |
| [macd_trend](strategies/macd_trend.md) | time-series | -0.10 | +0.66 | -0.3% | 2.5% | -10.3% | 4.2 | 0.00 | 2008-03-11 |
| [variance_carry](strategies/variance_carry.md) | volatility | -0.10 | +0.67 | -1.9% | 12.2% | -46.1% | 6.7 | 0.00 | 2008-08-14 |
| [mean_reversion](strategies/mean_reversion.md) | cross-sectional | -0.13 | +0.66 | -1.1% | 6.9% | -22.5% | 22.7 | 0.00 | 2008-02-15 |
| [credit_spread_timing](strategies/credit_spread_timing.md) | time-series | -0.17 | +0.79 | -0.6% | 3.5% | -16.8% | 3.0 | 0.00 | 2010-05-04 |
| [curve_steepener](strategies/curve_steepener.md) | fixed income | -0.17 | +0.64 | -2.3% | 10.7% | -59.2% | 14.5 | 0.00 | 2008-05-05 |
| [adx_trend](strategies/adx_trend.md) | time-series | -0.24 | +0.65 | -1.9% | 6.8% | -36.7% | 9.2 | 0.00 | 2008-03-12 |
| [consecutive_down](strategies/consecutive_down.md) | time-series | -0.26 | +0.86 | -0.4% | 1.5% | -12.4% | 15.2 | 0.00 | 2008-11-14 |
| [sparse_basket](strategies/sparse_basket.md) | stat-arb | -0.28 | +0.89 | -1.1% | 3.7% | -24.6% | 23.1 | 0.00 | 2009-04-29 |
| [supertrend](strategies/supertrend.md) | time-series | -0.31 | +0.68 | -2.8% | 8.2% | -54.0% | 4.8 | 0.00 | 2008-02-15 |
| [short_term_reversal](strategies/short_term_reversal.md) | cross-sectional | -0.31 | +0.65 | -2.5% | 7.2% | -46.6% | 98.7 | 0.00 | 2008-05-05 |
| [kama_trend](strategies/kama_trend.md) | time-series | -0.32 | +0.65 | -1.9% | 5.4% | -32.9% | 19.9 | 0.00 | 2008-03-04 |
| [pca_residual](strategies/pca_residual.md) | stat-arb | -0.33 | +0.81 | -1.1% | 3.3% | -19.0% | 22.7 | 0.00 | 2010-08-17 |
| [ichimoku_trend](strategies/ichimoku_trend.md) | time-series | -0.34 | +0.64 | -3.2% | 8.5% | -60.1% | 11.5 | 0.00 | 2008-05-23 |
| [long_term_reversal](strategies/long_term_reversal.md) | cross-sectional | -0.37 | +0.74 | -1.4% | 3.5% | -20.5% | 3.6 | 0.00 | 2013-02-05 |
| [inflation_rotation](strategies/inflation_rotation.md) | macro | -0.38 | +0.67 | -4.6% | 10.7% | -64.1% | 8.2 | 0.00 | 2008-08-04 |
| [amihud_illiquidity](strategies/amihud_illiquidity.md) | cross-sectional | -0.39 | +0.64 | -2.8% | 6.6% | -45.3% | 2.5 | 0.00 | 2008-05-05 |
| [seasonal_rank](strategies/seasonal_rank.md) | cross-sectional | -0.42 | +0.73 | -3.0% | 6.7% | -42.3% | 24.1 | 0.00 | 2013-02-04 |
| [value_proxy](strategies/value_proxy.md) | cross-sectional | -0.48 | +0.74 | -3.4% | 6.8% | -40.8% | 2.7 | 0.00 | 2013-02-05 |
| [donchian](strategies/donchian.md) | time-series | -0.52 | +0.64 | -5.5% | 9.8% | -73.9% | 1.4 | 0.00 | 2008-04-23 |
| [implied_vs_realized](strategies/implied_vs_realized.md) | volatility | -0.53 | +0.65 | -6.2% | 11.0% | -75.3% | 8.9 | 0.00 | 2008-03-05 |
| [ou_reversion](strategies/ou_reversion.md) | time-series | -0.56 | +0.68 | -0.3% | 0.5% | -4.9% | 3.6 | 0.00 | 2008-07-25 |
| [range_reversion](strategies/range_reversion.md) | time-series | -1.36 | +0.67 | -3.3% | 2.5% | -46.8% | 28.8 | 0.00 | 2008-03-12 |

Could not run on this bundle:

- `basis_momentum`: KeyError: 'needs macro series named BASISMOM_<asset>; this bundle has none (build one with src.assets.bundles)'
- `carry_ts`: KeyError: 'needs macro series named CARRY_<asset>; this bundle has none (build one with src.assets.bundles)'
- `carry_xs`: KeyError: 'needs macro series named CARRY_<asset>; this bundle has none (build one with src.assets.bundles)'

Regenerate with `python -m experiments.library_survey`. Single runs of this kind are exploratory: see [the research reports](../README.md) for the pre-registered studies.
