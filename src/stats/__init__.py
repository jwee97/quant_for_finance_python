"""Statistics for research: regression with robust covariances, panel and asset-pricing regressions, time-series tests, event studies and inference.

    regression    ols, wls, gls, feasible_gls_ar1, rolling_ols, diagnostics (HC0-3, Newey-West, clustered, two-way clustered)
    panel         fama_macbeth, fama_macbeth_two_pass (Shanken), grs_test, panel_ols (fixed effects, Driscoll-Kraay), random_effects, hausman_test
    timeseries    adf_test, phillips_perron, kpss_test, variance_ratio, hurst_exponent, engle_granger, granger_causality, ljung_box, arch_lm
    event_study   event_study (market model, BMP, Corrado), buy_and_hold_abnormal_return
    inference     p_adjust, romano_wolf, storey_qvalues, Sharpe inference, deflated Sharpe, bootstrap_ci
"""

from .event_study import EventStudyResult, buy_and_hold_abnormal_return, event_study
from .inference import (bootstrap_ci, deflated_sharpe_ratio, expected_max_sharpe, haircut_sharpe, min_track_record_length, p_adjust, probabilistic_sharpe_ratio,
                        romano_wolf, sharpe_standard_error, storey_qvalues)
from .panel import (FamaMacBethResult, PanelResult, fama_macbeth, fama_macbeth_two_pass, grs_test, hausman_test, long_from_wide, panel_ols,
                    random_effects)
from .regression import (OLSResult, diagnostics, durbin_watson, feasible_gls_ar1, gls, jarque_bera, newey_west_lag, ols, rolling_ols, variance_inflation,
                         wls)
from .timeseries import (UnitRootResult, acf, adf_test, arch_lm, engle_granger, granger_causality, half_life, hurst_exponent, kpss_test, ljung_box,
                         phillips_perron, stationarity_summary, variance_ratio)

__all__ = [n for n in dir() if not n.startswith("_")]
