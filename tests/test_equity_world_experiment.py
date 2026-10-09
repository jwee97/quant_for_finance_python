"""The experiment behind the equity guides runs, is deterministic, and its tables have the shape the guides describe."""

from __future__ import annotations

import numpy as np

from experiments.equity_world import run


def test_the_experiment_returns_the_tables_the_guides_quote_and_is_repeatable():
    a, b = run(n_assets=40, n_years=8, seed=3), run(n_assets=40, n_years=8, seed=3)
    assert set(a) >= {"world", "ic", "marginal", "alpha", "nonlinear", "slopes"}
    assert a["ic"].equals(b["ic"]) and a["alpha"].equals(b["alpha"])
    ic = a["ic"]
    assert {"style", "prior_sign", "mean_ic", "t", "months"} <= set(ic.columns) and ic["style"].isin(["value", "quality", "momentum", "estimates", "valuation"]).all()
    value = ic[ic["style"] == "value"]
    assert value["mean_ic"].mean() > 0                                              # the value yields all share the planted cheapness
    assert list(a["marginal"].index) == ["b2p", "earnings_yield", "cfo2ev", "s2ev"] and set(a["marginal"].columns) >= {"slope", "t", "mean_ic_alone"}
    assert {"equal weights", "optimal weights", "optimal, Gram-Schmidt", "optimal, symmetric", "value alone"} <= set(a["alpha"].index)
    assert a["alpha"].loc["equal weights", "mean_ic"] > a["alpha"].loc["quality alone", "mean_ic"]
    assert list(a["nonlinear"].index) == ["linear terms only", "with icapx squared"] and np.isfinite(a["slopes"].loc["icapx^2", "slope"])
