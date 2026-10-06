"""The Stage 26 grid is the declared one, and a single rule evaluates the same however it is run."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from experiments.stage26_distributed import evaluate_rule, expand_grid
from src.backtest.costs import LinearCostModel
from src.backtest.engine import BacktestEngine
from src.features.volatility import rolling_volatility
from src.signals.alpha_engine import forward_returns
from src.utils.config import load_config


def test_the_declared_grid_has_1584_distinct_rules():
    tasks = expand_grid(load_config().get("distributed"))
    assert len(tasks) == 1584
    keys = {tuple((k, v) for k, v in sorted(t.items()) if k != "id") for t in tasks}
    assert len(keys) == len(tasks)
    assert sum(t["family"] == "momentum" for t in tasks) == 1440 and sum(t["family"] == "mean_reversion" for t in tasks) == 144
    assert all(t["vol_lookback"] is None for t in tasks if t["variant"] in ("raw", "ranked"))
    assert [t["id"] for t in tasks] == list(range(1584))


def test_one_rule_is_deterministic_and_causal(synthetic_market):
    returns = synthetic_market.returns()
    shared = SimpleNamespace(prices=synthetic_market.prices, returns=returns, investable=synthetic_market.investable,
                             volatility=rolling_volatility(returns, 63),
                             transform={"winsorize_quantile": 0.02, "cross_sectional": True, "scale": "zscore", "clip": 3.0, "target_vol": 0.10,
                                        "max_abs_weight": 0.5, "gross_leverage": 1.0, "long_only_book": False},
                             engine=BacktestEngine(cost_model=LinearCostModel(5.0), target_vol=0.10), fwd21=forward_returns(returns, 21))
    task = {"id": 0, "family": "momentum", "variant": "vol_scaled", "lookback": 63, "skip": 1, "vol_lookback": 63,
            "rebalance": "monthly", "signal_lag": 1, "cross_sectional": True}
    a, b = evaluate_rule(task, shared), evaluate_rule(task, shared)
    assert np.array_equal(a["net"], b["net"]) and a["ann_turnover"] > 0
    # changing the future must not change the past: perturb the last 100 days and compare the earlier returns
    shocked = shared.returns.copy()
    shocked.iloc[-100:] *= 3.0
    prices = 100.0 * (1.0 + shocked).cumprod().where(synthetic_market.investable)
    shared2 = SimpleNamespace(**{**vars(shared), "prices": prices, "returns": shocked, "volatility": rolling_volatility(shocked, 63),
                                 "fwd21": forward_returns(shocked, 21)})
    c = evaluate_rule(task, shared2)
    keep = len(a["net"]) - 120
    assert np.allclose(a["net"][:keep], c["net"][:keep], atol=1e-12)
