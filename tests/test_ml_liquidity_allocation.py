"""The cost-aware allocation told how large the account is: capital and traded volume enter the optimiser's choice, with only what was known at the decision."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from src.ml import config as mc
from src.ml.signals import build_weights, cost_aware_mean_variance

warnings.filterwarnings("ignore")
ASSETS = ["A", "B", "C", "D", "E", "F"]
CFG = {"method": "cost_aware_mean_variance", "cost_bps": 5.0, "window_days": 5, "opt_every": 5, "risk_aversion": 2.0, "gross": 1.0, "net": 0.0, "max_weight": 0.5, "cov_window": 120}


@pytest.fixture(scope="module")
def panel():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2022-01-03", periods=320)
    returns = pd.DataFrame(rng.normal(0.0, 0.012, (320, 6)), index=idx, columns=ASSETS)
    score = pd.DataFrame(rng.normal(0.0, 0.01, (320, 6)), index=idx, columns=ASSETS)
    score.iloc[:150] = np.nan
    volume = pd.DataFrame(np.tile([5e8, 5e8, 5e8, 5e8, 2e6, 1e6], (320, 1)) * rng.uniform(0.8, 1.2, (320, 6)), index=idx, columns=ASSETS)       # E and F are thin
    return score, returns, volume


def turnover(w: pd.DataFrame) -> pd.Series:
    return w.diff().abs()


def test_without_capital_nothing_changes_and_volume_is_not_needed(panel):
    score, returns, volume = panel
    plain, log = cost_aware_mean_variance(score, returns, None, CFG)
    with_volume, _ = cost_aware_mean_variance(score, returns, None, CFG, None, volume)
    assert np.array_equal(plain.to_numpy(), with_volume.to_numpy()) and log["liquidity"] is None and log["optimal"] > 20


def test_a_participation_limit_caps_every_trade_at_that_share_of_the_names_volume(panel):
    score, returns, volume = panel
    capital, share = 2e8, 0.1
    w, log = cost_aware_mean_variance(score, returns, None, {**CFG, "capital": capital, "max_participation": share}, None, volume)
    free, _ = cost_aware_mean_variance(score, returns, None, CFG)
    days = score.index[150::5]
    adv = volume.rolling(20).mean()
    limit = share * adv.loc[days] / capital                                                 # in weight: a tenth of the average day's dollar volume over the capital
    traded = (w.loc[days] - w.shift(1).loc[days]).abs()
    assert (traded <= limit + 1e-7).all().all()
    assert limit[["E", "F"]].max().max() < 0.002 and w[["E", "F"]].abs().max().max() < 0.05 and free[["E", "F"]].abs().max().max() > 0.2       # the thin names are barely held; unlimited, they were
    assert log["liquidity"]["capital"] == capital and log["liquidity"]["solves_at_a_trade_limit"] > 10 and log["optimal"] > 20
    assert turnover(w)[["A", "B", "C", "D"]].sum().sum() > 0.5                              # the liquid names still trade


def test_more_capital_means_a_smaller_position_in_a_thin_name_and_impact_alone_does_it_too(panel):
    score, returns, volume = panel
    held = {}
    for capital in (1e6, 1e8, 1e10):
        w, _ = cost_aware_mean_variance(score, returns, None, {**CFG, "capital": capital, "max_participation": 0.1}, None, volume)
        held[capital] = float(w[["E", "F"]].abs().mean().mean())
    assert held[1e6] > held[1e8] > held[1e10] and held[1e10] < 1e-4
    free, _ = cost_aware_mean_variance(score, returns, None, CFG)
    costly, log = cost_aware_mean_variance(score, returns, None, {**CFG, "capital": 1e9, "impact": 1.0}, None, volume)
    assert turnover(costly)[["E", "F"]].sum().sum() < 0.3 * turnover(free)[["E", "F"]].sum().sum()        # trading a thin name moves its price: the optimiser trades it less, with no hard limit set
    assert log["liquidity"]["impact"] == 1.0 and log["liquidity"]["max_participation"] is None


def test_the_limit_at_a_date_uses_only_the_volume_up_to_that_date(panel):
    score, returns, volume = panel
    cfg = {**CFG, "capital": 2e8, "max_participation": 0.1}
    base, _ = cost_aware_mean_variance(score, returns, None, cfg, None, volume)
    later = volume.copy()
    later.iloc[250:] *= 1000.0                                                              # the thin names become liquid, but only from day 250
    changed, _ = cost_aware_mean_variance(score, returns, None, cfg, None, later)
    assert np.array_equal(base.iloc[:250].to_numpy(), changed.iloc[:250].to_numpy())
    assert changed.iloc[260:][["E", "F"]].abs().max().max() > 5 * base.iloc[260:][["E", "F"]].abs().max().max()


def test_a_name_with_no_volume_is_not_traded_and_is_counted(panel):
    score, returns, volume = panel
    missing = volume.copy()
    missing["D"] = np.nan
    w, log = cost_aware_mean_variance(score, returns, None, {**CFG, "capital": 2e8, "max_participation": 0.1}, None, missing)
    assert w["D"].abs().max() < 1e-9 and log["liquidity"]["names_without_volume"] >= log["solves"]


def test_asking_for_liquidity_without_what_it_needs_is_refused(panel):
    score, returns, volume = panel
    with pytest.raises(ValueError, match="needs the traded volume"):
        cost_aware_mean_variance(score, returns, None, {**CFG, "capital": 1e8, "max_participation": 0.1})
    with pytest.raises(ValueError, match="needs signal.capital"):
        cost_aware_mean_variance(score, returns, None, {**CFG, "max_participation": 0.1}, None, volume)
    with pytest.raises(ValueError, match="does nothing on its own"):
        cost_aware_mean_variance(score, returns, None, {**CFG, "capital": 1e8}, None, volume)
    with pytest.raises(ValueError, match="needs the traded volume"):
        build_weights(score, {**mc.DEFAULTS["signal"], **CFG, "holding_period": 1, "capital": 1e8, "impact": 0.5}, returns=returns)
    w = build_weights(score, {**mc.DEFAULTS["signal"], **CFG, "holding_period": 1, "capital": 1e8, "max_participation": 0.1}, returns=returns, dollar_volume=volume)
    assert w.attrs["allocation"]["liquidity"]["max_participation"] == 0.1


def test_the_configuration_checks_the_liquidity_settings():
    def cfg(**signal):
        return mc.experiment_config({"signal": {"method": "cost_aware_mean_variance", "holding_period": 1, **signal}}, defaults=mc.DEFAULTS)

    assert cfg(capital=1e7, max_participation=0.05)["signal"]["capital"] == 1e7
    for bad, message in (({"max_participation": 0.1}, "signal.capital must be given"), ({"capital": 1e7, "max_participation": 1.5}, "share of a day's volume"), ({"capital": -1, "impact": 1.0}, "signal.capital must be given"),
                         ({"capital": 1e7, "impact": 50.0}, "between 0 and 10"), ({"capital": 1e7, "impact": 1.0, "adv_window": 1}, "adv_window")):
        with pytest.raises(mc.MLConfigError, match=message):
            cfg(**bad)
    with pytest.raises(mc.MLConfigError, match="belong to the cost_aware_mean_variance method"):
        mc.experiment_config({"signal": {"method": "top_n_long_short", "capital": 1e7, "max_participation": 0.1}}, defaults=mc.DEFAULTS)
    assert mc.config_hash(mc.experiment_config({}, defaults=mc.DEFAULTS)) == mc.config_hash(mc.experiment_config({"signal": {}}, defaults=mc.DEFAULTS))       # no new default: no existing hash moves


def test_in_an_experiment_the_volume_of_the_universe_reaches_the_optimiser_and_the_report_says_what_it_used(tmp_path):
    from src.ml import experiments as xp, registry as rg
    from tests.ml_helpers import make_bundle

    reg = rg.Registry(tmp_path / "ml")
    bundle = make_bundle(10, 1150, seed=71, alpha=1.0)
    assert bundle.volume is not None
    signal = {"method": "cost_aware_mean_variance", "holding_period": 1, "gross": 1.0, "net": 0.0, "max_weight": 0.3, "risk_aversion": 10.0, "cov_window": 126, "opt_every": 5}
    over = {"dataset": {"feature_groups": ["price", "momentum"], "target": "fwd_ret_5"}, "split": {"min_train_days": 400, "refit_every": 63}, "model": {"kind": "ridge", "params": {"alpha": 10.0}},
            "signal": signal, "backtest": {"cost_bps": 1.0, "slippage_bps": 1.0}}
    free = xp.run_experiment(over, bundle=bundle, registry=reg, explain=False, final_model=False, robustness=False)
    dollars = float((bundle.volume * bundle.prices).rolling(20).mean().median().median())
    capital = 200.0 * dollars                                                                # an account two hundred times a typical name's daily dollar volume
    tight = xp.run_experiment({**over, "signal": {**signal, "capital": capital, "max_participation": 0.05}, "name": "tight"}, bundle=bundle, registry=reg, explain=False, final_model=False, robustness=False)
    liq = reg.report(tight.id)["allocation"]["liquidity"]
    assert liq["capital"] == pytest.approx(capital) and liq["max_participation"] == 0.05 and liq["solves_at_a_trade_limit"] > 0 and reg.report(free.id)["allocation"]["liquidity"] is None
    pos = pd.read_parquet(reg.root / "experiments" / tight.id / "positions.parquet")
    assert tight.metrics["oos"]["ann_turnover"] < 0.2 * free.metrics["oos"]["ann_turnover"]   # each trade is capped at 5% of a day's volume: 0.025% of this account
    assert pos.abs().max().max() <= 0.3 + 1e-9 and tight.id != free.id
    stripped = make_bundle(10, 1150, seed=71, alpha=1.0, volume=False)
    with pytest.raises(xp.ExperimentError, match="needs the traded volume"):
        xp.run_experiment({**over, "signal": {**signal, "capital": capital, "max_participation": 0.05}, "name": "blind"}, bundle=stripped, registry=reg, explain=False, final_model=False, robustness=False)
