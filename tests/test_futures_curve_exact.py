"""The curve views of the Futures tab against calculations from their definitions: the change in the contract that held each rank, the principal components of planted factors, the shape of the curve through
time, carry by calendar month and its bands, what the signals board says, and the settings a futures strategy accepts."""

from __future__ import annotations

import inspect
import math

import numpy as np
import pandas as pd
import pytest

from src.assets import futures_analytics as fa
from src.assets.strategy_runs import STRONG, correlation_view, subset_bundle
from src.framework import MODELS, load_library
from src.utils.config import load_config
from src.webapp.asset_api import AssetApi
from tests.test_futures_strategies_exact import world

load_library()


def table(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["date", "contract", "expiry", "price"])


# ------------------------------------------------------------------------------------------------------------------ the change in the contract that held each rank
def test_rank_changes_follow_the_contract_that_held_the_rank_the_day_before_and_skip_an_expired_one():
    d = pd.bdate_range("2024-03-04", periods=5)                                              # Monday to Friday
    a, b, c = pd.Timestamp("2024-03-06"), pd.Timestamp("2024-03-20"), pd.Timestamp("2024-04-20")
    rows = []
    for day, pa, pb, pc in zip(d, [100, 101, 102, None, None], [102, 102.5, 103, 104, 103], [105, 104, 106, 107, 108]):
        if pa is not None:                                                                  # A is listed up to and on its expiry day, not after
            rows.append((day, "A", a, pa))
        rows += [(day, "B", b, pb), (day, "C", c, pc)]
    got = fa.rank_changes(table(rows), depth=3)
    nan = np.nan
    expected = pd.DataFrame({"F1": [math.log(101 / 100), math.log(102 / 101), nan, math.log(103 / 104)],
                             "F2": [math.log(102.5 / 102), math.log(103 / 102.5), math.log(104 / 103), math.log(108 / 107)],
                             "F3": [math.log(104 / 105), math.log(106 / 104), math.log(107 / 106), nan]}, index=d[1:])
    pd.testing.assert_frame_equal(got.reindex(d[1:]), expected, check_freq=False, check_names=False, rtol=1e-12)
    assert got.loc[d[3], "F1"] != got.loc[d[3], "F1"]                                        # the contract that was first on the 6th is gone on the 7th: no change to report


def test_rank_changes_leave_out_a_contract_that_skipped_a_day():
    d = pd.bdate_range("2024-03-04", periods=4)
    near, far = pd.Timestamp("2024-09-20"), pd.Timestamp("2024-10-20")
    rows = [(day, "A", near, 100.0 + i) for i, day in enumerate(d) if i != 2] + [(day, "B", far, 90.0 + i) for i, day in enumerate(d)]            # A has no price on the third day
    got = fa.rank_changes(table(rows), depth=2)
    nan = np.nan
    expected = pd.DataFrame({"F1": [math.log(101 / 100), nan, math.log(93 / 92)], "F2": [math.log(91 / 90), math.log(92 / 91), nan]}, index=d[1:])
    # day 3: A was first on day 2 and has no price then, so B (first on day 2) is the only contract with a daily change at rank 1; A's change from day 1 to day 4 spans two days and is left out
    pd.testing.assert_frame_equal(got.reindex(d[1:]), expected, check_freq=False, check_names=False, rtol=1e-12)


# ------------------------------------------------------------------------------------------------------------------ the principal components of planted factors
def test_the_components_of_planted_level_slope_and_curvature_moves_are_found_with_their_variance_shares():
    n = 1200
    t = np.arange(n)
    level, slope, hump = (a * np.cos(2 * math.pi * k * t / n) for a, k in ((3.0, 3), (2.0, 7), (1.0, 11)))     # full periods of different frequencies: exactly uncorrelated
    v1, v2, v3 = np.array([1, 1, 1, 1, 1.0]), np.array([-2, -1, 0, 1, 2.0]), np.array([2, -1, -2, -1, 2.0])
    changes = pd.DataFrame(np.outer(level, v1) + np.outer(slope, v2) + np.outer(hump, v3), columns=[f"F{i}" for i in range(1, 6)])
    out = fa.pca_of_changes(changes, 3)
    shares = np.array([9 * 5, 4 * 10, 1 * 14], dtype=float)
    shares /= shares.sum()
    assert [c["label"] for c in out["components"]] == ["level", "slope", "curvature"] and out["n_days"] == n
    assert [c["explained"] for c in out["components"]] == pytest.approx(shares.tolist(), rel=1e-9) and out["total_explained"] == pytest.approx(1.0, rel=1e-9)
    assert np.allclose(out["components"][0]["loadings"], v1 / np.linalg.norm(v1), atol=2e-5)               # level: up across the curve
    assert np.allclose(out["components"][1]["loadings"], v2 / np.linalg.norm(v2), atol=2e-5)               # slope: more on the back than the front
    assert np.allclose(out["components"][2]["loadings"], -v3 / np.linalg.norm(v3), atol=2e-5)              # curvature: signed so that the middle loads up (the planted vector has it down)


def test_a_component_with_no_clear_shape_is_called_higher_order_and_few_components_are_allowed():
    n = 400
    rng = np.random.default_rng(3)
    zigzag = np.array([1, -1, 1, -1, 1.0])
    changes = pd.DataFrame(np.outer(rng.normal(size=n), zigzag) + 0.001 * rng.normal(size=(n, 5)), columns=list("ABCDE"))
    out = fa.pca_of_changes(changes, 2)
    assert out["components"][0]["label"] == "higher order" and len(out["components"]) == 2 and out["components"][0]["explained"] > 0.99


def test_the_factors_of_a_simulated_curve_are_a_level_move_first_and_the_curve_needs_enough_history():
    data = fa.simulated_curve("CL")
    out = fa.term_structure_pca(data, 8)
    assert out["components"][0]["label"] == "level" and out["components"][0]["explained"] > 0.8 and out["ranks"] == list(range(1, 9))
    assert all(a >= b for a, b in zip([c["explained"] for c in out["components"]], [c["explained"] for c in out["components"]][1:]))
    short = fa.CurveData("X", fa.ROOTS["CL"], data.long[pd.to_datetime(data.long["date"]) >= pd.to_datetime(data.long["date"]).max() - pd.Timedelta(days=30)], None, "user_file")
    with pytest.raises(ValueError, match="at least 60 days"):
        fa.term_structure_pca(short, 8)


# ------------------------------------------------------------------------------------------------------------------ the shape of the curve through time
def test_the_shape_history_is_the_log_of_each_contract_over_the_front_on_every_step_th_date_back_from_the_last():
    d = pd.bdate_range("2024-03-04", periods=6)
    expiries = [pd.Timestamp("2024-09-20") + pd.Timedelta(days=30 * k) for k in range(4)]
    rng = np.random.default_rng(4)
    prices = 100 + np.cumsum(rng.normal(0, 1, (6, 4)), axis=0) + np.arange(4) * 2.0
    rows = [(day, f"M{k}", expiries[k], prices[i, k]) for i, day in enumerate(d) for k in range(4)]
    data = fa.CurveData("T", fa.ROOTS["CL"], table(rows), None, "user_file")
    got = fa.curve_shape_history(data, depth=4, step=2)
    idx = [1, 3, 5]                                                                           # every second date, back from the last
    assert got["dates"] == [str(d[i].date()) for i in idx] and got["ranks"] == [2, 3, 4] and got["step_days"] == 2
    for r, rank in enumerate(got["ranks"]):
        assert got["values"][r] == pytest.approx([round(math.log(prices[i, rank - 1] / prices[i, 0]), 5) for i in idx], abs=1e-12)
    with pytest.raises(ValueError, match="only one contract"):
        fa.curve_shape_history(fa.CurveData("T", fa.ROOTS["CL"], table([(day, "M0", expiries[0], 100.0) for day in d]), None, "user_file"), depth=4)


# ------------------------------------------------------------------------------------------------------------------ carry by calendar month and its bands
def test_carry_by_month_is_the_mean_median_and_share_of_positive_days_of_each_month_of_the_year():
    idx = pd.bdate_range("2022-01-03", "2023-12-29")
    rng = np.random.default_rng(6)
    carry = pd.Series(rng.normal(0.02, 0.1, len(idx)), index=idx)
    got = fa.carry_by_month(carry)
    assert [m["month"] for m in got] == list(range(1, 13))
    for m in got:
        v = carry[carry.index.month == m["month"]].to_numpy()
        assert m["mean"] == pytest.approx(v.mean(), rel=1e-12) and m["median"] == pytest.approx(np.median(v), rel=1e-12)
        assert m["positive"] == pytest.approx(float((v > 0).mean()), rel=1e-12) and m["days"] == len(v) and m["years"] == 2
    assert [m["month"] for m in fa.carry_by_month(carry[carry.index.month.isin([3, 4])])] == [3, 4]


def test_the_carry_bands_are_the_quantiles_of_the_year_before_each_day():
    data = fa.simulated_curve("NG")
    out = fa.carry_dashboard(data)
    series, bands = out["series"]["carry"], out["series"]["bands"]
    assert set(bands) == {"p10", "p25", "p50", "p75", "p90"} and all(len(v) == len(series) for v in bands.values())
    for q in (0.1, 0.25, 0.5, 0.75, 0.9):
        key = f"p{int(q * 100)}"
        assert bands[key][-1] == pytest.approx(float(np.quantile(series[-252:], q)), abs=2e-5)           # the last 252 days, with the five-decimal rounding of the payload
    assert bands["p10"][-1] <= bands["p25"][-1] <= bands["p50"][-1] <= bands["p75"][-1] <= bands["p90"][-1]
    assert [m["month"] for m in out["by_month"]] == list(range(1, 13)) and all(m["years"] >= 3 for m in out["by_month"])


# ------------------------------------------------------------------------------------------------------------------ the signals board
@pytest.fixture(scope="module")
def planted():
    return world()


def test_the_strength_of_a_rule_is_its_score_over_the_typical_size_of_its_own_score_capped_at_three(planted):
    out = fa.signals_board(planted, window=504)
    carry = next(r for r in out["signals"] if r["id"] == "carry_ts")
    score = MODELS.create("carry_ts").score(planted).where(planted.investable)
    last, typical = score.dropna(how="all").iloc[-1], score.abs().tail(504).mean()
    for asset in planted.assets:
        assert carry["score"][asset] == pytest.approx(float(last[asset]), rel=1e-12)
        assert carry["strength"][asset] == pytest.approx(round(float(np.clip(last[asset] / typical[asset], -3.0, 3.0)), 3), abs=1e-12)
    assert carry["date"] == str(planted.index[-1].date()) and out["last_day"] == carry["date"] and out["window"] == 504
    tsmom = next(r for r in out["signals"] if r["id"] == "tsmom")
    assert max(abs(v) for v in tsmom["strength"].values() if v is not None) <= 3.0


def test_agreement_counts_the_rules_beyond_a_quarter_of_their_usual_size_and_a_rule_that_cannot_run_says_why(planted):
    out = fa.signals_board(planted)
    shown = [r for r in out["signals"] if "strength" in r]
    for asset, counts in out["agreement"].items():
        values = [r["strength"][asset] for r in shown]
        assert counts["long"] == sum(1 for v in values if v is not None and v > STRONG) and counts["short"] == sum(1 for v in values if v is not None and v < -STRONG)
        assert counts["long"] + counts["short"] + counts["flat"] == len(shown)
    failed = {r["id"]: r["error"] for r in out["signals"] if "error" in r}
    assert "BASISMOM" in failed["basis_momentum"]                                              # the planted bundle has no basis-momentum series
    assert [r["id"] for r in out["signals"]] == [name for _, name, _ in fa.SIGNAL_BOARD]


def test_a_rule_that_needs_two_roots_in_a_class_has_no_signal_for_a_single_root(planted):
    out = fa.signals_board(subset_bundle(planted, ["UP"]))
    carry_xs = next(r for r in out["signals"] if r["id"] == "carry_xs")
    assert "error" in carry_xs and "more history" in carry_xs["error"] and out["agreement"]["UP"]["long"] + out["agreement"]["UP"]["short"] + out["agreement"]["UP"]["flat"] >= 1


# ------------------------------------------------------------------------------------------------------------------ the strategies the page offers and the settings they take
def test_every_futures_strategy_has_its_group_its_settings_and_its_sources():
    infos = [AssetApi._futures_strategy_info(k, v) for k, v in fa.FUTURES_STRATEGIES.items()]
    assert len(infos) == 15 and {i["group"] for i in infos} == {"Carry and curve", "Trend", "Cross-section", "Switches and reversion"}
    assert {i["needs"] for i in infos} <= {None, "cross", "class", "commodities"} and [i["id"] for i in infos][:3] == ["carry_ts", "carry_xs", "basis_momentum"]
    for info in infos:
        entry = next(e for e in MODELS.entries() if e.name == info["model"])
        accepted = [p for p in inspect.signature(entry.factory.__init__).parameters if p != "self"]
        assert [p["name"] for p in info["params"]] == accepted, info["id"]
        assert info["references"] and all(r["cite"] for r in info["references"]), info["id"]
        spec = fa.FUTURES_STRATEGIES[info["id"]]
        assert spec["lookback"] is None or spec["lookback"] in accepted, info["id"]               # the older single setting names a real one


def test_a_strategys_own_settings_reach_the_pipeline_win_over_the_single_lookback_and_are_checked():
    config = load_config()
    roots = ["CL", "NG", "GC"]
    out = fa.run_futures_strategy(config, "momentum_carry", roots, "weekly", 126, None, 2.0, None, None, {"trend_weight": 1.0, "skip": 5})
    assert out["spec"]["models"][0]["params"] == {"lookback": 126, "trend_weight": 1.0, "skip": 5} and out["label"] == "Trend and carry together"
    over = fa.run_futures_strategy(config, "momentum_carry", roots, "weekly", 126, None, 2.0, None, None, {"lookback": 200})
    assert over["spec"]["models"][0]["params"] == {"lookback": 200}
    with pytest.raises(ValueError, match="Trend and carry together.*trend_weight"):
        fa.run_futures_strategy(config, "momentum_carry", roots, "weekly", None, None, 2.0, None, None, {"trend_weight": 5.0})
    with pytest.raises(ValueError, match="Trend and carry together"):
        fa.run_futures_strategy(config, "momentum_carry", roots, "weekly", None, None, 2.0, None, None, {"no_such_setting": 1})
    assert out["correlation"]["assets"] == ["CL", "GC", "NG"] and np.allclose(np.diag(np.array(out["correlation"]["values"], dtype=float)), 1.0)


def test_the_correlation_view_is_the_pairwise_correlation_with_each_assets_annual_volatility():
    rng = np.random.default_rng(8)
    r = pd.DataFrame(rng.normal(0, 0.01, (300, 3)), columns=["A", "B", "C"])
    r["B"] = 0.6 * r["A"] + 0.8 * r["B"]
    r.loc[:260, "C"] = np.nan                                                               # C has only 39 days: too few for a pair
    out = correlation_view(r)
    assert out["assets"] == ["A", "B", "C"] and out["values"][0][1] == pytest.approx(round(float(np.corrcoef(r["A"], r["B"])[0, 1]), 3), abs=1e-12)
    assert out["values"][0][2] is None and out["values"][2][2] is None and out["values"][1][0] == out["values"][0][1]
    assert out["volatility"][0] == pytest.approx(round(float(r["A"].std(ddof=1) * math.sqrt(252)), 4), abs=1e-12)
    assert correlation_view(r[["A"]]) is None
