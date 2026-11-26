"""Attribution of what the Portfolio Lab's optimisers earned: identities that hold to machine precision, reconciliation against the Lab's own net return, the treatment of costs, cash, missing groups and
changing classifications, and point-in-time exposures. Small deterministic data with answers worked out by hand."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.portfolio import attribution as at
from src.portfolio import lab


def _returns(T=520, N=6, seed=0, noise=0.01):
    rng = np.random.default_rng(seed)
    f = rng.standard_normal((T, 2)) * 0.006
    B = rng.standard_normal((N, 2)) * 0.8 + np.array([1.0, 0.0])
    return pd.DataFrame(f @ B.T + rng.standard_normal((T, N)) * noise + 0.0003, index=pd.bdate_range("2019-01-01", periods=T), columns=[f"S{i}" for i in range(N)])


@pytest.fixture(scope="module")
def world():
    data = _returns()
    res = lab.compare(data, ["equal_weight", "min_variance", "max_sharpe"], lookback=120, step=40, cost_bps=25.0)
    return data, res


def _net(res, key):
    e = res["equity"][key]
    n = e.pct_change()
    n.iloc[0] = e.iloc[0] - 1.0
    return n


# ------------------------------------------------------------------------------------------------------------------------ the replay is the Lab's accounting
def test_the_replay_of_the_target_weights_equals_the_labs_own_net_return_to_rounding(world):
    data, res = world
    for key in res["weights"]:
        rp = at.replay(data, res["weights"][key], res["config"]["step"], res["config"]["cost_bps"])
        assert (rp["net"] - _net(res, key)).abs().max() < 1e-14, key
        assert np.allclose(rp["earning"].sum(axis=1), 1.0)


def test_the_first_rebalance_pays_the_cost_of_the_whole_book_and_later_ones_only_the_drift(world):
    data, res = world
    rp = at.replay(data, res["weights"]["equal_weight"], 40, 25.0)
    first = rp["cost"].iloc[0]
    assert first == pytest.approx(25e-4 * 1.0), "buying the whole book from cash: traded = 1"
    assert (rp["cost"].iloc[1:40] == 0).all() and rp["cost"].iloc[40] > 0, "no cost between rebalances; the next one trades only the drift"
    assert rp["cost"].iloc[40] < first


# ------------------------------------------------------------------------------------------------------------------------ factor attribution
@pytest.mark.parametrize("kind", ["market", "statistical"])
def test_factor_contributions_plus_specific_less_costs_equal_the_net_return_every_day_and_cumulatively(world, kind):
    data, res = world
    cfg = res["config"]
    for key in res["weights"]:
        a = at.factor_attribution(data, res["weights"][key], _net(res, key), cfg["lookback"], cfg["step"], cfg["cost_bps"], kind)
        rec = a["reconciliation"]
        assert rec["status"] == "reconciled" and rec["max_daily_residual"] < 1e-12 and abs(rec["cumulative_gap"]) < 1e-12, (key, rec)
        lines = pd.DataFrame(a["daily"])
        assert (lines.drop(columns="residual").sum(axis=1) - _net(res, key).reindex(pd.to_datetime(a["days"])).to_numpy()).abs().max() < 1e-12
        assert a["totals"]["costs"] < 0 and a["totals"]["financing"] == 0.0
        final = sum(a["totals"].values())
        assert final == pytest.approx(res["equity"][key].iloc[-1] - 1.0, abs=1e-12), "the cumulative contributions add up to the cumulative return"
        cum = pd.DataFrame(a["cumulative"])
        eq = res["equity"][key].reindex(pd.to_datetime(a["days"])).to_numpy() - 1.0
        assert np.abs(cum.sum(axis=1).to_numpy() - eq).max() < 1e-12, "and they do at every date, not only at the end"


def test_a_return_that_is_exactly_one_factor_has_no_asset_specific_part():
    rng = np.random.default_rng(3)
    f = rng.standard_normal(400) * 0.01
    load = np.array([1.0, 0.7, 1.3, 0.4, 1.1, 0.9])
    data = pd.DataFrame(np.outer(f, load), index=pd.bdate_range("2020-01-01", periods=400), columns=[f"S{i}" for i in range(6)])
    res = lab.compare(data, ["equal_weight", "min_variance"], lookback=120, step=40, cost_bps=0.0)
    a = at.factor_attribution(data, res["weights"]["min_variance"], _net(res, "min_variance"), 120, 40, 0.0, "statistical", n_factors=1)
    assert abs(a["totals"]["specific"]) < 1e-10 and max(abs(v) for v in pd.DataFrame(a["daily"])["specific"]) < 1e-12
    assert a["totals"]["PC1"] == pytest.approx(res["equity"]["min_variance"].iloc[-1] - 1.0, abs=1e-10), "one factor explains the whole return"


def test_the_exposures_of_a_rebalance_do_not_depend_on_anything_from_that_day_onward(world):
    data, res = world
    cfg = res["config"]
    key = "min_variance"
    first = at.factor_attribution(data, res["weights"][key], _net(res, key), cfg["lookback"], cfg["step"], cfg["cost_bps"], "market")["ex_ante"]
    p = data.index.get_indexer([res["weights"][key].index[1]])[0]
    tampered = data.copy()
    tampered.iloc[p:] = tampered.iloc[p:] * 3.0 + 0.01                                  # change every return from the second rebalance on
    again = at.factor_attribution(tampered, res["weights"][key], _net(res, key), cfg["lookback"], cfg["step"], cfg["cost_bps"], "market")["ex_ante"]
    assert first[0] == again[0] and first[1] == again[1], "the first two rebalances' ex-ante exposures and risk shares are unchanged"
    assert first[2] != again[2], "while a later one, whose window now contains changed days, does change"


def test_a_difference_from_the_authoritative_return_is_reported_as_residual_and_not_spread_over_the_factors(world):
    data, res = world
    cfg = res["config"]
    key = "equal_weight"
    net = _net(res, key)
    ok = at.factor_attribution(data, res["weights"][key], net, cfg["lookback"], cfg["step"], cfg["cost_bps"])
    bad_net = net.copy()
    bad_net.iloc[30] += 1e-4
    bad = at.factor_attribution(data, res["weights"][key], bad_net, cfg["lookback"], cfg["step"], cfg["cost_bps"])
    assert bad["reconciliation"]["status"] == "does_not_reconcile" and bad["reconciliation"]["max_daily_residual"] == pytest.approx(1e-4, abs=1e-12)
    assert pd.DataFrame(bad["daily"])["residual"].iloc[30] == pytest.approx(1e-4, abs=1e-12)
    for line in ("market", "specific", "costs"):
        assert bad["daily"][line] == ok["daily"][line], f"{line} is untouched: the gap is not hidden in it"


def test_ex_ante_exposures_and_risk_shares_are_reported_apart_from_the_realised_contributions(world):
    data, res = world
    a = at.factor_attribution(data, res["weights"]["max_sharpe"], _net(res, "max_sharpe"), 120, 40, 25.0, "statistical", 2)
    assert len(a["ex_ante"]) == len(res["weights"]["max_sharpe"]) and set(a["ex_ante"][0]["exposures"]) == {"PC1", "PC2"}
    assert all(0.0 <= r["factor_risk_share"] <= 1.0 and r["ex_ante_volatility"] > 0 for r in a["ex_ante"])


# ------------------------------------------------------------------------------------------------------------------------ Brinson-Fachler
def _brinson_inputs(wp, wb, r):
    idx = pd.bdate_range("2021-01-04", periods=len(wp))
    cols = ["A", "B"]
    wp, wb, r = (pd.DataFrame(x, index=idx, columns=cols, dtype=float) for x in (wp, wb, r))
    zero = pd.Series(0.0, index=idx)
    net_p, net_b = (wp * r).sum(axis=1), (wb * r).sum(axis=1)
    return r, wp, wb, zero, net_p, net_b


def test_brinson_fachler_by_hand_one_asset_per_group():
    # Wp = (0.6, 0.4), Wb = (0.5, 0.5), R = (10%, 2%): Rp = 6.8%, Rb = 6%
    r, wp, wb, z, np_, nb = _brinson_inputs([[0.6, 0.4]], [[0.5, 0.5]], [[0.10, 0.02]])
    out = at.brinson(r, wp, wb, {"A": "A", "B": "B"}, z, z, np_, nb)
    assert out["by_group"]["A"]["allocation"] == pytest.approx(0.1 * (0.10 - 0.06)) and out["by_group"]["B"]["allocation"] == pytest.approx(-0.1 * (0.02 - 0.06))
    assert out["totals"]["selection"] == pytest.approx(0.0) and out["totals"]["interaction"] == pytest.approx(0.0)
    assert out["totals"]["allocation"] == pytest.approx(0.068 - 0.06) and out["cumulative_net_active_return"] == pytest.approx(0.008)
    assert out["reconciliation"]["status"] == "reconciled"


def test_brinson_selection_and_interaction_by_hand_two_assets_in_one_group():
    # group G = {A, B}, group H = {C}; Wp: A .3 B .3 C .4; Wb: A .2 B .2 C .6; R: A 10% B 0 C 5%
    idx = pd.bdate_range("2021-01-04", periods=1)
    data = pd.DataFrame([[0.10, 0.0, 0.05]], index=idx, columns=list("ABC"))
    wp = pd.DataFrame([[0.3, 0.3, 0.4]], index=idx, columns=list("ABC"))
    wb = pd.DataFrame([[0.2, 0.2, 0.6]], index=idx, columns=list("ABC"))
    z = pd.Series(0.0, index=idx)
    npt, nb = (wp * data).sum(axis=1), (wb * data).sum(axis=1)
    out = at.brinson(data, wp, wb, {"A": "G", "B": "G", "C": "H"}, z, z, npt, nb)
    rb_total, rg_b, rg_p = 0.2 * 0.10 + 0.6 * 0.05, 0.05, 0.05                          # Rb = .05; group G: Rb_G = .05 and Rp_G = .05
    assert out["by_group"]["G"]["allocation"] == pytest.approx((0.6 - 0.4) * (rg_b - rb_total)) and out["by_group"]["G"]["selection"] == pytest.approx(0.4 * (rg_p - rg_b))
    assert sum(out["totals"][k] for k in ("allocation", "selection", "interaction")) == pytest.approx(float(npt.iloc[0] - nb.iloc[0]), abs=1e-15)


def test_brinson_effects_costs_and_residual_add_up_to_the_cumulative_net_active_return_over_many_days(world):
    data, res = world
    cfg = res["config"]
    sectors = {c: ("X" if i < 3 else "Y") for i, c in enumerate(data.columns)}
    out = at.attribute_lab_result(res, data, sectors=sectors)
    assert out["brinson_available"]
    for key in ("min_variance", "max_sharpe"):
        b = out["methods"][key]["brinson"]
        rec = b["reconciliation"]
        assert rec["status"] == "reconciled" and rec["daily_identity_error"] < 1e-12 and abs(rec["cumulative_gap"]) < 1e-12, rec
        assert sum(b["totals"].values()) == pytest.approx(b["cumulative_net_active_return"], abs=1e-12)
        want = (res["equity"][key].iloc[-1] - res["equity"]["equal_weight"].iloc[-1])
        assert b["cumulative_net_active_return"] == pytest.approx(want, abs=1e-12), "the active return is the Lab's own: the two equity curves' difference"
        assert b["totals"]["costs"] < 0 or key == "equal_weight"
        assert abs(b["path"]["allocation"][-1] + b["path"]["selection"][-1] + b["path"]["interaction"][-1] + b["path"]["costs"][-1] + b["path"]["residual"][-1] - want) < 1e-12


def test_brinson_names_what_it_could_not_classify_and_a_benchmark_that_does_not_sum_to_one_is_refused(world):
    data, res = world
    out = at.attribute_lab_result(res, data, sectors={"S0": "X", "S1": "X", "S2": "Y"})              # S3 to S5 have no group
    b = out["methods"]["min_variance"]["brinson"]
    assert any("Unclassified" in w and "S3" in w for w in b["warnings"]) and "Unclassified" in b["groups"] and b["reconciliation"]["status"] == "reconciled"
    rp = at.replay(data, res["weights"]["min_variance"], 40, 25.0)
    bench = at.replay(data, res["weights"]["equal_weight"], 40, 25.0)
    short = bench["earning"].copy()
    short.iloc[:, 0] = 0.0
    z = rp["cost"]
    with pytest.raises(at.AttributionError, match="benchmark's weights do not sum to one"):
        at.brinson(data, rp["earning"], short, {c: ("X" if i < 3 else "Y") for i, c in enumerate(data.columns)}, z, bench["cost"], rp["net"], bench["net"])


def test_cash_is_a_group_of_its_own_and_a_changing_classification_is_applied_by_date(world):
    data, res = world
    rp = at.replay(data, res["weights"]["min_variance"], 40, 25.0)
    bench = at.replay(data, res["weights"]["equal_weight"], 40, 25.0)
    partial = rp["earning"] * 0.9                                                       # ten percent in cash
    net_cash = (partial * data.loc[partial.index]).sum(axis=1) - rp["cost"]
    sectors = {c: ("X" if i < 3 else "Y") for i, c in enumerate(data.columns)}
    b = at.brinson(data, partial, bench["earning"], sectors, rp["cost"], bench["cost"], net_cash, bench["net"])
    assert "Cash" in b["groups"] and any("holds cash" in w for w in b["warnings"]) and b["reconciliation"]["status"] == "reconciled"
    cut = data.index[200]
    moved = {c: ("Y" if i < 3 else "X") for i, c in enumerate(data.columns)}
    c = at.brinson(data, rp["earning"], bench["earning"].loc[rp["earning"].index], [(data.index[0], sectors), (cut, moved)], rp["cost"], bench["cost"], rp["net"], bench["net"])
    assert any("changes 1 time" in w for w in c["warnings"]) and c["reconciliation"]["status"] == "reconciled"
    with pytest.raises(at.AttributionError, match="no classification is in force"):
        at.brinson(data, rp["earning"], bench["earning"].loc[rp["earning"].index], [(data.index[300], sectors)], rp["cost"], bench["cost"], rp["net"], bench["net"])


def test_brinson_is_off_without_a_usable_classification_and_says_why(world):
    data, res = world
    for sectors, word in ((None, "no classification"), ({"S0": "X", "S1": "X", "S2": "X", "S3": "X", "S4": "X", "S5": "X"}, "at least two groups"), ({"ZZZ": "X"}, "none of the chosen tickers")):
        out = at.attribute_lab_result(res, data, sectors=sectors)
        assert out["brinson_available"] is False and word in out["brinson_reason"] and all("brinson" not in v for v in out["methods"].values())
    out = at.attribute_lab_result(res, data, sectors={"S0": "X", "S5": "Y"}, benchmark="hrp")
    assert out["brinson_available"] is False and "was not run" in out["brinson_reason"]


# ------------------------------------------------------------------------------------------------------------------------ currency
def test_a_base_currency_return_is_local_plus_currency_plus_cross_term_and_they_add_up_exactly():
    idx = pd.bdate_range("2022-01-03", periods=3)
    local = pd.DataFrame({"US": [0.01, -0.02, 0.03], "EU": [0.02, 0.01, -0.01]}, index=idx)
    fx = pd.DataFrame({"US": [0.0, 0.0, 0.0], "EU": [0.005, -0.01, 0.02]}, index=idx)
    w = pd.DataFrame({"US": [0.5, 0.4, 0.6], "EU": [0.5, 0.6, 0.4]}, index=idx)
    out = at.currency_decomposition(w, local, fx)
    base = ((1 + local) * (1 + fx) - 1)
    assert np.allclose(out["base"], (w * base).sum(axis=1), atol=1e-15)
    assert out["currency"].iloc[0] == pytest.approx(0.5 * 0.005) and out["cross"].iloc[0] == pytest.approx(0.5 * 0.02 * 0.005) and out["local"].iloc[0] == pytest.approx(0.5 * 0.01 + 0.5 * 0.02)
    with pytest.raises(at.AttributionError):
        at.currency_decomposition(w, local, fx.iloc[:2])
