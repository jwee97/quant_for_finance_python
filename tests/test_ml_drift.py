"""Release 8, P3-01: input and predictive-performance drift on stable references.

The PSI and the multiplicity adjustments are worked by hand; the KS p-value is checked against SciPy where the observations are independent and shown to be more cautious where they are not; the reference
window is shown not to move when data arrive; an unsettled outcome is shown not to count; and the lineage of an alert is traced through the append-only record.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.ml import diagnostics as dx, drift as dr, experiments as xp, monitoring as mon, registry as rg
from tests.ml_helpers import make_bundle

SPEC = dr.DriftSpec(reference_days=300, recent_days=60, min_reference=100, min_recent=40)


def panel(T=700, n=6, seed=0, shift=0.0, ar=0.0, nan_share=0.0, shift_from=600):
    """A feature panel (date, asset) with four columns: stable noise, a column whose mean jumps at ``shift_from``, a sparse column and a persistent AR(1) column."""
    rng = np.random.default_rng(seed)
    ix = pd.bdate_range("2018-01-01", periods=T)
    mi = pd.MultiIndex.from_product([ix, [f"A{i}" for i in range(n)]], names=["date", "asset"])
    stable = rng.normal(size=(T, n))
    moved = rng.normal(size=(T, n)) + shift * (np.arange(T) >= shift_from)[:, None]
    sparse = rng.normal(size=(T, n))
    sparse[rng.random((T, n)) < nan_share] = np.nan
    persistent = np.zeros((T, n))
    for t in range(1, T):
        persistent[t] = ar * persistent[t - 1] + np.sqrt(1 - ar ** 2) * rng.normal(size=n)
    return pd.DataFrame({"stable": stable.reshape(-1), "moved": moved.reshape(-1), "sparse": sparse.reshape(-1), "persistent": persistent.reshape(-1)}, index=mi)


# ------------------------------------------------------------------------------------------------------------------ the pieces by hand
def test_psi_by_hand_and_the_empty_bin_floor():
    ref = np.arange(100.0)
    same = dr.psi(ref, ref)
    assert same == pytest.approx(0.0, abs=1e-12)
    recent = np.full(40, 95.0)                                                                       # every recent value falls in the top decile
    edges = np.quantile(ref, np.linspace(0, 1, 11))
    pe = np.full(10, 0.1)
    pa = np.full(10, 0.5 / 40)
    pa[-1] = 1.0
    expected = float(np.sum((pa - pe) * np.log(pa / pe)))
    assert dr.psi(ref, recent) == pytest.approx(expected, rel=1e-9) and edges[0] == 0.0
    assert np.isnan(dr.psi(ref[:30], recent)) and np.isnan(dr.psi(ref, recent[:10]))
    assert dr.psi(np.full(100, 3.0), np.full(40, 3.0)) == 0.0 and dr.psi(np.full(100, 3.0), np.full(40, 4.0)) == float("inf"), "a constant reference: nothing moved only if the recent values are that constant"


def test_psi_is_the_existing_monitoring_function_with_one_rule_added_for_a_constant_reference():
    rng = np.random.default_rng(3)
    ref, rec = rng.normal(0, 1, 400), rng.normal(0.4, 1.2, 120)
    assert dr.psi(ref, rec) == mon.psi(ref, rec) and dr.psi(ref, rec, bins=5) == mon.psi(ref, rec, bins=5), "the same function on ordinary data: drift extends monitoring, it does not replace it"
    assert np.isnan(dr.psi(ref[:30], rec)) and np.isnan(mon.psi(ref[:30], rec)), "too few observations is NaN in both"
    flat = np.full(100, 3.0)
    assert mon.psi(flat, np.full(40, 4.0)) == 0.0, "the existing function reads a feature that left its only value as stable"
    assert dr.psi(flat, np.full(40, 4.0)) == float("inf") and dr.psi(flat, np.full(40, 3.0)) == 0.0


def test_the_multiplicity_adjustments_by_hand():
    p = [0.01, 0.04, 0.03]
    assert dr.adjust(p, "bh") == pytest.approx([0.03, 0.04, 0.04]) and dr.adjust(p, "holm") == pytest.approx([0.03, 0.06, 0.06]) and dr.adjust(p, "none") == p
    assert dr.adjust([0.2, None, 0.01], "holm") == [pytest.approx(0.2), None, pytest.approx(0.02)] and dr.adjust([None], "bh") == [None]
    assert max(dr.adjust([0.6, 0.7, 0.9], "holm")) == 1.0


def test_the_ks_p_value_matches_scipy_for_independent_data_and_is_more_cautious_for_dependent_data():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=500), rng.normal(0.15, 1.0, size=500)
    ours, ref = dr.ks_effective(a, b), stats.ks_2samp(a, b, method="asymp")
    assert ours["statistic"] == pytest.approx(ref.statistic) and ours["p_value"] == pytest.approx(ref.pvalue, abs=0.05)
    assert ours["n_eff"]["reference"] == pytest.approx(500, rel=0.15)
    ar = np.zeros(500)
    br = np.zeros(500)
    for t in range(1, 500):
        ar[t] = 0.9 * ar[t - 1] + rng.normal()
        br[t] = 0.9 * br[t - 1] + rng.normal()
    dep, nominal = dr.ks_effective(ar, br), stats.ks_2samp(ar, br, method="asymp").pvalue
    assert dep["n_eff"]["reference"] < 120 and dep["p_value"] > nominal, "serial dependence makes the evidence worth fewer observations, so the p-value is larger than the nominal one"
    assert dr.effective_n([1.0, 1.0, 1.0, 1.0]) == 4.0 and dr.ks_effective([1.0, 2.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])["p_value"] is None


def test_the_reference_window_is_fixed_at_the_fit_cutoff_whatever_arrives_later():
    ix = pd.bdate_range("2018-01-01", periods=700)
    a = dr.reference_dates(ix, ix[500], 300)
    b = dr.reference_dates(ix[:650], ix[500], 300)
    assert list(a) == list(b) and a[-1] == ix[500] and len(a) == 300, "more data after the cutoff changes nothing about what the model was fitted on"
    assert len(dr.reference_dates(ix, ix[100], 300)) == 101, "a short history gives the dates there are, and the sample-count check decides"
    with pytest.raises(ValueError, match="at least 5"):
        dr.DriftSpec(reference_days=3)
    with pytest.raises(ValueError, match="correction"):
        dr.DriftSpec(correction="bonferroni")


# ------------------------------------------------------------------------------------------------------------------ input drift
def test_a_shifted_feature_alerts_a_stable_one_does_not_and_a_sparse_one_is_insufficient():
    X = panel(shift=2.0, nan_share=0.5)
    ix = X.index.get_level_values(0).unique()
    out = dr.feature_drift(X, SPEC, fit_cutoff=ix[550], as_of=ix[-1])
    by = {r["feature"]: r for r in out["features"]}
    assert by["moved"]["state"] == "alert" and by["moved"]["psi"] > 0.25 and by["moved"]["ks_p_adjusted"] < 0.01
    assert by["stable"]["state"] in ("ok", "watch") and by["stable"]["psi"] < 0.1
    assert by["sparse"]["state"] == "insufficient" and "coverage" in by["sparse"]["reason"] and by["sparse"]["psi"] is None, "a feature that is half missing is not 'ok': it is not assessed"
    assert out["counts"]["alert"] >= 1 and out["counts"]["insufficient"] == 1 and out["tests"]["features"] == 4 and out["tests"]["multiplicity"] == "bh"
    assert out["reference"]["fixed_at"] == str(ix[550].date()) and out["reference"]["dates"] == 300 and out["recent"]["dates"] == 60 and out["thresholds"]["note"].startswith("the PSI thresholds are conventional")
    again = dr.feature_drift(X.loc[X.index.get_level_values(0) <= ix[640]], SPEC, fit_cutoff=ix[550], as_of=ix[640])
    assert again["reference"] == out["reference"], "the reference did not move when the recent window did"


def test_serial_dependence_does_not_raise_a_false_alarm_and_a_recent_window_inside_the_reference_is_flagged():
    X = panel(ar=0.97, seed=3)
    ix = X.index.get_level_values(0).unique()
    out = dr.feature_drift(X, SPEC, fit_cutoff=ix[550], as_of=ix[-1])
    pers = next(r for r in out["features"] if r["feature"] == "persistent")
    stable = next(r for r in out["features"] if r["feature"] == "stable")
    assert pers["n_eff"]["reference"] < pers["n_reference"] / 10, "an AR(1) series with a coefficient of 0.97 is worth about 1/65 of its observations"
    assert stable["n_eff"]["reference"] > 0.9 * stable["n_reference"], "independent noise keeps (almost) all of them"
    assert pers["state"] != "alert"
    W = X["persistent"].unstack("asset")
    assert dr.panel_effective_n(W.iloc[251:551]) == pytest.approx(pers["n_eff"]["reference"]), "the stacking order of the panel does not hide the persistence"
    assert "dependence between assets" in out["tests"]["p_value_basis"]
    early = dr.feature_drift(X, SPEC, fit_cutoff=ix[650], as_of=ix[660])
    assert any("both windows" in n for n in early["notes"]), "a recent window that reaches back into the reference is said to understate the drift"
    demanding = dr.DriftSpec(reference_days=300, recent_days=60, min_reference=1000, min_recent=40)
    thin = dr.feature_drift(X.loc[X.index.get_level_values(0) <= ix[120]], demanding, fit_cutoff=ix[100], as_of=ix[120])             # 101 dates x 6 assets = 606 cells before the cutoff
    assert thin["counts"]["insufficient"] == 4 and all("too few reference observations" in r["reason"] for r in thin["features"]), "a reference window too short for the declared minimum is not assessed"


# ------------------------------------------------------------------------------------------------------------------ predictive performance
def ic_series(T=400, ref_mean=0.05, recent_mean=-0.03, break_at=300, seed=0, noise=0.1):
    rng = np.random.default_rng(seed)
    ix = pd.bdate_range("2019-01-01", periods=T)
    ic = pd.Series(np.where(np.arange(T) < break_at, ref_mean, recent_mean) + rng.normal(0, noise, T), index=ix)
    return ic, pd.Series(ix + pd.tseries.offsets.BDay(6), index=ix)                         # a label ends six sessions after its decision


def test_only_settled_decisions_count_and_a_fallen_signal_alerts_with_an_interval_that_excludes_zero():
    ic, le = ic_series()
    fit = ic.index[299]
    s = dr.performance_drift(ic, le, SPEC, horizon=5, fit_cutoff=fit, as_of=ic.index[-1])
    assert s["state"] == "alert" and s["recent_ic"] < 0 < s["reference_ic"] and s["difference"]["interval"][1] < 0 and s["difference"]["settings"]["lags"] >= 4
    assert s["unsettled_dates_excluded"] == 6 and s["settled_dates"] == 394, "the last decisions' labels had not ended by the as-of date, so they are not in either window"
    mask = dr.settled_mask(le, ic.index[-1])
    assert int(mask.sum()) == 394 and not mask.iloc[-1] and mask.iloc[0]
    same, le2 = ic_series(recent_mean=0.05, seed=1)
    assert dr.performance_drift(same, le2, SPEC, horizon=5, fit_cutoff=same.index[299], as_of=same.index[-1])["state"] in ("ok", "watch")
    few = dr.performance_drift(ic, le, SPEC, horizon=5, fit_cutoff=fit, as_of=ic.index[320])
    assert few["state"] == "insufficient" and "settled dates" in few["reason"], "twenty settled recent dates are not enough to say anything"
    leak = ic.copy()
    leak.iloc[-6:] = 10.0                                                                            # outcomes not yet settled cannot move the result
    s2 = dr.performance_drift(leak, le, SPEC, horizon=5, fit_cutoff=fit, as_of=ic.index[-1])
    assert s2["recent_ic"] == pytest.approx(s["recent_ic"]), "unsettled outcomes are invisible"


# ------------------------------------------------------------------------------------------------------------------ lineage and a saved experiment
def test_an_alert_is_traced_through_the_looks_that_saw_it(tmp_path):
    reg = rg.Registry(tmp_path / "ml")
    cur = {"features": [{"feature": "a", "state": "alert"}, {"feature": "b", "state": "ok"}]}
    first = dr.lineage(reg, "exp-1", "drift_report", cur)
    assert first["previous_record"] is None and first["new"] == ["a"] and first["consecutive_looks"] == {"a": 1} and first["looks_so_far"] == 1
    seq = dx.store(reg, "drift_report", "exp-1", {**cur, "lineage": first}, actor="test")
    second = dr.lineage(reg, "exp-1", "drift_report", {"features": [{"feature": "a", "state": "alert"}, {"feature": "b", "state": "alert"}]})
    assert second["previous_record"] == seq and second["persisting"] == ["a"] and second["new"] == ["b"] and second["consecutive_looks"] == {"a": 2, "b": 1} and second["looks_so_far"] == 2
    dx.store(reg, "drift_report", "exp-1", {"features": [{"feature": "a", "state": "alert"}], "lineage": second}, actor="test")
    third = dr.lineage(reg, "exp-1", "drift_report", {"features": [{"feature": "b", "state": "ok"}]})
    assert third["cleared"] == ["a"] and third["alerting_now"] == [] and third["looks_so_far"] == 3


def test_the_report_of_a_saved_experiment_fixes_its_reference_at_the_model_fit_cutoff(tmp_path):
    reg = rg.Registry(tmp_path / "ml")
    bundle = make_bundle(8, 1150, seed=99, alpha=1.0)
    exp = xp.run_experiment({"dataset": {"feature_groups": ["price", "momentum"], "target": "fwd_ret_5"}, "split": {"min_train_days": 400, "refit_every": 63}, "model": {"kind": "ridge", "params": {"alpha": 10.0}},
                             "signal": {"method": "top_n_long_short", "n": 3, "holding_period": 1}}, bundle=bundle, registry=reg, explain=False, robustness=False)
    spec = dr.DriftSpec(reference_days=400, recent_days=63, min_reference=200, min_recent=40, performance_min_dates=20)
    out = dr.report(reg, exp.id, spec=spec)
    cfg, ds, train, _ = xp.load_run(reg, exp.id)
    last_fit = [f for f in train.folds if not f.get("skipped")][-1]["fit_cutoff"]
    assert out["eligibility"] == "applied" and out["inputs"]["reference"]["fixed_at"] == str(pd.Timestamp(last_fit).date()) and out["spec_hash"] == spec.hash()
    assert out["performance"]["state"] in dr.STATES and out["performance"]["scope"].startswith("settled decisions on the validation segment")
    assert out["performance"]["settled_dates"] + out["performance"]["unsettled_dates_excluded"] > 0 and out["lineage"]["previous_record"] is None
    seq = dx.store(reg, "drift_report", exp.id, out, actor="test")
    nxt = dr.report(reg, exp.id, spec=spec)
    assert nxt["lineage"]["previous_record"] == seq and nxt["lineage"]["looks_so_far"] == 2
    assert "nothing here retrains" in nxt["note"]
