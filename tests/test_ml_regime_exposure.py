"""Release 7, P2-11: causal regime exposure.

The exposure of a state is worked by hand from numbers whose volatilities are known. The overlay is then run on a market that alternates between calm and turbulent blocks: it must lower the risk of
the turbulent stretches, must not change anything on a date because of what happens after it, must not care how an engine happens to number its states after a refit, and must say when it had no evidence.
"""

import numpy as np
import pandas as pd
import pytest

from src.ml import config as mc, experiments as xp, monitoring as mon, regime_exposure as rx, registry as rg
from tests.ml_helpers import make_bundle

NAMES = [f"A{i}" for i in range(6)]


def market(T=1500, block=126, seed=0, calm=0.005, wild=0.02):
    """Six assets whose daily volatility alternates between ``calm`` and ``wild`` in blocks of ``block`` days (calm first), and a constant dollar-neutral book."""
    rng = np.random.default_rng(seed)
    ix = pd.bdate_range("2015-01-05", periods=T)
    scale = np.where((np.arange(T) // block) % 2 == 0, calm, wild)
    rets = pd.DataFrame(rng.normal(0.0, 1.0, (T, 6)) * scale[:, None], index=ix, columns=NAMES)
    book = pd.DataFrame(np.tile([1.0 / 6] * 3 + [-1.0 / 6] * 3, (T, 1)), index=ix, columns=NAMES)
    return rets, book, pd.Series(scale, index=ix)


def cfg(**kw):
    return {"regime_exposure": "vol_quantile", "regime_states": 2, **kw}


# ------------------------------------------------------------------------------------------------------------------ the pieces, by hand
def test_the_state_variables_are_the_trailing_ones_computed_by_hand():
    rng = np.random.default_rng(1)
    ix = pd.bdate_range("2020-01-01", periods=80)
    r = pd.DataFrame(rng.normal(0, 0.01, (80, 4)), index=ix, columns=list("ABCD"))
    f = rx.market_features(r)
    mkt = r.mean(axis=1).to_numpy()
    for t in (30, 55, 79):
        window = mkt[t - 20: t + 1]                                                                  # the 21 sessions ending at t
        assert f["log_vol"].iloc[t] == pytest.approx(np.log(window.std(ddof=1) * np.sqrt(252.0)))
        disp = r.to_numpy()[t - 4: t + 1].std(axis=1, ddof=0).mean()                                 # dispersion of the cross-section, averaged over the last five sessions
        assert f["log_disp"].iloc[t] == pytest.approx(np.log(disp))
    assert f["mkt_ret"].iloc[7] == pytest.approx(mkt[7])
    cut = f.iloc[:50].copy()
    r2 = r.copy()
    r2.iloc[50:] = 0.5                                                                               # what happens after the 50th session cannot move the first 50 rows
    pd.testing.assert_frame_equal(rx.market_features(r2).iloc[:50], cut)


def test_the_volatility_bands_are_the_training_quantiles_and_the_state_of_a_date_is_known_at_its_close():
    rng = np.random.default_rng(2)
    lv = rng.normal(0.0, 1.0, 300)
    Z = np.column_stack([np.zeros(300), lv, np.zeros(300)])
    eng = rx.VolQuantileEngine(3).fit(Z)
    assert eng.edges_ == pytest.approx(np.quantile(lv, [1 / 3, 2 / 3]))
    P = eng.filter(Z)
    assert (P.sum(axis=1) == 1).all() and (P.argmax(axis=1) == np.searchsorted(eng.edges_, lv, side="right")).all()
    assert np.bincount(P.argmax(axis=1), minlength=3).tolist() == pytest.approx([100, 100, 100], abs=2), "terciles on the rows they were fitted on"
    Z2 = Z.copy()
    Z2[200:, 1] = 99.0                                                                               # a changed future leaves the earlier states alone
    assert (eng.filter(Z2)[:200] == P[:200]).all()
    Z3 = Z.copy()
    Z3[150, 1] = np.nan                                                                              # a date with no state keeps the previous belief
    assert (eng.filter(Z3)[150] == P[149]).all()
    with pytest.raises(ValueError, match="fewer than 60"):
        rx.VolQuantileEngine(3).fit(Z[:30])


def test_a_states_exposure_is_the_ratio_of_the_overall_to_its_own_volatility_and_neutral_without_evidence():
    rng = np.random.default_rng(3)
    calm, wild = rng.normal(0, 0.01, 300), rng.normal(0, 0.03, 300)
    nxt = np.concatenate([calm, wild, rng.normal(0, 0.05, 10)])
    P = np.zeros((610, 3))
    P[:300, 0], P[300:600, 1], P[600:, 2] = 1.0, 1.0, 1.0
    e, table = rx.exposures(P, nxt, floor=0.25, cap=1.0, min_days=60)
    vol_all = nxt.std(ddof=1)
    assert e[0] == 1.0 and table[0]["status"] == "estimated", "the calm state is at least as quiet as average: the cap, not a leverage"
    assert e[1] == pytest.approx(vol_all / wild.std(ddof=1)) and e[1] < 1.0
    assert e[2] == 1.0 and "insufficient" in table[2]["status"] and table[2]["n"] == 10, "ten rows are not evidence: the neutral exposure"
    e2, _ = rx.exposures(P, nxt, floor=0.9, cap=1.0, min_days=60)
    assert e2[1] == 0.9, "the floor binds"
    e3, t3 = rx.exposures(P, nxt, floor=0.25, cap=1.0, min_days=1000)
    assert (e3 == 1.0).all() and all("insufficient" in r["status"] for r in t3)
    e4, _ = rx.exposures(np.ones((610, 1)), nxt, floor=0.25, cap=1.0, min_days=60)
    assert e4[0] == 1.0, "one state is the whole sample: its volatility is the overall one"


def test_states_are_numbered_by_volatility_whatever_labels_an_engine_gives_them():
    rng = np.random.default_rng(4)
    lv = np.concatenate([rng.normal(-1, 0.1, 100), rng.normal(0, 0.1, 100), rng.normal(1, 0.1, 100)])
    P = np.zeros((300, 3))
    P[:100, 0], P[100:200, 1], P[200:, 2] = 1.0, 1.0, 1.0
    nxt = lv * 0.01 + rng.normal(0, 0.001, 300)
    ref = rx.exposures(P, nxt, 0.25, 1.0, 20)[0]
    for perm in ((2, 0, 1), (1, 2, 0), (2, 1, 0)):
        shuffled = P[:, list(perm)]
        order, mean = rx.order_by_volatility(shuffled, lv)
        assert np.allclose(shuffled[:, order], P) and np.all(np.diff(mean[order]) > 0)
        assert np.allclose(rx.exposures(shuffled[:, order], nxt, 0.25, 1.0, 20)[0], ref), "after the canonical numbering the mapping does not depend on the engine's labels"
    never = np.column_stack([P[:, 0], P[:, 1], np.zeros(300), P[:, 2]])
    assert rx.order_by_volatility(never, lv)[0][-1] == 2, "a state that never occurs is last"


# ------------------------------------------------------------------------------------------------------------------ the overlay through time
def test_the_exposure_is_lower_in_the_turbulent_blocks_and_the_risk_taken_is_more_even():
    rets, book, scale = market()
    out, log = rx.apply(book, rets, None, cfg())
    mult = (out.iloc[:, 0] / book.iloc[:, 0])
    start = log["warmup_days"]
    assert 250 <= start <= 300 and (mult.iloc[:start] == 1.0).all(), "before there is evidence the exposure is neutral"
    live = scale.index[start:]
    turbulent, calm = live[scale.loc[live] > 0.01], live[scale.loc[live] < 0.01]
    assert mult.loc[turbulent].mean() < 0.8 < 0.95 < mult.loc[calm].mean() <= 1.0 and mult.max() <= 1.0 + 1e-12 and mult.min() >= 0.25 - 1e-12, (
        "about 0.74 in the turbulent blocks (the ratio of the overall to the turbulent volatility); a little under 1 in the calm ones because the 21-day volatility needs about a month to notice a block has ended")
    raw = (book * rets.shift(-1)).sum(axis=1)
    got = (out * rets.shift(-1)).sum(axis=1)
    v = lambda x, ix: float(x.loc[ix].std(ddof=1))                                                  # noqa: E731
    assert v(got, turbulent) / v(got, calm) < v(raw, turbulent) / v(raw, calm), "the risk taken in the two kinds of days is closer after the overlay"
    assert v(got, live) < v(raw, live), "and the overlay only ever lowered the exposure, so the total risk is lower"
    assert log["engine"] == "vol_quantile" and log["n_refits"] >= 15 and log["failed_refits"] == 0 and log["smoothing"].startswith("none")
    first = log["refits"][0]
    assert first["status"] == "fitted" and first["ordering_calm_to_turbulent"] is True and [s["state"] for s in first["states"]] == [0, 1]
    assert first["states"][0]["mean_log_vol"] < first["states"][1]["mean_log_vol"] and first["states"][0]["exposure"] >= first["states"][1]["exposure"]
    assert 0.0 < log["mean_multiplier"] < 1.0 and log["min_multiplier"] >= 0.25 - 1e-12


def test_nothing_on_a_date_depends_on_what_happens_after_it():
    rets, book, _ = market()
    base, _ = rx.apply(book, rets, None, cfg())
    for d in (700, 1010, 1300):
        rng = np.random.default_rng(d)
        future = rets.copy()
        future.iloc[d + 1:] = rng.normal(0.0, 0.08, future.iloc[d + 1:].shape)                      # a different world after the date
        again, _ = rx.apply(book, future, None, cfg())
        np.testing.assert_array_equal(again.iloc[: d + 1].to_numpy(), base.iloc[: d + 1].to_numpy(), err_msg=f"the exposure up to row {d} moved when only later returns changed")
    changed, _ = rx.apply(book, rets.assign(A0=rets["A0"] * 3.0), None, cfg())
    assert not np.array_equal(changed.to_numpy(), base.to_numpy()), "the test can tell a different past from a same one"


def test_a_refit_that_reshuffles_the_engines_own_labels_changes_nothing(monkeypatch):
    rets, book, _ = market(T=1100)
    plain, plain_log = rx.apply(book, rets, None, cfg(regime_states=3))
    real, calls = rx.make_engine, {"n": 0}

    class Shuffled:
        def __init__(self, inner, perm):
            self.inner, self.perm = inner, perm

        def fit(self, Z):
            self.inner.fit(Z)
            return self

        def filter(self, Z):
            return self.inner.filter(Z)[:, self.perm]

        def facts(self):
            return self.inner.facts()

    def fake(name, k, seed=0):
        calls["n"] += 1
        return Shuffled(real(name, k, seed), np.random.default_rng(calls["n"]).permutation(k))

    monkeypatch.setattr(rx, "make_engine", fake)
    mixed, mixed_log = rx.apply(book, rets, None, cfg(regime_states=3))
    np.testing.assert_allclose(mixed.to_numpy(), plain.to_numpy(), atol=1e-12)
    assert any(r["labels_reordered"] for r in mixed_log["refits"]) and not any(r["labels_reordered"] for r in plain_log["refits"]), "the guard did reorder, and said so"


def test_without_evidence_the_overlay_is_neutral_and_says_why():
    rets, book, _ = market(T=1100)
    out, log = rx.apply(book, rets, None, cfg(regime_min_days=10_000))
    pd.testing.assert_frame_equal(out, book)
    assert all("insufficient" in s["status"] for r in log["refits"] for s in r["states"]) and log["mean_multiplier"] == pytest.approx(1.0)
    short, slog = rx.apply(book.iloc[:200], rets.iloc[:200], None, cfg())
    pd.testing.assert_frame_equal(short, book.iloc[:200])
    assert slog["warmup_days"] == 200 and slog["refits"] == [] and slog["mean_multiplier"] is None, "too little history for a first fit: nothing is changed and nothing is claimed"
    empty = book.copy()
    empty.iloc[:600] = 0.0                                                                           # days with no book carry no evidence: the clock starts when the book does
    _, elog = rx.apply(empty, rets, None, cfg())
    assert elog["warmup_days"] >= 600 + 200


def test_a_model_that_cannot_be_fitted_leaves_the_exposure_neutral_for_that_stretch(monkeypatch):
    rets, book, _ = market(T=900)

    class Broken:
        def fit(self, Z):
            raise ValueError("no convergence")

    monkeypatch.setattr(rx, "make_engine", lambda name, k, seed=0: Broken())
    out, log = rx.apply(book, rets, None, cfg())
    pd.testing.assert_frame_equal(out, book)
    assert log["failed_refits"] == len(log["refits"]) > 0 and log["refits"][0]["status"] == "failed" and "no convergence" in log["refits"][0]["reason"] and log["n_refits"] == 0


def test_a_hidden_markov_engine_is_filtered_not_smoothed():
    pytest.importorskip("hmmlearn")
    rets, book, _ = market(T=1000, block=100)
    base, log = rx.apply(book, rets, None, cfg(regime_exposure="hmm", regime_refit_every=126))
    assert log["n_refits"] >= 4 and log["failed_refits"] == 0 and (base.abs().to_numpy() <= book.abs().to_numpy() + 1e-12).all()
    d = 800
    future = rets.copy()
    future.iloc[d + 1:] = np.random.default_rng(5).normal(0.0, 0.08, future.iloc[d + 1:].shape)
    again, _ = rx.apply(book, future, None, cfg(regime_exposure="hmm", regime_refit_every=126))
    np.testing.assert_allclose(again.iloc[: d + 1].to_numpy(), base.iloc[: d + 1].to_numpy(), atol=1e-12)


# ------------------------------------------------------------------------------------------------------------------ configuration and the experiment
def test_only_causal_engines_are_available_and_the_overlay_cannot_lever_a_book():
    c = mc.experiment_config({"signal": {"regime_exposure": "vol_quantile"}}, defaults=mc.DEFAULTS)
    assert c["signal"]["regime_exposure"] == "vol_quantile" and not (set(c["signal"]) & {"cost_bps", "window_days"}), "it needs no forecast, so no derived settings"
    assert "regime_exposure" not in mc.experiment_config({}, defaults=mc.DEFAULTS)["signal"], "absent from the defaults: no existing configuration hash changes"
    for bad, msg in (({"regime_exposure": "smoothed_hmm"}, "never smoothed"), ({"regime_exposure": "crisis_calendar"}, "hindsight"), ({"regime_exposure": "hmm", "regime_cap": 1.5}, "never levers"),
                     ({"regime_exposure": "hmm", "regime_floor": 0.9, "regime_cap": 0.5}, "regime_floor"), ({"regime_exposure": "hmm", "regime_states": 1}, "regime_states"),
                     ({"regime_exposure": "hmm", "regime_refit_every": 2}, "regime_refit_every"), ({"regime_exposure": "hmm", "regime_min_days": 5}, "regime_min_days")):
        with pytest.raises(mc.MLConfigError, match=msg):
            mc.experiment_config({"signal": bad}, defaults=mc.DEFAULTS)


def test_in_an_experiment_the_exposure_is_reported_and_a_forward_test_refuses_it(tmp_path):
    reg = rg.Registry(tmp_path / "ml")
    bundle = make_bundle(10, 1500, seed=83, alpha=1.0)
    over = {"dataset": {"feature_groups": ["price", "momentum"], "target": "fwd_ret_5"}, "split": {"min_train_days": 400, "refit_every": 63}, "model": {"kind": "ridge", "params": {"alpha": 10.0}},
            "signal": {"method": "top_n_long_short", "n": 3, "holding_period": 1, "regime_exposure": "vol_quantile", "regime_states": 3}, "backtest": {"cost_bps": 5.0, "slippage_bps": 2.0, "rebalance": "weekly"}}
    exp = xp.run_experiment(over, bundle=bundle, registry=reg, explain=False, robustness=False)
    rep = reg.report(exp.id)["overlays"]["regime_exposure"]
    assert rep["engine"] == "vol_quantile" and rep["n_refits"] >= 5 and 0.25 <= rep["min_multiplier"] <= rep["mean_multiplier"] <= 1.0
    plain = xp.rebacktest(reg, exp.id, {"signal": {"regime_exposure": None}, "name": "static"})
    assert "overlays" not in reg.report(plain.id) and plain.cfg["signal"].get("regime_exposure") is None
    assert exp.metrics["oos"]["ann_vol"] <= plain.metrics["oos"]["ann_vol"] + 1e-9, "the exposure only lowers the book, so it cannot add risk"
    with pytest.raises(ValueError, match="forward testing does not support"):
        mon.ForwardTracker(reg, exp.id).step(bundle)
