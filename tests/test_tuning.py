"""The strategy tuner's core (``webapp.tuning``) with synthetic return streams: it finds a planted optimum, scores on the tuning window only, says in words whether the gain survived the hold-out,
stops when asked, counts every try and never repeats one. No pipeline runs here; ``test_tune_api`` runs the real one."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.framework import load_library
from src.framework.registry import MODELS
from src.ops.hpo import Categorical, Float, Int, Study
from src.webapp import tuning as T
from src.webapp.api import parameters

IDX = pd.bdate_range("2010-01-04", periods=2600)
SPLIT = IDX[int(round(len(IDX) * 0.7))]                     # the first day of the 30% hold-out
NOISE = np.random.default_rng(1).normal(0, 0.01, len(IDX))


def peak(look: float, centre: float = 60.0, width: float = 0.5, height: float = 0.0015) -> float:
    return height * float(np.exp(-(((np.log(look) - np.log(centre)) / width) ** 2)))


def run_with(edge, tuning_only: bool = False):
    """A strategy whose daily edge is ``edge(params)``: the same noise for every setting, so differences between tries are the edge. ``tuning_only`` plants the edge before the split only."""
    def run(params):
        e = np.full(len(IDX), edge(params))
        if tuning_only:
            e = np.where(IDX < SPLIT, e, 0.0)
        return T.Run(pd.Series(NOISE + e, index=IDX))
    return run


LOOKBACK = T.Setting("lookback", "int", 20, 5, 250, "linear", curated=True)


# ------------------------------------------------------------------------------------------------------------------ the settings of a strategy
def test_auto_range_is_a_third_to_three_times_the_default_and_never_past_one_for_a_small_float():
    assert T.auto_range("int", 21) == (7, 63, "linear")
    assert T.auto_range("int", 1) == (1, 3, "linear")
    assert T.auto_range("int", 0) is None and T.auto_range("int", -5) is None
    assert T.auto_range("float", 2.0) == (0.667, 6.0, "log")
    assert T.auto_range("float", 0.6) == (0.2, 1.0, "log")                       # three times 0.6 would be 1.8: a share cannot pass one
    assert T.auto_range("float", 1e-5) == (1e-6, 1e-4, "log")
    assert T.auto_range("float", 0.0) is None and T.auto_range("float", -1.0) is None
    assert T.auto_range("int", True) is None and T.auto_range("float", None) is None


def test_the_kalman_filter_has_curated_ranges_and_clip_is_left_unticked_because_only_its_product_with_scale_acts():
    entry = next(e for e in MODELS.entries() if e.name == "kalman_trend")
    settings = {s.name: s for s in T.settings_for("kalman_trend", parameters("model", "kalman_trend", entry.factory))}
    assert set(settings) == {"level_noise", "slope_noise", "vol_halflife", "clip", "scale"}
    assert all(s.curated for s in settings.values())
    assert settings["level_noise"].scale == "log" and (settings["level_noise"].low, settings["level_noise"].high) == (0.002, 0.5)
    assert settings["slope_noise"].low == 1e-7 and settings["slope_noise"].high == 1e-3
    assert [n for n, s in settings.items() if s.tune] == ["level_noise", "slope_noise", "vol_halflife", "scale"]
    assert "product" in settings["clip"].help and "Higher" in settings["level_noise"].help


def test_a_setting_about_how_a_model_is_fed_is_not_ticked_and_is_searched_only_above_its_default():
    params = [{"name": "alpha", "kind": "float", "default": 100.0, "choices": None, "required": False},
              {"name": "min_train", "kind": "int", "default": 1260, "choices": None, "required": False},
              {"name": "refit_every", "kind": "int", "default": 252, "choices": None, "required": False},
              {"name": "features", "kind": "choice", "default": "price", "choices": ["price", "price_macro"], "required": False},
              {"name": "note", "kind": "str", "default": "x", "choices": None, "required": False},
              {"name": "pairs", "kind": "json", "default": [], "choices": None, "required": False},
              {"name": "needed", "kind": "int", "default": None, "choices": None, "required": True}]
    got = {s.name: s for s in T.settings_for("no_such_strategy", params)}
    assert set(got) == {"alpha", "min_train", "refit_every", "features"}        # text, structured and required settings cannot be searched
    assert got["alpha"].tune and not got["min_train"].tune and not got["refit_every"].tune and not got["features"].tune
    assert got["min_train"].low == 1260 and got["min_train"].high > 1260       # a refit interval or a minimum history is a floor: look above it
    assert got["features"].kind == "choice" and got["features"].choices == ("price", "price_macro")


def test_at_most_six_settings_are_ticked_at_first_and_the_rest_wait_to_be_ticked():
    params = [{"name": f"p{i}", "kind": "int", "default": 10 + i, "choices": None, "required": False} for i in range(9)]
    settings = T.settings_for("no_such_strategy", params)
    assert len(settings) == 9 and sum(s.tune for s in settings) == T.MAX_TICKED


def test_every_strategy_in_the_library_gets_ranges_its_constructor_accepts():
    """Each range's low end, high end and default build the strategy (one setting changed at a time), so a beginner never ticks a setting that fails at once. A pair that must stay ordered has ranges that
    do not overlap, so no combination can break the order."""
    load_library()
    failures, tried = [], 0
    for entry in MODELS.entries():
        settings = T.settings_for(entry.name, parameters("model", entry.name, entry.factory))
        assert sum(s.tune for s in settings) <= T.MAX_TICKED
        for s in settings:
            for value in ([s.low, s.high, s.default] if s.kind in ("int", "float") else list(s.choices)):
                tried += 1
                try:
                    MODELS.create(entry.name, **{s.name: value})
                except Exception as exc:                                       # noqa: BLE001
                    failures.append((entry.name, s.name, value, str(exc)[:80]))
    assert tried > 500
    assert {(name, setting) for name, setting, *_ in failures} <= {("ichimoku_trend", "kijun")}, failures      # kijun alone, against the default senkou of 52: in a joint search senkou is 64 or more
    for name, a, b in [("ma_crossover", "fast", "slow"), ("trend_atr", "fast", "slow"), ("macd_trend", "fast", "slow"), ("kama_trend", "fast", "slow"), ("donchian", "exit", "entry"),
                       ("ibs_reversion", "entry", "exit"), ("stochastic_reversion", "entry", "exit"), ("ichimoku_trend", "tenkan", "kijun"), ("ichimoku_trend", "kijun", "senkou")]:
        ranges = T.CURATED[name]
        assert ranges[a][1] < ranges[b][0], (name, a, b)


# ------------------------------------------------------------------------------------------------------------------ windows, scores, metrics
def test_the_split_keeps_the_last_share_for_the_hold_out_and_refuses_one_too_short_to_say_anything():
    win = T.make_windows(IDX, 0.3)
    assert win.split == SPLIT and win.start == IDX[0] and win.end == IDX[-1]
    assert win.n_tune + win.n_hold == len(IDX) and win.tuning_end == IDX[win.n_tune - 1] and win.tuning_end < win.split
    with pytest.raises(T.TuningError, match="trading days to work with.*Use a longer history"):
        T.make_windows(IDX[:300], 0.3)
    with pytest.raises(T.TuningError):
        T.make_windows(IDX[:340], 0.5)


def test_the_consistency_score_ranks_a_steady_stream_above_one_that_depends_on_a_single_stretch():
    rng = np.random.default_rng(5)
    steady = pd.Series(0.0007 + rng.normal(0, 0.01, 1600), index=IDX[:1600])
    lucky = pd.Series(rng.normal(0, 0.01, 1600), index=IDX[:1600])
    lucky.iloc[:400] += 0.0028                                                 # the same total drift, all of it in the first quarter
    assert T.score(steady, "sharpe", True) > T.score(lucky, "sharpe", True)
    assert T.score(lucky, "sharpe", True) < T.score(lucky, "sharpe", False)     # the spread across the four parts is charged for


def test_metrics_of_a_short_or_empty_stretch_are_none_and_of_a_stream_are_the_usual_numbers():
    assert T.metrics(pd.Series(dtype=float))["sharpe"] is None
    r = pd.Series(NOISE[:500] + 0.0005, index=IDX[:500])
    m = T.metrics(r)
    assert m["days"] == 500 and m["sharpe"] == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(252)) and m["max_drawdown"] <= 0 and m["vol"] == pytest.approx(r.std(ddof=1) * np.sqrt(252))


# ------------------------------------------------------------------------------------------------------------------ the search
def test_the_search_finds_the_planted_optimum_runs_the_starting_settings_first_and_the_gain_holds_on_the_hold_out():
    out = T.tune(run_with(lambda p: peak(p["lookback"])), [LOOKBACK], trials=30, seconds=60, seed=3)
    assert out["start"]["number"] == 0 and out["start"]["params"] == {"lookback": 20}
    assert 40 <= out["best"]["params"]["lookback"] <= 90 and out["same_as_start"] is False
    assert out["best"]["tuning"]["sharpe"] > out["start"]["tuning"]["sharpe"] + 1
    assert out["best"]["holdout"]["sharpe"] > out["start"]["holdout"]["sharpe"] + 1
    assert out["verdict"]["level"] == "good" and "held up" in out["verdict"]["headline"]
    assert out["search"]["tried"] == 30 and out["search"]["failed"] == 0 and out["search"]["stopped"] == "limit"
    w = out["windows"]
    assert w["tuning"] == [str(IDX[0].date()), str(IDX[w["tuning_days"] - 1].date())] and w["holdout"][0] == str(SPLIT.date()) and w["holdout_days"] == len(IDX) - w["tuning_days"]
    row = out["settings"][0]
    assert row["name"] == "lookback" and row["importance"] > 0.3 and row["share"] == pytest.approx(1.0) and len(row["points"]) == 30 and row["sweet"][0] <= out["best"]["params"]["lookback"] <= row["sweet"][1]
    assert [h["number"] for h in out["history"]] == list(range(30)) and out["history"][0]["params"] == {"lookback": 20} and out["history"][-1]["best"] == pytest.approx(max(h["score"] for h in out["history"]))
    assert out["selection"]["n_trials"] == 30 and 0 <= out["selection"]["pbo"] <= 1 and out["selection"]["n_trials_counted"] == 30


def test_the_same_seed_gives_the_same_search_and_another_seed_a_different_one():
    run = run_with(lambda p: peak(p["lookback"]))
    a, b, c = (T.tune(run, [LOOKBACK], trials=12, seconds=60, seed=s) for s in (4, 4, 5))
    assert [t["params"] for t in a["history"]] == [t["params"] for t in b["history"]]
    assert [t["params"] for t in a["history"]] != [t["params"] for t in c["history"]]


def test_a_gain_that_exists_only_where_the_search_could_see_is_called_overfitted():
    out = T.tune(run_with(lambda p: peak(p["lookback"]), tuning_only=True), [LOOKBACK], trials=24, seconds=60, seed=2)
    assert out["best"]["tuning"]["sharpe"] > out["start"]["tuning"]["sharpe"] + 1
    assert out["best"]["holdout"]["sharpe"] == pytest.approx(out["start"]["holdout"]["sharpe"])
    assert out["verdict"]["level"] == "bad" and out["verdict"]["headline"].startswith("Probably overfitted")
    text = " ".join(out["verdict"]["reasons"])
    assert "untouched" in text and "Keep the starting settings" in text


def test_nothing_beating_the_starting_settings_is_a_result_and_says_keep_what_you_have():
    start = T.Setting("lookback", "int", 60, 5, 250, "linear")
    out = T.tune(run_with(lambda p: peak(p["lookback"], width=0.2)), [start], trials=20, seconds=60, seed=1)
    assert out["same_as_start"] is True and out["best"]["number"] == 0
    assert out["verdict"]["level"] == "keep" and out["verdict"]["headline"].startswith("Keep what you have")


def test_the_score_is_the_tuning_window_alone_the_hold_out_never_reaches_the_search():
    """Two strategies identical before the split and opposite after it score the same, and so are searched the same way."""
    seen = []

    def run(params):
        seen.append(params["lookback"])
        edge = peak(params["lookback"])
        e = np.where(IDX < SPLIT, edge, 50.0 * edge * (1 if params["lookback"] % 2 else -1))      # what happens after the split is wildly different, and must not matter
        return T.Run(pd.Series(NOISE + e, index=IDX))

    wild = T.tune(run, [LOOKBACK], trials=14, seconds=60, seed=6)
    calm = T.tune(run_with(lambda p: peak(p["lookback"]), tuning_only=True), [LOOKBACK], trials=14, seconds=60, seed=6)
    assert [t["params"] for t in wild["history"]] == [t["params"] for t in calm["history"]]
    assert [t["score"] for t in wild["history"]] == pytest.approx([t["score"] for t in calm["history"]])
    assert wild["selection"]["best_trial"] == calm["selection"]["best_trial"]


def test_a_try_is_never_repeated_and_a_small_space_ends_the_search_when_it_is_used_up():
    seen = []

    def run(params):
        seen.append(tuple(sorted(params.items())))
        return T.Run(pd.Series(NOISE + 0.0002 * params["k"] + (0.0001 if params["flag"] else 0.0), index=IDX))

    settings = [T.Setting("k", "int", 2, 1, 3, "linear"), T.Setting("flag", "bool", False, choices=(False, True))]
    out = T.tune(run, settings, trials=50, seconds=60, seed=0)
    assert out["search"]["tried"] == 6 and out["search"]["stopped"] == "exhausted" and len(set(seen)) == 6
    assert out["best"]["params"] == {"k": 3, "flag": True} and out["settings"][1]["choices_summary"][1]["choice"] is True


def test_stop_ends_the_search_after_the_try_it_is_on_and_keeps_and_reports_the_tries_so_far():
    flags = {"calls": 0}

    def stop():
        flags["calls"] += 1
        return flags["calls"] >= 4

    out = T.tune(run_with(lambda p: peak(p["lookback"])), [LOOKBACK], trials=40, seconds=60, seed=0, stop=stop)
    assert out["search"]["stopped"] == "user" and out["search"]["tried"] == 5 and out["search"]["budget"] == 40
    assert out["verdict"]["level"] in ("good", "mixed", "bad", "keep")


def test_the_time_limit_stops_a_search_but_not_before_six_tries():
    ticks = iter(range(0, 10_000, 10))                                          # ten "seconds" pass at every look at the clock
    out = T.tune(run_with(lambda p: peak(p["lookback"])), [LOOKBACK], trials=100, seconds=1, seed=0, clock=lambda: float(next(ticks)))
    assert out["search"]["stopped"] == "time" and out["search"]["tried"] == T.MIN_TRIALS


def test_progress_says_which_try_it_is_on_and_the_best_score_so_far():
    said = []
    T.tune(run_with(lambda p: peak(p["lookback"])), [LOOKBACK], trials=8, seconds=60, seed=0, progress=said.append)
    assert said[0] == "running the starting settings" and any(m.startswith("try 2 of up to 8 · best score so far") for m in said) and any(m.startswith("try 8 of up to 8") for m in said)


def test_tries_that_fail_are_counted_and_explained_and_the_search_goes_on():
    def run(params):
        if params["lookback"] % 3 == 0:
            raise ValueError("lookback must not be a multiple of three")
        return run_with(lambda p: peak(p["lookback"]))(params)

    out = T.tune(run, [LOOKBACK], trials=40, seconds=60, seed=0)
    assert out["search"]["failed"] > 0 and out["search"]["scored"] == out["search"]["tried"] - out["search"]["failed"]
    assert out["failures"][0]["reason"] == "lookback must not be a multiple of three" and out["failures"][0]["count"] == out["search"]["failed"]
    assert out["best"]["params"]["lookback"] % 3 != 0 and all((h["score"] is None) == (h["params"]["lookback"] % 3 == 0) for h in out["history"])


def test_starting_settings_that_cannot_run_stop_the_tuning_with_the_reason_and_so_do_almost_all_failing():
    def broken(params):
        raise ValueError("this strategy needs more tickers")

    with pytest.raises(ValueError, match="this strategy needs more tickers"):            # a run that refuses outright says why, in its own words
        T.tune(broken, [LOOKBACK], trials=10, seconds=60)
    with pytest.raises(T.TuningError, match=r"starting settings could not be scored: it has no score \(no trades"):   # a strategy that never trades has no ratio to improve
        T.tune(lambda params: T.Run(pd.Series(0.0, index=IDX)), [LOOKBACK], trials=10, seconds=60)

    def only_start_works(params):
        if params["lookback"] != 20:
            raise ValueError("no data for that window")
        return T.Run(pd.Series(NOISE, index=IDX))

    with pytest.raises(T.TuningError, match=r"almost every setting tried failed.*no data for that window \(x4\)"):
        T.tune(only_start_works, [LOOKBACK], trials=20, seconds=60)


def test_a_try_that_trades_on_too_few_days_of_the_tuning_window_is_a_failed_try_not_a_good_score():
    def run(params):
        r = pd.Series(NOISE + (0.001 if params["lookback"] > 30 else 0.0), index=IDX)
        return T.Run(r.iloc[1500:] if params["lookback"] % 2 else r)             # a long warm-up leaves it almost nothing of the tuning window

    out = T.tune(run, [LOOKBACK], trials=30, seconds=60, seed=0)
    assert any("traded on only" in f["reason"] for f in out["failures"])


def test_a_best_value_at_the_end_of_its_range_is_flagged_and_the_verdict_says_to_widen_the_range():
    edge = T.Setting("lookback", "int", 20, 5, 100, "linear")
    out = T.tune(run_with(lambda p: 0.00002 * p["lookback"]), [edge], trials=30, seconds=60, seed=0)       # more is always better: the best is at the top of the range
    assert out["settings"][0]["edge"] == "high" and out["best"]["params"]["lookback"] >= 90
    assert any("end of the range" in r and "lookback" in r for r in out["verdict"]["reasons"])


def test_earlier_looks_at_the_same_prices_lower_the_chance_that_the_winner_is_more_than_luck():
    run = run_with(lambda p: peak(p["lookback"]))
    fresh = T.tune(run, [LOOKBACK], trials=20, seconds=60, seed=3)
    seen = T.tune(run, [LOOKBACK], trials=20, seconds=60, seed=3, prior_trials=400)
    assert seen["selection"]["n_trials_counted"] == 420 and fresh["selection"]["n_trials_counted"] == 20
    assert seen["selection"]["deflated_sharpe_probability"] < fresh["selection"]["deflated_sharpe_probability"]
    assert any("counting 400 earlier ones" in r for r in seen["verdict"]["reasons"])


def test_the_comparison_with_holding_the_tickers_and_the_table_by_ticker_use_the_same_days_and_arithmetic():
    rng = np.random.default_rng(8)
    assets = pd.DataFrame(rng.normal(0.0003, 0.01, (len(IDX), 3)), index=IDX, columns=["AAA", "BBB", "CCC"])

    def run(params):
        w = pd.DataFrame(0.0, index=IDX, columns=assets.columns)
        w["AAA"], w["BBB"] = params["lookback"] / 500.0, -params["lookback"] / 1000.0
        returns = (w.shift(1) * assets).sum(axis=1).iloc[1:]
        return T.Run(returns, w)

    out = T.tune(run, [LOOKBACK], trials=10, seconds=60, seed=0, assets=assets)
    assert out["benchmark"]["label"].startswith("Hold every ticker") and out["benchmark"]["holdout"]["days"] == out["windows"]["holdout_days"]
    held = assets.loc[SPLIT:].mean(axis=1)
    assert out["benchmark"]["holdout"]["sharpe"] == pytest.approx(held.mean() / held.std(ddof=1) * np.sqrt(252), rel=1e-6)
    rows = {r["ticker"]: r for r in out["tickers"]}
    total_best = sum(r["best_holdout"] for r in rows.values())
    returns = run(out["best"]["params"]).returns
    assert total_best == pytest.approx(100 * returns.loc[SPLIT:].sum(), rel=1e-6)                 # the tickers' shares add up to the strategy's own return on those days
    assert rows["CCC"]["start_holdout"] == 0 and rows["CCC"]["best_holdout"] == 0


def test_a_verdict_is_a_good_one_only_when_half_the_gain_survives_and_the_strategy_still_earns():
    def pack(t, h):
        return {"tuning": {"sharpe": t}, "holdout": {"sharpe": h}}

    win = T.make_windows(IDX, 0.3)
    judged = lambda s, b: T.judge("sharpe", pack(*s), pack(*b), *T.spans(win), False, None, 30, 0)   # noqa: E731
    assert judged((0.1, 0.1), (0.7, 0.5))["level"] == "good"                       # 0.4 of the 0.6 gain survived
    assert judged((0.1, 0.1), (0.7, 0.25))["level"] == "mixed" and "Only part" in judged((0.1, 0.1), (0.7, 0.25))["headline"]
    assert judged((0.1, 0.1), (0.7, 0.12))["headline"] == "No real difference on the untouched days"
    assert judged((0.1, 0.1), (0.7, 0.05))["level"] == "bad"
    assert judged((-0.5, -0.4), (0.3, -0.1))["headline"].startswith("Better than the starting settings, but still losing")
    unknown = T.judge("calmar", {"tuning": {"calmar": 0.1}, "holdout": {"calmar": None}}, {"tuning": {"calmar": 0.3}, "holdout": {"calmar": 0.2}}, *T.spans(win), False, None, 30, 0)
    assert unknown["level"] == "mixed" and "Could not judge" in unknown["headline"]
    growth = T.judge("cagr", {"tuning": {"cagr": 0.02}, "holdout": {"cagr": 0.01}}, {"tuning": {"cagr": 0.06}, "holdout": {"cagr": 0.04}}, *T.spans(win), False, None, 30, 0)
    assert growth["level"] == "good" and "+4.0 points" in " ".join(growth["reasons"])


def test_the_comparison_with_holding_every_ticker_is_in_the_verdict_when_it_beats_the_strategy():
    win = T.make_windows(IDX, 0.3)
    start, best = {"tuning": {"sharpe": 0.1}, "holdout": {"sharpe": 0.1}}, {"tuning": {"sharpe": 0.6}, "holdout": {"sharpe": 0.5}}
    behind = T.judge("sharpe", start, best, *T.spans(win), False, None, 30, 0, None, {"holdout": {"sharpe": 0.8}})
    ahead = T.judge("sharpe", start, best, *T.spans(win), False, None, 30, 0, None, {"holdout": {"sharpe": 0.2}})
    assert any("not beaten simply holding the tickers" in r for r in behind["reasons"]) and any("ahead of it" in r for r in ahead["reasons"])


def test_a_request_the_tuner_cannot_run_is_refused_with_a_sentence():
    run = run_with(lambda p: 0.0)
    for kwargs, message in [({"objective": "nonsense"}, "measure must be one of"), ({"holdout": 0.9}, "between 10% and 50%"), ({"trials": 1}, "between 2 and"), ({"trials": 1000}, "between 2 and")]:
        with pytest.raises(T.TuningError, match=message):
            T.tune(run, [LOOKBACK], **{"trials": 10, **kwargs})
    with pytest.raises(T.TuningError, match="tick at least one setting"):
        T.tune(run, [], trials=10)
    with pytest.raises(T.TuningError, match="lowest value to try must be below the highest"):
        T.tune(run, [T.Setting("lookback", "int", 20, 50, 5, "linear")], trials=10)
    with pytest.raises(T.TuningError, match="above zero"):
        T.tune(run, [T.Setting("noise", "float", 0.1, 0.0, 1.0, "log")], trials=10)


# ------------------------------------------------------------------------------------------------------------------ the optimiser hooks the tuner relies on
def test_a_study_stops_when_its_callback_says_so_and_keeps_every_try_it_ran():
    study = Study({"x": Float(0, 1)}, "maximize", 0)
    study.optimize(lambda p: p["x"], n_trials=30, method="random", callback=lambda st, trial: st.n_trials >= 4)
    assert study.n_trials == 4 and study.best.value == max(t.value for t in study.trials)


def test_a_unique_search_never_repeats_a_configuration_and_ends_when_none_is_left():
    study = Study({"k": Int(1, 3), "flag": Categorical(False, True)}, "maximize", 0)
    study.optimize(lambda p: p["k"] + (0.5 if p["flag"] else 0.0), n_trials=50, method="tpe", unique=True, n_startup=3)
    keys = [(t.params["k"], t.params["flag"]) for t in study.trials]
    assert len(keys) == 6 and len(set(keys)) == 6
    study2 = Study({"k": Int(1, 3)}, "maximize", 0)
    study2._run(lambda p: float(p["k"]), {"k": 2})                              # a try run by hand is not repeated either
    study2.optimize(lambda p: float(p["k"]), n_trials=10, method="random", unique=True)
    assert sorted(t.params["k"] for t in study2.trials) == [1, 2, 3]
