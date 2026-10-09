"""Alpha-generating styles: each model finds the thing planted in synthetic data, ignores the lack of it, and uses nothing it could not have known."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.framework import MODELS, bundle_from_prices, load_library
from src.framework.validate import check_causality
from src.strategies import alpha_styles
from src.strategies.alpha_styles import headline_score

load_library()


def _bundle(returns: pd.DataFrame, volume: pd.DataFrame | None = None, classes: dict | None = None):
    prices = 100.0 * (1.0 + returns.fillna(0.0)).cumprod()
    return bundle_from_prices(prices, volume=volume, asset_class=classes or {c: "equity" for c in returns.columns}, name="synthetic")


def _noise(n: int, k: int, seed: int, sd: float = 0.01, names=None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-04", periods=n)
    return pd.DataFrame(rng.normal(0.0, sd, (n, k)), index=idx, columns=names or [f"A{i}" for i in range(k)])


# ------------------------------------------------------------------------------------------------------------------------------- jensen alpha
def test_jensen_alpha_prefers_steady_alpha_to_noisy_alpha_and_both_to_none():
    r = _noise(2200, 8, 1, 0.006)
    market = r.mean(axis=1)
    r["STEADY"] = 0.9 * market + np.random.default_rng(2).normal(0, 0.003, len(r)) + 0.001
    r["NOISY"] = 0.9 * market + np.random.default_rng(3).normal(0, 0.025, len(r)) + 0.001
    s = MODELS.create("jensen_alpha").score(_bundle(r))
    late = s.iloc[-300:].mean()
    assert late["STEADY"] == late.max()
    assert late["STEADY"] > late["NOISY"] + 0.1                                                              # the same alpha in a quarter of the noise is worth far more
    assert late["NOISY"] < 0.15 and late.drop(["STEADY", "NOISY"]).abs().max() < 0.12


def test_jensen_alpha_skips_the_last_month_and_needs_a_history():
    r = _noise(1500, 6, 4, 0.01)
    model = MODELS.create("jensen_alpha", window=500, skip=21)
    s = model.score(_bundle(r))
    assert s.iloc[:250].isna().all().all()                                                                   # half a window of history before the first score
    assert not s.iloc[300:].isna().all().all()
    unskipped = MODELS.create("jensen_alpha", window=500, skip=0).score(_bundle(r))
    pd.testing.assert_frame_equal(s.iloc[100:], unskipped.shift(21).iloc[100:], check_exact=False, atol=1e-12)


@pytest.mark.parametrize("kwargs", [{"window": 50}, {"skip": -1}])
def test_jensen_alpha_rejects_bad_parameters(kwargs):
    with pytest.raises(ValueError):
        MODELS.create("jensen_alpha", **kwargs)


# ------------------------------------------------------------------------------------------------------------------------------- adaptive autocorrelation
def _ar1(n: int, rho: float, seed: int, sd: float = 0.01) -> np.ndarray:
    e = np.random.default_rng(seed).normal(0, sd, n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + e[t]
    return x


def test_adaptive_autocorrelation_continues_where_returns_trend_and_fades_where_they_revert():
    n = 3000
    idx = pd.bdate_range("2010-01-04", periods=n)
    r = pd.DataFrame({"MOM": _ar1(n, 0.25, 1), "REV": _ar1(n, -0.25, 2), "WHITE": _ar1(n, 0.0, 3)}, index=idx)
    s = MODELS.create("adaptive_autocorrelation").score(_bundle(r))
    late = slice(400, None)
    agree = lambda a: float((np.sign(s[a].iloc[late]) == np.sign(r[a].iloc[late])).mean())                  # noqa: E731
    assert agree("MOM") > 0.85                                                                              # same sign as yesterday's move
    assert agree("REV") < 0.15                                                                              # the opposite sign
    assert s["WHITE"].iloc[late].abs().mean() < 0.5 * s["MOM"].iloc[late].abs().mean()                      # no significant autocorrelation, little conviction
    next_day = r.shift(-1).iloc[late]
    assert (s["MOM"].iloc[late] * next_day["MOM"]).mean() > 0 and (s["REV"].iloc[late] * next_day["REV"]).mean() > 0


# ------------------------------------------------------------------------------------------------------------------------------- squeeze breakout
def _squeezed_series(jump: float, quiet_first: bool = False) -> pd.DataFrame:
    """300 volatile days, 80 quiet ones (a squeeze), then a one-day jump and a drift: ``quiet_first`` puts the quiet spell at the start so there is no squeeze at the breakout."""
    rng = np.random.default_rng(5)
    loud, quiet = rng.normal(0, 0.012, 300), rng.normal(0, 1.0, 80) * np.geomspace(0.012, 0.0003, 80)        # a spell in which volatility keeps shrinking: the last day is the narrowest
    r = np.r_[quiet, loud] if quiet_first else np.r_[loud, quiet]
    r = np.r_[r, [jump], np.full(30, 0.004)]
    return pd.DataFrame({"X": r}, index=pd.bdate_range("2012-01-02", periods=len(r)))


def test_squeeze_breakout_goes_long_on_the_first_close_above_the_band_after_a_squeeze_and_exits_below_the_mid_band():
    r = _squeezed_series(0.03)
    b = _bundle(r)
    s = MODELS.create("squeeze_breakout").score(b)["X"]
    jump_day = 300 + 80
    assert s.iloc[jump_day - 1] <= 0 and s.iloc[jump_day] == 1.0                                              # not long until the breakout bar closes above the band, and long from it
    assert s.iloc[jump_day:jump_day + 25].eq(1.0).all()                                                       # held while the price stays above the middle band
    crash = r.copy()
    crash.iloc[jump_day + 5, 0] = -0.2
    out = MODELS.create("squeeze_breakout").score(_bundle(crash))["X"]
    assert out.iloc[jump_day + 5] == 0.0                                                                      # a close back below the middle band ends it


def test_squeeze_breakout_needs_the_squeeze_to_have_been_there_the_day_before():
    quiet_then_loud = _squeezed_series(0.03, quiet_first=True)
    s = MODELS.create("squeeze_breakout").score(_bundle(quiet_then_loud))["X"]
    assert s.iloc[-31] == 0.0                                                                               # the jump after a volatile spell is no breakout from a squeeze


# ------------------------------------------------------------------------------------------------------------------------------- abnormal volume drift
def _volume_event_bundle():
    n = 400
    idx = pd.bdate_range("2013-01-02", periods=n)
    rng = np.random.default_rng(9)
    r = pd.DataFrame(rng.uniform(-0.004, 0.004, (n, 3)), index=idx, columns=["HEAVY", "QUIET", "NONE"])        # bounded noise: no ordinary day is a two-sigma move
    volume = pd.DataFrame(rng.uniform(0.9e6, 1.1e6, (n, 3)), index=idx, columns=r.columns)
    r.iloc[200, 0], volume.iloc[200, 0] = 0.06, 5e6                                                         # a big move on heavy volume
    r.iloc[200, 1], volume.iloc[200, 1] = 0.06, 1e6                                                         # the same move on ordinary volume
    return _bundle(r, volume), 200


def test_abnormal_volume_drift_continues_a_heavy_volume_move_and_fades_a_quiet_one_over_ten_days():
    b, e = _volume_event_bundle()
    s = MODELS.create("abnormal_volume_drift", hold=10, reversal=0.5).score(b)
    heavy, quiet = s["HEAVY"], s["QUIET"]
    assert np.allclose(heavy.iloc[e:e + 10].to_numpy(), [1.0 - j / 10 for j in range(10)])
    assert np.allclose(quiet.iloc[e:e + 10].to_numpy(), [-0.5 * (1.0 - j / 10) for j in range(10)])
    assert heavy.iloc[e + 10] == 0.0 and heavy.iloc[e - 1] == 0.0                                           # nothing before the event or after the window


def test_abnormal_volume_drift_needs_volume():
    r = _noise(300, 2, 1)
    with pytest.raises(KeyError, match="volume"):
        MODELS.create("abnormal_volume_drift").score(_bundle(r))


# ------------------------------------------------------------------------------------------------------------------------------- event study
def _event_world(drift_days: int = 5, n_each: int = 120, seed: int = 21, drift: float = 0.01):
    """Eight assets; events of type DRIFT are followed by ``drift_days`` days of +1% a day, events of type NULL by nothing. Returns the bundle, the events and the positions."""
    n = 3200
    r = _noise(n, 8, seed, 0.01)
    rng = np.random.default_rng(seed + 1)
    rows = []
    for kind, plant in (("DRIFT", True), ("NULL", False)):
        slots = np.sort(rng.choice(np.arange(150, n - 20, 25), size=n_each, replace=False))
        for pos in slots:
            asset = int(rng.integers(0, 8))
            rows.append({"date": r.index[pos], "ticker": r.columns[asset], "type": kind, "pos": int(pos), "asset": asset})
            if plant:
                r.iloc[pos + 1:pos + 1 + drift_days, asset] += drift
    events = pd.DataFrame(rows)
    return _bundle(r), events


def test_event_study_drift_learns_the_planted_drift_from_a_file_and_stays_out_of_the_null_events(tmp_path):
    b, events = _event_world()
    path = tmp_path / "events.csv"
    events[["date", "ticker", "type"]].to_csv(path, index=False)
    model = MODELS.create("event_study_drift", path=str(path), window=5, min_events=30, tstat=3.0)
    s = model.score(b).fillna(0.0).to_numpy()
    drift, null = events[events["type"] == "DRIFT"], events[events["type"] == "NULL"]
    # once it has learned (after 40 of the 120 drift events had finished), the day of the event it expects a gain; for the null events it stays flat
    late_drift = drift.iloc[60:]
    clear = lambda row: not ((drift["asset"] == row["asset"]) & ((drift["pos"] - row["pos"]).abs() <= 5)).any()                          # noqa: E731  (no drift window of the same asset overlaps it)
    late_null = null[(null["pos"] > late_drift["pos"].min()) & null.apply(clear, axis=1)]
    on_drift = np.array([s[p, a] for p, a in zip(late_drift["pos"], late_drift["asset"])])
    on_null = np.array([s[p, a] for p, a in zip(late_null["pos"], late_null["asset"])])
    assert len(late_null) > 20
    assert (on_drift > 0.2).mean() > 0.9
    assert np.abs(on_null).mean() < 0.05
    # nothing before some type has min_events completed events behind it
    first_ready = min(np.sort(drift["pos"].to_numpy())[29], np.sort(null["pos"].to_numpy())[29]) + 5
    assert not s[:first_ready].any()


def test_event_study_drift_uses_only_events_whose_whole_window_has_finished():
    """Five identical events with a clear drift: the sixth is traded, but not before the fifth one's window is complete."""
    n, h = 800, 5
    r = _noise(n, 4, 31, 0.002)
    spots = [100, 200, 300, 400, 500, 600]
    for p in spots:
        r.iloc[p + 1:p + 1 + h, 0] += 0.01
    events = pd.DataFrame({"pos": spots, "asset": 0, "kind": "A"})
    b = _bundle(r)
    model = MODELS.create("event_study_drift", window=h, min_events=5, tstat=0.0)
    model.events = lambda data: events                                                                      # five finished events are needed; hand them over directly
    s = model.score(b)["A0"]
    assert s.iloc[spots[4]:spots[4] + h].eq(0.0).all()                                                       # the fifth event's own window: only four events are complete
    assert s.iloc[spots[5]] > 0.5                                                                           # the sixth, once five windows have finished
    assert s.iloc[spots[5] + h:].abs().max() < 1e-12                                                         # and nothing after its window is over


def test_event_study_drift_is_causal_with_a_file_and_with_price_events(tmp_path):
    b, events = _event_world(n_each=60)
    path = tmp_path / "events.csv"
    events[["date", "ticker", "type"]].to_csv(path, index=False)
    for params in ({"path": str(path), "window": 5, "min_events": 20}, {}):
        check = check_causality(MODELS.create("event_study_drift", **params), b, cutoff=b.index[2400])
        assert check["ok"], (params, check)


def test_event_study_drift_splits_a_type_by_the_sign_of_its_size(tmp_path):
    b, events = _event_world(n_each=40)
    table = events[["date", "ticker", "type"]].copy()
    table["size"] = np.where(np.arange(len(table)) % 2 == 0, 1.0, -1.0)
    path = tmp_path / "events.csv"
    table.to_csv(path, index=False)
    model = MODELS.create("event_study_drift", path=str(path))
    kinds = set(model.events(b)["kind"])
    assert kinds == {"DRIFT_up", "DRIFT_down", "NULL_up", "NULL_down"}


def test_event_study_drift_rolls_weekend_events_forward_and_drops_tickers_it_does_not_hold(tmp_path):
    r = _noise(300, 3, 2)
    b = _bundle(r)
    table = pd.DataFrame({"date": ["2010-06-05", "2010-06-02", "2010-06-02"], "ticker": ["A0", "A1", "NOPE"], "type": ["X", "X", "X"]})
    path = tmp_path / "events.csv"
    table.to_csv(path, index=False)
    ev = MODELS.create("event_study_drift", path=str(path)).events(b)
    assert len(ev) == 2
    assert b.index[ev.loc[ev["asset"] == 0, "pos"].iloc[0]] == pd.Timestamp("2010-06-07")                    # the next trading day


def test_event_study_drift_explains_a_missing_or_malformed_events_file(tmp_path, monkeypatch):
    b = _bundle(_noise(300, 3, 2))
    monkeypatch.setattr(alpha_styles, "USER_DATA", tmp_path)
    with pytest.raises(KeyError, match="needs the file"):
        MODELS.create("event_study_drift", path="nothing_here.csv").score(b)
    (tmp_path / "bad.csv").write_text("when,what\n2010-01-05,x\n")
    with pytest.raises(KeyError, match="date, ticker and type"):
        MODELS.create("event_study_drift", path="bad.csv").score(b)


# ------------------------------------------------------------------------------------------------------------------------------- headlines
def test_headline_score_reads_tone_from_a_finance_word_list():
    assert headline_score("Company beats estimates and raises guidance") == 1.0
    assert headline_score("Shares plunge as regulator opens probe") == -1.0
    assert headline_score("Company beats estimates but warns of weak demand") == pytest.approx((1 - 2) / 3)
    assert headline_score("The company holds its annual meeting") == 0.0
    assert headline_score("") == 0.0 and headline_score(None) == 0.0


def _headline_file(tmp_path, rows):
    path = tmp_path / "headlines.csv"
    pd.DataFrame(rows, columns=["date", "ticker", "headline"]).to_csv(path, index=False)
    return str(path)


def test_news_sentiment_acts_from_the_day_after_the_headline_and_decays_with_its_half_life(tmp_path):
    b = _bundle(_noise(300, 3, 7, names=["UP", "DOWN", "QUIET"]))
    d = b.index[100]
    path = _headline_file(tmp_path, [[d, "UP", "Record profit as company beats estimates"], [d, "DOWN", "Shares plunge on fraud probe"]])
    s = MODELS.create("news_sentiment", path=path, half_life=3.0, lag=1, scale=1.0).score(b)
    assert s[["UP", "DOWN", "QUIET"]].iloc[:101].abs().sum().sum() == 0                                      # nothing on the day itself or before
    first = s.iloc[101]
    assert first["UP"] > 0.5 and first["DOWN"] < -0.5 and first["QUIET"] == 0
    assert s["UP"].iloc[101] == pytest.approx(np.tanh(1.0))
    assert s["UP"].iloc[104] == pytest.approx(np.tanh(0.5))                                                  # one half-life later
    assert 0 < s["UP"].iloc[130] < 0.01


def test_news_sentiment_takes_a_score_column_in_place_of_the_word_list_and_stacks_scores(tmp_path):
    b = _bundle(_noise(200, 2, 7, names=["X", "Y"]))
    d = b.index[120]
    path = tmp_path / "scores.csv"
    pd.DataFrame({"date": [d, d], "ticker": ["X", "X"], "score": [0.4, 0.3]}).to_csv(path, index=False)
    s = MODELS.create("news_sentiment", path=str(path), half_life=5.0, lag=0, scale=1.0).score(b)
    assert s["X"].iloc[120] == pytest.approx(np.tanh(0.7))


def test_news_sentiment_is_causal_and_explains_what_it_needs(tmp_path, monkeypatch):
    b = _bundle(_noise(500, 3, 8, names=["X", "Y", "Z"]))
    rows = [[b.index[i], t, h] for i, (t, h) in zip(range(30, 480, 15), (("X", "company beats estimates"), ("Y", "plunge in profit and fraud probe"), ("Z", "record growth")) * 10)]
    path = _headline_file(tmp_path, rows)
    assert check_causality(MODELS.create("news_sentiment", path=path), b, cutoff=b.index[300])["ok"]
    monkeypatch.setattr(alpha_styles, "USER_DATA", tmp_path / "empty")
    with pytest.raises(KeyError, match="needs the file"):
        MODELS.create("news_sentiment").score(b)
    (tmp_path / "wrong.csv").write_text("date,ticker\n2010-01-05,X\n")
    with pytest.raises(KeyError, match="headline"):
        MODELS.create("news_sentiment", path=str(tmp_path / "wrong.csv")).score(b)


# ------------------------------------------------------------------------------------------------------------------------------- panel signals
def _panel_bundle():
    return _bundle(_noise(400, 4, 11, names=["A", "B", "C", "D"]))


def test_panel_signal_is_usable_after_its_lag_and_expires(tmp_path):
    b = _panel_bundle()
    d = b.index[100]
    pd.DataFrame({"date": [d], "A": [3.0], "B": [1.0], "C": [-1.0], "D": [-3.0]}).to_csv(tmp_path / "signals.csv", index=False)
    s = MODELS.create("panel_signal", path=str(tmp_path / "signals.csv"), lag=2, expiry=10, standardise=False).score(b)
    assert s.iloc[:102].isna().all().all()                                                                   # published on day 100, usable on day 102
    assert s.iloc[102]["A"] == 3.0 and s.iloc[102]["D"] == -3.0
    assert s.iloc[111].notna().all() and s.iloc[113].isna().all()                                            # carried for `expiry` days, then stale


def test_panel_signal_standardises_across_companies_flips_sign_and_reads_the_long_format(tmp_path):
    b = _panel_bundle()
    dates = [b.index[120], b.index[220]]
    wide = pd.DataFrame({"date": dates, "A": [10.0, 5.0], "B": [20.0, 6.0], "C": [30.0, 7.0], "D": [60.0, 8.0]})
    long = wide.melt(id_vars="date", var_name="ticker", value_name="value")
    wide.to_csv(tmp_path / "wide.csv", index=False)
    long.to_csv(tmp_path / "long.csv", index=False)
    a = MODELS.create("panel_signal", path=str(tmp_path / "wide.csv"), lag=1, expiry=60).score(b)
    c = MODELS.create("panel_signal", path=str(tmp_path / "long.csv"), lag=1, expiry=60).score(b)
    pd.testing.assert_frame_equal(a, c)
    row = a.iloc[121]
    assert row.mean() == pytest.approx(0.0, abs=1e-12) and row.std() == pytest.approx(1.0) and row["D"] == row.max()
    flipped = MODELS.create("panel_signal", path=str(tmp_path / "wide.csv"), lag=1, expiry=60, direction=-1.0).score(b)
    pd.testing.assert_frame_equal(flipped, -a)


def test_panel_signal_change_turns_a_level_into_a_revision(tmp_path):
    b = _panel_bundle()
    dates = [b.index[120], b.index[160]]
    pd.DataFrame({"date": dates, "A": [1.0, 3.0], "B": [1.0, 1.0], "C": [2.0, 2.0], "D": [2.0, 0.0]}).to_csv(tmp_path / "s.csv", index=False)
    s = MODELS.create("panel_signal", path=str(tmp_path / "s.csv"), lag=0, expiry=100, change=20, standardise=False).score(b)
    # ten days after the second update the 20-day change spans it: A went from 1 to 3, D from 2 to 0; once the window no longer spans it, the revision is gone
    assert s["A"].iloc[170] == 2.0 and s["D"].iloc[170] == -2.0 and s["B"].iloc[170] == 0.0 and s["C"].iloc[170] == 0.0
    assert s["A"].iloc[180] == 0.0 and s["D"].iloc[180] == 0.0
    assert s.iloc[:140].isna().all().all()                                                                   # the first update has nothing to be compared with 20 days earlier


def test_panel_signal_is_causal_and_explains_what_it_needs(tmp_path, monkeypatch):
    b = _panel_bundle()
    dates = list(b.index[20:380:40])
    rng = np.random.default_rng(1)
    pd.DataFrame({"date": dates, **{t: rng.normal(size=len(dates)) for t in "ABCD"}}).to_csv(tmp_path / "signals.csv", index=False)
    assert check_causality(MODELS.create("panel_signal", path=str(tmp_path / "signals.csv")), b, cutoff=b.index[250])["ok"]
    monkeypatch.setattr(alpha_styles, "USER_DATA", tmp_path / "empty")
    with pytest.raises(KeyError, match="needs the file"):
        MODELS.create("panel_signal").score(b)


@pytest.mark.parametrize("name, kwargs", [("panel_signal", {"lag": -1}), ("panel_signal", {"expiry": 0}), ("panel_signal", {"direction": 0.0}), ("news_sentiment", {"half_life": 0.0}),
                                          ("news_sentiment", {"lag": -1}), ("event_study_drift", {"window": 1}), ("event_study_drift", {"min_events": 2}), ("squeeze_breakout", {"quantile": 1.5}),
                                          ("abnormal_volume_drift", {"hold": 1}), ("adaptive_autocorrelation", {"window": 20})])
def test_the_new_models_reject_nonsense_parameters(name, kwargs):
    with pytest.raises(ValueError):
        MODELS.create(name, **kwargs)
