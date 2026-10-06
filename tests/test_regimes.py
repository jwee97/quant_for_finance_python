"""Regime models: filtering, causality, change-point detection and named rules.

The HMM filter is checked against the library it wraps (log-likelihood and
smoothed posterior), because a filter that merely looks plausible is how a
subtle look-ahead gets in. Causality is then tested the way the rest of the
platform tests it: replace the future, require the past not to move.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.regime_rules import (
    bear_market_state,
    inflation_shock,
    liquidity_crisis,
    volatility_regime,
)
from src.models.regimes import (
    HMMFit,
    alarm_episodes,
    bocpd,
    evaluate_detector,
    fit_hmm,
    forward_filter,
    hmm_posteriors,
    log_emission,
    naive_shock_detector,
    persistence,
    walk_forward_gmm,
    walk_forward_hmm,
)

TRANSMAT = np.array([[0.985, 0.015], [0.04, 0.96]])
MEANS = np.array([[0.08, 0.01, 0.05], [-0.15, 0.02, -0.05]])
COVARS = np.array([np.diag([0.5, 0.15, 0.4]), np.diag([5.0, 0.30, 3.0])])


def simulate_hmm(T: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    states = np.empty(T, dtype=int)
    states[0] = 0
    for t in range(1, T):
        states[t] = rng.choice(2, p=TRANSMAT[states[t - 1]])
    noise = rng.standard_normal((T, 3))
    X = np.empty((T, 3))
    for k in range(2):
        mask = states == k
        X[mask] = MEANS[k] + noise[mask] @ np.linalg.cholesky(COVARS[k]).T
    index = pd.bdate_range("2008-01-02", periods=T)
    return pd.DataFrame(X, index=index, columns=["equity", "rates", "real"]), states


@pytest.fixture(scope="module")
def sample():
    return simulate_hmm(1500, seed=3)


@pytest.fixture(scope="module")
def library_model(sample):
    from hmmlearn.hmm import GaussianHMM

    X, _ = sample
    model = GaussianHMM(2, covariance_type="full", n_iter=200, tol=1e-6, random_state=0).fit(X.to_numpy())
    fit = HMMFit(model.startprob_.copy(), model.transmat_.copy(), model.means_.copy(),
                 np.asarray(model.covars_).copy(), float(model.score(X.to_numpy())))
    return model, fit


# ----------------------------------------------------------------------------- the filter
def test_forward_filter_log_likelihood_equals_the_library_score(sample, library_model):
    X, _ = sample
    model, fit = library_model
    alpha, loglik = forward_filter(log_emission(X, fit), fit.startprob, fit.transmat)
    assert loglik == pytest.approx(model.score(X.to_numpy()), rel=1e-10)
    np.testing.assert_allclose(alpha.sum(axis=1), 1.0, atol=1e-12)


def test_smoothed_posterior_equals_the_library_posterior(sample, library_model):
    X, _ = sample
    model, fit = library_model
    posterior = hmm_posteriors(X, fit)
    np.testing.assert_allclose(posterior["smoothed"], model.predict_proba(X.to_numpy()), atol=1e-8)


def test_filtered_and_smoothed_agree_only_at_the_last_observation(sample, library_model):
    X, _ = sample
    _, fit = library_model
    posterior = hmm_posteriors(X, fit)
    np.testing.assert_allclose(posterior["filtered"][-1], posterior["smoothed"][-1], atol=1e-12)
    gap = np.abs(posterior["filtered"][:-1] - posterior["smoothed"][:-1]).max(axis=1)
    assert gap.max() > 0.2 and (gap > 0.01).mean() > 0.05                 # smoothing does use the future


def test_filtered_probabilities_do_not_depend_on_later_data(sample, library_model):
    X, _ = sample
    _, fit = library_model
    full = hmm_posteriors(X, fit)
    head = hmm_posteriors(X.iloc[:900], fit)["filtered"]
    np.testing.assert_allclose(head, full["filtered"][:900], atol=1e-13)


def test_smoothed_probabilities_do_depend_on_later_data_where_the_state_is_ambiguous(sample, library_model):
    """The control: cut the sample where the filter is least sure and the smoothed value moves."""
    X, _ = sample
    _, fit = library_model
    full = hmm_posteriors(X, fit)
    ambiguity = np.abs(full["filtered"][:, 1] - 0.5)
    candidates = 300 + np.argsort(ambiguity[300:1200])[:10]
    moves = []
    for t in candidates:
        head = hmm_posteriors(X.iloc[:t + 1], fit)["smoothed"]
        moves.append(abs(head[t, 1] - full["smoothed"][t, 1]))
    assert max(moves) > 0.05


def test_fit_recovers_the_simulated_regimes(sample):
    X, states = sample
    fit = fit_hmm(X, n_states=2, n_init=3, seed=1)
    assert fit.covars[1][0, 0] > 3.0 * fit.covars[0][0, 0]                # ordered by equity variance
    assert abs(fit.transmat[0, 0] - TRANSMAT[0, 0]) < 0.03
    assert abs(fit.transmat[1, 1] - TRANSMAT[1, 1]) < 0.05
    filtered = hmm_posteriors(X, fit)["filtered"]
    accuracy = ((filtered[:, 1] > 0.5).astype(int) == states).mean()
    assert accuracy > 0.88


def test_state_order_is_by_variance_whatever_the_starting_labels():
    X, _ = simulate_hmm(1200, seed=5)
    orders = []
    for seed in range(3):
        fit = fit_hmm(X, n_states=2, n_init=1, seed=seed)
        orders.append(fit.covars[1][0, 0] > fit.covars[0][0, 0])
    assert all(orders)


# --------------------------------------------------------------------- walk-forward fitting
SETTINGS = dict(n_states=2, min_train=400, refit_every=200, n_init=2, n_init_refit=1, seed=4)


def test_walk_forward_is_nan_before_the_first_refit_and_a_distribution_after(sample):
    X, _ = sample
    result = walk_forward_hmm(X, **SETTINGS)
    assert result.prob.iloc[:400].isna().all().all()
    np.testing.assert_allclose(result.prob.iloc[400:].sum(axis=1), 1.0, atol=1e-12)
    assert result.first_date == X.index[400]
    assert [w["start"] for w in result.windows] == [400, 600, 800, 1000, 1200, 1400]


def test_walk_forward_probabilities_do_not_depend_on_data_after_the_day():
    X, _ = simulate_hmm(1400, seed=6)
    base = walk_forward_hmm(X, **SETTINGS)
    cut = 1000
    altered = X.copy()
    rng = np.random.default_rng(8)
    altered.iloc[cut:] = 3.0 * rng.standard_normal(altered.iloc[cut:].shape)
    changed = walk_forward_hmm(altered, **SETTINGS)
    pd.testing.assert_frame_equal(base.prob.iloc[:cut], changed.prob.iloc[:cut])
    assert not np.allclose(base.prob.iloc[cut:].to_numpy(), changed.prob.iloc[cut:].to_numpy())


def test_each_refit_is_estimated_on_rows_before_its_window_only():
    X, _ = simulate_hmm(1000, seed=7)
    result = walk_forward_hmm(X, **SETTINGS)
    first = result.windows[0]
    direct = fit_hmm(X.iloc[:first["start"]], 2, SETTINGS["n_init"], 300, 1e-4, SETTINGS["seed"])
    np.testing.assert_allclose(first["fit"].transmat, direct.transmat, atol=1e-9)
    np.testing.assert_allclose(first["fit"].means, direct.means, atol=1e-9)


def test_the_fit_cache_is_keyed_on_content_so_it_cannot_hide_a_dependence_on_the_future():
    X, _ = simulate_hmm(1000, seed=9)
    cache: dict = {}
    first = walk_forward_hmm(X, cache=cache, **SETTINGS)
    n_after_first = len(cache)
    again = walk_forward_hmm(X, cache=cache, **SETTINGS)
    assert len(cache) == n_after_first                                    # identical input: all served from the cache
    pd.testing.assert_frame_equal(first.prob, again.prob)
    altered = X.copy()
    altered.iloc[100] += 5.0                                              # a TRAINING row of every window
    walk_forward_hmm(altered, cache=cache, **SETTINGS)
    assert len(cache) > n_after_first                                     # new content, new fits


def test_gmm_labels_flip_far_more_often_than_hmm_labels():
    X, _ = simulate_hmm(1600, seed=11)
    hmm = walk_forward_hmm(X, **SETTINGS)
    gmm = walk_forward_gmm(X, n_components=2, min_train=400, refit_every=200, n_init=2, seed=4)
    assert gmm.iloc[400:].notna().all().all()
    hmm_labels = hmm.prob.iloc[400:].to_numpy().argmax(axis=1)
    gmm_labels = gmm.iloc[400:].to_numpy().argmax(axis=1)
    run = lambda labels: persistence(pd.Series(labels, index=X.index[400:]))["mean_run_days"]
    assert run(hmm_labels) > 1.5 * run(gmm_labels)


# ----------------------------------------------------------------------------- BOCPD
def _shift_series(kind: str, seed: int = 0, calm: int = 600, after: int = 300):
    rng = np.random.default_rng(seed)
    if kind == "variance":
        values = np.concatenate([rng.normal(0, 1, calm), rng.normal(0, 3, after)])
    else:
        values = np.concatenate([rng.normal(0, 1, calm), rng.normal(3, 1, after)])
    return pd.Series(values, index=pd.bdate_range("2010-01-04", periods=len(values)))


def test_bocpd_with_constant_hazard_puts_exactly_the_hazard_on_a_change_today():
    x = _shift_series("variance")
    result = bocpd(x, hazard_lambda=200.0, burn_in=250, short_runs=(0, 10))
    np.testing.assert_allclose(result.short_mass["le_0"], 1.0 / 200.0, atol=1e-9)
    assert (result.short_mass["le_10"] >= result.short_mass["le_0"]).all()
    assert result.run_length_bins.sum(axis=1).sub(1.0).abs().max() < 1e-9


@pytest.mark.parametrize("kind", ["variance", "mean"])
def test_bocpd_alarms_soon_after_a_shift_and_is_quiet_before_it(kind):
    x = _shift_series(kind)
    result = bocpd(x, hazard_lambda=250.0, burn_in=250, short_runs=(10,))
    mass = result.short_mass["le_10"]
    change = x.index[600]
    quiet = mass.loc[:change - pd.Timedelta(days=1)]
    assert (quiet >= 0.5).mean() < 0.05
    after = mass.loc[change:]
    first_alarm = after[after >= 0.5]
    assert len(first_alarm) > 0
    delay = int(x.index.get_loc(first_alarm.index[0])) - 600
    assert delay <= 12


def test_bocpd_posterior_up_to_a_date_does_not_depend_on_later_data():
    x = _shift_series("variance", seed=2)
    base = bocpd(x, burn_in=250, short_runs=(10,))
    altered = x.copy()
    altered.iloc[700:] = 50.0 * np.random.default_rng(1).standard_normal(len(altered) - 700)
    changed = bocpd(altered, burn_in=250, short_runs=(10,))
    cut = x.index[699]
    np.testing.assert_allclose(base.short_mass.loc[:cut], changed.short_mass.loc[:cut], atol=1e-12)
    np.testing.assert_allclose(base.expected_run_length.loc[:cut], changed.expected_run_length.loc[:cut], atol=1e-9)


def test_alarm_episodes_merge_flicker_and_the_detector_scores_delay_and_false_alarms():
    index = pd.bdate_range("2012-01-02", periods=400)
    alarm = pd.Series(False, index=index)
    alarm.iloc[[50, 51, 54]] = True                       # one episode (gaps below the merge gap): FALSE
    alarm.iloc[[200, 201]] = True                         # starts at the event day plus 0 -> detected, delay 0
    alarm.iloc[[310, 311, 312]] = True                    # event at 300 -> detected with delay 10
    alarm.iloc[[395]] = True                              # false
    episodes = alarm_episodes(alarm, merge_gap=5)
    assert len(episodes) == 4 and episodes[0] == (index[50], index[54])
    events = {"a": index[200].strftime("%Y-%m-%d"), "b": index[300].strftime("%Y-%m-%d"),
              "c": index[150].strftime("%Y-%m-%d")}
    scored = evaluate_detector(alarm, events, search_days=20, merge_gap=5)
    table = scored["events"].set_index("event")
    assert bool(table.loc["a", "detected"]) and table.loc["a", "delay_days"] == 0
    assert bool(table.loc["b", "detected"]) and table.loc["b", "delay_days"] == 10
    assert not bool(table.loc["c", "detected"])           # the episode at 200 is 50 days later: outside its window
    assert scored["n_false_alarms"] == 2                  # episodes at 50 and 395
    assert scored["n_detected"] == 2 and scored["n_events"] == 3


def test_an_alarm_that_starts_before_the_event_is_a_false_alarm_not_a_detection():
    index = pd.bdate_range("2012-01-02", periods=200)
    alarm = pd.Series(False, index=index)
    alarm.iloc[95:105] = True                             # starts 5 days BEFORE the event at 100
    scored = evaluate_detector(alarm, {"e": index[100].strftime("%Y-%m-%d")}, search_days=30)
    assert scored["n_detected"] == 0 and scored["n_false_alarms"] == 1


def test_naive_detector_fires_on_a_volatility_jump_and_not_in_calm():
    rng = np.random.default_rng(3)
    r = pd.Series(np.concatenate([rng.normal(0, 1, 500), rng.normal(0, 3, 30)]),
                  index=pd.bdate_range("2012-01-02", periods=530))
    alarm = naive_shock_detector(r, 5, 250, 2.0)
    assert alarm.iloc[260:500].mean() < 0.08
    assert alarm.iloc[505:530].mean() > 0.5


# ------------------------------------------------------------------ rule-based regimes
def test_bear_state_confirms_at_minus_20_and_ends_at_plus_20_from_the_trough():
    price = pd.Series([100, 110, 120, 100, 96, 95, 80, 90, 95, 96, 97, 110.0],
                      index=pd.bdate_range("2020-01-01", periods=12))
    state = bear_market_state(price, 0.20, 0.20)
    # peak 120; -20% is 96, first reached at index 4; trough 80; +20% is 96, reached at index 9
    assert state.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]


def test_bear_state_is_causal():
    rng = np.random.default_rng(0)
    price = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 600))),
                      index=pd.bdate_range("2015-01-01", periods=600))
    full = bear_market_state(price)
    altered = price.copy()
    altered.iloc[400:] *= 0.3
    pd.testing.assert_series_equal(full.iloc[:400], bear_market_state(altered).iloc[:400])


def _spike_path(seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(np.concatenate([rng.normal(0, 0.007, 800), rng.normal(0, 0.03, 40),
                                     rng.normal(0, 0.007, 300)]),
                     index=pd.bdate_range("2010-01-01", periods=1140))


def test_volatility_regime_flags_a_spike_and_ranks_against_its_own_past():
    calm_share, spike_share = [], []
    for seed in range(25):
        frame = volatility_regime(_spike_path(seed), halflife=21, high_percentile=0.80,
                                  low_percentile=0.20, min_history=252)
        assert frame["percentile"].dropna().between(0, 1).all()
        assert frame["percentile"].iloc[:251].isna().all()
        calm_share.append(frame["high"].iloc[300:700].mean())
        spike_share.append(frame["high"].iloc[815:840].mean())
    assert min(spike_share) > 0.9                                         # always flagged inside the spike
    # In stationary data a percentile flag is on about 20% of days; the volatility is
    # autocorrelated, so any single path wanders, but the average over paths does not.
    assert 0.12 < float(np.mean(calm_share)) < 0.30


def test_volatility_regime_is_causal():
    r = _spike_path(1)
    frame = volatility_regime(r)
    altered = r.copy()
    altered.iloc[900:] *= 10
    pd.testing.assert_frame_equal(frame.iloc[:900], volatility_regime(altered).iloc[:900])


def test_inflation_shock_needs_both_conditions():
    index = pd.bdate_range("2021-01-01", periods=400)
    rng = np.random.default_rng(2)
    stock = pd.Series(rng.normal(0, 0.01, 400), index=index)
    bond = pd.Series(0.5 * stock.to_numpy() + rng.normal(0, 0.003, 400), index=index)    # correlated
    hedge = pd.Series(-0.5 * stock.to_numpy() + rng.normal(0, 0.003, 400), index=index)  # classic hedge
    cpi = pd.Series(0.05, index=index)
    assert inflation_shock(cpi, stock, bond)["shock"].iloc[200:].mean() == 1.0
    assert inflation_shock(cpi, stock, hedge)["shock"].iloc[200:].mean() == 0.0
    assert inflation_shock(cpi * 0.5, stock, bond)["shock"].iloc[200:].mean() == 0.0


def test_liquidity_crisis_needs_a_volatility_spike_and_a_credit_drawdown():
    index = pd.bdate_range("2022-01-03", periods=100)
    credit = pd.Series(0.0, index=index)
    safe = pd.Series(0.0, index=index)
    credit.iloc[60:81] = -0.004                                           # about -8% over 21 days
    spike = pd.Series(0.95, index=index)
    calm = pd.Series(0.40, index=index)
    assert liquidity_crisis(spike, credit, safe)["crisis"].iloc[80] == 1.0
    assert liquidity_crisis(calm, credit, safe)["crisis"].iloc[80] == 0.0
    assert liquidity_crisis(spike, credit * 0.0, safe)["crisis"].iloc[80] == 0.0


# ------------------------------------------------------------------- degenerate fits
def test_a_state_collapsed_onto_one_observation_is_flagged_and_does_not_crash_the_filter():
    degenerate = HMMFit(np.array([0.5, 0.5]), np.array([[0.9, 0.1], [0.1, 0.9]]),
                        np.zeros((2, 3)),
                        np.array([np.eye(3), 0.03 * np.outer([1.0, 0.05, 0.44], [1.0, 0.05, 0.44])]), loglik=0.0)
    healthy = HMMFit(np.array([0.5, 0.5]), np.array([[0.9, 0.1], [0.1, 0.9]]), np.zeros((2, 3)),
                     np.array([np.eye(3), 4.0 * np.eye(3)]), loglik=0.0)
    assert degenerate.is_degenerate and not healthy.is_degenerate
    X = np.random.default_rng(0).normal(size=(50, 3))
    log_b = log_emission(X, degenerate)                       # jitter, not a LinAlgError
    assert np.isfinite(log_b).all()


def test_fit_hmm_prefers_a_sound_start_over_a_degenerate_one_with_a_higher_likelihood(monkeypatch):
    transmat = np.array([[0.9, 0.1], [0.1, 0.9]])
    start = np.array([0.5, 0.5])
    collapsed = np.outer([1.0, 0.05, 0.44], [1.0, 0.05, 0.44])
    bad = HMMFit(start, transmat, np.zeros((2, 3)), np.array([np.eye(3), 0.03 * collapsed]), loglik=-1.0)
    good = HMMFit(start, transmat, np.zeros((2, 3)), np.array([np.eye(3), 4.0 * np.eye(3)]), loglik=-50.0)
    queue = [bad, good, bad]
    monkeypatch.setattr("src.models.regimes._fit_hmmlearn", lambda *args, **kwargs: queue.pop(0))
    chosen = fit_hmm(np.zeros((10, 3)), n_states=2, n_init=3)
    assert not chosen.is_degenerate and chosen.loglik == -50.0


def test_fit_hmm_and_filter_survive_a_single_300_percent_day():
    X, _ = simulate_hmm(900, seed=13)
    X = X.copy()
    X.iloc[500] = [312.0, 15.0, 138.0]                        # a price series spliced to a time-reversed copy of itself
    fit = fit_hmm(X, n_states=2, n_init=3, seed=2)
    posteriors = hmm_posteriors(X, fit)
    assert np.isfinite(posteriors["filtered"]).all() and np.isfinite(posteriors["smoothed"]).all()
