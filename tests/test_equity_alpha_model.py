"""The alpha model: standardise, orthogonalise, weigh by the IC covariance, attribute. Closed forms first, then a world whose truth was planted."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.equity import alpha_model as am
from src.equity.synthetic import simulate_factor_panel

CORR = np.array([[1.0, 0.6, 0.0, 0.0], [0.6, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])


def _world(seed=0, n_assets=200, n_periods=180, corr=None, premia=(0.004, 0.003, 0.0, -0.002), **kw):
    return simulate_factor_panel(n_assets=n_assets, n_periods=n_periods, premia=premia, corr=corr, seed=seed, **kw)


def _corr_by_date(frames: dict[str, pd.DataFrame], t: int) -> np.ndarray:
    X = np.column_stack([f.iloc[t].to_numpy() for f in frames.values()])
    ok = np.isfinite(X).all(axis=1)
    return np.corrcoef(X[ok], rowvar=False)


# ------------------------------------------------------------------------------------------------------------------ the simulator itself
def test_the_simulator_plants_the_premia_and_the_correlation_it_was_asked_for():
    w = _world(0, n_periods=240, corr=CORR)
    # the average premium each factor earned over time is its planted mean (within sampling error)
    assert np.abs(w.premia.mean() - w.mean_premia).max() < 0.0012
    assert abs(am.factor_correlation(w.factors).iloc[0, 1] - 0.6) < 0.03
    assert abs(am.factor_correlation(w.factors).iloc[0, 2]) < 0.03
    with pytest.raises(ValueError):
        simulate_factor_panel(premia=(0.1, 0.1), names=["a"])
    with pytest.raises(ValueError):
        simulate_factor_panel(corr=np.ones((3, 3)) * 0.5, premia=(0.1, 0.1))


# ------------------------------------------------------------------------------------------------------------------ standardise
def test_standardising_removes_the_units_the_level_and_the_outliers():
    w = _world(1)
    f = w.factors["f1"]
    a, b = am.standardize(f), am.standardize(1000.0 * f + 5.0)
    assert np.abs(a - b).max().max() < 1e-9                                 # the units do not matter
    assert np.abs(a.mean(axis=1)).max() < 1e-12 and np.abs(a.std(axis=1) - 1).max() < 1e-12
    spiked = f.copy()
    spiked.iloc[10, 5] = 1e6
    z = am.standardize(spiked)
    assert z.abs().max().max() < 4.5                                        # winsorised before it was scaled, one asset cannot flatten the others
    squashed = am.standardize(spiked, winsorize_quantile=0.0).iloc[10].drop(spiked.columns[5])
    assert squashed.std() < 0.001 and squashed.abs().max() < 0.08                    # without it the other 199 are squashed together (all about -1/sqrt(200)) and cannot be told apart
    inv = pd.DataFrame(True, index=f.index, columns=f.columns)
    inv.iloc[:, :20] = False
    z = am.standardize(f, inv)
    assert z.iloc[:, :20].isna().all().all() and z.iloc[:, 20:].notna().all().all()


# ------------------------------------------------------------------------------------------------------------------ orthogonalise
def test_gram_schmidt_makes_every_date_orthogonal_and_keeps_what_the_factors_span():
    w = _world(2, corr=CORR)
    std = am.standardize_all(w.factors)
    gs = am.gram_schmidt(std)
    names = list(std)
    for t in (0, 50, 120, 179):
        c = _corr_by_date(gs, t)
        assert np.abs(c - np.eye(4)).max() < 1e-10
        assert np.abs(np.array([gs[n].iloc[t].std() for n in names]) - 1).max() < 1e-12
    assert np.abs(gs["f1"] - std["f1"]).max().max() < 1e-12                  # the first factor keeps everything
    # the k-th original is a combination of the first k outputs: nothing was lost
    for t in (3, 77):
        Q = np.column_stack([gs[n].iloc[t].to_numpy() for n in names])
        for k, n in enumerate(names):
            y = std[n].iloc[t].to_numpy()
            X = np.column_stack([np.ones(len(y)), Q[:, : k + 1]])
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            assert np.abs(y - X @ beta).max() < 1e-10, (t, n)


def test_the_order_decides_who_keeps_the_shared_part():
    w = _world(3, corr=CORR)
    std = am.standardize_all(w.factors)
    ab = am.gram_schmidt(std, ["f1", "f2", "f3", "f4"])
    ba = am.gram_schmidt(std, ["f2", "f1", "f3", "f4"])
    assert np.abs(ab["f1"] - std["f1"]).max().max() < 1e-12 and np.abs(ba["f2"] - std["f2"]).max().max() < 1e-12
    assert np.abs(ab["f2"] - std["f2"]).max().max() > 0.5 and np.abs(ba["f1"] - std["f1"]).max().max() > 0.5
    # what is left of f2 after f1 is correlated sqrt(1 - 0.6^2) = 0.8 with the original f2
    dates = range(0, 180, 3)
    corr = np.mean([np.corrcoef(ab["f2"].iloc[t], std["f2"].iloc[t])[0, 1] for t in dates])
    assert abs(corr - 0.8) < 0.03
    # the independent factors are barely touched
    assert np.mean([np.corrcoef(ab["f4"].iloc[t], std["f4"].iloc[t])[0, 1] for t in dates]) > 0.9
    with pytest.raises(ValueError, match="every factor"):
        am.gram_schmidt(std, ["f1", "f2"])
    with pytest.raises(ValueError, match="every factor"):
        am.gram_schmidt(std, ["f1", "f1", "f3", "f4"])


def test_a_factor_that_adds_nothing_comes_out_as_zeros_not_as_noise():
    w = _world(4, corr=CORR, n_periods=40)
    std = am.standardize_all(w.factors)
    both = {"f1": std["f1"], "f2": std["f2"], "sum": std["f1"] + 2.0 * std["f2"]}
    gs = am.gram_schmidt(both)
    assert gs["sum"].abs().max().max() < 1e-9 and gs["sum"].notna().all().all()
    assert np.abs(_corr_by_date({k: gs[k] for k in ("f1", "f2")}, 5) - np.eye(2)).max() < 1e-10


def test_missing_values_are_neutral_or_the_asset_is_dropped():
    w = _world(5, corr=CORR, n_periods=30, n_assets=60)
    std = am.standardize_all(w.factors)
    hole = {k: v.copy() for k, v in std.items()}
    hole["f2"].iloc[:, :6] = np.nan
    zero = am.gram_schmidt(hole, missing="zero")
    drop = am.gram_schmidt(hole, missing="drop")
    assert zero["f2"].iloc[:, :6].notna().all().all() and zero["f1"].iloc[:, :6].notna().all().all()      # still ranked on what is known
    assert drop["f1"].iloc[:, :6].isna().all().all() and drop["f2"].iloc[:, :6].isna().all().all()
    assert np.abs(_corr_by_date(drop, 7) - np.eye(4)).max() < 1e-10                    # the rest are orthogonal
    both_gone = {k: v.copy() for k, v in std.items()}
    for k in both_gone:
        both_gone[k].iloc[:, :3] = np.nan
    assert am.gram_schmidt(both_gone)["f1"].iloc[:, :3].isna().all().all()             # nothing is known about them: stay missing
    sparse = {k: v.copy() for k, v in std.items()}
    for k in sparse:
        sparse[k].iloc[3] = np.nan                                                      # a date with no data stays missing
    assert am.gram_schmidt(sparse)["f1"].iloc[3].isna().all()
    with pytest.raises(ValueError, match="missing"):
        am.gram_schmidt(std, missing="ignore")


def test_symmetric_orthogonalisation_is_orthonormal_closest_and_independent_of_order():
    w = _world(6, corr=CORR)
    std = am.standardize_all(w.factors)
    sy = am.symmetric_orthogonalize(std)
    gs = am.gram_schmidt(std)
    for t in (0, 60, 150):
        assert np.abs(_corr_by_date(sy, t) - np.eye(4)).max() < 1e-10
    rev = am.symmetric_orthogonalize({k: std[k] for k in reversed(list(std))})
    assert max(np.abs(rev[k] - sy[k]).max().max() for k in std) < 1e-9
    # of all orthonormal sets, Loewdin's is the closest to the originals (in total squared distance), so it is at least as close as Gram-Schmidt's
    dist = lambda out: sum(((out[k] - std[k]) ** 2).sum().sum() for k in std)
    assert dist(sy) < dist(gs)
    # and no factor is privileged: every one keeps a similar correlation with its own original (Gram-Schmidt leaves f1 exact and the rest less)
    keep = {k: np.mean([np.corrcoef(sy[k].iloc[t], std[k].iloc[t])[0, 1] for t in range(0, 180, 6)]) for k in std}
    assert min(keep["f1"], keep["f2"]) > 0.85 and keep["f3"] > 0.97 and keep["f4"] > 0.97


def test_the_factor_correlation_matrix_is_the_average_over_dates():
    w = _world(7, corr=CORR, n_periods=60)
    std = am.standardize_all(w.factors)
    c = am.factor_correlation(std)
    manual = np.mean([_corr_by_date(std, t) for t in range(60)], axis=0)
    assert np.abs(c.to_numpy() - manual).max() < 1e-12 and list(c.index) == list(std)
    assert np.abs(am.factor_correlation(am.gram_schmidt(std)) - np.eye(4)).max().max() < 1e-10


# ------------------------------------------------------------------------------------------------------------------ weigh
def test_the_maximum_ir_weights_are_the_inverse_covariance_times_the_means_and_beat_any_other():
    rng = np.random.default_rng(0)
    k = 5
    B = rng.normal(size=(k, k))
    cov = pd.DataFrame(B @ B.T / k * 0.01 + 0.002 * np.eye(k))
    mean = pd.Series(rng.normal(0.03, 0.03, k))
    w = am.max_ir_weights(mean, cov)
    ref = np.linalg.solve(cov.to_numpy(), mean.to_numpy())
    assert np.abs(w.to_numpy() - ref / np.abs(ref).sum()).max() < 1e-6 and abs(w.abs().sum() - 1) < 1e-12
    best = am.best_possible_ir(mean, cov)
    assert abs(am.composite_ir(w, mean, cov) - best) < 1e-6
    assert abs(best - np.sqrt(mean @ np.linalg.solve(cov, mean))) < 1e-6
    random_w = [am.composite_ir(pd.Series(rng.normal(size=k)), mean, cov) for _ in range(2000)]
    assert max(random_w) < best + 1e-9
    # a factor can earn a weight with a flat IC of its own because it hedges the others: that is what the covariance buys
    mean2 = pd.Series([0.05, 0.0])
    cov2 = pd.DataFrame([[1.0, 0.9], [0.9, 1.0]]) * 0.01
    w2 = am.max_ir_weights(mean2, cov2)
    assert w2[1] < -0.3
    assert am.composite_ir(w2, mean2, cov2) > am.composite_ir(pd.Series([1.0, 0.0]), mean2, cov2) * 1.5


def test_non_negative_weights_never_go_short_and_agree_when_nothing_would_be():
    rng = np.random.default_rng(1)
    k = 6
    B = rng.normal(size=(k, k))
    cov = pd.DataFrame(B @ B.T / k * 0.01 + 0.002 * np.eye(k))
    mean = pd.Series([0.05, 0.04, 0.0, -0.03, 0.02, -0.01])
    free, long_only = am.max_ir_weights(mean, cov), am.max_ir_weights(mean, cov, nonnegative=True)
    assert (long_only >= 0).all() and abs(long_only.sum() - 1) < 1e-9
    assert am.composite_ir(long_only, mean, cov) <= am.composite_ir(free, mean, cov) + 1e-9
    others = [am.composite_ir(pd.Series(np.abs(rng.normal(size=k))), mean, cov) for _ in range(2000)]
    assert max(others) <= am.composite_ir(long_only, mean, cov) + 1e-9           # nothing non-negative does better
    good_mean = pd.Series(np.full(k, 0.03))
    diag = pd.DataFrame(np.diag(np.linspace(0.005, 0.02, k)))
    assert np.abs(am.max_ir_weights(good_mean, diag, nonnegative=True) - am.max_ir_weights(good_mean, diag)).max() < 1e-6
    assert (am.max_ir_weights(pd.Series([-0.01, -0.02]), pd.DataFrame(np.eye(2)), nonnegative=True) == 0).all()          # nothing earns: no weights, not a short book


def test_for_orthonormal_factors_the_ic_of_the_composite_is_w_ic_over_the_norm():
    w = _world(8, corr=CORR, n_periods=100)
    sy = am.symmetric_orthogonalize(am.standardize_all(w.factors))
    ic = am.information_coefficients(sy, w.forward, "pearson")
    weights = pd.Series([0.5, -0.2, 0.1, 0.7], index=ic.columns)
    composite = am.combine_factors(sy, weights)
    got = am.information_coefficients({"c": composite}, w.forward, "pearson")["c"]
    want = (ic * weights.to_numpy()).sum(axis=1) / np.linalg.norm(weights.to_numpy())
    assert np.abs(got - want).max() < 1e-12
    mean, cov = am.ic_moments(ic)
    assert abs(got.mean() / got.std() - am.composite_ir(weights, mean, cov)) < 1e-9          # and so the information ratio is w'mu / sqrt(w' Sigma w)


def test_the_ic_covariance_can_be_shrunk_toward_its_diagonal():
    w = _world(9, corr=CORR, n_periods=80)
    ic = am.information_coefficients(am.standardize_all(w.factors), w.forward)
    m0, c0 = am.ic_moments(ic)
    m1, c1 = am.ic_moments(ic, shrink=1.0)
    mh, ch = am.ic_moments(ic, shrink=0.5)
    assert (m0 == m1).all() and np.abs(np.diag(c0) - np.diag(c1)).max() < 1e-15
    assert np.abs(c1.to_numpy() - np.diag(np.diag(c1.to_numpy()))).max() == 0
    assert np.abs(ch.to_numpy() - 0.5 * (c0.to_numpy() + c1.to_numpy())).max() < 1e-15
    single = am.ic_moments(ic[["f1"]])
    assert single[1].shape == (1, 1)


def test_combining_factors_is_scale_free_and_follows_time_varying_weights():
    w = _world(10, corr=CORR, n_periods=30, n_assets=80)
    std = am.standardize_all(w.factors)
    base = pd.Series([0.4, 0.3, 0.2, 0.1], index=list(std))
    a, b = am.combine_factors(std, base), am.combine_factors(std, 25.0 * base)
    assert np.abs(a - b).max().max() < 1e-12 and np.abs(a.std(axis=1) - 1).max() < 1e-12
    only_first = am.combine_factors(std, pd.Series([1.0, 0, 0, 0], index=list(std)))
    assert np.abs(only_first - std["f1"]).max().max() < 1e-12
    moving = pd.DataFrame({"f1": [1.0, 0.0], "f2": [0.0, 1.0], "f3": 0.0, "f4": 0.0}, index=[std["f1"].index[5], std["f1"].index[15]])
    c = am.combine_factors(std, moving)
    assert c.iloc[:5].isna().all().all()                                             # no weights yet, no alpha
    assert np.abs(c.iloc[5:15] - std["f1"].iloc[5:15]).max().max() < 1e-12 and np.abs(c.iloc[15:] - std["f2"].iloc[15:]).max().max() < 1e-12
    holey = {k: v.copy() for k, v in std.items()}
    holey["f1"].iloc[:, :4] = np.nan
    assert am.combine_factors(holey, base).iloc[:, :4].notna().all().all()           # an asset missing one factor is ranked on the others


# ------------------------------------------------------------------------------------------------------------------ walk-forward
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_walk_forward_model_finds_the_planted_signs_and_beats_equal_weights_out_of_sample(seed):
    w = _world(seed, corr=CORR)
    names = list(w.factors)
    model = am.optimal_alpha(w.factors, w.forward, horizon=1, window=36, min_obs=12, shrink=0.3, orthogonalize="gram_schmidt")
    late = model.weights.iloc[60:].mean()
    assert late["f1"] > 0.1 and late["f4"] < -0.1                             # the premium factor is bought, the one that pays the other way is sold
    assert abs(late["f3"]) < min(abs(late["f1"]), abs(late["f4"]))           # the factor with no premium gets less weight than the ones with one
    ic_model = am.information_coefficients({"a": model.alpha}, w.forward, "pearson")["a"].iloc[48:]
    equal = am.combine_factors(am.gram_schmidt(am.standardize_all(w.factors)), pd.Series(0.25, index=names))
    ic_equal = am.information_coefficients({"a": equal}, w.forward, "pearson")["a"].iloc[48:]
    assert ic_model.mean() > ic_equal.mean() + 0.02
    assert ic_model.mean() / ic_model.std() > ic_equal.mean() / ic_equal.std()
    assert model.expected_ir.iloc[60:].between(0.0, 5.0).all()
    assert list(model.factors) == names and model.alpha.shape == w.forward.shape


def test_before_enough_ics_have_matured_the_weights_are_equal_and_after_that_they_are_not():
    w = _world(11, corr=CORR, n_periods=60)
    for horizon in (1, 3):
        model = am.optimal_alpha(w.factors, w.forward, horizon=horizon, min_obs=12)
        first_real = 12 + horizon - 1                                         # row i uses the ICs of rows 0 .. i - horizon, so 12 of them exist from i = 11 + horizon
        assert (model.weights.iloc[:first_real] == 0.25).all().all()
        assert not (model.weights.iloc[first_real] == 0.25).all()
        assert model.expected_ir.iloc[:first_real].isna().all() and model.expected_ir.iloc[first_real:].notna().all()


@pytest.mark.parametrize("horizon", [1, 3])
def test_the_alpha_at_a_date_never_depends_on_returns_that_had_not_ended(horizon):
    w = _world(12, corr=CORR, n_periods=80)
    j = 50
    base = am.optimal_alpha(w.factors, w.forward, horizon=horizon, min_obs=12)
    changed = w.forward.copy()
    changed.iloc[j:] = -changed.iloc[j:] + 0.5 * np.random.default_rng(0).normal(size=changed.iloc[j:].shape)
    other = am.optimal_alpha(w.factors, changed, horizon=horizon, min_obs=12)
    safe = j + horizon                                                         # rows before this one use only the ICs of rows < j
    assert base.alpha.iloc[:safe].equals(other.alpha.iloc[:safe]) and base.weights.iloc[:safe].equals(other.weights.iloc[:safe])
    assert np.abs(base.alpha.iloc[safe + 5:] - other.alpha.iloc[safe + 5:]).max().max() > 0.1          # and the test would have noticed if they did


def test_the_non_negative_option_leaves_a_factor_out_instead_of_selling_it():
    w = _world(13, corr=CORR)
    free = am.optimal_alpha(w.factors, w.forward, orthogonalize=True)
    long_only = am.optimal_alpha(w.factors, w.forward, orthogonalize=True, nonnegative=True)
    assert (long_only.weights >= -1e-12).all().all()
    assert free.weights["f4"].iloc[60:].mean() < -0.1 and long_only.weights["f4"].iloc[60:].mean() < 0.05


def test_the_orthogonalisation_options_and_the_investable_mask():
    w = _world(14, corr=CORR, n_periods=60, n_assets=80)
    sym = am.optimal_alpha(w.factors, w.forward, orthogonalize="symmetric")
    gs = am.optimal_alpha(w.factors, w.forward, orthogonalize="gram_schmidt", order=["f2", "f1", "f3", "f4"])
    raw = am.optimal_alpha(w.factors, w.forward)
    assert np.abs(_corr_by_date(sym.factors, 30) - np.eye(4)).max() < 1e-10 and np.abs(_corr_by_date(gs.factors, 30) - np.eye(4)).max() < 1e-10
    assert np.abs(_corr_by_date(raw.factors, 30) - np.eye(4)).max() > 0.3
    inv = pd.DataFrame(True, index=w.forward.index, columns=w.forward.columns)
    inv.iloc[:, :10] = False
    masked = am.optimal_alpha(w.factors, w.forward, investable=inv)
    assert masked.alpha.iloc[:, :10].isna().all().all() and masked.alpha.iloc[20:, 10:].notna().all().all()
    with pytest.raises(ValueError, match="orthogonalize"):
        am.optimal_alpha(w.factors, w.forward, orthogonalize="nope")


# ------------------------------------------------------------------------------------------------------------------ attribute
def test_fama_macbeth_separates_a_factor_that_earns_from_a_proxy_that_only_resembles_it():
    # the second factor is 0.8 correlated with the first and has no premium of its own: alone it looks good, with the first held fixed it earns nothing
    w = simulate_factor_panel(n_assets=300, n_periods=240, premia=(0.004, 0.0), corr=[[1.0, 0.8], [0.8, 1.0]], seed=21, names=["real", "proxy"])
    table = am.marginal_contributions(w.factors, w.realized)
    assert table.loc["real", "t"] > 4 and table.loc["real", "slope"] > 0.002
    assert abs(table.loc["proxy", "t"]) < 3 and abs(table.loc["proxy", "slope"]) < 0.0012
    assert table.loc["proxy", "mean_ic_alone"] > 0.5 * table.loc["real", "mean_ic_alone"] > 0.01      # stand-alone, the proxy looks like most of the real one
    assert set(table.columns) == {"slope", "t", "ir", "hit_rate", "mean_ic_alone"} and table["hit_rate"].between(0, 1).all()
    # orthogonalising with the real factor first leaves the proxy only what is new in it, which has no IC and the same t-statistic as before ...
    gs = am.marginal_contributions(w.factors, w.realized, orthogonalize=True, order=["real", "proxy"])
    assert abs(gs.loc["proxy", "t"] - table.loc["proxy", "t"]) < 0.3 and abs(gs.loc["proxy", "mean_ic_alone"]) < 0.01
    # ... but the order decides who is credited with the shared part: put the proxy first and it takes the credit, which the multivariate regression never gives it
    swapped = am.marginal_contributions(w.factors, w.realized, orthogonalize=True, order=["proxy", "real"])
    assert swapped.loc["proxy", "t"] > 5 and swapped.loc["real", "mean_ic_alone"] < 0.7 * table.loc["real", "mean_ic_alone"]


def test_fama_macbeth_slopes_are_in_return_units_and_recover_the_planted_premia():
    w = simulate_factor_panel(n_assets=300, n_periods=240, premia=(0.004, -0.002, 0.0), seed=22)
    table = am.marginal_contributions(w.factors, w.realized)
    # the factors are standardised, so a slope is the return of one standard deviation of exposure
    assert abs(table.loc["f1", "slope"] - 0.004) < 0.0012 and abs(table.loc["f2", "slope"] + 0.002) < 0.0012 and abs(table.loc["f3", "slope"]) < 0.0012
    assert table.loc["f2", "t"] < -2 and abs(table.loc["f3", "t"]) < 3
