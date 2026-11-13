"""Random-feature ridge (the 'virtue of complexity' model), the triple-barrier target and the fractional-differencing features."""

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge

from src.ml import features as F, models as md, targets as tg
from src.ml.complexity import RandomFeatureRidge
from src.ml.param_schema import ParamError, effective_params

warnings.filterwarnings("ignore")


# ------------------------------------------------------------------------------------------------------------ random-feature ridge
def test_one_draw_is_a_ridge_on_the_sines_and_cosines_of_random_projections():
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(300, 6)), rng.normal(size=300)
    m = RandomFeatureRidge(components=40, bandwidth=1.5, alpha=3.0, draws=1, random_state=4).fit(X, y)
    W = m.projections_[0]
    assert W.shape == (6, 20) and m.complexity_ == pytest.approx(40 / 300)
    Z = np.concatenate([np.sin(X @ W), np.cos(X @ W)], axis=1) / np.sqrt(20)
    ref = Ridge(alpha=3.0).fit(Z, y)
    assert np.allclose(m.predict(X), ref.predict(Z), atol=1e-8)
    assert np.std(X @ W) == pytest.approx(1.5, rel=0.15)                                  # the projections are scaled so their spread is the bandwidth, whatever the number of inputs


def test_many_random_features_approach_the_gaussian_kernel_ridge():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(250, 3))
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] * X[:, 2] + 0.1 * rng.normal(size=250)
    bandwidth, alpha = 1.0, 0.5
    m = RandomFeatureRidge(components=6000, bandwidth=bandwidth, alpha=alpha, draws=1, random_state=0).fit(X, y)
    yc = y - y.mean()
    kernel = KernelRidge(kernel="rbf", gamma=bandwidth ** 2 / (2 * 3), alpha=alpha).fit(X, yc)         # E[z(x)'z(x')] = exp(-|x - x'|^2 sigma^2 / 2) with sigma^2 = bandwidth^2 / d
    Xn = rng.normal(size=(200, 3))
    assert np.corrcoef(m.predict(Xn), kernel.predict(Xn) + y.mean())[0, 1] > 0.98


def test_it_learns_a_nonlinear_signal_a_ridge_cannot_and_more_draws_steady_the_prediction():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(3000, 5))
    f = lambda A: np.abs(A[:, 0]) - 0.8 + np.sin(2 * A[:, 1])                             # noqa: E731
    y = f(X) + 0.5 * rng.normal(size=3000)
    Xn = rng.normal(size=(1500, 5))
    rff = RandomFeatureRidge(components=800, bandwidth=1.5, alpha=1.0, draws=3, random_state=0).fit(X, y)
    assert np.corrcoef(rff.predict(Xn), f(Xn))[0, 1] > 0.9 > abs(np.corrcoef(Ridge(alpha=1.0).fit(X, y).predict(Xn), f(Xn))[0, 1]) + 0.3
    spread = lambda draws: np.std([RandomFeatureRidge(components=60, alpha=1.0, draws=draws, random_state=s).fit(X[:500], y[:500]).predict(Xn[:200]) for s in range(6)], axis=0).mean()   # noqa: E731
    assert spread(8) < 0.6 * spread(1)
    imp = rff.feature_importances_
    assert imp.sum() == pytest.approx(1.0) and set(np.argsort(imp)[-2:]) == {0, 1}        # the fitted function responds to the two inputs that matter


def test_sample_weights_are_those_of_a_weighted_ridge_and_bad_settings_are_refused():
    rng = np.random.default_rng(3)
    X, y, w = rng.normal(size=(200, 4)), rng.normal(size=200), rng.uniform(0.2, 2.0, size=200)
    m = RandomFeatureRidge(components=30, alpha=2.0, draws=1, random_state=1).fit(X, y, sample_weight=w)
    W = m.projections_[0]
    Z = np.concatenate([np.sin(X @ W), np.cos(X @ W)], axis=1) / np.sqrt(15)
    assert np.allclose(m.predict(X), Ridge(alpha=2.0).fit(Z, y, sample_weight=w).predict(Z), atol=1e-8)
    with pytest.raises(ValueError):
        RandomFeatureRidge(components=1).fit(X, y)
    with pytest.raises(ParamError):
        effective_params("rff_ridge", {"bandwidth": 0})
    with pytest.raises(ValueError, match="regression"):
        md.make_model("rff_ridge", {}, "classification", 0)


def test_the_catalogue_model_fits_a_panel_and_survives_a_round_trip():
    rng = np.random.default_rng(5)
    dates = pd.bdate_range("2020-01-01", periods=120)
    idx = pd.MultiIndex.from_product([dates, list("ABCDEF")], names=["date", "asset"])
    X = pd.DataFrame(rng.normal(size=(len(idx), 5)), index=idx, columns=[f"f{i}" for i in range(5)])
    y = pd.Series(0.02 * np.tanh(2 * X["f0"]) + 0.01 * rng.normal(size=len(idx)), index=idx)
    model = md.make_model("rff_ridge", {"components": 200, "alpha": 5.0}, "regression", 1).fit(X, y)
    pred = model.predict(X)
    assert np.corrcoef(pred, y)[0, 1] > 0.5 and model.importance().idxmax() == "f0"
    assert np.allclose(md.Model.from_bytes(model.to_bytes()).predict(X), pred)
    assert "rff_ridge" in md.search_space.__code__.co_consts or md.search_space("rff_ridge")


# ------------------------------------------------------------------------------------------------------------ triple barrier
def frame(path, vol=0.01, warm=70):
    """A price frame whose first ``warm`` days alternate +vol and -vol (so the trailing daily volatility is about ``vol``) and then follow ``path`` (prices relative to the last warm-up price)."""
    r = np.where(np.arange(warm) % 2 == 0, vol, -vol)
    base = 100.0 * np.cumprod(1.0 + r)
    prices = np.concatenate([base, base[-1] * np.asarray(path, dtype=float)])
    p = pd.DataFrame({"A": prices}, index=pd.bdate_range("2021-01-04", periods=len(prices)))
    return p, p.pct_change()


def test_the_first_barrier_touched_decides_whatever_happens_afterwards():
    h, lag, warm = 5, 1, 70
    band = 0.01 * np.sqrt(h)                                                              # about 2.2%: the trailing volatility times sqrt(h)
    up_then_crash, down_then_rally = [1.0, 1.0 + 1.5 * band, 0.8, 0.8, 0.8, 0.8, 0.8], [1.0, 1.0 - 1.5 * band, 1.3, 1.3, 1.3, 1.3, 1.3]
    for path, want in ((up_then_crash, 1.0), (down_then_rally, 0.0)):
        p, r = frame(path)
        out = tg.make_target(f"barrier_{h}", p, r, lag)
        t = warm - 1                                                                      # decision at the last warm-up close; entry one day later at relative price 1.0
        assert out["A"].iloc[t] == want
        assert tg.make_target(f"direction_{h}", p, r, lag)["A"].iloc[t] == 1.0 - want      # the last day alone says the opposite


def test_without_a_touch_the_sign_of_the_return_at_the_horizon_decides_and_an_unfinished_window_has_no_label():
    h, lag, warm = 5, 1, 70
    band = 0.01 * np.sqrt(h)
    drift_up, drift_down = [1.0] + [1.0 + 0.4 * band] * 6, [1.0] + [1.0 - 0.4 * band] * 6
    for path, want in ((drift_up, 1.0), (drift_down, 0.0)):
        p, r = frame(path)
        assert tg.make_target(f"barrier_{h}", p, r, lag)["A"].iloc[warm - 1] == want
    p, r = frame([1.0, 1.0 + 2 * band, 1.0, 1.0])                                         # the barrier is touched on day 1, but days 3 to 5 have not happened yet
    out = tg.make_target(f"barrier_{h}", p, r, lag)
    assert np.isnan(out["A"].iloc[warm - 1]) and out["A"].iloc[:30].isna().all()          # and no label before the volatility exists
    assert tg.is_classification("barrier_5") and tg.parse_target("barrier_10") == ("barrier", 10) and tg.label_end(5, 1) == 6


def test_barrier_labels_do_not_change_when_later_prices_are_appended():
    rng = np.random.default_rng(7)
    p = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, size=(400, 3)), axis=0)), index=pd.bdate_range("2020-01-01", periods=400), columns=list("ABC"))
    full = tg.make_target("barrier_10", p, p.pct_change(), 1)
    part = tg.make_target("barrier_10", p.iloc[:300], p.iloc[:300].pct_change(), 1)
    known = part.notna().to_numpy()
    assert known.sum() > 600 and (full.iloc[:300].to_numpy()[known] == part.to_numpy()[known]).all() and set(np.unique(part.stack())) == {0.0, 1.0}
    assert 0.3 < part.stack().mean() < 0.7


# ------------------------------------------------------------------------------------------------------------ fractional differencing
def test_the_weights_are_the_binomial_series_and_the_orders_0_and_1_are_the_price_and_the_return():
    assert np.allclose(F.fracdiff_weights(1.0, 5), [1, -1, 0, 0, 0, 0]) and np.allclose(F.fracdiff_weights(0.0, 4), [1, 0, 0, 0, 0])
    w = F.fracdiff_weights(0.4, 4)
    assert np.allclose(w, [1, -0.4, -0.12, -0.064, -0.0416])
    rng = np.random.default_rng(0)
    p = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, size=(300, 2)), axis=0)), index=pd.bdate_range("2020-01-01", periods=300), columns=["A", "B"])
    logp = np.log(p)
    assert np.allclose(F.fracdiff(p, 1.0, 20).iloc[20:], logp.diff().iloc[20:])            # order 1: the log return
    assert np.allclose(F.fracdiff(p, 0.0, 20).iloc[20:], (logp - logp.shift(20)).iloc[20:])       # order 0: the log price measured from the window's start
    assert F.fracdiff(p, 0.5, 20).iloc[:20].isna().all().all()


def test_a_fractional_difference_has_no_unit_keeps_memory_and_never_looks_ahead():
    rng = np.random.default_rng(1)
    p = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, size=(700, 1)), axis=0)), index=pd.bdate_range("2019-01-01", periods=700), columns=["A"])
    a = F.fracdiff(p, 0.3)
    assert np.allclose(F.fracdiff(p * 37.0, 0.3).dropna(), a.dropna())                    # a price in another currency or after a split gives the same feature
    changed = p.copy()
    changed.iloc[500:] *= 1.5
    assert np.allclose(F.fracdiff(changed, 0.3).iloc[:500].dropna(), a.iloc[:500].dropna())       # later prices do not reach back
    level = (np.log(p) - np.log(p).shift(126))["A"]
    corr = lambda d: F.fracdiff(p, d)["A"].corr(level)                                    # noqa: E731
    assert corr(0.3) > corr(0.7) > np.log(p)["A"].diff().corr(level)                      # the lower the order, the more of the level it remembers
    holes = p.copy()
    holes.iloc[400] = np.nan
    assert F.fracdiff(holes, 0.5).iloc[400:527].isna().all().all() and F.fracdiff(holes, 0.5).iloc[527:].notna().all().all()


def test_the_features_are_registered_in_a_group_of_their_own_that_the_default_dataset_does_not_use():
    from src.ml import config as mc

    names = [f.name for f in F.FEATURES.values() if f.group == "fracdiff"]
    assert names == ["fracdiff_03", "fracdiff_05", "fracdiff_07"] and "fracdiff" in F.GROUPS and "fracdiff" not in mc.DEFAULTS["dataset"]["feature_groups"]
    assert all(F.FEATURES[n].lookback == 127 and F.FEATURES[n].memory == "finite" for n in names)
