"""Economic-outlook strategies: what each state has paid is learned from matured history only, a planted effect is found, and a null leaves the book flat."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.framework import MODELS, bundle_from_prices, load_library
from src.framework.validate import check_causality
from src.strategies.economic_outlook import CREDIT_STATES, CURVE_STATES, conditional_tstat

load_library()


# ------------------------------------------------------------------------------------------------------------------------------- the learning step
def _blocks(n: int, block: int, k: int = 2) -> pd.Series:
    idx = pd.bdate_range("2005-01-03", periods=n)
    return pd.Series((np.arange(n) // block) % k, index=idx, dtype=float)


def test_conditional_tstat_finds_the_planted_effect_of_each_state_and_is_nan_until_there_is_evidence():
    n = 4000
    state = _blocks(n, 200)
    rng = np.random.default_rng(1)
    # next-day return depends on today's state: +0.2% after state 1, -0.2% after state 0
    mean = np.where(state.shift(1).fillna(0).to_numpy() == 1, 0.002, -0.002)
    r = pd.DataFrame({"X": mean + rng.normal(0, 0.01, n), "NOISE": rng.normal(0, 0.01, n)}, index=state.index)
    t = conditional_tstat(state, r, horizon=5, min_obs=300)
    late = t.iloc[2500:]
    assert (late.loc[state.iloc[2500:] == 1, "X"] > 2).mean() > 0.9
    assert (late.loc[state.iloc[2500:] == 0, "X"] < -2).mean() > 0.9
    assert late["NOISE"].abs().max() < 4.5
    assert t.iloc[:250].isna().all().all()                                                                   # fewer than min_obs matured observations: nothing yet


def test_conditional_tstat_uses_only_pairs_that_have_matured():
    n, h = 1500, 21
    state = _blocks(n, 90, 3)
    rng = np.random.default_rng(2)
    r = pd.DataFrame(rng.normal(0, 0.01, (n, 2)), index=state.index, columns=["A", "B"])
    cut = 1000
    r2 = r.copy()
    r2.iloc[cut + 1:] = rng.normal(0, 0.05, r2.iloc[cut + 1:].shape)                                          # the future changes completely after the cut ...
    a, b = conditional_tstat(state, r, h, 100), conditional_tstat(state, r2, h, 100)
    pd.testing.assert_frame_equal(a.iloc[:cut + 1], b.iloc[:cut + 1])                                        # ... the past statistics do not notice
    assert not a.iloc[cut + 1:].equals(b.iloc[cut + 1:])


def test_conditional_tstat_standard_error_allows_for_the_overlap_of_the_windows():
    """With a constant true daily mean the same information is in one-day and in 21-day windows, so the overlap-corrected t-statistics agree; treating the 21-day windows as independent
    would inflate it by sqrt(21)."""
    n = 6000
    state = pd.Series(0.0, index=pd.bdate_range("2000-01-03", periods=n))
    rng = np.random.default_rng(3)
    r = pd.DataFrame({"X": 0.0004 + rng.normal(0, 0.01, n)}, index=state.index)
    t1, t21 = conditional_tstat(state, r, 1, 500)["X"].iloc[-1], conditional_tstat(state, r, 21, 500)["X"].iloc[-1]
    assert t1 > 0 and t21 > 0 and t21 / t1 == pytest.approx(1.0, rel=0.3)


def test_conditional_tstat_rejects_a_zero_horizon():
    with pytest.raises(ValueError):
        conditional_tstat(_blocks(100, 10), pd.DataFrame({"A": np.zeros(100)}, index=_blocks(100, 10).index), 0)


# ------------------------------------------------------------------------------------------------------------------------------- yield-curve strategy
def _curve_world(n: int = 3600, seed: int = 5):
    """Yields that trend for 120 days at a time (level and slope in different directions); BOND earns +/-0.15% a day after bull/bear states, NOISE earns nothing."""
    idx = pd.bdate_range("2004-01-05", periods=n)
    rng = np.random.default_rng(seed)
    phase = np.arange(n) // 120
    level_dir = np.where(rng.random(phase.max() + 1) < 0.5, -1.0, 1.0)[phase]
    slope_dir = np.where(rng.random(phase.max() + 1) < 0.5, -1.0, 1.0)[phase]
    d2 = np.cumsum(level_dir * 0.01 - 0.5 * slope_dir * 0.01 + rng.normal(0, 0.002, n))
    d10 = np.cumsum(level_dir * 0.01 + 0.5 * slope_dir * 0.01 + rng.normal(0, 0.002, n))
    macro = pd.DataFrame({"DGS2": 4.0 + d2, "DGS10": 5.0 + d10}, index=idx)
    base = pd.DataFrame(rng.normal(0, 0.008, (n, 2)), index=idx, columns=["BOND", "NOISE"])
    stub = bundle_from_prices(100 * (1 + base).cumprod(), asset_class={"BOND": "rates", "NOISE": "equity"}, macro=macro, name="curve")
    model = MODELS.create("curve_quadrant", min_obs=400, tstat=2.0)
    state = model.states(stub)
    bull = (state.isin([0.0, 1.0])).astype(float).where(state.notna())                                       # the level fell
    effect = (0.0015 * np.where(bull == 1.0, 1.0, -1.0)).astype(float)
    effect = pd.Series(np.where(state.notna(), effect, 0.0), index=idx).shift(1).fillna(0.0)
    returns = base.copy()
    returns["BOND"] += effect
    return bundle_from_prices(100 * (1 + returns).cumprod(), asset_class={"BOND": "rates", "NOISE": "equity"}, macro=macro, name="curve"), model, bull


def test_curve_quadrant_learns_that_bonds_rise_when_yields_fall_and_leaves_an_unrelated_asset_alone():
    b, model, bull = _curve_world()
    s = model.score(b)
    late = s.iloc[2200:]
    assert late.loc[bull.iloc[2200:] == 1.0, "BOND"].mean() > 0.5                                             # after the level has fallen: long
    assert late.loc[bull.iloc[2200:] == 0.0, "BOND"].mean() < -0.5                                            # after it has risen: short
    assert late["NOISE"].abs().mean() < 0.15
    assert s.iloc[:300].isna().all().all()                                                                   # no state has min_obs matured observations yet
    names = model.named_states(b)
    assert set(names.unique()) <= set(CURVE_STATES) | {"unknown"}


def test_curve_quadrant_is_causal_and_names_what_it_needs():
    b, model, _ = _curve_world(n=2600)
    assert check_causality(model, b, cutoff=b.index[2000])["ok"]
    no_macro = bundle_from_prices(100 * (1 + pd.DataFrame(np.random.default_rng(0).normal(0, 0.01, (500, 2)), index=pd.bdate_range("2015-01-05", periods=500), columns=["A", "B"])).cumprod(), name="x")
    with pytest.raises(KeyError, match="macro series"):
        MODELS.create("curve_quadrant").score(no_macro)


def test_curve_quadrant_restricts_itself_to_the_assets_it_is_given():
    b, _, _ = _curve_world(n=2600)
    s = MODELS.create("curve_quadrant", min_obs=400, tstat=1.0, assets=("BOND",)).score(b)
    assert s["NOISE"].isna().all() and s["BOND"].notna().any()


@pytest.mark.parametrize("kwargs", [{"window": 2}, {"horizon": 0}, {"min_obs": 10}, {"tstat": -1.0}, {"scale": 0.0}])
def test_the_learned_state_models_reject_nonsense(kwargs):
    for name in ("curve_quadrant", "credit_cycle_rotation"):
        with pytest.raises(ValueError):
            MODELS.create(name, **kwargs)


# ------------------------------------------------------------------------------------------------------------------------------- credit strategy
def _credit_world(n: int = 3600, seed: int = 9, macro: bool = False):
    """Credit beats Treasuries for 250 days and lags for 250, repeatedly; the risky asset RISK earns +/-0.12% a day the day after those phases, TREAS earns nothing."""
    idx = pd.bdate_range("2004-01-05", periods=n)
    rng = np.random.default_rng(seed)
    cycle = ((np.arange(n) // 250) % 2) == 0                                                                  # True: credit outperforming (spreads tightening)
    credit = rng.normal(0, 0.003, n) + np.where(cycle, 0.0008, -0.0008)
    base = pd.DataFrame({"HYG": credit, "IEF": rng.normal(0, 0.003, n), "RISK": rng.normal(0, 0.008, n)}, index=idx)
    classes = {"HYG": "credit", "IEF": "rates", "RISK": "equity"}
    frame = None
    if macro:
        frame = pd.DataFrame({"BAA10Y": 2.5 - np.cumsum(np.where(cycle, 0.004, -0.004)) * 0.0 + np.where(cycle, -1.0, 1.0) * np.minimum(np.arange(n) % 250, 125) * 0.004}, index=idx)
    stub = bundle_from_prices(100 * (1 + base).cumprod(), asset_class=classes, macro=frame, name="credit")
    model = MODELS.create("credit_cycle_rotation", min_obs=400, tstat=2.0)
    state = model.states(stub)
    widening = (state >= 2).astype(float).where(state.notna())                                               # codes 2 and 3: spreads widening
    effect = pd.Series(np.where(widening == 1.0, -0.0012, 0.0012), index=idx).where(state.notna(), 0.0).shift(1).fillna(0.0)
    returns = base.copy()
    returns["RISK"] += effect
    return bundle_from_prices(100 * (1 + returns).cumprod(), asset_class=classes, macro=frame, name="credit"), model, widening


def test_credit_cycle_rotation_learns_that_risk_pays_when_spreads_tighten_and_loses_when_they_widen():
    b, model, widening = _credit_world()
    s = model.score(b)
    late = s.iloc[2400:]
    assert late.loc[widening.iloc[2400:] == 0.0, "RISK"].mean() > 0.5
    assert late.loc[widening.iloc[2400:] == 1.0, "RISK"].mean() < -0.5
    assert late["IEF"].abs().mean() < 0.2
    assert set(model.named_states(b).unique()) <= set(CREDIT_STATES) | {"unknown"}


def test_credit_cycle_rotation_uses_the_baa_spread_when_the_bundle_has_one_and_is_causal():
    b, model, _ = _credit_world(n=2600, macro=True)
    assert model.spread(b).equals(b.macro["BAA10Y"].reindex(b.index).ffill())
    assert check_causality(model, b, cutoff=b.index[2000])["ok"]
    assert check_causality(_credit_world(n=2600)[1], _credit_world(n=2600)[0], cutoff=b.index[2000])["ok"]    # and the price stand-in too


def test_credit_cycle_rotation_needs_a_spread_or_credit_and_treasury_assets():
    r = pd.DataFrame(np.random.default_rng(1).normal(0, 0.01, (600, 2)), index=pd.bdate_range("2015-01-05", periods=600), columns=["A", "B"])
    with pytest.raises(KeyError, match="asset of class"):
        MODELS.create("credit_cycle_rotation").score(bundle_from_prices(100 * (1 + r).cumprod(), name="x"))
