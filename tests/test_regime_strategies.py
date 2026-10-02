"""Permutation tests for persistent labels, and the regime-aware portfolio rules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.portfolio.regime_aware import blend_books, derisk_overlay, full_sample_gate, gate_book, training_gate
from src.validation.permutation import (
    _all_shift_statistics,
    _all_shift_t_statistics,
    circular_shift_test,
    lagged_pair,
    shuffle_test,
)


# ------------------------------------------------------------------ the permutation machinery
def test_fft_shift_statistics_equal_a_brute_force_loop():
    rng = np.random.default_rng(0)
    v = rng.normal(size=61)
    f = rng.random(61) < 0.3
    stat, n_in, n_out = _all_shift_statistics(v, f)
    for k in (0, 1, 7, 30, 60):
        shifted = np.roll(f, k)                      # flag moved forward by k: shifted[t] = f[t - k]
        brute = v[shifted].mean() - v[~shifted].mean()
        assert stat[k] == pytest.approx(brute, abs=1e-12)
    assert n_in == f.sum() and n_out == (~f).sum()


def test_fft_welch_t_statistics_equal_a_brute_force_loop():
    rng = np.random.default_rng(1)
    v = rng.normal(size=70)
    f = rng.random(70) < 0.35
    t_stat = _all_shift_t_statistics(v, f)
    for k in (0, 3, 20, 69):
        shifted = np.roll(f, k)
        a, b = v[shifted], v[~shifted]
        brute = (a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        assert t_stat[k] == pytest.approx(brute, abs=1e-10)


def _rare_stress_flag(T: int, rng, p_enter: float = 0.01, p_stay: float = 0.95) -> np.ndarray:
    flag = np.zeros(T, dtype=bool)
    for t in range(1, T):
        flag[t] = (rng.random() < p_stay) if flag[t - 1] else (rng.random() < p_enter)
    return flag


def test_studentising_is_what_keeps_the_test_honest_for_rare_high_variance_regimes():
    """The null is TRUE (mean zero in both states) but the regime is rare and six times as volatile.
    The plain difference of means has far more sampling error on the real flag than on shifted flags,
    so its shift distribution understates the noise and it 'finds' effects that are not there."""
    rng = np.random.default_rng(0)
    plain, studentised = [], []
    for rep in range(150):
        flag = _rare_stress_flag(1500, rng)
        if flag.sum() < 30:
            continue
        returns = np.where(flag, 3.0, 0.5) * rng.standard_normal(1500)
        out = circular_shift_test(returns[1:], flag[:-1], n_shifts=400, min_shift=40, seed=rep)
        plain.append(out["p_value_difference"])
        studentised.append(out["p_value"])
    assert np.mean(np.array(plain) < 0.05) > 0.20             # grossly oversized (about 37% in this design)
    assert np.mean(np.array(studentised) < 0.05) < 0.10       # about 4%


def _persistent_flag(T: int, rng, stay: float = 0.985) -> np.ndarray:
    flag = np.empty(T, dtype=bool)
    flag[0] = rng.random() < 0.5
    for t in range(1, T):
        flag[t] = flag[t - 1] if rng.random() < stay else not flag[t - 1]
    return flag


def _persistent_outcome(T: int, rng, phi: float = 0.98) -> np.ndarray:
    """A slowly wandering series, standing in for volatility clustering."""
    out = np.empty(T)
    out[0] = 0.0
    shocks = rng.standard_normal(T)
    for t in range(1, T):
        out[t] = phi * out[t - 1] + 0.2 * shocks[t]
    return out


def test_circular_shift_is_correctly_sized_and_the_naive_shuffle_is_not():
    """Independent persistent label and persistent outcome: no relationship exists, so a 5% test
    should reject about 5% of the time. Shuffling the label ignores its persistence and rejects far
    more often."""
    rng = np.random.default_rng(1)
    circular, naive = [], []
    for rep in range(120):
        flag = _persistent_flag(1200, rng)
        outcome = _persistent_outcome(1200, rng)
        circular.append(circular_shift_test(outcome, flag, n_shifts=400, min_shift=40, seed=rep)["p_value"])
        naive.append(shuffle_test(outcome, flag, n_permutations=200, seed=rep)["p_value"])
    circular_rate = float(np.mean(np.array(circular) < 0.05))
    naive_rate = float(np.mean(np.array(naive) < 0.05))
    assert circular_rate < 0.13                      # nominal 5%, finite-sample slack
    assert naive_rate > 0.35                         # grossly oversized


def test_circular_shift_finds_a_planted_regime_effect():
    rng = np.random.default_rng(2)
    flag = _persistent_flag(1500, rng)
    outcome = 0.15 * flag + rng.standard_normal(1500) * 0.5
    result = circular_shift_test(outcome, flag, n_shifts=1000, min_shift=40, seed=3)
    assert result["p_value"] < 0.01
    assert result["mean_in"] > result["mean_out"] and result["statistic"] > 0.08


def test_circular_shift_refuses_degenerate_inputs():
    assert circular_shift_test(np.arange(50.0), np.ones(50, dtype=bool)) == {}
    assert circular_shift_test(np.arange(50.0), np.arange(50) % 2 == 0, min_shift=63) == {}


def test_lagged_pair_aligns_the_state_at_t_with_the_outcome_on_the_next_day():
    index = pd.bdate_range("2020-01-01", periods=6)
    flag = pd.Series([1, 0, 1, 0, 1, 0], index=index, dtype=float)
    outcome = pd.Series(np.arange(6, dtype=float), index=index)
    f, o = lagged_pair(flag, outcome, lag=1)
    assert o.tolist() == [1, 2, 3, 4, 5] and f.tolist() == [1, 0, 1, 0, 1]


# ------------------------------------------------------------------------- overlays and gate
def _books():
    index = pd.bdate_range("2020-01-01", periods=5)
    ew = pd.DataFrame(0.25, index=index, columns=["A", "B", "C", "SHY"])
    rp = pd.DataFrame({"A": 0.1, "B": 0.2, "C": 0.3, "SHY": 0.4}, index=index)
    return ew, rp


def test_derisk_overlay_moves_exactly_the_stated_share_into_cash_and_keeps_gross_fixed():
    ew, _ = _books()
    p = pd.Series([0.0, 0.5, 1.0, np.nan, 0.2], index=ew.index)
    out = derisk_overlay(ew, p, cash="SHY", derisk_max=0.5)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-14)
    assert out.loc[ew.index[0]].tolist() == [0.25] * 4                    # p = 0: untouched
    assert out.loc[ew.index[2], "A"] == pytest.approx(0.125)              # p = 1: half the risky weight goes
    assert out.loc[ew.index[2], "SHY"] == pytest.approx(0.25 * 0.5 + 0.5)
    assert out.loc[ew.index[3]].tolist() == [0.25] * 4                    # undefined probability = no information


def test_derisk_overlay_leaves_dates_without_a_book_empty():
    ew, _ = _books()
    ew.iloc[0] = np.nan
    out = derisk_overlay(ew, pd.Series(1.0, index=ew.index), "SHY", 0.5)
    assert out.iloc[0].isna().all() and out.iloc[1].notna().all()


def test_blend_books_is_a_convex_combination():
    ew, rp = _books()
    p = pd.Series([0.0, 1.0, 0.3, 0.5, 0.0], index=ew.index)
    out = blend_books(ew, rp, p)
    np.testing.assert_allclose(out.iloc[0], ew.iloc[0])
    np.testing.assert_allclose(out.iloc[1], rp.iloc[1])
    np.testing.assert_allclose(out.iloc[2], 0.7 * ew.iloc[2] + 0.3 * rp.iloc[2])
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-14)


def test_gate_book_scales_rows_and_defaults_to_open():
    ew, _ = _books()
    gate = pd.Series([1.0, 0.0, np.nan, 1.0, 0.0], index=ew.index)
    out = gate_book(ew, gate)
    assert out.sum(axis=1).tolist() == [1.0, 0.0, 1.0, 1.0, 0.0]


def _gate_fixture():
    """20 days, one window starting at row 12; label alternates, so each state has known pairs."""
    index = pd.bdate_range("2021-01-01", periods=20)
    labels = np.arange(20) % 2
    alpha = np.zeros((20, 2))
    alpha[np.arange(20), labels] = 1.0
    windows = [{"start": 12, "end": 20, "alpha": alpha}]
    # return on day u: positive when the label two days earlier was 0, negative when it was 1
    returns = pd.Series([0.01 if (u - 2) % 2 == 0 else -0.01 for u in range(20)], index=index)
    return index, windows, returns


def test_training_gate_closes_states_that_lost_money_on_the_training_sample():
    index, windows, returns = _gate_fixture()
    gate, log = training_gate(returns, windows, index, min_days=3, horizon=2)
    assert gate.iloc[:12].isna().all()
    labels = windows[0]["alpha"].argmax(axis=1)
    expected = np.where(labels[12:] == 0, 1.0, 0.0)                       # state 1 had negative training mean
    np.testing.assert_array_equal(gate.iloc[12:].to_numpy(), expected)
    by_state = log.set_index("state")
    assert by_state.loc[0, "mean_daily_return"] > 0 > by_state.loc[1, "mean_daily_return"]
    assert bool(by_state.loc[0, "gate_open"]) and not bool(by_state.loc[1, "gate_open"])


def test_training_gate_stays_open_when_a_state_has_too_little_evidence():
    index, windows, returns = _gate_fixture()
    gate, log = training_gate(returns, windows, index, min_days=50, horizon=2)
    assert (gate.iloc[12:] == 1.0).all()                                  # never enough pairs: no evidence, no gate


def test_training_gate_ignores_returns_not_yet_realised_at_the_refit():
    index, windows, returns = _gate_fixture()
    base, _ = training_gate(returns, windows, index, min_days=3, horizon=2)
    altered = returns.copy()
    altered.iloc[12:] = 5.0 * np.random.default_rng(0).standard_normal(8)  # everything from the refit on
    changed, _ = training_gate(altered, windows, index, min_days=3, horizon=2)
    pd.testing.assert_series_equal(base, changed)


def test_training_gate_uses_returns_up_to_the_last_training_row_and_no_later():
    index, windows, returns = _gate_fixture()
    # make the LAST legal pair (t = 9, return on day 11) decisive for state 1 and check it is used
    altered = returns.copy()
    altered.iloc[:12] = 0.01
    altered.iloc[11] = -10.0                                               # pairs with label[9] = 1
    _, log = training_gate(altered, windows, index, min_days=2, horizon=2)
    state_one = log.set_index("state").loc[1]
    assert state_one["mean_daily_return"] < 0 and not bool(state_one["gate_open"])


def test_full_sample_gate_is_the_look_ahead_control_it_claims_to_be():
    """The first day's gate changes when only LATE returns change: the whole history decides every day."""
    index, windows, returns = _gate_fixture()
    probabilities = pd.DataFrame(windows[0]["alpha"], index=index)
    gate, _ = full_sample_gate(returns, probabilities, min_days=3, horizon=2)
    assert gate.iloc[0] == 1.0
    altered = returns.copy()
    altered.iloc[10:] = -0.5                                               # only the second half of the sample
    changed, log = full_sample_gate(altered, probabilities, min_days=3, horizon=2)
    assert changed.iloc[0] == 0.0
    assert not bool(log.set_index("state").loc[0, "gate_open"])

