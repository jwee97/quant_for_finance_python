"""The Alpha Lab's library: IC statistics and decay, calibration to expected residual return, breadth and the transfer coefficient, and the signal registry. Every number is checked against a world whose
truth is known: an AR(1) signal whose decay rate is set, a covariance whose optimal portfolio is closed-form, a registry whose files are inspected."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.alphalab import breadth as br
from src.alphalab import calibrate as cal
from src.alphalab import ic as icm
from src.alphalab.registry import RegistryError, SignalRegistry, fingerprint, pit_violations

T, N, PHI, C = 1500, 60, 0.9, 0.10


@pytest.fixture(scope="module")
def world():
    """An AR(1) signal ``a`` (unit variance, persistence 0.9, so a time constant of 9.5 periods) and returns ``r[t+1] = 0.1 a[t] + noise``: the IC at lag k is 0.1 x 0.9^(k-1) / 1.005."""
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2015-01-01", periods=T)
    a = np.zeros((T, N))
    a[0] = rng.standard_normal(N)
    for t in range(1, T):
        a[t] = PHI * a[t - 1] + np.sqrt(1 - PHI ** 2) * rng.standard_normal(N)
    r = np.zeros((T, N))
    r[1:] = C * a[:-1] + rng.standard_normal((T - 1, N))
    cols = [f"S{i:02d}" for i in range(N)]
    return pd.DataFrame(a, index=dates, columns=cols), pd.DataFrame(r * 0.01, index=dates, columns=cols)


def test_decay_recovers_the_time_constant_the_signal_was_built_with(world):
    sig, ret = world
    curve = icm.lagged_ic(sig, ret, lags=(1, 2, 3, 5, 8, 12, 20, 30))
    fit = icm.fit_decay(curve)
    tau = -1.0 / np.log(PHI)
    assert fit["significant"] and fit["ic1"] == pytest.approx(0.10, abs=0.015)
    assert fit["tau"] == pytest.approx(tau, rel=0.25) and fit["half_life"] == pytest.approx(tau * np.log(2), rel=0.25) and fit["r2"] > 0.9
    assert fit["periods_to_90pct"] == pytest.approx(-fit["tau"] * np.log(0.1))
    assert icm.delay_cost(fit["tau"], 0) == 0 and icm.delay_cost(fit["tau"], fit["tau"]) == pytest.approx(1 - np.exp(-1))


def test_pure_noise_has_no_half_life(world):
    _, ret = world
    noise = pd.DataFrame(np.random.default_rng(1).standard_normal(ret.shape), index=ret.index, columns=ret.columns)
    fit = icm.fit_decay(icm.lagged_ic(noise, ret, lags=(1, 2, 3, 5, 8)))
    assert not fit["significant"] and np.isnan(fit["half_life"]), "a decay rate fitted to noise is noise: it must not be reported"


def test_ic_summary_stability_regimes_and_rolling(world):
    sig, ret = world
    ic = icm.ic_series(sig, ret.shift(-1))
    s = icm.ic_summary(ic, horizon=1)
    assert s["mean"] == pytest.approx(0.10, abs=0.015) and s["t"] > 10 and s["hit_rate"] > 0.7 and s["stability"] == 1.0 and s["n"] > 1400
    overlapping = icm.ic_series(sig, cal.forward_residual(ret, 21))                       # 21-day windows: neighbouring ICs share 20 of 21 days
    assert icm.ic_summary(overlapping, horizon=21)["t"] < 0.6 * icm.ic_summary(overlapping, horizon=1)["t"], "overlapping windows must lower the t-statistic"
    regime = pd.Series(np.where(np.arange(len(ic)) % 2 == 0, "calm", "stress"), index=ic.index)
    by = icm.ic_by_regime(ic, regime)
    assert set(by.index) == {"calm", "stress"} and by["n"].sum() == ic.notna().sum()
    roll = icm.rolling_ic(ic, 63)
    assert roll["mean"].dropna().between(-0.1, 0.4).all() and len(roll) == len(ic)
    flipped = ic.copy()
    flipped.iloc[len(flipped) // 2:] *= -1
    assert icm.stability(flipped) < 1.0, "a signal that works then stops is not stable"
    assert np.isnan(icm.ic_summary(ic.iloc[:2])["mean"])


def test_ic_known_at_waits_for_the_forward_window(world):
    sig, ret = world
    ic = icm.ic_series(sig, ret.shift(-1))
    assert icm.known_at(ic, 5).iloc[5] == ic.iloc[0] and icm.known_at(ic, 5).iloc[:5].isna().all()


def test_calibrated_alpha_has_a_reliability_slope_of_one_and_a_wrong_scale_is_caught(world):
    sig, ret = world
    f = cal.calibrate(sig, ret, horizon=1, window=252, min_obs=60)
    fwd = cal.forward_residual(ret, 1)
    rel = cal.reliability(f.alpha.iloc[300:], fwd.iloc[300:])
    assert 0.8 < rel["slope"] < 1.25 and rel["verdict"] == "calibrated", rel
    assert rel["bins"]["realised"].is_monotonic_increasing and len(rel["bins"]) == 5
    assert cal.reliability(2.0 * f.alpha.iloc[300:], fwd.iloc[300:])["verdict"].startswith("overconfident")
    assert cal.reliability(0.4 * f.alpha.iloc[300:], fwd.iloc[300:])["verdict"].startswith("too timid")
    assert f.ic_hat.iloc[:60].isna().all(), "no IC is believed before enough have matured"
    # nothing at date t uses a later return: truncating the future leaves the alpha unchanged
    g = cal.calibrate(sig.iloc[:900], ret.iloc[:900], horizon=1, window=252, min_obs=60)
    pd.testing.assert_frame_equal(f.alpha.iloc[:880], g.alpha.iloc[:880], rtol=1e-9, atol=1e-12)


def test_a_signal_with_no_skill_is_shrunk_toward_zero(world):
    _, ret = world
    noise = pd.DataFrame(np.random.default_rng(3).standard_normal(ret.shape), index=ret.index, columns=ret.columns)
    f = cal.calibrate(noise, ret, horizon=1, window=252, min_obs=60)
    raw = cal.estimate_ic(f.ic, 1, 252, 60, shrink=False)
    assert f.ic_hat.abs().mean() < raw.abs().mean() and f.ic_hat.abs().max() < 0.05


def test_effective_breadth_counts_independent_bets():
    rng = np.random.default_rng(0)
    indep = pd.DataFrame(rng.standard_normal((4000, 30)))
    assert br.effective_cross_section(indep) == pytest.approx(30, rel=0.12)
    one_factor = pd.DataFrame(rng.standard_normal((4000, 1)) + 0.15 * rng.standard_normal((4000, 30)))
    assert br.effective_cross_section(one_factor) < 1.5, "thirty assets driven by one factor are one bet"
    b = br.effective_breadth(indep, horizon=21)
    assert b["forecasts_per_year"] == pytest.approx(12) and b["breadth"] == pytest.approx(b["n_effective"] * 12)
    assert br.fundamental_law(0.05, 400, 0.8) == pytest.approx(0.8)
    assert br.implied_ic(0.8, 400, 0.8) == pytest.approx(0.05)


def test_signal_autocorrelation_lowers_breadth(world):
    sig, ret = world
    slow = br.effective_breadth(ret, sig, horizon=1)
    assert slow["rho"] == pytest.approx(PHI, abs=0.05) and slow["breadth"] < br.effective_breadth(ret, None, horizon=1)["breadth"] * 0.2


def test_transfer_coefficient_is_the_ir_ratio_and_one_for_the_unconstrained_optimum():
    rng = np.random.default_rng(5)
    n = 12
    m = rng.standard_normal((n, n))
    cov = m @ m.T / n + 0.5 * np.eye(n)
    alpha = rng.standard_normal(n) * 0.02
    opt = np.linalg.solve(cov, alpha)
    assert br.transfer_coefficient(alpha, cov, opt) == pytest.approx(1.0) and br.transfer_coefficient(alpha, cov, 7.0 * opt) == pytest.approx(1.0)
    h = np.clip(opt, 0, None)
    tc = br.transfer_coefficient(alpha, cov, h)
    ir_h = h @ alpha / np.sqrt(h @ cov @ h)
    assert tc == pytest.approx(ir_h / br.unconstrained_ir(alpha, cov)) and tc < 1
    lo = br.constrained_transfer(alpha, cov, lb=0.0)
    assert lo["ok"] and (lo["holdings"] >= -1e-8).all() and lo["tc"] <= 1 + 1e-9 and lo["tc"] >= tc - 1e-6, "the best long-only portfolio transfers at least as much as a clipped optimum"
    free = br.constrained_transfer(alpha, cov, lb=-1e3, ub=1e3)
    assert free["tc"] == pytest.approx(1.0, abs=1e-4)
    assert br.transfer_coefficient(alpha, cov, np.zeros(n)) != br.transfer_coefficient(alpha, cov, np.zeros(n)), "zero holdings have no defined TC"


def test_registry_versions_lags_and_the_point_in_time_check(tmp_path, world):
    sig, _ = world
    reg = SignalRegistry(tmp_path / "sig")
    a = reg.register("mom", sig, lag=2, horizon=5, description="momentum")
    assert a["version"] == 1 and a["new"] and a["lag"] == 2 and a["dates"] == T
    again = reg.register("mom", sig, lag=2, horizon=5)
    assert again["version"] == 1 and not again["new"], "the same panel is the same version"
    changed = reg.register("mom", sig * 1.01, lag=2, horizon=5)
    assert changed["version"] == 2 and [v["version"] for v in reg.versions("mom")] == [1, 2]
    assert reg.register("mom", sig, lag=3, horizon=5)["version"] == 3, "a different lag is a different signal"
    known = reg.known("mom", 1)
    pd.testing.assert_frame_equal(known.iloc[2:], sig.shift(2).iloc[2:], check_freq=False)
    assert known.iloc[:2].isna().all().all()
    assert reg.as_of("mom", sig.index[10], 1).equals(sig.iloc[8]), "at date t only the row from lag periods before is known"
    assert fingerprint(sig) == fingerprint(sig[sig.columns[::-1]]) and reg.listing()[0]["versions"] == 3
    reg2 = SignalRegistry(tmp_path / "sig")
    assert reg2.names() == ["mom"] and reg2.load("mom", 2).shape == sig.shape, "the registry is on disk, not in memory"
    reg.delete("mom")
    assert reg.names() == [] and not (tmp_path / "sig" / "mom").exists()
    with pytest.raises(RegistryError, match="no signal"):
        reg.meta("mom")


def test_registry_refuses_bad_input(tmp_path, world):
    sig, _ = world
    reg = SignalRegistry(tmp_path)
    for name, panel, kw, word in (("../x", sig, {}, "name"), ("ok", sig.reset_index(drop=True), {}, "frame of dates"), ("ok", sig, {"lag": -1}, "lag"), ("ok", sig.iloc[::-1], {}, "increasing"), ("ok", sig.iloc[:0], {}, "non-empty")):
        with pytest.raises(RegistryError, match=word):
            reg.register(name, panel, **kw)


def test_pit_violations_counts_values_known_after_their_date(world):
    sig, _ = world
    sub = sig.iloc[:50, :3]
    info = pd.DataFrame({c: sub.index for c in sub.columns}, index=sub.index)
    assert pit_violations(sub, info)["ok"]
    late = info.copy()
    late.iloc[10:20, 0] = late.iloc[10:20, 0] + pd.Timedelta(days=30)
    v = pit_violations(sub, late)
    assert v["violations"] == 10 and not v["ok"] and v["cells"] == 150
