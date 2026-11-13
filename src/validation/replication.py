"""Replication benchmarks: the platform's implementations of published methods, each checked against the publication.

Every benchmark states what kind of check it is, because the kinds are not worth the same:

``exact``         the platform's code reproduces the paper's formula or its worked number to rounding. The reference is transcribed here from the paper and shares no code with the implementation.
``simulation``    the paper's claim is a statistical property (a test's size, an estimator's lower loss); it is checked on simulated data where the truth is known. This validates the implementation's
                  behaviour, not a number from the paper.
``approximate``   the platform uses the paper's own approximation, or a reduced form of its method; the size of the gap is measured and reported.
``method_only``   the paper's empirical result needs data the platform does not have. The method is reproduced on simulated data built to the paper's design; nothing here reproduces the paper's numbers.

``run_all()`` returns one row per benchmark with the measured value, the reference, the tolerance and whether it held. ``python -m src replicate`` prints them. Nothing is hidden in a pass: a row carries the
numbers, so a tolerance that is too generous can be seen.

Sources: Ledoit and Wolf (2004) "Honey, I shrunk the sample covariance matrix", Journal of Portfolio Management 30(4); Bailey and Lopez de Prado (2012) "The Sharpe ratio efficient frontier", Journal of
Risk 15(2), and (2014) "The deflated Sharpe ratio", Journal of Portfolio Management 40(5); Bailey, Borwein, Lopez de Prado and Zhu (2017) "The probability of backtest overfitting", Journal of Computational
Finance 20(4); White (2000) "A reality check for data snooping", Econometrica 68(5); Hansen (2005) "A test for superior predictive ability", Journal of Business and Economic Statistics 23(4); Almgren and
Chriss (2000) "Optimal execution of portfolio transactions", Journal of Risk 3(2); Gu, Kelly and Xiu (2020) "Empirical asset pricing via machine learning", Review of Financial Studies 33(5).
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import pandas as pd

KINDS = ("exact", "simulation", "approximate", "method_only")


def _row(key: str, paper: str, claim: str, kind: str, value: float, reference: float, tolerance: float, note: str = "", relative: bool = False, one_sided: str | None = None) -> dict:
    """One benchmark's result. ``one_sided``: ``"at_most"`` or ``"at_least"`` when the claim is a bound and not an equality."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    gap = abs(value - reference) / max(abs(reference), 1e-300) if relative else abs(value - reference)
    if one_sided == "at_most":
        held = value <= reference + tolerance
    elif one_sided == "at_least":
        held = value >= reference - tolerance
    else:
        held = gap <= tolerance
    return {"id": key, "paper": paper, "claim": claim, "kind": kind, "value": float(value), "reference": float(reference), "gap": float(gap), "tolerance": float(tolerance), "relative": bool(relative),
            "bound": one_sided, "passed": bool(held), "note": note}


# ---------------------------------------------------------------------------------------------------------------------- Ledoit and Wolf (2004)
def ledoit_wolf_2004_reference(x: np.ndarray) -> tuple[np.ndarray, float]:
    """The shrinkage estimator toward the constant-correlation target, transcribed from Appendix B of the paper (and equal to the authors' published ``covCor`` routine): ``(matrix, intensity)`` with the
    paper's ``1 / T`` normalisation. Written from the paper; it shares no code with ``portfolio.covariance``."""
    t, n = x.shape
    x = x - x.mean(axis=0)
    s = x.T @ x / t
    var = np.diag(s)
    sd = np.sqrt(var)
    rbar = ((s / np.outer(sd, sd)).sum() - n) / (n * (n - 1))
    f = rbar * np.outer(sd, sd)
    np.fill_diagonal(f, var)
    pi_mat = (x ** 2).T @ (x ** 2) / t - s ** 2
    theta = (x ** 3).T @ x / t - var[:, None] * s
    np.fill_diagonal(theta, 0.0)
    rho = np.trace(pi_mat) + rbar * (np.outer(1.0 / sd, sd) * theta).sum()
    gamma = float(((f - s) ** 2).sum())
    delta = max(0.0, min(1.0, (pi_mat.sum() - rho) / gamma / t))
    return delta * f + (1.0 - delta) * s, float(delta)


def _factor_returns(t: int, n: int, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Fat-tailed returns with a factor structure and their true covariance."""
    rng = np.random.default_rng(seed)
    loadings = rng.normal(0.0, 1.0, (n, k))
    idio = rng.uniform(0.004, 0.02, n)
    scale = math.sqrt(5.0 / 3.0)                                                               # a Student-t with 5 degrees of freedom has variance 5/3
    x = (rng.standard_t(5, (t, k)) / scale * 0.01) @ loadings.T + rng.standard_t(5, (t, n)) / scale * idio
    return x, 0.01 ** 2 * loadings @ loadings.T + np.diag(idio ** 2)


def ledoit_wolf_intensity() -> dict:
    from ..portfolio.covariance import ledoit_wolf_shrinkage

    worst = 0.0
    for t, n, k, seed in ((250, 10, 2, 0), (120, 30, 3, 1), (60, 40, 4, 2), (1000, 12, 3, 3)):
        x, _ = _factor_returns(t, n, k, seed)
        ours, delta = ledoit_wolf_shrinkage(pd.DataFrame(x), "constant_correlation", annualise=False)
        ref, delta_ref = ledoit_wolf_2004_reference(x)
        worst = max(worst, abs(delta - delta_ref), float(np.abs(ours.to_numpy() - ref * t / (t - 1.0)).max() / np.abs(ref).max()))
    return _row("ledoit_wolf_2004_intensity", "Ledoit and Wolf (2004)", "the shrinkage intensity and the shrunk matrix toward the constant-correlation target equal the paper's Appendix B", "exact", worst, 0.0,
                1e-10, "four simulated panels; the platform's matrix uses 1/(T-1) where the paper uses 1/T, so the comparison is against the paper's matrix times T/(T-1)")


def _market_returns(t: int, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Fat-tailed returns of a one-factor market model (every loading positive, as for stocks: the case the paper's target is built for) and their true covariance."""
    rng = np.random.default_rng(seed)
    beta, idio = rng.uniform(0.5, 1.5, n), rng.uniform(0.008, 0.02, n)
    scale = math.sqrt(5.0 / 3.0)
    x = (rng.standard_t(5, (t, 1)) / scale * 0.01) * beta + rng.standard_t(5, (t, n)) / scale * idio
    return x, 0.01 ** 2 * np.outer(beta, beta) + np.diag(idio ** 2)


def ledoit_wolf_loss() -> dict:
    from ..portfolio.covariance import ledoit_wolf_shrinkage

    ratios = {}
    for name, make in (("market model", lambda seed: _market_returns(60, 30, seed)), ("two factors of mixed sign", lambda seed: _factor_returns(60, 30, 2, seed))):
        shrunk_loss = sample_loss = 0.0
        for seed in range(100, 160):
            x, truth = make(seed)
            shrunk, _ = ledoit_wolf_shrinkage(pd.DataFrame(x), "constant_correlation", annualise=False)
            shrunk_loss += float(((shrunk.to_numpy() - truth) ** 2).sum())
            sample_loss += float(((np.cov(x, rowvar=False, ddof=1) - truth) ** 2).sum())
        ratios[name] = shrunk_loss / sample_loss
    text = "; ".join(f"{k}: {v:.3f}" for k, v in ratios.items())
    return _row("ledoit_wolf_2004_loss", "Ledoit and Wolf (2004)", "with few observations per asset the shrunk matrix has a lower expected squared (Frobenius) error than the sample matrix", "simulation",
                max(ratios.values()), 1.0, 0.0, f"60 panels of 60 days by 30 assets for each of two true covariances; total squared error of the shrunk matrix over the sample matrix's: {text}. "
                "The benchmark's value is the larger of the two ratios: below one in both, so the gain does not depend on the truth having the target's shape", one_sided="at_most")


# ---------------------------------------------------------------------------------------------------------------------- Bailey and Lopez de Prado
def probabilistic_sharpe_example() -> dict:
    from ..backtest.metrics import probabilistic_sharpe_ratio

    value = probabilistic_sharpe_ratio(0.458 * math.sqrt(12), 0.0, 24, skew=-2.448, kurtosis=10.164, periods_per_year=12)
    return _row("bailey_lopez_de_prado_2012_psr", "Bailey and Lopez de Prado (2012)", "the paper's worked example: a monthly Sharpe ratio of 0.458 over 24 months with skewness -2.448 and kurtosis 10.164 "
                "has a probabilistic Sharpe ratio of 0.913 against zero (0.982 if the returns were normal)", "exact", value, 0.913, 0.001,
                "the reference is the paper's printed figure, to its three decimals")


def expected_maximum_sharpe() -> dict:
    from ..stats.inference import expected_max_sharpe

    rng = np.random.default_rng(0)
    worst = 0.0
    for n in (10, 100, 1000):
        simulated = float(rng.standard_normal((20_000, n)).max(axis=1).mean())
        worst = max(worst, expected_max_sharpe(n, 1.0) / simulated - 1.0)
    return _row("bailey_lopez_de_prado_2014_expected_max", "Bailey and Lopez de Prado (2014)", "the expected maximum of N skill-less Sharpe ratios, which the deflated Sharpe ratio uses as its bar, is the "
                "paper's closed-form approximation", "approximate", worst, 0.0, 0.035, "the largest relative excess of the formula over a simulated maximum of 10, 100 and 1,000 standard normals: the formula "
                "is the paper's own approximation and runs about 2% high at ten trials (a higher bar, so it errs toward rejecting)")


def deflated_sharpe_false_discovery() -> dict:
    """Pick the best of 50 skill-less strategies, 300 times. Its plain t-test 'finds' skill most of the time; the deflated Sharpe ratio, told the trials and their spread, must not."""
    from ..stats.inference import deflated_sharpe_ratio

    rng = np.random.default_rng(1)
    hits, runs = 0, 300
    for _ in range(runs):
        trials = rng.standard_normal((500, 50)) * 0.01
        sharpes = trials.mean(axis=0) / trials.std(axis=0, ddof=1)
        best = trials[:, int(np.argmax(sharpes))]
        hits += deflated_sharpe_ratio(best, trial_sharpes=sharpes)["probability"] > 0.95
    return _row("bailey_lopez_de_prado_2014_dsr_size", "Bailey and Lopez de Prado (2014)", "the best of many skill-less backtests is not declared skilful by the deflated Sharpe ratio more often than its level",
                "simulation", hits / runs, 0.05, 0.03, "300 selections of the best of 50 noise strategies; the share with a deflated Sharpe ratio above 0.95", one_sided="at_most")


def backtest_overfitting_probability() -> dict:
    from .multiple_testing import pbo_cscv

    rng = np.random.default_rng(2)
    noise = float(np.mean([pbo_cscv(rng.standard_normal((480, 20)) * 0.01, blocks=8)["pbo"] for _ in range(20)]))
    skilled = rng.standard_normal((480, 20)) * 0.01
    skilled[:, 3] += 0.004                                                                     # one strategy with a real edge
    found = float(pbo_cscv(skilled, blocks=8)["pbo"])
    value = abs(noise - 0.5) + found
    return _row("bailey_et_al_2017_pbo", "Bailey, Borwein, Lopez de Prado and Zhu (2017)", "the probability of backtest overfitting is about one half when no strategy has skill and near zero when one has",
                "simulation", value, 0.0, 0.15, f"20 noise studies gave a mean of {noise:.3f} (one half is right); a study with one real edge gave {found:.3f}; the value is the sum of the two errors")


# ---------------------------------------------------------------------------------------------------------------------- White (2000) and Hansen (2005)
# Measured once (500 studies each, 10 skill-less candidates, 300 bootstrap draws, 10% level) and recorded so that the limitation is a number: the share of true nulls rejected by (Reality Check, SPA).
# One candidate and an independent bootstrap is at its level; more candidates and longer blocks relative to the sample push both tests above it. A block of 21 days on one year of data is too long.
SMALL_SAMPLE_SIZE = {"250 days, 1 candidate, block 1": (0.108, 0.108), "250 days, 10 candidates, block 1": (0.124, 0.120), "250 days, 10 candidates, block 5": (0.130, 0.152),
                     "250 days, 10 candidates, block 21": (0.158, 0.230), "1000 days, 10 candidates, block 5": (0.088, 0.096)}


def _snooping_rates(with_poor: bool, edge: float, runs: int = 120, n: int = 250, k: int = 10, seed: int = 3, block: float = 5.0) -> tuple[float, float]:
    """The shares of ``runs`` studies in which the Reality Check and the SPA test (consistent p-value) reject at 10%."""
    from .multiple_testing import bootstrap_means, hansen_spa, white_reality_check

    rng = np.random.default_rng(seed)
    rc = spa = 0
    for r in range(runs):
        x = rng.standard_normal((n, k)) * 0.01
        x[:, 0] += edge
        if with_poor:
            x = np.hstack([x, rng.standard_normal((n, 3 * k)) * 0.03 - 0.004])                 # many clearly losing and noisy alternatives
        boot = bootstrap_means(x, 300, block, seed + r)
        rc += white_reality_check(x, boot=boot)["p_value"] <= 0.10
        spa += hansen_spa(x, boot=boot)["p_consistent"] <= 0.10
    return rc / runs, spa / runs


def snooping_test_size() -> dict:
    rc, spa = _snooping_rates(with_poor=False, edge=0.0, runs=200, n=1000, block=10.0)
    return _row("white_2000_hansen_2005_size", "White (2000); Hansen (2005)", "when no candidate beats the benchmark, neither test rejects more often than its level", "simulation", max(rc, spa), 0.10, 0.045,
                f"200 studies of 10 skill-less candidates over 1,000 days, mean block 10 (the selection layer's rule for that length), at the 10% level: the Reality Check rejected {rc:.3f} and the SPA test "
                f"{spa:.3f}; the tolerance is two standard errors of a 10% rate over 200 studies. With short samples or long blocks both tests over-reject: see ``SMALL_SAMPLE_SIZE`` for measured rates",
                one_sided="at_most")


def spa_power_with_poor_alternatives() -> dict:
    rc, spa = _snooping_rates(with_poor=True, edge=0.0016)
    return _row("hansen_2005_power", "Hansen (2005)", "adding poor and noisy alternatives costs the Reality Check power and the SPA test much less: SPA finds a real edge at least as often", "simulation",
                spa - rc, 0.0, 0.0, f"120 studies with one real edge among 10 candidates and 30 poor ones: the SPA test found it in {spa:.3f} of them, the Reality Check in {rc:.3f}", one_sided="at_least")


# ---------------------------------------------------------------------------------------------------------------------- Almgren and Chriss (2000)
def almgren_chriss_trajectory() -> dict:
    """With linear temporary impact, a flat volume profile and a flat variance profile the platform's quadratic program is the paper's problem, whose solution is closed form: the shares remaining after
    trade ``j`` of ``N`` are ``X sinh(kappa (N - j)) / sinh(kappa N)`` with ``2 (cosh(kappa) - 1) = (risk penalty per step) / (impact cost per squared share)`` (the paper's equations 17 to 19, per step)."""
    from ..algo.market import Market, Order
    from ..algo.optimize import optimal_schedule, quadratic_form

    worst, n = 0.0, 20
    flat = tuple([1.0 / n] * n)
    for risk_aversion in (1e-4, 1e-3, 1e-2):
        market = Market(n=n, beta=1.0, volume_fractions=flat, variance_fractions=flat, volume_noise=0.0, day_noise=0.0)
        order = Order(1, 150_000.0)
        hessian, _ = quadratic_form(order, market, risk_aversion)
        risk = hessian[n - 1, n - 2] / 2.0                                                     # the risk penalty on the square of the shares still held after a step
        impact = hessian[n - 1, n - 1] / 2.0 - risk                                            # the cost of a squared share traded in a step
        kappa = math.acosh(1.0 + risk / (2.0 * impact))
        closed = order.shares * np.sinh(kappa * (n - np.arange(n + 1))) / math.sinh(kappa * n)
        remaining = order.shares - np.concatenate([[0.0], np.cumsum(optimal_schedule(order, market, risk_aversion, exact=False))])
        worst = max(worst, float(np.abs(remaining - closed).max() / order.shares))
    return _row("almgren_chriss_2000_trajectory", "Almgren and Chriss (2000)", "the optimal liquidation path under linear impact is the paper's hyperbolic-sine trajectory", "exact", worst, 0.0, 1e-6,
                "three risk aversions, 20 intervals; the largest gap between the solver's path and the closed form, as a share of the order")


def almgren_chriss_frontier() -> dict:
    from ..algo.market import Market, Order
    from ..algo.optimize import cost_and_risk, optimal_schedule

    market, order = Market(beta=1.0, volume_noise=0.0, day_noise=0.0), Order(1, 150_000.0)
    points = [cost_and_risk(optimal_schedule(order, market, lam, exact=False), order, market) for lam in (0.0, 1e-4, 1e-3, 1e-2, 1e-1)]
    cost, risk = np.array([p["cost_bps"] for p in points]), np.array([p["risk_bps"] for p in points])
    violations = int((np.diff(cost) < -1e-9).sum() + (np.diff(risk) > 1e-9).sum())
    return _row("almgren_chriss_2000_frontier", "Almgren and Chriss (2000)", "along the efficient frontier expected cost rises and timing risk falls as risk aversion grows", "exact", violations, 0.0, 0.0,
                "five risk aversions from zero to 0.1 on the default intraday profile: the count of steps that break the ordering")


# ---------------------------------------------------------------------------------------------------------------------- Gu, Kelly and Xiu (2020)
def r2_out_of_sample(realised, predicted) -> float:
    """The paper's out-of-sample R squared, equation 19: ``1 - sum((r - rhat)^2) / sum(r^2)``. The denominator is NOT demeaned: the benchmark is a forecast of zero, because the historical mean of a stock's
    return is so noisy that beating it would flatter every model."""
    r, p = np.asarray(realised, dtype=float), np.asarray(predicted, dtype=float)
    return float(1.0 - ((r - p) ** 2).sum() / (r ** 2).sum())


def gkx_panel(n_stocks: int = 200, n_months: int = 180, n_chars: int = 100, predictive_r2: float = 0.05, nonlinear: bool = True, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A simulated panel on the design of the paper's Internet Appendix A: 200 stocks, 180 months, 100 persistent characteristics ranked into (-1, 1) each month, one persistent macro series, 200 predictors
    (the characteristics and their products with the macro series), a signal of three terms (linear, or with a square, a product and a sign as in the paper's model (b)), fat-tailed noise and a predictive
    R squared of 5%. Returns ``(features, return, month)``."""
    rng = np.random.default_rng(seed)
    rho = rng.uniform(0.9, 1.0, n_chars)
    c = np.zeros((n_months, n_stocks, n_chars))
    state = rng.standard_normal((n_stocks, n_chars))
    for t in range(n_months):
        state = rho * state + np.sqrt(1.0 - rho ** 2) * rng.standard_normal((n_stocks, n_chars))
        c[t] = 2.0 * (np.argsort(np.argsort(state, axis=0), axis=0) + 1.0) / (n_stocks + 1.0) - 1.0       # ranked into (-1, 1) each month, as the paper does
    macro = np.zeros(n_months)
    for t in range(1, n_months):
        macro[t] = 0.95 * macro[t - 1] + math.sqrt(1.0 - 0.95 ** 2) * rng.standard_normal()
    if nonlinear:
        signal = c[:, :, 0] ** 2 + c[:, :, 0] * c[:, :, 1] + np.sign(c[:, :, 2] * macro[:, None])
    else:
        signal = c[:, :, 0] + c[:, :, 1] + c[:, :, 2] * macro[:, None]
    noise_sd = 0.05
    signal = noise_sd * math.sqrt(predictive_r2 / (1.0 - predictive_r2)) * (signal - signal.mean()) / signal.std()
    noise = rng.standard_t(5, (n_months, n_stocks)) * noise_sd / math.sqrt(5.0 / 3.0)
    features = np.concatenate([c, c * macro[:, None, None]], axis=2).reshape(n_months * n_stocks, 2 * n_chars)
    return features, (signal + noise).reshape(-1), np.repeat(np.arange(n_months), n_stocks)


def gkx_scores(seed: int = 0, nonlinear: bool = True) -> dict:
    """Out-of-sample R squared (the paper's) of three models on one simulated panel: every model is fitted on the first half, the penalty and the tree depth are chosen on the next quarter, and the last
    quarter is scored once."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge

    x, y, month = gkx_panel(seed=seed, nonlinear=nonlinear)
    train, valid, test = month < 90, (month >= 90) & (month < 135), month >= 135
    on_validation = lambda model: r2_out_of_sample(y[valid], model.predict(x[valid]))            # noqa: E731
    ols = LinearRegression().fit(x[train], y[train])
    ridge = max((Ridge(alpha=a).fit(x[train], y[train]) for a in (10.0, 1000.0, 100_000.0)), key=on_validation)
    trees = max((HistGradientBoostingRegressor(max_depth=d, learning_rate=0.05, max_iter=100, min_samples_leaf=200, random_state=0).fit(x[train], y[train]) for d in (1, 2)), key=on_validation)
    return {name: r2_out_of_sample(y[test], model.predict(x[test])) for name, model in (("ols", ols), ("ridge", ridge), ("trees", trees))}


def gkx_method() -> dict:
    """The paper's comparison on its own simulation design. The claims checked are its qualitative ones (its Table A.1 and section 3): least squares on every predictor has a NEGATIVE out-of-sample R
    squared, a penalised linear model a small positive one, and a tree ensemble recovers most of the true 5% when the truth is nonlinear."""
    scores = gkx_scores()
    held = scores["ols"] < 0.0 < scores["ridge"] < scores["trees"] < 0.05
    return _row("gu_kelly_xiu_2020_method", "Gu, Kelly and Xiu (2020)", "on the paper's nonlinear simulation, least squares on every predictor overfits (negative out-of-sample R squared), a penalised linear model "
                "is slightly positive, and boosted trees recover most of the true predictability", "method_only", 1.0 if held else 0.0, 1.0, 0.0,
                f"200 stocks, 180 months, 200 predictors, true predictive R squared 5%: out-of-sample R squared of {100 * scores['ols']:.2f}% (least squares), {100 * scores['ridge']:.2f}% (ridge) and "
                f"{100 * scores['trees']:.2f}% (boosted trees). The R squared is the paper's (no demeaning). The paper's empirical figures come from CRSP stocks from 1957 to 2016 with 94 characteristics, "
                "which the platform does not have: nothing here reproduces them")


BENCHMARKS: tuple[Callable[[], dict], ...] = (ledoit_wolf_intensity, ledoit_wolf_loss, probabilistic_sharpe_example, expected_maximum_sharpe, deflated_sharpe_false_discovery, backtest_overfitting_probability,
                                             snooping_test_size, spa_power_with_poor_alternatives, almgren_chriss_trajectory, almgren_chriss_frontier, gkx_method)


def run_all(only: tuple[str, ...] | None = None) -> list[dict]:
    """Every benchmark's row (or those whose id starts with one of ``only``)."""
    rows = []
    for fn in BENCHMARKS:
        if only is None or any(fn.__name__.startswith(o) or o in fn.__name__ for o in only):
            rows.append(fn())
    return rows


def format_table(rows: list[dict]) -> str:
    lines = [f"{'benchmark':42s} {'kind':12s} {'value':>12s} {'reference':>12s} {'tolerance':>10s}  result"]
    for r in rows:
        bound = {"at_most": "<=", "at_least": ">="}.get(r["bound"], "=")
        lines.append(f"{r['id']:42s} {r['kind']:12s} {r['value']:12.6g} {bound + format(r['reference'], '.6g'):>12s} {r['tolerance']:10.3g}  {'held' if r['passed'] else 'DID NOT HOLD'}")
        lines.append(f"    {r['paper']}: {r['claim']}")
        if r["note"]:
            lines.append(f"    {r['note']}")
    counts = {k: sum(r["kind"] == k for r in rows) for k in KINDS}
    lines.append("")
    lines.append(f"{sum(r['passed'] for r in rows)} of {len(rows)} held: " + ", ".join(f"{v} {k}" for k, v in counts.items() if v))
    return "\n".join(lines)
