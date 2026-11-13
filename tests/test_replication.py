"""The replication benchmarks: every one holds, each says what kind of check it is, and the exact ones are re-derived here without the module's help."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.portfolio.covariance import ledoit_wolf_shrinkage
from src.validation import replication as rp

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def rows():
    return {r["id"]: r for r in rp.run_all()}


def test_every_benchmark_holds_and_names_its_kind_its_paper_and_its_numbers(rows):
    assert len(rows) == len(rp.BENCHMARKS) == 11
    for key, r in rows.items():
        assert r["passed"], f"{key}: {r['value']} against {r['reference']} (tolerance {r['tolerance']}): {r['note']}"
        assert r["kind"] in rp.KINDS and r["paper"] and r["claim"] and math.isfinite(r["value"]) and math.isfinite(r["reference"])
    kinds = {k: sorted(key for key, r in rows.items() if r["kind"] == k) for k in rp.KINDS}
    assert kinds["exact"] == ["almgren_chriss_2000_frontier", "almgren_chriss_2000_trajectory", "bailey_lopez_de_prado_2012_psr", "ledoit_wolf_2004_intensity"]
    assert kinds["method_only"] == ["gu_kelly_xiu_2020_method"] and kinds["approximate"] == ["bailey_lopez_de_prado_2014_expected_max"] and len(kinds["simulation"]) == 5


def test_an_exact_benchmark_is_exact_and_an_empirical_paper_is_not_claimed_to_be_reproduced(rows):
    for key in ("ledoit_wolf_2004_intensity", "almgren_chriss_2000_trajectory"):
        assert rows[key]["gap"] < 1e-9 and rows[key]["tolerance"] <= 1e-6                    # exact means to rounding, not to a generous tolerance
    gkx = rows["gu_kelly_xiu_2020_method"]
    assert "nothing here reproduces them" in gkx["note"] and "CRSP" in gkx["note"] and gkx["kind"] == "method_only"
    assert rows["bailey_lopez_de_prado_2014_expected_max"]["value"] > 0.01                   # the approximation's error is reported as a number, not rounded away


def test_a_benchmark_that_does_not_hold_says_so():
    bad = rp._row("x", "paper", "claim", "exact", 1.02, 1.0, 0.01)
    assert not bad["passed"] and bad["gap"] == pytest.approx(0.02)
    assert rp._row("x", "p", "c", "simulation", 0.12, 0.10, 0.03, one_sided="at_most")["passed"] and not rp._row("x", "p", "c", "simulation", 0.14, 0.10, 0.03, one_sided="at_most")["passed"]
    assert rp._row("x", "p", "c", "simulation", 0.2, 0.0, 0.0, one_sided="at_least")["passed"]
    with pytest.raises(ValueError):
        rp._row("x", "p", "c", "proved", 1.0, 1.0, 0.0)
    text = rp.format_table([bad])
    assert "DID NOT HOLD" in text and "0 of 1 held" in text


# ---------------------------------------------------------------------------------------------------------------------- Ledoit and Wolf, from the paper
def test_the_shrinkage_intensity_is_the_papers_entry_by_entry():
    """Appendix B of the paper with every sum written as a loop, on returns of four assets whose correlations differ (two move together, one alone, one against): nothing vectorised, nothing shared with the
    implementation or with the module's own transcription."""
    rng = np.random.default_rng(4)
    common = rng.standard_t(6, (200, 1)) * 0.01
    x = rng.standard_t(6, (200, 4)) * np.array([0.006, 0.008, 0.015, 0.007]) + common * np.array([1.0, 1.2, 0.0, -0.6])
    t, n = x.shape
    c = x - x.mean(axis=0)
    s = [[sum(c[k, i] * c[k, j] for k in range(t)) / t for j in range(n)] for i in range(n)]
    rbar = sum(s[i][j] / math.sqrt(s[i][i] * s[j][j]) for i in range(n) for j in range(n) if i != j) / (n * (n - 1))
    f = [[s[i][i] if i == j else rbar * math.sqrt(s[i][i] * s[j][j]) for j in range(n)] for i in range(n)]
    pi = sum(sum((c[k, i] * c[k, j] - s[i][j]) ** 2 for k in range(t)) / t for i in range(n) for j in range(n))
    theta = lambda a, i, j: sum((c[k, a] ** 2 - s[a][a]) * (c[k, i] * c[k, j] - s[i][j]) for k in range(t)) / t      # noqa: E731
    rho = sum(sum((c[k, i] ** 2 - s[i][i]) ** 2 for k in range(t)) / t for i in range(n))
    rho += sum(rbar / 2 * (math.sqrt(s[j][j] / s[i][i]) * theta(i, i, j) + math.sqrt(s[i][i] / s[j][j]) * theta(j, i, j)) for i in range(n) for j in range(n) if i != j)
    gamma = sum((f[i][j] - s[i][j]) ** 2 for i in range(n) for j in range(n))
    delta = max(0.0, min(1.0, (pi - rho) / gamma / t))
    ours, ours_delta = ledoit_wolf_shrinkage(pd.DataFrame(x), "constant_correlation", annualise=False)
    assert 0.0 < delta < 1.0 and ours_delta == pytest.approx(delta, rel=1e-10)
    assert rp.ledoit_wolf_2004_reference(x)[1] == pytest.approx(delta, rel=1e-10)
    want = (delta * np.array(f) + (1 - delta) * np.array(s)) * t / (t - 1)
    assert np.allclose(ours.to_numpy(), want, rtol=1e-10)


def test_the_diagonal_target_and_a_flat_asset_are_handled():
    rng = np.random.default_rng(5)
    x = pd.DataFrame(rng.standard_normal((120, 4)) * 0.01)
    _, delta = ledoit_wolf_shrinkage(x, "diagonal", annualise=False)
    c = x.to_numpy() - x.to_numpy().mean(axis=0)
    s = c.T @ c / 120
    pi = (c ** 2).T @ (c ** 2) / 120 - s ** 2
    gamma = float((s ** 2).sum() - (np.diag(s) ** 2).sum())
    assert delta == pytest.approx(float(np.clip((pi.sum() - np.trace(pi)) / gamma / 120, 0, 1)), rel=1e-10)        # the diagonal target's off-diagonal entries are constants: its rho is the diagonal alone
    x[3] = 0.0
    matrix, d = ledoit_wolf_shrinkage(x, "constant_correlation", annualise=False)
    assert np.isfinite(matrix.to_numpy()).all() and 0.0 <= d <= 1.0 and matrix.iloc[3, 3] == 0.0


# ---------------------------------------------------------------------------------------------------------------------- the other exact ones, by hand
def test_the_papers_probabilistic_sharpe_example_by_hand(rows):
    z = 0.458 * math.sqrt(23) / math.sqrt(1 + 2.448 * 0.458 + (10.164 - 1) / 4 * 0.458 ** 2)
    assert rows["bailey_lopez_de_prado_2012_psr"]["value"] == pytest.approx(stats.norm.cdf(z), rel=1e-12)
    assert round(stats.norm.cdf(z), 3) == 0.913 and round(stats.norm.cdf(0.458 * math.sqrt(23) / math.sqrt(1 + 0.458 ** 2 / 2)), 3) == 0.982


def test_the_closed_form_liquidation_path_satisfies_the_papers_difference_equation():
    """Almgren and Chriss: the optimal holdings satisfy ``x[j-1] - 2 x[j] + x[j+1] = k x[j]`` with ``k = 2 (cosh(kappa) - 1)``. Check that the closed form does, then that the minimum of the paper's
    objective, found by solving its first-order conditions as a linear system, is that path, and that no nearby path costs less."""
    n, total, impact, risk = 12, 1000.0, 2.0e-6, 3.0e-7
    kappa = math.acosh(1 + risk / (2 * impact))
    closed = total * np.sinh(kappa * (n - np.arange(n + 1))) / math.sinh(kappa * n)
    assert closed[0] == pytest.approx(total) and closed[-1] == 0.0
    assert np.allclose(closed[:-2] - 2 * closed[1:-1] + closed[2:], risk / impact * closed[1:-1], rtol=1e-9)

    def objective(inner):
        x = np.concatenate([[total], inner, [0.0]])
        return impact * (np.diff(x) ** 2).sum() + risk * (x[1:] ** 2).sum()

    m = n - 1                                                                                # the unknown holdings x[1] .. x[n-1]
    system = np.zeros((m, m))
    for j in range(m):                                                                       # d/dx[j]: 2 impact (2 x[j] - x[j-1] - x[j+1]) + 2 risk x[j] = 0
        system[j, j] = 2 * impact + risk
        if j > 0:
            system[j, j - 1] = -impact
        if j < m - 1:
            system[j, j + 1] = -impact
    rhs = np.zeros(m)
    rhs[0] = impact * total
    found = np.linalg.solve(system, rhs)
    assert np.allclose(found, closed[1:-1], rtol=1e-10)
    rng = np.random.default_rng(0)
    best = objective(closed[1:-1])
    assert all(objective(closed[1:-1] + rng.normal(0, 5.0, m)) > best for _ in range(200))


def test_the_papers_r_squared_is_measured_against_a_forecast_of_zero():
    r, p = np.array([0.02, -0.01, 0.03, 0.01]), np.array([0.01, 0.0, 0.02, 0.01])
    assert rp.r2_out_of_sample(r, p) == pytest.approx(1 - 0.0003 / 0.0015)
    assert rp.r2_out_of_sample(r, np.zeros(4)) == 0.0 and rp.r2_out_of_sample(r, np.full(4, r.mean())) > 0      # the historical mean beats zero here, which is why the paper does not demean
    x, y, month = rp.gkx_panel(n_stocks=40, n_months=24, n_chars=6, seed=1)
    assert x.shape == (960, 12) and len(y) == 960 and month.max() == 23 and np.abs(x[:, :6]).max() < 1.0
    ranks = x[:40, 0]
    assert sorted(np.round((ranks + 1) / 2 * 41).astype(int).tolist()) == list(range(1, 41))                    # each month's characteristic is a rank spread over (-1, 1)


def test_the_gkx_ordering_holds_on_another_draw_and_reverses_where_the_truth_is_linear():
    other = rp.gkx_scores(seed=1)
    assert other["ols"] < 0.0 < other["ridge"] < other["trees"] < 0.05
    linear = rp.gkx_scores(seed=0, nonlinear=False)
    assert linear["ridge"] > linear["ols"] and linear["ridge"] > 0.01                        # a linear truth: the penalised linear model finds it, and trees have no advantage to claim


def test_the_command_prints_the_table_and_its_exit_status_says_whether_everything_held(capsys, tmp_path):
    import json

    from src.cli import main

    out = tmp_path / "rows.json"
    assert main(["replicate", "ledoit", "almgren", "--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert "ledoit_wolf_2004_intensity" in text and "almgren_chriss_2000_trajectory" in text and "4 of 4 held" in text and "gu_kelly" not in text
    saved = json.loads(out.read_text())
    assert [r["id"] for r in saved] == ["ledoit_wolf_2004_intensity", "ledoit_wolf_2004_loss", "almgren_chriss_2000_trajectory", "almgren_chriss_2000_frontier"] and all(r["passed"] for r in saved)
    assert main(["replicate", "no-such-benchmark"]) == 2
