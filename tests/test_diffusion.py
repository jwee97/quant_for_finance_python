import numpy as np
import pytest

from src.models.diffusion import Diffusion, cosine_schedule, kupiec_pof, pinball_loss


def test_schedule_is_valid_and_noises_to_nearly_pure_noise():
    b = cosine_schedule(100)
    assert (b > 0).all() and (b < 1).all() and np.prod(1 - b) < 1e-3


def test_kupiec_is_calibrated_and_detects_miscoverage():
    rng = np.random.default_rng(0)
    p_values = [kupiec_pof(int(rng.binomial(500, 0.05)), 500, 0.05)["p_value"] for _ in range(400)]
    assert 0.02 < np.mean(np.array(p_values) < 0.05) < 0.12             # about 5% false rejections under the null (the test is exact only asymptotically)
    assert kupiec_pof(60, 500, 0.05)["p_value"] < 0.001                 # 12% exceedances against a 5% promise
    assert kupiec_pof(25, 500, 0.05)["p_value"] == pytest.approx(1.0)   # exactly the promised count


def test_pinball_loss_is_minimised_by_the_true_quantile():
    rng = np.random.default_rng(1)
    y = rng.normal(size=20_000)
    grid = np.linspace(-3, 0, 61)
    losses = [pinball_loss(y, q, 0.05).mean() for q in grid]
    assert abs(grid[int(np.argmin(losses))] - (-1.645)) < 0.15


def test_diffusion_recovers_the_covariance_and_the_fat_tail_direction_of_a_known_distribution():
    pytest.importorskip("torch")
    rng = np.random.default_rng(2)
    L = np.array([[1.0, 0, 0], [0.8, 0.6, 0], [-0.3, 0.2, 0.9]])
    X = rng.standard_t(6, size=(1500, 3)) @ L.T
    model = Diffusion(steps=60, epochs=150, seed=3).fit(X)
    S = model.sample(3000, seed=5)
    assert S.shape == (3000, 3) and np.isfinite(S).all()
    assert np.abs(np.corrcoef(S.T) - np.corrcoef(X.T)).max() < 0.12
    assert np.abs(S.std(axis=0) / X.std(axis=0) - 1).max() < 0.15
    assert np.allclose(model.sample(10, seed=9), model.sample(10, seed=9))       # deterministic for a seed


def test_stage36_decision_matches_the_declared_rule():
    import json
    from pathlib import Path

    import pandas as pd

    root = Path(__file__).resolve().parents[1]
    path = root / "reports" / "tables" / "stage36_var_tests.csv"
    if not path.exists():
        pytest.skip("stage 36 not run")
    t = pd.read_csv(path)
    declared = t[t["status"] == "declared"]
    assert len(declared) == 2 and set(declared["b"]) == {"gaussian", "bootstrap"}
    expected = bool(declared["passes"].astype(bool).all())
    entry = next(json.loads(l) for l in (root / "experiments" / "registry.jsonl").read_text().splitlines() if '"stage36_diffusion"' in l)
    assert entry["decision"] == ("retain" if expected else "reject")
    s = pd.read_csv(root / "reports" / "tables" / "stage36_var_summary.csv")
    assert (s["n_origins"] == s["n_origins"].iloc[0]).all()
