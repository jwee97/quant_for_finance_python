from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.factors import load_factors

ROOT = Path(__file__).resolve().parents[1]


def test_french_factors_load_as_decimals_and_match_the_manifest():
    import hashlib
    import json

    d = ROOT / "data" / "raw" / "factors"
    if not d.exists():
        pytest.skip("factors not cached")
    f = load_factors(d)
    assert {"Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF", "Mom"} <= set(f.columns)
    assert f.index.is_monotonic_increasing and f["Mkt-RF"].abs().max() < 0.25 and f["RF"].between(0, 0.001).all()
    manifest = json.loads((ROOT / "data" / "metadata" / "factors_manifest.json").read_text())
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((d / name).read_bytes()).hexdigest() == digest


def test_regression_recovers_planted_loadings_and_intercept():
    from experiments.stage40_factors import FACTORS, regress

    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2015-01-01", periods=3000)
    F = pd.DataFrame(rng.normal(0, 0.01, (3000, 6)), index=idx, columns=FACTORS)
    y = 0.0002 + 0.6 * F["Mkt-RF"] - 0.3 * F["HML"] + rng.normal(0, 0.002, 3000)
    res = regress(pd.Series(y, index=idx), F)
    assert abs(res["beta_Mkt-RF"] - 0.6) < 0.03 and abs(res["beta_HML"] + 0.3) < 0.03 and abs(res["alpha_annual"] - 0.0504) < 0.02
    assert res["r_squared"] > 0.9


def test_momentum_books_load_on_the_momentum_factor():
    path = ROOT / "reports" / "tables" / "stage40_factor_regressions.csv"
    if not path.exists():
        pytest.skip("stage 40 not run")
    t = pd.read_csv(path, index_col=0)
    assert t.loc["M3_momentum", "t_Mom"] > 5 and t.loc["M0_equal_weight", "beta_Mkt-RF"] > 0.3
