"""Optimiser robustness: how stable the HERC and NCO clusters are, and how the Wasserstein ambiguity radius moves the robust CVaR book."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml import alloc_adapters as ad


def block_panel(n=1500, seed=0, within=0.7, blocks=3, per=3, vol=0.01):
    rng = np.random.default_rng(seed)
    k = blocks * per
    cols = [f"S{i}" for i in range(k)]
    f = rng.standard_normal((n, blocks))
    e = rng.standard_normal((n, k))
    x = np.zeros((n, k))
    for j in range(k):
        x[:, j] = np.sqrt(within) * f[:, j // per] + np.sqrt(1 - within) * e[:, j]
    return pd.DataFrame(x * vol, index=pd.bdate_range("2016-01-01", periods=n), columns=cols)


def test_clusters_with_a_clear_block_structure_are_stable_through_time_and_under_resampling():
    s = ad.cluster_stability(block_panel(), "herc", window=252, step=63, n_boot=25)
    assert s["clusters"]["min"] >= 2 and s["consecutive_ari"]["mean"] > 0.7 and s["bootstrap"]["ari_mean"] > 0.7
    n = ad.cluster_stability(block_panel(), "nco", window=252, step=63, n_boot=25)
    assert n["consecutive_ari"]["mean"] > 0.6 and n["book_turnover_between_dates"]["mean"] < 0.35


def test_clusters_of_unrelated_assets_are_unstable_and_the_diagnostic_says_so():
    flat = block_panel(within=0.0, seed=3)
    s = ad.cluster_stability(flat, "herc", window=252, step=63, n_boot=25)
    clear = ad.cluster_stability(block_panel(), "herc", window=252, step=63, n_boot=25)
    assert s["consecutive_ari"]["mean"] < clear["consecutive_ari"]["mean"] - 0.2 and s["bootstrap"]["ari_mean"] < clear["bootstrap"]["ari_mean"] - 0.2
    assert s["book_turnover_between_dates"]["mean"] > clear["book_turnover_between_dates"]["mean"]


def test_the_cluster_partition_is_the_librarys_own_rule():
    p = block_panel(seed=1)
    lab = ad.cluster_labels("herc", p.cov())
    groups = {tuple(sorted(np.flatnonzero(lab == g))) for g in set(lab)}
    assert groups == {(0, 1, 2), (3, 4, 5), (6, 7, 8)}
    with pytest.raises(ValueError, match="herc and nco only"):
        ad.cluster_labels("erc", p.cov())


def test_a_larger_wasserstein_radius_asks_for_a_less_concentrated_book_and_a_larger_worst_case_loss():
    rng = np.random.default_rng(5)
    n = 400
    x = np.column_stack([rng.standard_normal(n) * 0.010, rng.standard_normal(n) * 0.020, rng.standard_normal(n) * 0.015, rng.standard_normal(n) * 0.030])
    x = x - x.mean(axis=0)
    out = ad.wasserstein_radius_sensitivity(x, list("ABCD"), (0.0, 0.1, 0.5, 2.0))
    rows = out["rows"]
    assert all(r["ok"] for r in rows) and rows[0]["l1_distance_from_radius_zero"] == 0.0
    wc = [r["worst_case_cvar"] for r in rows]
    eff = [r["effective_n"] for r in rows]
    assert wc == sorted(wc) and eff[-1] >= eff[0] - 1e-9 and rows[1]["relative_radius"] == out["adapter_relative_radius"]
    assert "assumption" in out["note"]
