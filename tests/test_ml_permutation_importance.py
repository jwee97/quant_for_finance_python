"""Permutation importance: what it computes, and what it says when it cannot be computed.

Root cause of the chart with no bars (reproduced): on a universe of TWO assets a cross-sectional rank IC does not exist (with two assets the rank correlation is +1, -1 or undefined), so the drop in rank IC is
undefined for every feature. The server sent 20 rows whose ``mean_drop`` was null, and the chart drew an empty value axis: ECharts' default axis for no data is 0 to 1 in steps of 0.2, which is exactly the
"0.200 0.400 0.600 0.800" in the report. The fix is not a synthetic value: the server now says whether the importance is computable and why not (``permutation_status``), and the page shows that instead."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from src.ml import experiments as xe, explainability as xp, registry as rg
from tests.ml_helpers import make_bundle

warnings.filterwarnings("ignore")
BASE = {"dataset": {"feature_groups": ["price", "momentum"]}, "split": {"min_train_days": 504}, "model": {"kind": "ridge", "params": {}}}


def run(n_assets: int, tmp_path, explain=True):
    reg = rg.Registry(tmp_path / f"ml{n_assets}{explain}")
    exp = xe.run_experiment({**BASE, "name": "perm"}, bundle=make_bundle(n_assets, 1300, seed=3, alpha=1.0), registry=reg, explain=explain, robustness=False)
    return reg.report(exp.id)["importance"]


def test_with_three_or_more_assets_the_drops_are_finite_signed_and_carry_intervals(tmp_path):
    imp = run(4, tmp_path)
    st, perm = imp["permutation_status"], pd.DataFrame(imp["permutation"])
    assert st["computable"] is True and st["n_valid"] == st["n_features"] == len(perm) and st["reason"] is None and st["min_assets"] == 3
    assert np.isfinite(perm["mean_drop"]).all() and (perm["mean_drop"] > 0).any() and (perm["mean_drop"] < 0).any(), "a diverging chart needs both signs to be possible, and the sample has them"
    assert list(perm["mean_drop"]) == sorted(perm["mean_drop"], reverse=True)
    has = perm.dropna(subset=["ci_low"])
    assert len(has) and ((has["ci_low"] <= has["mean_drop"]) & (has["mean_drop"] <= has["ci_high"])).all()
    assert "does not re-read the protected outcomes" in st["sample"] and "positive" in st["sign"]


def test_with_two_assets_every_drop_is_undefined_and_the_server_says_why(tmp_path):
    imp = run(2, tmp_path)
    st, perm = imp["permutation_status"], pd.DataFrame(imp["permutation"])
    assert len(perm) > 0 and perm["mean_drop"].isna().all(), "the rows exist but carry no value: this is what drew the empty axis"
    assert st["computable"] is False and st["n_valid"] == 0 and st["n_assets"] == 2
    assert "2 asset(s)" in st["reason"] and "at least 3" in st["reason"] and "undefined for every feature" in st["reason"]


def test_a_run_without_the_explanation_says_it_was_switched_off(tmp_path):
    st = run(4, tmp_path, explain=False)["permutation_status"]
    assert st["computable"] is False and "switched off" in st["reason"]


def test_status_counts_only_finite_drops_and_names_the_partial_case():
    perm = pd.DataFrame({"mean_drop": [0.1, np.nan, -0.02, np.inf]}, index=list("abcd"))
    st = xp.permutation_status(perm, 8)
    assert st["n_valid"] == 2 and st["n_features"] == 4 and st["computable"] and "2 of 4 features have no defined drop" in st["reason"]
    assert xp.permutation_status(pd.DataFrame(), 8)["computable"] is False
    assert "at least 3" in xp.permutation_status(pd.DataFrame({"mean_drop": [np.nan]}), 2)["reason"]


def test_a_planted_feature_is_the_most_important_and_noise_is_not(tmp_path):
    """The sign convention: shuffling a feature the model needs lowers the rank IC (a positive drop); a feature with no relation to the outcome has a drop near zero, of either sign."""
    imp = run(6, tmp_path)
    perm = pd.DataFrame(imp["permutation"]).set_index("feature")
    top, bottom = perm["mean_drop"].iloc[0], perm["mean_drop"].iloc[-1]
    assert top > 0.05 and abs(bottom) < top, (top, bottom)
