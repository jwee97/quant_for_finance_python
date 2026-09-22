"""Shared alpha-research harness used by Stages 3, 4 and 5.

One function evaluates a whole signal family the same way every time, so that
momentum and mean reversion are judged on identical terms and neither gets an
accidental advantage from a different evaluation protocol.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.returns import forward_returns
from src.models.regression import (
    cross_sectional_ic,
    ic_summary,
    per_asset_regression,
    pooled_panel_regression,
    time_series_ic,
)


def evaluate_family(
    features: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    horizons=(1, 5, 10, 21, 42, 63),
    breadth: int | None = None,
    sample: tuple | None = None,
) -> pd.DataFrame:
    """IC statistics for every (feature, horizon) pair.

    ``sample`` optionally restricts the evaluation to a date window, which is
    how the development sample is kept separate from the validation sample.
    """
    forward_cache = {h: forward_returns(returns, h) for h in horizons}
    rows = []
    for name, feature in features.items():
        for horizon in horizons:
            forward = forward_cache[horizon]
            signal, target = feature, forward
            if sample is not None:
                start, end = sample
                signal = signal.loc[(signal.index >= start) & (signal.index <= end)]
                target = target.loc[(target.index >= start) & (target.index <= end)]
            ic = cross_sectional_ic(signal, target)
            summary = ic_summary(ic, horizon=horizon, breadth=breadth)
            if not summary:
                continue
            ts = time_series_ic(signal, target)
            summary.update(
                {
                    "feature": name,
                    "horizon": horizon,
                    "ts_ic_mean": float(ts.mean()) if len(ts) else np.nan,
                    "ts_ic_share_positive": float((ts > 0).mean()) if len(ts) else np.nan,
                    "n_assets_evaluated": int(len(ts)),
                }
            )
            rows.append(summary)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    front = ["feature", "horizon", "mean_ic", "std_ic", "ic_ir", "t_stat_overlap_adjusted",
             "p_value", "hit_rate", "ts_ic_mean", "ts_ic_share_positive"]
    ordered = front + [c for c in out.columns if c not in front]
    return out.loc[:, ordered].sort_values(["feature", "horizon"]).reset_index(drop=True)


def regression_study(feature: pd.DataFrame, returns: pd.DataFrame, horizon: int = 21) -> dict:
    """Pooled and per-asset regressions of forward returns on one feature."""
    forward = forward_returns(returns, horizon)
    pooled = pooled_panel_regression(feature, forward, horizon)
    per_asset = per_asset_regression(feature, forward, horizon)
    return {
        "pooled": pooled,
        "per_asset": per_asset,
        "share_negative_beta": float((per_asset["beta"] < 0).mean()) if len(per_asset) else np.nan,
        "share_significant": float((per_asset["p_beta"] < 0.05).mean()) if len(per_asset) else np.nan,
    }


def decide(mean_ic: float, p_value: float, expected_sign: int = 1,
           alpha: float = 0.05, weak_alpha: float = 0.20) -> tuple[str, str]:
    """Turn evidence into a decision, using a rule fixed before looking.

    Writing the rule down in code rather than applying judgement per result is
    what stops the decision from drifting to fit whatever the data produced.
    """
    if not np.isfinite(mean_ic) or not np.isfinite(p_value):
        return "reject", "insufficient evidence to evaluate"
    right_sign = np.sign(mean_ic) == np.sign(expected_sign)
    if right_sign and p_value < alpha:
        return "retain", f"IC {mean_ic:+.4f} has the hypothesised sign and p={p_value:.4f} < {alpha}"
    if right_sign and p_value < weak_alpha:
        return "investigate", f"IC {mean_ic:+.4f} has the right sign but p={p_value:.4f} is weak"
    if not right_sign and p_value < alpha:
        return "reject", f"IC {mean_ic:+.4f} is significant with the WRONG sign (p={p_value:.4f})"
    return "reject", f"IC {mean_ic:+.4f}, p={p_value:.4f}: no evidence of predictive content"


def best_by_horizon(table: pd.DataFrame, horizon: int, by: str = "mean_ic",
                    ascending: bool = False) -> pd.Series:
    subset = table[table["horizon"] == horizon]
    if subset.empty:
        return pd.Series(dtype=float)
    return subset.sort_values(by, ascending=ascending).iloc[0]


def multiple_testing_adjustment(table: pd.DataFrame, fdr: float = 0.10,
                                p_column: str = "p_value") -> pd.DataFrame:
    """Benjamini-Hochberg FDR control across a whole signal family.

    A family of 12 features at 6 horizons is 72 tests. At a naive 5% threshold
    roughly 4 of them are expected to look significant even if every signal is
    pure noise, which is precisely how a parameter grid manufactures a
    discovery (Ch. 22 §22.2.4). BH controls the expected share of false
    positives among the rejected hypotheses instead of the per-test error
    rate, so the surviving claims mean something at the family level.

    Adds ``p_rank``, ``bh_threshold``, ``significant_raw`` and
    ``significant_bh`` columns; the table is returned sorted by p-value.
    """
    if table.empty or p_column not in table:
        return table
    out = table.sort_values(p_column).reset_index(drop=True).copy()
    n = len(out)
    out["p_rank"] = np.arange(1, n + 1)
    out["bh_threshold"] = out["p_rank"] / n * fdr
    out["significant_raw"] = out[p_column] < 0.05
    below = out[p_column] <= out["bh_threshold"]
    # BH: reject every hypothesis up to the largest rank that clears its own
    # threshold, not just the ones that individually clear it.
    cutoff = int(out.index[below].max()) if bool(below.any()) else -1
    out["significant_bh"] = out.index <= cutoff
    return out


def family_verdict(table: pd.DataFrame, fdr: float = 0.10, expected_sign: int = 1) -> dict:
    """Summarise what survives multiple-testing control for a whole family."""
    adjusted = multiple_testing_adjustment(table, fdr)
    if adjusted.empty:
        return {"n_tests": 0, "n_significant_raw": 0, "n_significant_bh": 0}
    right_sign = np.sign(adjusted["mean_ic"]) == np.sign(expected_sign)
    survivors = adjusted[adjusted["significant_bh"] & right_sign]
    return {
        "n_tests": int(len(adjusted)),
        "n_significant_raw": int(adjusted["significant_raw"].sum()),
        "n_expected_false_positives_at_5pct": float(0.05 * len(adjusted)),
        "n_significant_bh": int(adjusted["significant_bh"].sum()),
        "n_survivors_right_sign": int(len(survivors)),
        "survivors": [f"{r.feature}@h{int(r.horizon)}" for r in survivors.itertuples()][:10],
        "min_p_value": float(adjusted["p_value"].min()),
        "table": adjusted,
    }
