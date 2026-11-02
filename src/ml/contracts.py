"""Typed model outputs: what a model's number MEANS, checked where it is used (P0-09, P0-11).

A model's score is not a bare array. A ranker's score orders assets and says nothing about a return; a quantile model's score is a median; a volatility forecast is not an attractiveness score; an agent's
output is a book. ``output_kind`` names what the primary score is, ``check_signal_use`` refuses a position rule that would misread it, and the validators below enforce what each kind promises (probabilities
that sum to one, scales that are positive, quantiles that do not cross). ``reinterpret_outputs`` in the signal settings is the one explicit way to override the refusal, and it is written to the report.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1
OUTPUT_KINDS = ("expected_return", "ranking_score", "class_probability", "quantile_forecast", "predictive_distribution", "regime_probability", "latent_features", "portfolio_weights",
                "volatility_forecast", "drawdown_forecast")
UNITS = {"expected_return": "fractional return over the horizon", "ranking_score": "dimensionless; higher ranks higher; no return meaning", "class_probability": "centred class score; probabilities from predict_proba",
         "quantile_forecast": "fractional return over the horizon; the score is the median", "predictive_distribution": "fractional return over the horizon",
         "regime_probability": "probability", "latent_features": "dimensionless", "portfolio_weights": "weights as a fraction of equity",
         "volatility_forecast": "volatility of the target's own definition", "drawdown_forecast": "fractional drawdown over the horizon"}
RANKERS = {"lgbm_ranker", "xgb_ranker"}
QUANTILE_MODELS = {"lgbm_quantile", "tft"}
AGENTS = {"ppo", "sac", "recurrent_ppo"}
RANK_RULES = {"top_n_long_short", "long_only_top_quantile", "proportional", "vol_scaled", "rank_normalised", "threshold", "regime_filtered", "confidence_weighted"}
SCORE_KINDS = {"expected_return", "ranking_score", "quantile_forecast", "class_probability"}


class OutputContractError(ValueError):
    """A model output used as something it is not, or an output that breaks what its kind promises."""


def output_kind(kind: str, task: str, target: str = "") -> str:
    if kind in AGENTS:
        return "portfolio_weights"
    if target.startswith("fwd_vol"):
        return "volatility_forecast"
    if target.startswith("fwd_dd"):
        return "drawdown_forecast"
    if kind in RANKERS:
        return "ranking_score"
    if kind in QUANTILE_MODELS:
        return "quantile_forecast"
    if task == "classification":
        return "class_probability"
    return "expected_return"


def check_signal_use(kind: str, task: str, target: str, signal_cfg: dict) -> str:
    """The output kind of the model, after checking that the position rule may read it. Raises ``OutputContractError`` with what to do instead."""
    ok = output_kind(kind, task, target)
    method = signal_cfg.get("method")
    if signal_cfg.get("reinterpret_outputs"):
        return ok
    if method == "target_weights":
        if ok != "portfolio_weights":
            raise OutputContractError(f"target_weights reads the model's output as portfolio weights, but a {kind} model on {target} outputs a {ok.replace('_', ' ')}; choose a rank-based rule, or set signal.reinterpret_outputs to treat it as weights anyway")
    elif ok == "portfolio_weights":
        raise OutputContractError(f"the {kind} agent sizes its own positions (its output is portfolio weights), but the position rule '{method}' would rank and rescale them again; choose target_weights, or set signal.reinterpret_outputs to do it anyway")
    elif ok in ("volatility_forecast", "drawdown_forecast"):
        raise OutputContractError(f"a {kind} model on {target} forecasts {ok.replace('_', ' ')}, which is not how attractive an asset is; map it to a strategy explicitly, or set signal.reinterpret_outputs to rank assets by it as it is")
    elif method == "cost_aware_mean_variance":
        from .targets import parse_target

        tkind = parse_target(target)[0]
        if ok != "expected_return" or tkind not in ("fwd_ret", "excess_ret", "sector_ret", "resid_ret"):
            raise OutputContractError(f"the cost-aware optimiser weighs the forecast against risk and cost in return units, but a {kind} model on {target} outputs a {ok.replace('_', ' ')}"
                                      f"{'' if ok != 'expected_return' else ' of a ' + tkind.replace('_', ' ') + ' target'}, which is not a return; choose a rank-based rule, or set signal.reinterpret_outputs to read it as a return anyway")
        if tkind != "fwd_ret" and abs(float(signal_cfg.get("net", 0.0))) > 1e-12:
            raise OutputContractError(f"a {tkind.replace('_', ' ')} forecast is a return relative to a benchmark or group, which says nothing about the market's own direction: the optimiser can only hold it dollar neutral (set signal.net to 0)")
    elif method in RANK_RULES and ok not in SCORE_KINDS:
        raise OutputContractError(f"the position rule '{method}' cannot read a {ok.replace('_', ' ')}")
    return ok


def build_contract(kind: str, task: str, target: str, ds, cfg: dict, config_hash: str | None = None) -> dict:
    """The versioned output contract of a run, for the report. Fields that cannot be known from the configuration are None, never guessed."""
    ok = output_kind(kind, task, target)
    features = list(ds.features)
    return {"schema_version": SCHEMA_VERSION, "model_version_id": config_hash, "target_definition_hash": hashlib.sha256(json.dumps([target, ds.horizon, ds.lag]).encode()).hexdigest()[:16],
            "output_kind": ok, "units": UNITS[ok], "horizon": int(ds.horizon), "horizon_unit": "trading_days", "asset_order": list(ds.assets),
            "feature_schema_hash": hashlib.sha256(json.dumps(features).encode()).hexdigest()[:16], "decision_at": "close of day t", "information_cutoff": "close of day t",
            "available_at": None, "fit_information_cutoff": "per refit: see the folds", "normalization_version": "cross_sectional_z_v1" if cfg["dataset"].get("scaling", "cross_sectional") else None,
            "missing_value_policy": "median impute for linear and neural models, native for trees", "state_schema_version": 2 if kind in AGENTS else None,
            "class_labels": [0, 1] if task == "classification" else None, "quantile_levels": [cfg["model"]["params"].get("alpha_low", 0.1), 0.5, cfg["model"]["params"].get("alpha_high", 0.9)] if kind in QUANTILE_MODELS else None,
            "distribution_family": "normal" if kind in ("ngboost",) else None}


# ------------------------------------------------------------------------------------------------------------ validators
def validate_probabilities(p, name: str = "probabilities", atol: float = 1e-6) -> np.ndarray:
    """Finite, within [0, 1], and (for a matrix) every row summing to one."""
    a = np.asarray(p, dtype=float)
    if not np.isfinite(a).all():
        raise OutputContractError(f"{name} are not all finite")
    if ((a < -atol) | (a > 1 + atol)).any():
        raise OutputContractError(f"{name} fall outside [0, 1]")
    if a.ndim == 2 and not np.allclose(a.sum(axis=1), 1.0, atol=atol):
        raise OutputContractError(f"{name} do not sum to one on every row")
    return np.clip(a, 0.0, 1.0)


def validate_quantiles(levels, values, repair: bool = True) -> tuple[np.ndarray, float]:
    """Quantile levels strictly increasing inside (0, 1), values finite with one column per level. Crossing quantiles are rearranged by sorting (``repair``) and the share of rows that crossed is returned
    so that the repair is never invisible; with ``repair`` False a crossing raises."""
    lv, v = np.asarray(levels, dtype=float), np.asarray(values, dtype=float)
    if lv.ndim != 1 or len(lv) < 1 or not ((lv > 0) & (lv < 1)).all() or (np.diff(lv) <= 0).any():
        raise OutputContractError("quantile levels must be strictly increasing and inside (0, 1)")
    if v.ndim != 2 or v.shape[1] != len(lv):
        raise OutputContractError(f"expected one column per quantile level ({len(lv)}), got shape {v.shape}")
    if not np.isfinite(v).all():
        raise OutputContractError("quantile forecasts are not all finite")
    crossed = (np.diff(v, axis=1) < 0).any(axis=1)
    if crossed.any() and not repair:
        raise OutputContractError(f"{int(crossed.sum())} rows have crossing quantiles")
    return np.sort(v, axis=1), float(crossed.mean()) if len(v) else 0.0


def pinball_loss(y, q, level: float) -> float:
    """Mean pinball (quantile) loss of the forecasts ``q`` of the ``level`` quantile of ``y``."""
    e = np.asarray(y, dtype=float) - np.asarray(q, dtype=float)
    return float(np.mean(np.maximum(level * e, (level - 1.0) * e)))


def validate_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    """A predictive distribution frame: finite ``mean`` and strictly positive finite ``std``, and ``lower <= mean <= upper``."""
    need = {"mean", "std", "lower", "upper"}
    if not need <= set(frame.columns):
        raise OutputContractError(f"a predictive distribution needs the columns {sorted(need)}")
    if not np.isfinite(frame[list(need)].to_numpy(dtype=float)).all():
        raise OutputContractError("the predictive distribution has a value that is not finite")
    if (frame["std"] <= 0).any():
        raise OutputContractError("the predictive distribution has a scale that is not positive")
    if ((frame["lower"] > frame["mean"] + 1e-12) | (frame["mean"] > frame["upper"] + 1e-12)).any():
        raise OutputContractError("the predictive distribution has a mean outside its interval")
    return frame
