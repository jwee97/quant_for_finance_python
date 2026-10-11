"""Configuration of the machine-learning platform: the defaults, ``config/ml_research.yaml`` merged over them, and the experiment configuration that is saved beside every result.

An experiment is described by one nested dictionary with the sections ``dataset``, ``split``, ``model``, ``signal``, ``backtest`` and (optionally) ``robustness``. ``experiment_config`` merges what you give
over the platform defaults, validates it and returns a plain dictionary; ``config_hash`` is a short fingerprint of the merged configuration, so two runs with the same hash are the same experiment, and
``dataset_key`` fingerprints only the part that decides the data (used to count how many models have been tried on the same data).
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "ml_research.yaml"

DEFAULTS: dict = {
    "registry": {"root": "artifacts/ml"},
    "dataset": {"universe": "etf", "feature_groups": ["price", "momentum", "mean_reversion", "volatility", "volume", "relative", "macro"], "target": "fwd_ret_5", "horizon": 5, "lag": 1,
                "min_history": 252, "winsor": 0.01, "normalize": "cross_sectional_z", "benchmark": "equal_weight", "start": None, "end": None, "regime_detector": None},
    "split": {"scheme": "walk_forward", "window": "expanding", "rolling_days": 1260, "min_train_days": 756, "refit_every": 21, "embargo": 0, "validation_fraction": 0.20, "test_fraction": 0.15,
              "forward_fraction": 0.15},
    "model": {"kind": "ridge", "params": {}},
    "signal": {"method": "top_n_long_short", "n": 3, "quantile": 0.3, "threshold": 0.0, "gross": 1.0, "net": 0.0, "max_weight": 0.35, "sector_cap": None, "holding_period": 5},
    "backtest": {"rebalance": "weekly", "cost_bps": 5.0, "slippage_bps": 2.0, "target_vol": 0.10, "max_leverage": 1.5, "extra_lag": 0},
    "robustness": {"enabled": False, "n_draws": 20, "cost_multiples": [1, 2, 3, 5], "delays": [0, 1, 2, 5], "noise_levels": [0.0, 0.25, 0.5, 1.0], "seed": 0},
    "search": {"n_random": 12, "n_bayes": 12, "generations": 3, "population": 8, "objective_weights": {"turnover": 0.5, "max_drawdown": 0.3, "instability": 0.2, "train_validation_gap": 0.5}, "top_k": 3},
    "forward": {"horizon_days": 5, "track_window": 63, "psi_alert": 0.25},
}
SECTIONS = ("dataset", "split", "model", "signal", "backtest", "robustness")
HASH_SECTIONS = ("dataset", "split", "model", "signal", "backtest")                  # what decides the numbers; the robustness switch only decides which extra reports are written
NORMALIZATIONS = ("none", "cross_sectional_z", "rank")
SCHEMES = ("walk_forward", "expanding", "rolling")
REBALANCES = ("daily", "weekly", "monthly")
HOLDING_PERIODS = (1, 5, 20)


class MLConfigError(ValueError):
    """A configuration the user can fix; the message says which field and why."""


def merge(base: dict, over: dict | None) -> dict:
    """``over`` laid on ``base`` recursively (dictionaries merge, everything else is replaced); neither input is modified."""
    out = copy.deepcopy(base)
    for key, value in (over or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict) and key != "params":
            out[key] = merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_ml_config(path: str | Path | None = None) -> dict:
    """The platform defaults with ``config/ml_research.yaml`` (or ``path``) merged over them; a missing or empty file leaves the defaults."""
    p = Path(path) if path else CONFIG_PATH
    loaded = yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else None
    return merge(DEFAULTS, loaded if isinstance(loaded, dict) else {})


def _check(cond: bool, message: str) -> None:
    if not cond:
        raise MLConfigError(message)


def validate(cfg: dict) -> dict:
    """Refuse an experiment that cannot mean what it says. Returns ``cfg`` for chaining."""
    d, s, sig, bt = cfg["dataset"], cfg["split"], cfg["signal"], cfg["backtest"]
    _check(int(d["horizon"]) >= 1 and int(d["horizon"]) <= 252, "dataset.horizon must be between 1 and 252 days")
    _check(int(d["lag"]) >= 0 and int(d["lag"]) <= 20, "dataset.lag must be between 0 and 20 days (the backtest engine's signal lag is 1)")
    _check(d["normalize"] in NORMALIZATIONS, f"dataset.normalize must be one of {NORMALIZATIONS}")
    _check(0.0 <= float(d["winsor"]) < 0.25, "dataset.winsor must be in [0, 0.25)")
    _check(int(d["min_history"]) >= 30, "dataset.min_history must be at least 30 days")
    _check(bool(d["feature_groups"]), "choose at least one feature group")
    _check(s["scheme"] in SCHEMES, f"split.scheme must be one of {SCHEMES}")
    _check(s["window"] in ("expanding", "rolling"), "split.window must be expanding or rolling")
    _check(int(s["refit_every"]) >= 1 and int(s["min_train_days"]) >= 60, "split.refit_every >= 1 and split.min_train_days >= 60")
    for k in ("validation_fraction", "test_fraction", "forward_fraction"):
        _check(0.0 <= float(s[k]) < 0.6, f"split.{k} must be in [0, 0.6)")
    _check(float(s["validation_fraction"]) + float(s["test_fraction"]) + float(s["forward_fraction"]) <= 0.7, "validation + test + forward fractions must leave at least 30% for the first training window")
    _check(bt["rebalance"] in REBALANCES, f"backtest.rebalance must be one of {REBALANCES}")
    _check(float(bt["cost_bps"]) >= 0 and float(bt["slippage_bps"]) >= 0 and float(bt["cost_bps"]) + float(bt["slippage_bps"]) <= 500, "costs must be non-negative and total at most 500 bps")
    _check(bt["target_vol"] is None or 0.01 <= float(bt["target_vol"]) <= 1.0, "backtest.target_vol must be between 1% and 100%, or null")
    _check(int(bt["extra_lag"]) >= 0 and int(bt["extra_lag"]) <= 20, "backtest.extra_lag must be between 0 and 20")
    _check(int(sig["holding_period"]) >= 1 and int(sig["holding_period"]) <= 60, "signal.holding_period must be between 1 and 60 days")
    _check(0.0 < float(sig["max_weight"]) <= 1.0 and float(sig["gross"]) > 0, "signal.max_weight in (0, 1] and signal.gross > 0")
    _check(int(sig["n"]) >= 1 and 0.0 < float(sig["quantile"]) <= 0.5, "signal.n >= 1 and signal.quantile in (0, 0.5]")
    _check(d.get("regime_detector") in (None, "vol_state", "hmm", "gmm"), "dataset.regime_detector must be null, vol_state, hmm or gmm")
    _check(isinstance(cfg["model"].get("kind"), str) and bool(cfg["model"]["kind"]), "model.kind is required")
    _check(cfg["model"]["kind"] != "regime_ridge" or bool(d.get("regime_detector")), "the regime_ridge model needs dataset.regime_detector")
    _check(isinstance(cfg["model"].get("params", {}), dict), "model.params must be a mapping")
    return cfg


def experiment_config(overrides: dict | None = None, name: str | None = None, defaults: dict | None = None) -> dict:
    """The full, validated configuration of one experiment: ``overrides`` over the platform configuration. ``name`` (else generated from the model, target and universe) labels it."""
    base = defaults if defaults is not None else load_ml_config()
    unknown = set(overrides or {}) - set(SECTIONS) - {"name", "seed", "notes", "registry", "search", "forward"}
    _check(not unknown, f"unknown configuration sections {sorted(unknown)}; allowed: {list(SECTIONS)} plus name, seed and notes")
    cfg = merge(base, overrides)
    cfg["name"] = str(name or (overrides or {}).get("name") or f"{cfg['model']['kind']}_{cfg['dataset']['target']}_{_universe_label(cfg['dataset']['universe'])}")[:80]
    cfg["seed"] = int((overrides or {}).get("seed", cfg.get("seed", 0)))
    cfg["notes"] = str((overrides or {}).get("notes", cfg.get("notes", "")))[:500]
    cfg["dataset"]["horizon"], cfg["dataset"]["lag"] = int(cfg["dataset"]["horizon"]), int(cfg["dataset"]["lag"])
    return validate(cfg)


def _universe_label(universe) -> str:
    return universe if isinstance(universe, str) else f"{len(universe)}tickers"


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))


def config_hash(cfg: dict) -> str:
    """A 12-character fingerprint of everything that decides an experiment's result (the name and notes do not)."""
    keep = {k: cfg[k] for k in HASH_SECTIONS + ("seed",) if k in cfg}
    return hashlib.sha256(_canon(keep).encode()).hexdigest()[:12]


def dataset_key(cfg: dict) -> str:
    """A fingerprint of the data a model sees (the dataset section only): experiments sharing it are looks at the same data, which is what the trial count of a deflated Sharpe ratio counts."""
    return hashlib.sha256(_canon(cfg["dataset"]).encode()).hexdigest()[:12]
