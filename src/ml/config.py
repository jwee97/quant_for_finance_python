"""Configuration of the machine-learning platform: the defaults, ``config/ml_research.yaml`` merged over them, and the experiment configuration that is saved beside every result.

An experiment is described by one nested dictionary with the sections ``dataset``, ``split``, ``model``, ``signal``, ``backtest`` and (optionally) ``robustness``. ``experiment_config`` merges what you give
over the platform defaults, validates it and returns a plain dictionary; ``config_hash`` is a short fingerprint of the merged configuration, so two runs with the same hash are the same experiment, and
``dataset_key`` fingerprints the sample (universe and dates; used to count how many models have been tried on the same data) and ``build_key`` the whole dataset section (used to reuse a built panel).
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
    "signal": {"method": "top_n_long_short", "n": 3, "quantile": 0.3, "threshold": 0.0, "gross": 1.0, "net": 0.0, "max_weight": 0.35, "sector_cap": None, "holding_period": 5,
               "regime_on_states": [0], "reinterpret_outputs": False},
    "backtest": {"rebalance": "weekly", "cost_bps": 5.0, "slippage_bps": 2.0, "target_vol": 0.10, "max_leverage": 1.5, "extra_lag": 0},
    "robustness": {"enabled": False, "n_draws": 20, "cost_multiples": [1, 2, 3, 5], "delays": [0, 1, 2, 5], "noise_levels": [0.0, 0.25, 0.5, 1.0], "seed": 0},
    "search": {"n_random": 12, "n_bayes": 12, "generations": 3, "population": 8, "objective_weights": {"turnover": 0.5, "max_drawdown": 0.3, "instability": 0.2, "train_validation_gap": 0.5}, "top_k": 3},
    "forward": {"horizon_days": 5, "track_window": 63, "psi_alert": 0.25},
}
SECTIONS = ("dataset", "split", "model", "signal", "backtest", "robustness", "execution")
HASH_SECTIONS = ("dataset", "split", "model", "signal", "backtest", "execution")      # what decides the numbers (``execution`` only when present: it is absent from the defaults, so no existing hash changes); the robustness switch only decides which extra reports are written
NORMALIZATIONS = ("none", "cross_sectional_z", "rank")
SCHEMES = ("walk_forward", "expanding", "rolling")
REBALANCES = ("daily", "weekly", "monthly")
HOLDING_PERIODS = (1, 5, 10, 20)


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
    _check(s.get("sample_weighting", "none") in ("none", "uniqueness"), "split.sample_weighting must be none or uniqueness (absent from the defaults so that no existing configuration hash changes)")
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
    from .risk_models import METHODS as COV_METHODS

    # the optional settings of the Release 6 and 7 rules: absent from the defaults, so no existing configuration hash changes
    _check(0.0 <= float(sig.get("risk_aversion", 5.0)) <= 100.0 and float(sig.get("turnover_penalty", 0.0)) >= 0.0, "signal.risk_aversion in [0, 100] and signal.turnover_penalty >= 0")
    _check(sig.get("cov_method", "lw_constant_correlation") in COV_METHODS, f"signal.cov_method must be one of {COV_METHODS}")
    _check(60 <= int(sig.get("cov_window", 252)) <= 1000 and 1 <= int(sig.get("opt_every") or 1) <= 60, "signal.cov_window in [60, 1000] and signal.opt_every in [1, 60]")
    _check(sig.get("alloc_fallback", "none") in ("none", "hold", "equal_weight"), "signal.alloc_fallback must be none, hold or equal_weight")
    if sig["method"] == "cost_aware_mean_variance":
        _check(int(sig["holding_period"]) == 1, "the cost-aware optimiser trades edge against cost itself: averaging its books over a holding period would change the units of its cost term (set signal.holding_period to 1)")
    if sig.get("no_trade_margin") is not None:
        _check(float(sig["no_trade_margin"]) >= 0.0, "signal.no_trade_margin is not negative (a multiple of the incremental cost)")
        _check(sig.get("no_trade_hold_margin") is None or 0.0 <= float(sig["no_trade_hold_margin"]) <= float(sig["no_trade_margin"]), "signal.no_trade_hold_margin is between 0 and the margin: a trading episode may continue on less than it took to start")
    if sig.get("trade_filter_k") is not None:
        _check(0.0 <= float(sig["trade_filter_k"]) <= 10.0 and float(sig.get("trade_filter_cost_margin", 1.0)) >= 0.0 and 40 <= int(sig.get("trade_filter_window", 252)) <= 1000,
               "signal.trade_filter_k in [0, 10], trade_filter_cost_margin >= 0 and trade_filter_window in [40, 1000]")
    if sig.get("regime_exposure") is not None:
        from .regime_exposure import ENGINES

        _check(sig["regime_exposure"] in ENGINES, f"signal.regime_exposure must be one of {list(ENGINES)}: only causal (filtered) states can set an exposure, never smoothed probabilities or a calendar of crises chosen with hindsight")
        _check(2 <= int(sig.get("regime_states", 3)) <= 5 and 0.0 <= float(sig.get("regime_floor", 0.25)) <= float(sig.get("regime_cap", 1.0)) <= 1.0 and float(sig.get("regime_cap", 1.0)) > 0.0,
               "signal.regime_states in [2, 5] and 0 <= regime_floor <= regime_cap <= 1 (the overlay lowers a book, it never levers it)")
        _check(5 <= int(sig.get("regime_refit_every", 63)) <= 252 and 20 <= int(sig.get("regime_min_days", 60)) <= 500 and 60 <= int(sig.get("regime_min_train", 252)) <= 1000,
               "signal.regime_refit_every in [5, 252], regime_min_days in [20, 500] and regime_min_train in [60, 1000]")
    if sig["method"] == "calibrated_sizing":
        _check(60 <= int(sig.get("calib_window", 504)) <= 2000 and float(sig.get("calib_min_t", 2.0)) >= 0.0 and 5 <= int(sig.get("stability_window", 21)) <= 250,
               "signal.calib_window in [60, 2000], calib_min_t >= 0 and stability_window in [5, 250]")
    derived = sig["method"] in ("cost_aware_mean_variance", "calibrated_sizing") or sig.get("no_trade_margin") is not None or sig.get("trade_filter_k") is not None
    if derived:                                                                             # what these rules need from the other sections, set here so that one place decides each of them
        sig["cost_bps"] = float(bt["cost_bps"]) + float(bt["slippage_bps"])
        sig["window_days"] = int(d["horizon"])                                               # the forecast horizon: the window the covariance is scaled to
        sig["label_offset"] = int(d["lag"]) + int(d["horizon"])                             # outcomes settle this many sessions after the decision
        sig["rebalance"] = bt["rebalance"]
        sig["lag_total"] = int(d["lag"]) + int(bt["extra_lag"])
    else:
        for k in ("cost_bps", "window_days", "label_offset", "rebalance", "lag_total"):
            sig.pop(k, None)
    _check(d.get("regime_detector") in (None, "vol_state", "hmm", "gmm"), "dataset.regime_detector must be null, vol_state, hmm or gmm")
    if d.get("sessions") is not None:                                                  # optional, and absent from the defaults so that no existing configuration hash changes
        from .sessions import SessionError, validate_settings

        try:
            validate_settings(d["sessions"])
        except SessionError as exc:
            raise MLConfigError(str(exc)) from None
    _check(isinstance(cfg["model"].get("kind"), str) and bool(cfg["model"]["kind"]), "model.kind is required")
    _check(cfg["model"]["kind"] != "regime_ridge" or bool(d.get("regime_detector")), "the regime_ridge model needs dataset.regime_detector")
    _check(isinstance(cfg["model"].get("params", {}), dict), "model.params must be a mapping")
    if cfg.get("execution") is not None:                                               # optional, and absent from the defaults so that no existing configuration hash changes
        from .execution import validate_settings

        try:
            validate_settings(cfg["execution"])
        except ValueError as exc:
            raise MLConfigError(str(exc)) from None
        if cfg["execution"].get("mode", "realistic_cash_equity") == "vectorized_legacy":
            del cfg["execution"]                                                        # the legacy mode is the absence of the section: one meaning, one hash
        else:
            _check(int(d["lag"]) + int(bt["extra_lag"]) >= 1, "realistic execution needs dataset.lag + backtest.extra_lag >= 1: a decision made from a close cannot trade at that close")
    return cfg


def experiment_config(overrides: dict | None = None, name: str | None = None, defaults: dict | None = None) -> dict:
    """The full, validated configuration of one experiment: ``overrides`` over the platform configuration. ``name`` (else generated from the model, target and universe) labels it."""
    base = defaults if defaults is not None else load_ml_config()
    unknown = set(overrides or {}) - set(SECTIONS) - {"name", "seed", "notes", "registry", "search", "forward", "study"}
    _check(not unknown, f"unknown configuration sections {sorted(unknown)}; allowed: {list(SECTIONS)} plus name, seed, notes and study")
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
    """A fingerprint of the SAMPLE a configuration looks at: the universe and the date range, not which features or target it builds from it. Experiments sharing it are looks at the same data, which is what
    the trial count of a deflated Sharpe ratio counts: trying five horizons and three feature sets on the same ETFs is eight looks, not eight different datasets."""
    d = cfg["dataset"]
    return hashlib.sha256(_canon({"universe": d["universe"], "start": d.get("start"), "end": d.get("end")}).encode()).hexdigest()[:12]


def build_key(cfg: dict) -> str:
    """A fingerprint of the whole ``dataset`` section: two configurations with the same build key build the same panel from the same data, so one can reuse the other's."""
    return hashlib.sha256(_canon(cfg["dataset"]).encode()).hexdigest()[:12]
