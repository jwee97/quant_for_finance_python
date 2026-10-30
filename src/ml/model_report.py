"""What a model reports beyond its predictions, for the experiment report: training curves, ranking metrics, prediction intervals, regime probabilities, forecast heads, ensemble weights, latent factors, the
graph, an agent's rewards, and the cost and provenance of the run (resources, library versions, seed, artifact checksums).

Every section is built from out-of-sample material only (the folds' predictions and the labels that came after them) and is optional: a model that does not report a thing simply has no such section, and
a section that cannot be computed is left out rather than failing the experiment.
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys

import numpy as np
import pandas as pd

from . import evaluation as ev
from .models import _INFO
from .targets import forward_return

MAX_POINTS = 300
LIBRARIES = ("numpy", "pandas", "scipy", "sklearn", "torch", "lightgbm", "xgboost", "ngboost", "statsmodels", "hmmlearn", "gymnasium", "stable_baselines3", "sb3_contrib", "optuna")


def _thin(n: int, limit: int = MAX_POINTS) -> np.ndarray:
    return np.unique(np.linspace(0, n - 1, min(n, limit)).round().astype(int)) if n else np.array([], dtype=int)


def _dates(index) -> list[str]:
    return [str(pd.Timestamp(x).date()) for x in index]


def _num(v):
    return None if v is None or not np.isfinite(v) else float(v)


# ------------------------------------------------------------------------------------------------------------ training
def training_section(diagnostics: list) -> dict | None:
    rows = [d for d in diagnostics if d.get("train_loss")]
    if not rows:
        return None
    pick = sorted({0, len(rows) // 2, len(rows) - 1})
    folds = [{"fold": rows[i]["fold"], "train_loss": rows[i]["train_loss"], "val_loss": rows[i]["val_loss"], "best_epoch": rows[i]["best_epoch"], "stopped_early": rows[i]["stopped_early"],
              "lr": rows[i].get("lr")} for i in pick]
    return {"folds": folds, "n_fits": len(rows), "epochs_median": float(np.median([d["epochs_run"] for d in rows])), "stopped_early_share": float(np.mean([bool(d["stopped_early"]) for d in rows])),
            "best_val_loss": float(np.mean([min(d["val_loss"]) for d in rows])), "train_seconds": float(sum(d.get("seconds", 0.0) for d in rows))}


def rewards_section(diagnostics: list) -> dict | None:
    rows = [d for d in diagnostics if d.get("rewards")]
    if not rows:
        return None
    first, last = rows[0], rows[-1]
    return {"algo": first["algo"], "reward": first["reward"], "timesteps": first["timesteps"], "constraints": first["constraints"], "curve": {"timesteps": last["rewards"]["timesteps"], "mean_reward": last["rewards"]["mean_reward"]},
            "first_curve": {"timesteps": first["rewards"]["timesteps"], "mean_reward": first["rewards"]["mean_reward"]}, "in_sample": last["train_curve"], "fits": len(rows)}


# ------------------------------------------------------------------------------------------------------------ ranking
def ranking_section(scores: pd.DataFrame, fwd: pd.DataFrame, labels: pd.Series | None, levels: int = 5, seed: int = 0) -> dict | None:
    """NDCG@k and precision@k of the daily ordering against the realised return buckets, with what a random ordering scores for comparison."""
    both = scores.notna() & fwd.notna()
    s, f = scores.where(both), fwd.where(both)
    counts = both.sum(axis=1)
    counts = counts[counts > 0]
    n_assets = int(counts.median()) if len(counts) else 0
    if n_assets < 4:
        return None
    k = max(1, round(n_assets / 4))
    rng = np.random.default_rng(seed)
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    ndcg, rand, prec, days = [], [], [], []
    for date in s.index[both.sum(axis=1) >= max(4, n_assets // 2)]:
        a, b = s.loc[date].dropna().to_numpy(), f.loc[date].dropna().to_numpy()
        m = len(a)
        grade = np.clip(np.ceil(pd.Series(b).rank(pct=True).to_numpy() * levels).astype(int) - 1, 0, levels - 1)
        gain = 2.0 ** grade - 1.0
        kk = min(k, m)
        order = np.argsort(-a)[:kk]
        ideal = np.sort(gain)[::-1][:kk]
        idcg = float((ideal * disc[:kk]).sum())
        if idcg <= 0:
            continue
        ndcg.append(float((gain[order] * disc[:kk]).sum() / idcg))
        rand.append(float(np.mean([(gain[rng.permutation(m)[:kk]] * disc[:kk]).sum() for _ in range(8)]) / idcg))
        prec.append(float(np.mean(grade[order] >= levels - 2)))
        days.append(date)
    if not ndcg:
        return None
    frame = pd.DataFrame({"ndcg": ndcg, "random": rand, "precision": prec}, index=pd.DatetimeIndex(days))
    out = {"k": k, "levels": levels, "ndcg": float(frame["ndcg"].mean()), "ndcg_random": float(frame["random"].mean()), "precision_top": float(frame["precision"].mean()),
           "precision_random": float(2.0 / levels), "n_days": int(len(frame))}
    if labels is not None:
        lab = labels.reindex(frame.index)
        out["by_segment"] = {seg: {"ndcg": float(frame.loc[lab == seg, "ndcg"].mean()), "precision": float(frame.loc[lab == seg, "precision"].mean()), "n_days": int((lab == seg).sum())}
                             for seg in ("validation", "test", "forward") if (lab == seg).sum() >= 10}
    roll = frame["ndcg"].rolling(63, min_periods=20).mean().dropna()
    idx = _thin(len(roll))
    out["series"] = {"dates": _dates(roll.index[idx]), "ndcg": [_num(v) for v in roll.to_numpy()[idx]], "random": float(frame["random"].mean())}
    return out


# ------------------------------------------------------------------------------------------------------------ distributions
def intervals_section(dist: pd.DataFrame, y: pd.Series, nominal: float, labels: pd.Series | None) -> dict | None:
    y = y.reindex(dist.index)
    ok = y.notna()
    if ok.sum() < 30:
        return None
    d, t = dist[ok], y[ok]
    inside = (t >= d["lower"]) & (t <= d["upper"])
    z = (t - d["mean"]) / d["std"].replace(0.0, np.nan)
    out = {"nominal": float(nominal), "coverage": float(inside.mean()), "mean_width": float((d["upper"] - d["lower"]).mean()), "rmse": float(np.sqrt(((t - d["mean"]) ** 2).mean())),
           "z_std": _num(z.std()), "n": int(len(d))}
    if labels is not None:
        seg = labels.reindex(d.index.get_level_values(0)).to_numpy()
        out["by_segment"] = {s: {"coverage": float(inside.to_numpy()[seg == s].mean()), "n": int((seg == s).sum())} for s in ("validation", "test", "forward") if (seg == s).sum() >= 30}
    q = pd.qcut(d["std"].rank(method="first"), 5, labels=False)                                   # does a larger predicted spread go with larger errors?
    out["by_uncertainty"] = [{"bucket": int(b) + 1, "mean_std": float(d["std"][q == b].mean()), "rmse": float(np.sqrt(((t - d["mean"])[q == b] ** 2).mean())), "coverage": float(inside[q == b].mean())}
                             for b in sorted(q.unique())]
    assets = d.index.get_level_values(1).value_counts().index[:5]
    series = {}
    for a in sorted(assets):
        sub = d.xs(a, level=1)
        idx = _thin(len(sub), 160)
        sub = sub.iloc[idx]
        actual = t.xs(a, level=1).reindex(sub.index)
        series[a] = {"dates": _dates(sub.index), "mean": [_num(v) for v in sub["mean"]], "lower": [_num(v) for v in sub["lower"]], "upper": [_num(v) for v in sub["upper"]], "actual": [_num(v) for v in actual]}
    out["series"] = series
    return out


def regimes_section(reg: pd.DataFrame, diag: dict | None, labels: pd.Series | None) -> dict | None:
    if reg is None or reg.empty:
        return None
    idx = _thin(len(reg))
    hard = reg.to_numpy().argmax(axis=1)
    runs = np.diff(np.flatnonzero(np.diff(np.r_[-1, hard, -1]) != 0)) if len(hard) else np.array([])
    out = {"dates": _dates(reg.index[idx]), "columns": list(reg.columns), "probabilities": [[_num(v) for v in row] for row in reg.to_numpy()[idx]],
           "share": [float((hard == i).mean()) for i in range(reg.shape[1])], "mean_run_days": float(runs.mean()) if len(runs) else None}
    if diag:
        out["regimes"], out["transition"], out["engine"] = diag.get("regimes"), diag.get("transition"), diag.get("engine")
    return out


def heads_section(heads: pd.DataFrame, ds, labels: pd.Series | None) -> dict | None:
    rows = []
    for col in heads.columns:
        h = int(str(col)[1:])
        if h <= 0:
            continue
        pred = heads[col].unstack("asset").reindex(columns=ds.assets)
        fwd = forward_return(ds.prices, h, ds.lag).reindex(index=pred.index, columns=ds.assets)
        ic = ev.rank_ic_wide(pred, fwd).dropna()
        if len(ic) >= 20:
            rows.append({"horizon": h, "mean_ic": float(ic.mean()), "t": float(ic.mean() / ic.std(ddof=1) * np.sqrt(len(ic))) if ic.std(ddof=1) > 0 else None, "n_days": int(len(ic)),
                         "primary": h == ds.horizon})
    return {"horizons": rows} if rows else None


def weights_section(w: pd.DataFrame, diag: dict | None) -> dict | None:
    if w is None or w.empty:
        return None
    idx = _thin(len(w))
    out = {"dates": _dates(w.index[idx]), "members": list(w.columns), "weights": [[_num(v) for v in row] for row in w.to_numpy()[idx]], "mean": {c: float(w[c].mean()) for c in w.columns},
           "turnover": float(w.diff().abs().sum(axis=1).mean())}
    if diag:
        out["calibration"], out["weighting"], out["regime_weights"] = diag.get("calibration"), diag.get("weighting"), diag.get("regime_weights")
    return out


def latent_section(latent: pd.DataFrame, ds) -> dict | None:
    fwd = forward_return(ds.prices, ds.horizon, ds.lag).reindex(columns=ds.assets)
    rows = []
    for col in latent.columns:
        z = latent[col].unstack("asset").reindex(columns=ds.assets)
        ic = ev.rank_ic_wide(z, fwd.reindex(z.index)).dropna()
        rows.append({"factor": col, "mean_ic": _num(ic.mean()), "std": _num(latent[col].std()), "n_days": int(len(ic))})
    return {"factors": rows, "dim": int(latent.shape[1])} if rows else None


def portfolio_section(port: pd.DataFrame, ds) -> dict | None:
    y = ds.Y[ds.target].groupby(level=0).mean().reindex(port.index)
    ok = y.notna()
    if ok.sum() < 30:
        return None
    pred, real = port["predicted"][ok], y[ok]
    idx = _thin(len(pred))
    return {"correlation": _num(np.corrcoef(pred, real)[0, 1]), "n_days": int(ok.sum()), "series": {"dates": _dates(pred.index[idx]), "predicted": [_num(v) for v in pred.to_numpy()[idx]], "actual": [_num(v) for v in real.to_numpy()[idx]]}}


# ------------------------------------------------------------------------------------------------------------ the run itself
def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=3, check=False)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def versions() -> dict:
    out = {"python": platform.python_version()}
    for name in LIBRARIES:
        mod = sys.modules.get(name)                                                  # only what the run actually imported
        if mod is not None and getattr(mod, "__version__", None):
            out[name] = str(mod.__version__)
    return out


def resources_section(diagnostics: list, seconds_total: float, train_seconds: float) -> dict:
    devices = sorted({d["device"] for d in diagnostics if d.get("device")})
    peaks = [d["peak_memory_mb"] for d in diagnostics if d.get("peak_memory_mb")]
    out = {"train_seconds": float(train_seconds), "total_seconds": float(seconds_total), "devices": devices or ["cpu"], "n_params": next((d["n_params"] for d in diagnostics if d.get("n_params")), None),
           "peak_memory_mb": float(max(peaks)) if peaks else None, "notes": sorted({d["device_note"] for d in diagnostics if d.get("device_note")}), "cpu_count": os.cpu_count(),
           "platform": platform.platform()}
    torch = sys.modules.get("torch")
    if torch is not None:
        out["torch_threads"] = int(torch.get_num_threads())
        out["cuda"] = bool(torch.cuda.is_available())
    return out


def reproducibility_section(cfg: dict, config_hash: str, ds, final_model_bytes: bytes | None) -> dict:
    torch = sys.modules.get("torch")
    return {"seed": int(cfg["seed"]), "config_hash": config_hash, "data_fingerprint": ds.meta.get("data_fingerprint"), "versions": versions(), "git_commit": _git_commit(),
            "deterministic_torch": bool(torch.are_deterministic_algorithms_enabled()) if torch is not None else None,
            "model_artifact": None if final_model_bytes is None else {"bytes": len(final_model_bytes), "sha256": hashlib.sha256(final_model_bytes).hexdigest()}}


def build_report(train, ds, cfg: dict, scores: pd.DataFrame, fwd: pd.DataFrame, labels_by_date: pd.Series | None, nominal: float | None, seconds_total: float, config_hash: str,
                 final_model_bytes: bytes | None) -> dict:
    """The ``model_extras`` section of an experiment report."""
    kind = train.kind
    info = _INFO.get(kind)
    diag = train.diagnostics
    last = diag[-1] if diag else {}
    out = {"kind": kind, "family": info.family if info else None, "outputs": list(info.outputs) if info else []}
    y = ds.Y[train.target]
    steps = {
        "training": lambda: training_section(diag),
        "rewards": lambda: rewards_section(diag),
        "ranking": lambda: ranking_section(scores, fwd, labels_by_date),
        "intervals": lambda: intervals_section(train.extras["distribution"], y, nominal or last.get("interval_level") or 0.8, labels_by_date) if "distribution" in train.extras else None,
        "regimes": lambda: regimes_section(train.extras.get("regimes"), last, labels_by_date) if "regimes" in train.extras else None,
        "heads": lambda: heads_section(train.extras["heads"], ds, labels_by_date) if "heads" in train.extras else None,
        "weights": lambda: weights_section(train.extras["weights"], last) if "weights" in train.extras else None,
        "latent": lambda: latent_section(train.extras["latent"], ds) if "latent" in train.extras else None,
        "portfolio": lambda: portfolio_section(train.extras["portfolio"], ds) if "portfolio" in train.extras else None,
        "graph": lambda: {"graph": last["graph"], "attention": last.get("attention")} if last.get("graph") else None,
        "attention": lambda: {"steps": last["attention"]} if last.get("attention") and not last.get("graph") else None,
    }
    for name, make in steps.items():
        try:
            section = make()
        except Exception:                                                              # a report section that cannot be computed must not lose the experiment
            section = None
        if section:
            out[name] = section
    out["resources"] = resources_section(diag, seconds_total, train.seconds)
    out["reproducibility"] = reproducibility_section(cfg, config_hash, ds, final_model_bytes)
    return out
