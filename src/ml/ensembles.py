"""Combining strategies: the strategy combination engine, bandit allocation, strategy selectors and meta-labelling.

The inputs are return streams: the out-of-sample net returns of experiments (each already costed by the backtest engine). Weights over strategies are re-estimated every ``rebalance`` days from data up to
and including that day and applied to the returns of the days after it, so a combination never sees the day it is allocating to. Changing the weights between strategies has a cost of its own that is NOT
charged here (the strategies' own trading costs are in their returns); keep ``rebalance`` long (monthly) and treat the result as an upper bound on what combining can add.

Methods (``combine``):

``equal``              one over N
``inverse_vol``        weight proportional to one over trailing volatility
``risk_parity``        equal risk contributions from a shrunk covariance matrix (``portfolio.risk_parity``)
``mean_variance``      long-only mean-variance with the means shrunk halfway to their average (an optimiser that trusts sample means chases noise)
``bayesian``           the same with Bayes-Stein shrinkage of the means (Jorion 1986): the shrinkage weight is estimated from the data
``regime_conditioned`` mean-variance on the days that were in the same market regime as the day of allocation
``online_learning``    multiplicative weights (Hedge): each day, strategies that earned more get more, at a learning rate ``eta``
``bandit``             Thompson-sampling probability matching: the weight of a strategy is the chance, under the posterior of its mean return, that it is the best one
``best_trailing``      all in the strategy with the best trailing Sharpe ratio (a selector, not a blend)
``regime_selector``    all in the strategy with the best Sharpe ratio in the current regime

``meta_label_scores`` is different: it turns ONE model's predictions into a better-sized book by training a second model to say when the first is right (Lopez de Prado 2018, chapter 3).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from ..portfolio.constraints import Constraints
from ..portfolio.covariance import ledoit_wolf_shrinkage
from ..portfolio.mean_variance import mean_variance_weights
from ..portfolio.risk_parity import risk_parity_weights
from .evaluation import sharpe_of, strategy_summary
from .robustness import market_states
from .visualization import similarity_map

ANN = 252.0
METHODS = ("equal", "inverse_vol", "risk_parity", "mean_variance", "bayesian", "regime_conditioned", "online_learning", "bandit", "best_trailing", "regime_selector")


@dataclass
class Combination:
    method: str
    weights: pd.DataFrame                  # date x strategy, stamped at the allocation date
    returns: pd.Series                     # the combined return stream
    params: dict
    summary: dict
    turnover: float                        # annual one-way turnover of the strategy weights


def _long_only() -> Constraints:
    return Constraints(min_weight=0.0, max_weight=1.0, gross_leverage=1.0, net_exposure=1.0, group_limits={}, group_map={}, max_turnover=None, long_only=True)


def _cov(window: pd.DataFrame) -> pd.DataFrame:
    try:
        return ledoit_wolf_shrinkage(window, "constant_correlation", annualise=True)[0]
    except ValueError:
        return window.cov() * ANN + 1e-8 * np.eye(window.shape[1])


def _normalise(w: pd.Series) -> pd.Series:
    w = w.clip(lower=0.0)
    total = w.sum()
    return w / total if total > 0 else pd.Series(1.0 / len(w), index=w.index)


def _w_equal(window, **_):
    return pd.Series(1.0 / window.shape[1], index=window.columns)


def _w_inverse_vol(window, **_):
    return _normalise(1.0 / window.std(ddof=1).replace(0.0, np.nan).fillna(window.std(ddof=1).max() + 1e-9))


def _w_risk_parity(window, **_):
    return _normalise(risk_parity_weights(_cov(window)))


def _mv(mu: pd.Series, window: pd.DataFrame, risk_aversion: float) -> pd.Series:
    res = mean_variance_weights(mu, _cov(window), risk_aversion, _long_only())
    return _normalise(res.weights)


def _w_mean_variance(window, risk_aversion=5.0, **_):
    mu = window.mean() * ANN
    return _mv(0.5 * mu + 0.5 * mu.mean(), window, risk_aversion)


def bayes_stein_means(window: pd.DataFrame) -> pd.Series:
    """Jorion's Bayes-Stein estimator: the sample means pulled toward the minimum-variance portfolio's mean with a data-driven weight ``lambda = (N + 2) / ((N + 2) + T (m - m0)' S^-1 (m - m0))``."""
    t, n = window.shape
    m, S = window.mean().to_numpy() * ANN, _cov(window).to_numpy()
    inv = np.linalg.pinv(S)
    ones = np.ones(n)
    m0 = float(ones @ inv @ m / (ones @ inv @ ones))
    d = m - m0
    lam = (n + 2.0) / ((n + 2.0) + t / ANN * float(d @ inv @ d))
    return pd.Series((1.0 - lam) * m + lam * m0, index=window.columns)


def _w_bayesian(window, risk_aversion=5.0, **_):
    return _mv(bayes_stein_means(window), window, risk_aversion)


def _w_regime_conditioned(window, history: pd.DataFrame | None = None, regimes: pd.Series | None = None, risk_aversion=5.0, **_):
    if history is None or regimes is None:
        return _w_mean_variance(window, risk_aversion)
    now = regimes.reindex(history.index).iloc[-1]
    same = history[regimes.reindex(history.index) == now]
    if len(same) < 60 or pd.isna(now):
        return _w_mean_variance(window, risk_aversion)
    mu = same.mean() * ANN
    return _mv(0.5 * mu + 0.5 * mu.mean(), same, risk_aversion)


def _w_best_trailing(window, **_):
    s = window.mean() / window.std(ddof=1).replace(0.0, np.nan)
    w = pd.Series(0.0, index=window.columns)
    if s.notna().any():
        w[s.idxmax()] = 1.0
        return w
    return _w_equal(window)


def _w_regime_selector(window, history: pd.DataFrame | None = None, regimes: pd.Series | None = None, **_):
    if history is None or regimes is None:
        return _w_best_trailing(window)
    now = regimes.reindex(history.index).iloc[-1]
    same = history[regimes.reindex(history.index) == now]
    return _w_best_trailing(same) if len(same) >= 60 and not pd.isna(now) else _w_best_trailing(window)


def thompson_weights(history: pd.DataFrame, discount: float = 0.995, n_draws: int = 400, seed: int = 0) -> pd.Series:
    """The probability, under a normal posterior for each strategy's mean daily return (exponentially discounted history), that the strategy has the highest mean: probability matching, the weights of
    Thompson sampling."""
    n = len(history)
    age = np.arange(n)[::-1]
    w = discount ** age
    wsum = w.sum()
    mean = (history.to_numpy() * w[:, None]).sum(axis=0) / wsum
    var = (w[:, None] * (history.to_numpy() - mean) ** 2).sum(axis=0) / wsum
    n_eff = wsum ** 2 / (w ** 2).sum()
    se = np.sqrt(np.maximum(var, 1e-12) / max(n_eff, 2.0))
    rng = np.random.default_rng(seed + n)
    draws = mean + se * rng.standard_normal((n_draws, history.shape[1]))
    wins = np.bincount(draws.argmax(axis=1), minlength=history.shape[1]) / n_draws
    return pd.Series(wins, index=history.columns)


def hedge_weights(R: pd.DataFrame, eta: float = 5.0, vol_window: int = 63) -> pd.DataFrame:
    """Multiplicative weights updated every day: ``w <- w exp(eta r / sigma)`` normalised, with ``r / sigma`` each strategy's return in units of its own trailing volatility so ``eta`` means the same for
    a calm strategy and a wild one. Weights are floored at 1% so a strategy that has done badly can come back."""
    sig = R.rolling(vol_window, min_periods=20).std(ddof=1).shift(1).bfill().replace(0.0, np.nan).fillna(R.std(ddof=1))
    z = (R / sig).clip(-4, 4).fillna(0.0).to_numpy()
    n = R.shape[1]
    w = np.full(n, 1.0 / n)
    out = np.empty_like(z)
    for t in range(len(z)):
        out[t] = w
        w = w * np.exp(eta * 0.01 * z[t])
        w = np.maximum(w / w.sum(), 0.01)
        w = w / w.sum()
    return pd.DataFrame(out, index=R.index, columns=R.columns)


def combine(R: pd.DataFrame, method: str, lookback: int = 252, rebalance: int = 21, min_history: int = 126, regimes: pd.Series | None = None, risk_aversion: float = 5.0, eta: float = 5.0,
            seed: int = 0) -> Combination:
    """Weights over the columns of ``R`` (daily returns of strategies) by ``method``, and the resulting combined return stream."""
    if method not in METHODS:
        raise ValueError(f"unknown combination method '{method}'; choose from {list(METHODS)}")
    R = R.dropna(how="all")
    if R.shape[1] < 2:
        raise ValueError("combine at least two strategies")
    R = R.fillna(0.0)
    if len(R) < min_history + rebalance:
        raise ValueError(f"the strategies share only {len(R)} days; need at least {min_history + rebalance}")
    if method in ("regime_conditioned", "regime_selector") and regimes is None:
        raise ValueError(f"the {method} method needs a regime series")
    if method == "online_learning":
        W = hedge_weights(R, eta)
        W.iloc[:min_history] = np.nan
    else:
        W = pd.DataFrame(np.nan, index=R.index, columns=R.columns)
        fns = {"equal": _w_equal, "inverse_vol": _w_inverse_vol, "risk_parity": _w_risk_parity, "mean_variance": _w_mean_variance, "bayesian": _w_bayesian, "regime_conditioned": _w_regime_conditioned,
               "best_trailing": _w_best_trailing, "regime_selector": _w_regime_selector}
        for i in range(min_history, len(R), rebalance):
            window = R.iloc[max(0, i - lookback + 1): i + 1]
            if method == "bandit":
                W.iloc[i] = thompson_weights(R.iloc[: i + 1], seed=seed)
            else:
                W.iloc[i] = fns[method](window, risk_aversion=risk_aversion, history=R.iloc[: i + 1], regimes=regimes)
        W = W.ffill()
    ret = (W.shift(1) * R).sum(axis=1, min_count=1).dropna()
    turnover = float(W.diff().abs().sum(axis=1).mean() * ANN / 2.0) if W.notna().any().any() else float("nan")
    summary = strategy_summary(ret)
    return Combination(method, W.dropna(how="all"), ret, {"lookback": lookback, "rebalance": rebalance, "risk_aversion": risk_aversion, "eta": eta}, summary, turnover)


def _segment_sharpes(r: pd.Series, labels: pd.Series | None) -> dict:
    """The Sharpe ratio of a return stream on each segment of ``labels`` (``train``, ``validation``, ``test``, ``forward``) it has at least 40 days in."""
    if labels is None:
        return {}
    lab = labels.reindex(r.index)
    return {seg: sharpe_of(r[lab == seg]) for seg in ("validation", "test", "forward") if int((lab == seg).sum()) >= 40}


def compare_methods(R: pd.DataFrame, methods: tuple = METHODS, regimes: pd.Series | None = None, labels: pd.Series | None = None, **kw) -> dict:
    """Every method on the same strategies and the same evaluation window (the latest start among them), with the strategies themselves for reference. A method that cannot run is listed with the reason.
    With ``labels`` (the segment of each date) every row also carries the method's Sharpe ratio on the validation, test and forward segments."""
    rows, curves, errors = [], {}, {}
    results = {}
    for m in methods:
        try:
            results[m] = combine(R, m, regimes=regimes, **kw)
        except ValueError as exc:
            errors[m] = str(exc)
    if not results:
        raise ValueError("no method could run: " + "; ".join(errors.values()))
    start = max(c.returns.index[0] for c in results.values())
    for m, c in results.items():
        r = c.returns.loc[start:]
        eq = (1.0 + r).cumprod()
        rows.append({"method": m, "sharpe": sharpe_of(r), "ann_return": float(r.mean() * ANN), "ann_vol": float(r.std(ddof=1) * math.sqrt(ANN)), "max_drawdown": float((eq / eq.cummax() - 1.0).min()),
                     "turnover": c.turnover, "final_weights": {k: float(v) for k, v in c.weights.iloc[-1].items()}, "segments": _segment_sharpes(r, labels)})
        curves[m] = eq
    singles = {col: sharpe_of(R[col].loc[start:]) for col in R.columns}
    return {"start": str(start.date()), "rows": sorted(rows, key=lambda r: -(r["sharpe"] if np.isfinite(r["sharpe"]) else -9)), "strategy_sharpes": singles,
            "strategy_segments": {col: _segment_sharpes(R[col].loc[start:], labels) for col in R.columns}, "errors": errors,
            "curves": {"dates": [str(d.date()) for d in next(iter(curves.values())).index[::5]], **{m: [round(float(v), 5) for v in eq.iloc[::5]] for m, eq in curves.items()}},
            "similarity": similarity_map(R.loc[start:])}


def strategy_regimes(returns: pd.DataFrame, investable: pd.DataFrame | None = None) -> pd.Series:
    """Volatility-tercile regime labels (``low``, ``mid``, ``high``) of the equal-weight market, known at the previous close; for the regime-conditioned combinations."""
    v, _ = market_states(returns, investable)
    return v


# ------------------------------------------------------------------------------------------------------------------ meta-labelling
def meta_label_scores(ds, train, model_kind: str = "logistic", params: dict | None = None, min_rows: int = 400, seed: int = 0) -> dict:
    """A second model that says when the first is right. The label of a primary prediction is 1 if the position its sign implies made money (``sign(score) x forward return > 0``). The secondary model
    sees the same features plus the primary score and is fitted walk-forward on the PAST out-of-sample primary predictions whose labels were complete (the same purge as the primary), then scores the
    next block. The meta score is ``sign(primary) x max(2 P(right) - 1, 0)``: the primary's side, sized by the secondary's confidence, and no bet at all when the secondary thinks the primary is more likely wrong than right (the secondary never reverses the side). Returns the meta scores (wide), the secondary model's AUC and coverage."""
    from sklearn.metrics import roc_auc_score

    from .models import make_model
    from .targets import forward_return

    prim = train.predictions
    fwd_wide = forward_return(ds.prices, ds.horizon, ds.lag).reindex(index=ds.dates, columns=ds.assets)
    fwd_long = pd.Series(fwd_wide.to_numpy().reshape(-1), index=pd.MultiIndex.from_product([ds.dates, ds.assets], names=["date", "asset"]))
    f = fwd_long.reindex(prim.index)
    right = ((np.sign(prim) * f) > 0).astype(float).where(f.notna() & (prim != 0))
    X = ds.X.reindex(prim.index).copy()
    X["primary_score"] = prim
    X["primary_abs"] = prim.abs()
    purge = ds.lag + ds.horizon
    dates = prim.index.get_level_values(0)
    pos = ds.dates.get_indexer(dates)
    out = pd.Series(np.nan, index=prim.index)
    prob = pd.Series(np.nan, index=prim.index)
    blocks = [(a, b) for a, b in zip([f_["test_start"] for f_ in train.folds if not f_.get("skipped")], [f_["test_end"] for f_ in train.folds if not f_.get("skipped")])]
    n_models = 0
    for a, b in blocks:
        lo, hi = ds.dates.get_indexer([pd.Timestamp(a), pd.Timestamp(b)])
        past = (pos <= lo - purge) & right.notna().to_numpy()
        now = (pos >= lo) & (pos <= hi)
        if past.sum() < min_rows or not now.any() or right[past].nunique() < 2:
            continue
        model = make_model(model_kind, params, "classification", seed).fit(X[past], right[past], pos[past])
        p = model.predict_proba(X[now])
        prob[now] = p
        out[now] = np.sign(prim[now]) * np.clip(2.0 * p - 1.0, 0.0, 1.0)
        n_models += 1
    ok = prob.notna() & right.notna()
    auc = float(roc_auc_score(right[ok], prob[ok])) if ok.sum() > 50 and right[ok].nunique() == 2 else float("nan")
    return {"scores": out.unstack("asset").reindex(index=ds.dates, columns=ds.assets), "probability": prob, "auc": auc, "n_models": n_models, "coverage": float(prob.notna().mean()),
            "primary_hit_rate": float(right[right.notna()].mean()), "meta_hit_rate": float(right[ok & (prob > 0.5)].mean()) if (ok & (prob > 0.5)).any() else float("nan")}


# ------------------------------------------------------------------------------------------------------------------ on saved experiments
def load_streams(registry, ids: list[str]) -> tuple[pd.DataFrame, dict[str, pd.Series], dict[str, str]]:
    """The daily net returns of saved experiments (one column per experiment id), each one's segment label by date, and the experiments' names."""
    cols, labels, names = {}, {}, {}
    for i in ids:
        rec = registry.get_experiment(i)
        path = registry.root / "experiments" / i / "returns.parquet"
        if not path.exists():
            raise ValueError(f"experiment '{i}' has no saved returns")
        r = pd.read_parquet(path)
        cols[i], labels[i], names[i] = r["net"], r["segment"], rec["name"]
    return pd.DataFrame(cols), labels, names


def combine_experiments(registry, ids: list[str], methods: tuple = METHODS, lookback: int = 252, rebalance: int = 21, min_history: int = 126, risk_aversion: float = 5.0, eta: float = 5.0, seed: int = 0) -> dict:
    """The strategy combination engine on saved experiments: every method in ``methods`` allocates across their out-of-sample net returns. Segment Sharpe ratios use the first experiment's segments.
    The regime-based methods use the volatility regime of the first experiment's universe (they are listed under ``errors`` when its saved panel is unavailable)."""
    if len(ids) < 2:
        raise ValueError("choose at least two experiments to combine")
    R, labels, names = load_streams(registry, ids)
    regimes = None
    try:
        from .datasets import Dataset

        report = registry.report(ids[0])
        fsets = [f for f in registry.list_feature_sets() if f["id"].startswith(report["build_key"])]
        if fsets:
            d = Dataset.load(fsets[0]["path"])
            regimes = strategy_regimes(d.returns, d.investable).reindex(R.index).ffill()
    except Exception:                                                                       # the regime methods then report that they have no regimes; the others do not need them
        regimes = None
    out = compare_methods(R, tuple(methods), regimes, labels[ids[0]], lookback=lookback, rebalance=rebalance, min_history=min_history, risk_aversion=risk_aversion, eta=eta, seed=seed)
    out["names"] = names
    out["experiments"] = list(ids)
    out["note"] = ("Each method re-estimates its weights every {r} days from the returns up to that day and applies them afterwards. The strategies' returns include the days they were selected on "
                   "(validation); read the test and forward columns for evidence.").format(r=rebalance)
    return out


def meta_label_experiment(registry, exp_id: str, model_kind: str = "logistic", params: dict | None = None, signal: dict | None = None, progress=None, save: bool = True, min_rows: int = 400, seed: int = 0):
    """Meta-label a saved experiment: fit the secondary model on its out-of-sample bets, size the book with it (proportional to the meta score by default: the primary's side, no bet where the secondary
    doubts it) and run the result through the same backtest, segments and diagnostics. Returns ``(experiment, meta summary)``; the new run is a child of the original and counts as another look."""
    from .config import experiment_config, merge
    from .experiments import ExperimentError, evaluate_run, load_run

    cfg0, ds, train, info = load_run(registry, exp_id)
    if progress:
        progress("fitting the secondary model on the primary's past bets")
    try:
        meta = meta_label_scores(ds, train, model_kind, params, min_rows, seed)
    except ValueError as exc:
        raise ExperimentError(str(exc)) from None
    flat = pd.Series(meta["scores"].to_numpy().reshape(-1), index=pd.MultiIndex.from_product([ds.dates, ds.assets], names=["date", "asset"])).dropna()
    if flat.empty:
        raise ExperimentError(f"the secondary model never had {min_rows} past bets to learn from: the primary's out-of-sample run is too short for meta-labelling")
    cfg = experiment_config(merge(cfg0, {"signal": {"method": "proportional", **(signal or {})}, "notes": f"meta-labelled version of {exp_id} ({model_kind} secondary model)"}), name=f"{cfg0['name']}-meta")
    sized = replace(train, predictions=flat, insample=pd.Series(dtype=float), probabilities=None, models=[], fold_rows=[], importance=pd.DataFrame(), task="regression", kind=f"meta({train.kind})")
    exp = evaluate_run(cfg, ds, sized, info, registry, progress, save, exp_id, explain=False, final_model=False, robustness=False)
    return exp, {k: meta[k] for k in ("auc", "n_models", "coverage", "primary_hit_rate", "meta_hit_rate")}


def compare_experiments(registry, ids: list[str], max_points: int = 700) -> dict:
    """Side by side: each saved experiment's settings and headline numbers by segment, their equity curves over the days they share (each starting at 1), the correlation of their daily returns and a
    two-dimensional map of how alike they are."""
    if not 2 <= len(ids) <= 8:
        raise ValueError("choose between two and eight experiments")
    R, labels, names = load_streams(registry, ids)
    rows = []
    for i in ids:
        rep = registry.report(i)
        cfg, st = rep["config"], rep["segments"]["stats"]
        rows.append({"id": i, "name": names[i], "model": cfg["model"]["kind"], "target": cfg["dataset"]["target"], "universe": cfg["dataset"]["universe"] if isinstance(cfg["dataset"]["universe"], str) else "tickers",
                     "signal": cfg["signal"]["method"], "rebalance": cfg["backtest"]["rebalance"], "ic": rep["ic"]["all"].get("mean"), "ic_ir": rep["ic"]["all"].get("ir"),
                     "generalization": rep["overfit"].get("generalization_score"), "trials": rep["trials"]["dataset_trials"], "turnover": rep["metrics"]["oos"].get("ann_turnover"),
                     "segments": {seg: {k: v.get(k) for k in ("sharpe", "ann_return", "ann_vol", "max_drawdown", "n_days")} for seg, v in st.items()}, "warnings": list(rep.get("warnings", []))[:3]})
    common = R.dropna()
    note = None
    if len(common) < 60:
        common = R.fillna(0.0).loc[R.apply(lambda c: c.first_valid_index()).max():]
        note = "The experiments share fewer than 60 days of returns; the curves start at the latest start among them and count missing days as flat."
    if len(common) < 20:
        raise ValueError("the experiments share too few days to compare")
    curves = (1.0 + common).cumprod()
    keep = np.unique(np.append(np.linspace(0, len(curves) - 1, min(len(curves), max_points)).round().astype(int), len(curves) - 1))
    lab = labels[ids[0]].reindex(curves.index).ffill()
    corr = common.corr().fillna(0.0)
    return {"rows": rows, "dates": [str(d.date()) for d in curves.index[keep]], "curves": {i: [round(float(v), 5) for v in curves[i].to_numpy()[keep]] for i in ids},
            "segment": [str(x) for x in lab.to_numpy()[keep]], "correlation": {"ids": list(ids), "matrix": [[round(float(v), 3) for v in row] for row in corr.to_numpy()]},
            "similarity": similarity_map(common, min_overlap=20), "common_days": int(len(common)), "start": str(common.index[0].date()), "end": str(common.index[-1].date()), "note": note}
