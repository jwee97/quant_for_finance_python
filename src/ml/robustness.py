"""Ten ways to try to break a strategy, and a control that says what luck looks like.

Each test re-uses the experiment's out-of-sample predictions and its fitted fold models (nothing is refit unless a test says so) and returns the Sharpe ratios under the stress, a verdict (``pass``, ``warn`` or
``fail``) by a simple stated rule, and a sentence explaining what the test means. The verdict rules are deliberately crude: they exist so that a strategy that only works in one lucky configuration
is visibly different from one that works in many.

1. ``random_start``      begin the evaluation on random dates inside the out-of-sample period: does the result depend on when you started?
2. ``universe_subsets``  trade random subsets of the assets: does it depend on one or two names?
3. ``higher_costs``      double, triple, quintuple the costs
4. ``delayed_execution`` trade one, two, five days later than planned
5. ``noisy_features``    add noise to the features at prediction time (the fitted models are reused): is the signal fragile?
6. ``parameter_perturbation`` change the signal settings (holdings, rebalance, book size)
7. ``bootstrap``         resample the daily returns in blocks: how wide is the uncertainty of the Sharpe ratio?
8. ``regime_split``      Sharpe in low-, mid- and high-volatility markets
9. ``crisis_windows``    Sharpe inside named crises that fall in the out-of-sample period
10. ``bull_bear``        Sharpe when the market was above and below its 200-day average

``random_signals`` is the control: the same positions rules, costs and rebalancing applied to RANDOM scores. If the model's Sharpe is not clearly above what random scores earn through the same machinery,
the machinery (a trend in the universe, a long bias, a cost model) is doing the work.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import equal_weight_benchmark, label_days, run_backtest, trim_to_signal
from .evaluation import rank_ic_wide, sharpe_of
from .signals import build_weights

ANN = 252.0
DEFAULT_CRISES = {"Global financial crisis": ("2008-09-01", "2009-03-31"), "COVID crash": ("2020-02-15", "2020-04-30"), "2022 inflation bear market": ("2022-01-03", "2022-10-14")}


@dataclass
class Context:
    """Everything the tests need about an experiment."""

    dataset: object
    train: object                         # TrainResult
    cfg: dict
    scores: pd.DataFrame                  # out-of-sample scores, wide
    first_signal: pd.Timestamp
    base_net: pd.Series
    labels: pd.Series                     # segment of every date
    vol: pd.DataFrame | None = None
    confidence: pd.DataFrame | None = None
    regime_on: pd.Series | None = None


def _row(name: str, net: pd.Series, **extra) -> dict:
    r = net.dropna()
    eq = (1.0 + r).cumprod()
    return {"scenario": name, "sharpe": sharpe_of(r), "ann_return": float(r.mean() * ANN) if len(r) else float("nan"), "max_drawdown": float((eq / eq.cummax() - 1.0).min()) if len(r) else float("nan"),
            "n_days": int(len(r)), **extra}


def _verdict(passed: bool, warned: bool) -> str:
    return "pass" if passed else ("warn" if warned else "fail")


def _weights(ctx: Context, scores: pd.DataFrame | None = None, signal_overrides: dict | None = None) -> pd.DataFrame:
    sig = {**ctx.cfg["signal"], **(signal_overrides or {})}
    return build_weights(ctx.scores if scores is None else scores, sig, ctx.vol, ctx.confidence, ctx.regime_on, ctx.dataset.group or ctx.dataset.asset_class, ctx.dataset.investable)


def _run(ctx: Context, weights: pd.DataFrame, cost_multiple: float = 1.0, extra_lag: int | None = None, bt_overrides: dict | None = None) -> pd.Series:
    bt = {**ctx.cfg["backtest"], **(bt_overrides or {})}
    res = run_backtest(weights, ctx.dataset.returns, ctx.dataset.investable, bt, ctx.dataset.lag, cost_multiple=cost_multiple, extra_lag=extra_lag)
    return trim_to_signal(res, ctx.first_signal, ctx.dataset.lag)["net"]


def random_start(ctx: Context) -> dict:
    n = int(ctx.cfg["robustness"]["n_draws"])
    rng = np.random.default_rng(int(ctx.cfg["robustness"]["seed"]))
    idx = ctx.base_net.index
    starts = rng.integers(0, max(int(len(idx) * 0.6), 1), n)
    rows = [_row(f"start {idx[s].date()}", ctx.base_net.iloc[s:]) for s in starts]
    sharpes = np.array([r["sharpe"] for r in rows], dtype=float)
    share = float(np.nanmean(sharpes > 0))
    return {"name": "random_start", "title": "Random start dates", "description": "The out-of-sample evaluation begun on random dates in the first 60% of the period.",
            "rows": rows, "summary": {"share_positive": share, "median": float(np.nanmedian(sharpes)), "p10": float(np.nanpercentile(sharpes, 10)), "p90": float(np.nanpercentile(sharpes, 90))},
            "verdict": _verdict(share >= 0.8, share >= 0.6), "detail": f"Sharpe was positive from {share:.0%} of the start dates (median {np.nanmedian(sharpes):.2f})."}


def universe_subsets(ctx: Context) -> dict:
    n = int(ctx.cfg["robustness"]["n_draws"])
    rng = np.random.default_rng(int(ctx.cfg["robustness"]["seed"]) + 1)
    assets = ctx.dataset.assets
    k = max(3, int(round(0.7 * len(assets))))
    if len(assets) < 5:
        return {"name": "universe_subsets", "title": "Universe subsets", "description": "Random subsets of the assets.", "rows": [], "verdict": "na", "detail": "Fewer than five assets: subsets would be too small to mean anything."}
    rows = []
    for i in range(n):
        keep = list(rng.choice(assets, size=min(k, len(assets) - 1), replace=False))
        s = ctx.scores.copy()
        s.loc[:, [a for a in assets if a not in keep]] = np.nan
        rows.append(_row(f"subset {i + 1}: {len(keep)} assets", _run(ctx, _weights(ctx, s))))
    sharpes = np.array([r["sharpe"] for r in rows], dtype=float)
    share = float(np.nanmean(sharpes > 0))
    return {"name": "universe_subsets", "title": "Universe subsets", "description": f"{n} random subsets of {k} of the {len(assets)} assets (the scores are the model's own: a cross-sectional feature would change slightly without the dropped names).",
            "rows": rows, "summary": {"share_positive": share, "median": float(np.nanmedian(sharpes)), "p10": float(np.nanpercentile(sharpes, 10))},
            "verdict": _verdict(share >= 0.8, share >= 0.6), "detail": f"Positive in {share:.0%} of the subsets (median Sharpe {np.nanmedian(sharpes):.2f})."}


def higher_costs(ctx: Context) -> dict:
    mults = [float(m) for m in ctx.cfg["robustness"]["cost_multiples"]]
    w = _weights(ctx)
    rows = [_row(f"{m:g}x costs", _run(ctx, w, cost_multiple=m), multiple=m) for m in mults]
    by = {r["multiple"]: r["sharpe"] for r in rows}
    base, double = by.get(1.0, rows[0]["sharpe"]), by.get(2.0, np.nan)
    ok = np.isfinite(double) and double > 0 and double >= 0.5 * base
    return {"name": "higher_costs", "title": "Higher costs", "description": "The same positions with the fee and slippage multiplied.", "rows": rows,
            "verdict": _verdict(ok, np.isfinite(double) and double > 0), "detail": f"Sharpe {base:.2f} at the assumed costs and {double:.2f} at twice them."}


def delayed_execution(ctx: Context) -> dict:
    w = _weights(ctx)
    rows = [_row(f"+{d} day{'s' if d != 1 else ''} late", _run(ctx, w, extra_lag=int(ctx.cfg["backtest"].get("extra_lag", 0)) + int(d)), delay=int(d)) for d in ctx.cfg["robustness"]["delays"]]
    by = {r["delay"]: r["sharpe"] for r in rows}
    base, one = by.get(0, rows[0]["sharpe"]), by.get(1, np.nan)
    ok = np.isfinite(one) and one > 0 and one >= 0.6 * base
    return {"name": "delayed_execution", "title": "Delayed execution", "description": "The same positions traded later than the signal.", "rows": rows,
            "verdict": _verdict(ok, np.isfinite(one) and one > 0), "detail": f"Sharpe {base:.2f} on time and {one:.2f} one day late."}


def noisy_features(ctx: Context) -> dict:
    ds, tr = ctx.dataset, ctx.train
    usable = [(model, idx) for model, idx in zip(tr.models, tr.fold_rows) if model is not None and len(idx)]
    description = "Gaussian noise (as a multiple of each feature's standard deviation) added to the features when the fitted models predict."
    if not usable:                                                                          # a re-backtest re-uses saved predictions: the models of each refit are not kept with an experiment
        return {"name": "noisy_features", "title": "Noisy features", "description": description, "rows": [], "verdict": "na",
                "detail": "Not run: the models of each refit are not kept with a saved experiment, so a re-backtest cannot re-predict. Train with the stress tests switched on to include it."}
    levels = [float(x) for x in ctx.cfg["robustness"]["noise_levels"]]
    rng = np.random.default_rng(int(ctx.cfg["robustness"]["seed"]) + 2)
    sd = ds.X.std(ddof=0).replace(0.0, 1.0).to_numpy(dtype=float)
    rows = []
    for lvl in levels:
        pieces = []
        for model, idx in usable:
            X = ds.X.iloc[idx]
            Xn = X + lvl * sd * rng.standard_normal(X.shape) if lvl > 0 else X
            pieces.append(pd.Series(model.predict(Xn), index=X.index))
        s = pd.concat(pieces).sort_index().unstack("asset").reindex(index=ds.dates, columns=ds.assets)
        fwd = ds.Y[ds.target].unstack("asset").reindex(index=ds.dates, columns=ds.assets)
        ic = rank_ic_wide(s, fwd)
        rows.append(_row(f"noise {lvl:g} sd", _run(ctx, _weights(ctx, s)), level=lvl, mean_ic=float(ic.mean())))
    by = {r["level"]: r["sharpe"] for r in rows}
    base, half = by.get(0.0, rows[0]["sharpe"]), by.get(0.5, np.nan)
    ok = np.isfinite(half) and half > 0 and half >= 0.5 * base
    return {"name": "noisy_features", "title": "Noisy features", "description": description,
            "rows": rows, "verdict": _verdict(ok, np.isfinite(half) and half > 0), "detail": f"Sharpe {base:.2f} clean and {half:.2f} with noise half the size of the features."}


def parameter_perturbation(ctx: Context) -> dict:
    sig = ctx.cfg["signal"]
    variants = [("as configured", {}, {})]
    n = int(sig.get("n", 3))
    for dn in (-1, 1):
        if n + dn >= 1:
            variants.append((f"n = {n + dn}", {"n": n + dn}, {}))
    for h in (1, 5, 20):
        if h != int(sig.get("holding_period", 1)):
            variants.append((f"holding {h}d", {"holding_period": h}, {}))
    for rb in ("daily", "weekly", "monthly"):
        if rb != ctx.cfg["backtest"]["rebalance"]:
            variants.append((f"rebalance {rb}", {}, {"rebalance": rb}))
    rows = [_row(label, _run(ctx, _weights(ctx, signal_overrides=so), bt_overrides=bo)) for label, so, bo in variants]
    sharpes = np.array([r["sharpe"] for r in rows], dtype=float)
    share = float(np.nanmean(sharpes > 0))
    return {"name": "parameter_perturbation", "title": "Parameter perturbation", "description": "The signal settings changed one at a time: book size, holding period, rebalance frequency.", "rows": rows,
            "summary": {"share_positive": share, "min": float(np.nanmin(sharpes)), "max": float(np.nanmax(sharpes))}, "verdict": _verdict(share == 1.0, share >= 0.6),
            "detail": f"Positive in {share:.0%} of {len(rows)} variants (range {np.nanmin(sharpes):.2f} to {np.nanmax(sharpes):.2f})."}


def block_bootstrap(returns: np.ndarray, n_draws: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Circular block bootstrap: ``n_draws`` series as long as ``returns`` built from random blocks of ``block`` consecutive days (wrapping around), preserving short-range dependence."""
    n = len(returns)
    n_blocks = int(math.ceil(n / block))
    starts = rng.integers(0, n, (n_draws, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    return returns[idx.reshape(n_draws, -1)[:, :n]]


def bootstrap(ctx: Context) -> dict:
    r = ctx.base_net.dropna().to_numpy()
    if len(r) < 60:
        return {"name": "bootstrap", "title": "Bootstrap", "description": "Block bootstrap of the daily returns.", "rows": [], "verdict": "na", "detail": "Too few days."}
    rng = np.random.default_rng(int(ctx.cfg["robustness"]["seed"]) + 3)
    draws = block_bootstrap(r, 1000, 21, rng)
    sd = draws.std(axis=1, ddof=1)
    sharpes = np.where(sd > 0, draws.mean(axis=1) / sd * math.sqrt(ANN), np.nan)
    p_neg = float(np.nanmean(sharpes <= 0))
    q = np.nanpercentile(sharpes, [5, 50, 95])
    return {"name": "bootstrap", "title": "Block bootstrap", "description": "1,000 resamples of the daily returns in 21-day blocks (circular).",
            "rows": [{"scenario": "5th percentile", "sharpe": float(q[0])}, {"scenario": "median", "sharpe": float(q[1])}, {"scenario": "95th percentile", "sharpe": float(q[2])}],
            "summary": {"p_sharpe_le_0": p_neg, "p5": float(q[0]), "p50": float(q[1]), "p95": float(q[2])}, "verdict": _verdict(p_neg < 0.05, p_neg < 0.2),
            "detail": f"In {p_neg:.0%} of resamples the Sharpe ratio was zero or negative; the 5th to 95th percentile range is {q[0]:.2f} to {q[2]:.2f}."}


def market_states(returns: pd.DataFrame, investable: pd.DataFrame | None = None) -> tuple[pd.Series, pd.Series]:
    """Causal volatility terciles (``low``, ``mid``, ``high``) and bull/bear flags of the equal-weight market, each known at the close of the previous day (so a day is classified before it is traded)."""
    mkt = equal_weight_benchmark(returns, investable)
    vol = mkt.rolling(21, min_periods=15).std() * math.sqrt(ANN)
    pct = vol.expanding(min_periods=126).apply(lambda w: (w[:-1] < w[-1]).mean() if len(w) > 1 else np.nan, raw=True)
    vstate = pd.Series(np.where(pct < 1 / 3, "low", np.where(pct < 2 / 3, "mid", "high")), index=vol.index).where(pct.notna()).shift(1)
    level = (1.0 + mkt).cumprod()
    bull = (level > level.rolling(200, min_periods=120).mean()).where(level.rolling(200, min_periods=120).mean().notna()).shift(1)
    tren = pd.Series(np.where(bull == True, "bull", "bear"), index=bull.index).where(bull.notna())               # noqa: E712
    return vstate, tren


def regime_split(ctx: Context) -> dict:
    vstate, _ = market_states(ctx.dataset.returns, ctx.dataset.investable)
    st = vstate.reindex(ctx.base_net.index)
    rows = [_row(f"{k} volatility", ctx.base_net[st == k]) for k in ("low", "mid", "high") if (st == k).sum() >= 20]
    pos = sum(1 for r in rows if r["sharpe"] > 0)
    return {"name": "regime_split", "title": "Volatility regimes", "description": "Sharpe in low-, mid- and high-volatility markets (terciles of the market's 21-day volatility against its own history, known the day before).",
            "rows": rows, "verdict": "na" if len(rows) < 2 else _verdict(pos == len(rows), pos >= max(len(rows) - 1, 1)), "detail": f"Positive in {pos} of {len(rows)} volatility regimes."}


def crisis_windows(ctx: Context, windows: dict | None = None) -> dict:
    wins = windows or DEFAULT_CRISES
    rows = []
    for name, (a, b) in wins.items():
        seg = ctx.base_net.loc[a:b]
        if len(seg) >= 15:
            rows.append(_row(name, seg, total_return=float((1.0 + seg).prod() - 1.0)))
    if not rows:
        return {"name": "crisis_windows", "title": "Crisis windows", "description": "Named crises inside the out-of-sample period.", "rows": [], "verdict": "na", "detail": "No crisis window falls inside the out-of-sample period."}
    pos = sum(1 for r in rows if r["total_return"] > 0)
    return {"name": "crisis_windows", "title": "Crisis windows", "description": "Named crises inside the out-of-sample period.", "rows": rows,
            "verdict": _verdict(pos == len(rows), pos >= 1), "detail": f"Made money in {pos} of {len(rows)} crisis windows."}


def bull_bear(ctx: Context) -> dict:
    _, trend = market_states(ctx.dataset.returns, ctx.dataset.investable)
    st = trend.reindex(ctx.base_net.index)
    rows = [_row(f"{k} market", ctx.base_net[st == k]) for k in ("bull", "bear") if (st == k).sum() >= 20]
    pos = sum(1 for r in rows if r["sharpe"] > 0)
    return {"name": "bull_bear", "title": "Bull and bear markets", "description": "Sharpe when the equal-weight market was above or below its 200-day average (the day before).", "rows": rows,
            "verdict": "na" if len(rows) < 2 else _verdict(pos == 2, pos == 1), "detail": f"Positive in {pos} of {len(rows)} trends."}


def random_signals(ctx: Context) -> dict:
    """The control: random scores through the same signal rules, rebalancing and costs. Reports where the model's Sharpe falls among them."""
    n = int(ctx.cfg["robustness"]["n_draws"])
    rng = np.random.default_rng(int(ctx.cfg["robustness"]["seed"]) + 4)
    mask = ctx.scores.notna()
    sharpes = []
    for _ in range(n):
        rnd = pd.DataFrame(rng.standard_normal(ctx.scores.shape), index=ctx.scores.index, columns=ctx.scores.columns).where(mask)
        sharpes.append(sharpe_of(_run(ctx, _weights(ctx, rnd))))
    sharpes = np.array(sharpes, dtype=float)
    base = sharpe_of(ctx.base_net)
    pct = float(np.nanmean(sharpes < base))
    return {"name": "random_signals", "title": "Random-signal control", "description": f"{n} runs with random scores through the same position rules, rebalancing and costs.",
            "rows": [{"scenario": "random scores: 5th percentile", "sharpe": float(np.nanpercentile(sharpes, 5))}, {"scenario": "random scores: median", "sharpe": float(np.nanmedian(sharpes))},
                     {"scenario": "random scores: 95th percentile", "sharpe": float(np.nanpercentile(sharpes, 95))}, {"scenario": "the model", "sharpe": base}],
            "summary": {"model_percentile": pct, "random_median": float(np.nanmedian(sharpes)), "random_p95": float(np.nanpercentile(sharpes, 95))}, "verdict": _verdict(pct >= 0.95, pct >= 0.8),
            "detail": f"The model's Sharpe {base:.2f} beats {pct:.0%} of the random-score runs (their median is {np.nanmedian(sharpes):.2f})."}


TESTS = (random_start, universe_subsets, higher_costs, delayed_execution, noisy_features, parameter_perturbation, bootstrap, regime_split, crisis_windows, bull_bear, random_signals)


def run_all(ctx: Context, only: tuple | None = None, progress=None) -> dict:
    """Run the tests (all by default) and score them: the share that passed, counting a warning as half."""
    results = []
    for fn in TESTS:
        if only and fn.__name__ not in only:
            continue
        if progress:
            progress(f"robustness: {fn.__name__}")
        try:
            results.append(fn(ctx))
        except Exception as exc:                                                    # one failing test must not hide the others
            results.append({"name": fn.__name__, "title": fn.__name__, "description": "", "rows": [], "verdict": "na", "detail": f"could not run: {type(exc).__name__}: {exc}"})
    scored = [r for r in results if r["verdict"] != "na"]
    score = float(sum({"pass": 1.0, "warn": 0.5, "fail": 0.0}[r["verdict"]] for r in scored) / len(scored)) if scored else float("nan")
    return {"tests": results, "score": score, "n_scored": len(scored), "n_pass": sum(r["verdict"] == "pass" for r in results), "n_warn": sum(r["verdict"] == "warn" for r in results),
            "n_fail": sum(r["verdict"] == "fail" for r in results)}


def segment_labels_for(ctx_dates: pd.DatetimeIndex, plan_labels: pd.Series) -> pd.Series:
    return label_days(ctx_dates, plan_labels)
