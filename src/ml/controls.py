"""Random-signal controls: where does a model's result sit among results that cannot contain skill?

A Sharpe ratio of 0.5 means little until you know what the *same policy, costs and trading pace* produce from scores that carry no information. The control replays a saved run's position rule with random
scores in place of the model's, many times, and reports the share of those draws that did at least as well.

The random scores keep what a naive shuffle would destroy: their **persistence**. A model's ranks change slowly (a name is still near the top tomorrow), and the policy trades only when ranks move, so day-by-day
shuffled scores would trade far more than the model and look worse than they should. Each asset's random score is an AR(1) process whose coefficient is the model's own rank autocorrelation, so the draws have
the same pace of change. Their cross-section is Gaussian and their relation to returns is nil by construction.

A *matched* benchmark is reported beside it: the market scaled by the book's causal beta (``benchmarks.beta_matched``), because a long-only book earns the market's return for being exposed to it, and that is not
skill. The replay is checked against the saved run first: if it does not reproduce the saved Sharpe, the control says so and is not to be trusted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import benchmarks as bm
from .backtest import run_backtest, trim_to_signal
from .experiments import confidence_from_scores, trailing_vol
from .signals import build_weights

VERSION = "controls-v1"


def _sharpe(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / x.std(ddof=1) * np.sqrt(252.0)) if len(x) > 5 and x.std(ddof=1) > 0 else float("nan")


def _replay(cfg: dict, scores: pd.DataFrame, returns: pd.DataFrame, lag: int, investable: pd.DataFrame | None = None) -> dict:
    vol = trailing_vol(returns)
    w = build_weights(scores, cfg["signal"], vol, confidence_from_scores(scores, None), None, None, investable, None, returns)
    res = run_backtest(w, returns, investable, cfg["backtest"], lag, "control")
    first = scores.dropna(how="all").index[0]
    parts = trim_to_signal(res, first, int(cfg["signal"].get("lag_total", lag)))
    return {"net": parts["net"], "weights": parts["weights"], "turnover": parts["turnover"], "trades": parts["trades"]}


def rank_persistence(scores: pd.DataFrame) -> float:
    """The average one-day autocorrelation of the cross-sectional ranks: 0 for scores reshuffled every day, close to 1 for scores that barely change."""
    r = scores.rank(axis=1, pct=True)
    x, y = r.iloc[1:].to_numpy(), r.shift(1).iloc[1:].to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)
    xs, ys = x[ok] - 0.5, y[ok] - 0.5
    return float((xs * ys).sum() / np.sqrt((xs ** 2).sum() * (ys ** 2).sum())) if ok.sum() > 10 else 0.0


def random_scores(like: pd.DataFrame, rho: float, rng: np.random.Generator) -> pd.DataFrame:
    """Gaussian scores with the same shape and missing cells as ``like`` and day-to-day persistence ``rho`` (an AR(1) per asset)."""
    n, k = like.shape
    z = rng.standard_normal((n, k))
    s = np.zeros((n, k))
    s[0] = z[0]
    rho = float(np.clip(rho, 0.0, 0.995))
    for t in range(1, n):
        s[t] = rho * s[t - 1] + np.sqrt(1.0 - rho ** 2) * z[t]
    return pd.DataFrame(s, index=like.index, columns=like.columns).where(like.notna())


def run(cfg: dict, scores: pd.DataFrame, returns: pd.DataFrame, saved_net: pd.Series | None = None, *, investable: pd.DataFrame | None = None, n: int = 40, seed: int = 0, lag: int = 1, progress=None) -> dict:
    """``cfg`` is the saved run's configuration, ``scores`` its date x asset scores, ``returns`` the daily returns. ``saved_net`` (the run's recorded net returns) is used to check the replay."""
    if n < 5 or n > 200:
        raise ValueError("n is between 5 and 200 draws")
    progress = progress or (lambda msg: None)
    me = _replay(cfg, scores, returns, lag, investable)
    sh = _sharpe(me["net"])
    check = None
    if saved_net is not None:
        common = me["net"].index.intersection(saved_net.index)
        check = {"saved_sharpe": _sharpe(saved_net.loc[common]), "replayed_sharpe": _sharpe(me["net"].loc[common]), "days": int(len(common))}
        check["reproduces"] = bool(np.isfinite(check["saved_sharpe"]) and abs(check["saved_sharpe"] - check["replayed_sharpe"]) < 0.15)
    rho = rank_persistence(scores)
    rng = np.random.default_rng(seed)
    draws = []
    for i in range(n):
        progress(f"random-signal draw {i + 1} of {n}")
        r = _replay(cfg, random_scores(scores, rho, rng), returns, lag, investable)
        draws.append({"sharpe": _sharpe(r["net"]), "turnover": float(r["turnover"].mean() * 252) if len(r["turnover"]) else float("nan"), "n_trades": int((r["trades"].abs() > 1e-9).sum().sum()),
                      "gross": float(r["weights"].abs().sum(axis=1).mean())})
    d = pd.DataFrame(draws)
    ok = d["sharpe"].dropna()
    bench = bm.equal_weight(returns).reindex(me["net"].index)
    matched, beta = bm.beta_matched(me["net"], bench)
    active = (me["net"] - matched).dropna()
    mine_turn = float(me["turnover"].mean() * 252) if len(me["turnover"]) else float("nan")
    return {"version": VERSION, "model_sharpe": sh, "draws": int(len(ok)), "rank_persistence": rho, "replay_check": check,
            "random": {"mean": float(ok.mean()), "std": float(ok.std(ddof=1)) if len(ok) > 1 else None, "p05": float(ok.quantile(0.05)), "p50": float(ok.quantile(0.5)), "p95": float(ok.quantile(0.95)), "max": float(ok.max()),
                       "mean_turnover": float(d["turnover"].mean()), "mean_trades": float(d["n_trades"].mean()), "sharpes": [float(x) for x in ok.round(4)]},
            "model": {"turnover": mine_turn, "n_trades": int((me["trades"].abs() > 1e-9).sum().sum()), "gross": float(me["weights"].abs().sum(axis=1).mean())},
            "share_of_random_at_least_as_good": float((ok >= sh).mean()) if np.isfinite(sh) else None,
            "matched_benchmark": {"beta_mean": float(beta.mean()), "active_ann_return": float(active.mean() * 252) if len(active) else None, "active_sharpe": _sharpe(active), "matched_sharpe": _sharpe(matched),
                                  "equal_weight_sharpe": _sharpe(bench), "days": int(len(active))},
            "reading": ("The model's Sharpe ratio is compared with the same position rule, costs and rank persistence fed random scores. A share near 0.5 means the model is indistinguishable from noise at this pace of trading; "
                        "a share below 0.05 is some evidence of information beyond it (not proof: the draws are few, the sample is one history, and a choice of policy on this history is not corrected for).")}
