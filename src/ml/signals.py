"""From predictions to positions: eight ways to turn a panel of scores into target weights, and the controls that keep the book tradable.

A score is "higher is better" and is comparable across assets on a date (not across dates). Every method returns target weights stamped at the date of the score; the backtest engine applies the
execution lag, the rebalance schedule, drift and costs. Gross exposure is ``sum |w|`` and net exposure ``sum w``: with ``gross = 1`` and ``net = 0`` the long book and the short book are 0.5 each.

Methods (``build_weights(method=...)``):

``top_n_long_short``       the ``n`` best equally weighted long, the ``n`` worst equally weighted short
``long_only_top_quantile`` the best ``quantile`` of the assets, equally weighted, fully invested up to ``gross``
``proportional``           weight proportional to the cross-sectionally demeaned score
``vol_scaled``             the same divided by the asset's volatility, so each position takes a similar risk
``rank_normalised``        weight proportional to the centred rank (robust to outlying scores)
``threshold``              long above ``+threshold``, short below ``-threshold``, flat in between
``regime_filtered``        ``top_n_long_short`` that is switched off when the regime says so
``confidence_weighted``    ``proportional`` times the prediction's confidence
``uncertainty_weighted``   each score shrunk toward zero by ``kappa`` times the asset's excess predictive uncertainty, then weighted by score over std ** ``uncertainty_power``
``uncertainty_mean_variance``  those views through the inverse of the (Ledoit-Wolf) covariance of trailing returns: the covariance-aware version of the same idea
``target_weights``         the scores ARE the target weights (a reinforcement-learning policy's output), only clipped to the gross exposure

Controls: ``cap_weights`` (a maximum per position, redistributing the excess), ``group_caps`` (a maximum gross per sector or asset class), ``holding_period`` (the average of the last ``H`` daily books, i.e. staggered
tranches that each stay for ``H`` days) and ``extra_lag`` (a delay on top of the engine's own).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

UNCERTAINTY_METHODS = ("uncertainty_weighted", "uncertainty_mean_variance")
METHODS = ("top_n_long_short", "long_only_top_quantile", "proportional", "vol_scaled", "rank_normalised", "threshold", "regime_filtered", "confidence_weighted", *UNCERTAINTY_METHODS, "target_weights")


def _scale_sides(raw: np.ndarray, gross: float, net: float) -> np.ndarray:
    """Scale each row so the long side sums to ``(gross + net) / 2`` and the short side to ``-(gross - net) / 2`` (a side with no positions stays empty)."""
    out = np.zeros_like(raw)
    long_t, short_t = (gross + net) / 2.0, (gross - net) / 2.0
    pos, neg = np.where(raw > 0, raw, 0.0), np.where(raw < 0, -raw, 0.0)
    ps, ns = pos.sum(axis=1, keepdims=True), neg.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        out += np.where(ps > 0, pos / ps * long_t, 0.0)
        out -= np.where(ns > 0, neg / ns * short_t, 0.0)
    return out


def _wide(frame: pd.DataFrame) -> np.ndarray:
    return frame.to_numpy(dtype=float)


def top_n_long_short(score: pd.DataFrame, n: int = 3, gross: float = 1.0, net: float = 0.0) -> pd.DataFrame:
    """The best ``n`` long and the worst ``n`` short per date, equally weighted within each side. With fewer than ``2n`` valid scores each side takes half of them; with fewer than 2, no position."""
    s = _wide(score)
    valid = np.isfinite(s)
    order = np.argsort(np.where(valid, s, np.nan), axis=1)                       # NaN sorts last
    k = valid.sum(axis=1)
    take = np.minimum(n, k // 2)
    raw = np.zeros_like(s)
    ranks = np.empty_like(order)
    rows = np.arange(len(s))[:, None]
    ranks[rows, order] = np.arange(s.shape[1])[None, :]
    long_mask = valid & (ranks >= (k - take)[:, None])
    short_mask = valid & (ranks < take[:, None])
    raw[long_mask] = 1.0
    raw[short_mask] = -1.0
    return pd.DataFrame(_scale_sides(raw, gross, net), index=score.index, columns=score.columns)


def long_only_top_quantile(score: pd.DataFrame, quantile: float = 0.3, gross: float = 1.0) -> pd.DataFrame:
    """The top ``quantile`` of the valid scores (at least one asset), equally weighted, summing to ``gross``."""
    s = _wide(score)
    valid = np.isfinite(s)
    k = valid.sum(axis=1)
    take = np.maximum(1, np.ceil(quantile * k)).astype(int)
    order = np.argsort(np.where(valid, -s, np.inf), axis=1)
    ranks = np.empty_like(order)
    ranks[np.arange(len(s))[:, None], order] = np.arange(s.shape[1])[None, :]
    raw = np.where(valid & (ranks < take[:, None]) & (k[:, None] > 0), 1.0, 0.0)
    return pd.DataFrame(_scale_sides(raw, gross, gross), index=score.index, columns=score.columns)


def proportional(score: pd.DataFrame, gross: float = 1.0, net: float = 0.0, weight: pd.DataFrame | None = None) -> pd.DataFrame:
    """Weight proportional to the demeaned score (times ``weight`` if given), the long and short sides scaled to the targets. For ``net == gross`` (long only) the positive part of the score is used."""
    s = score.sub(score.mean(axis=1), axis=0) if net < gross - 1e-12 else score.clip(lower=0.0)
    if weight is not None:
        s = s * weight
    return pd.DataFrame(_scale_sides(_wide(s.fillna(0.0)), gross, net), index=score.index, columns=score.columns)


def vol_scaled(score: pd.DataFrame, vol: pd.DataFrame, gross: float = 1.0, net: float = 0.0) -> pd.DataFrame:
    """``proportional`` with each score divided by the asset's volatility (``vol`` known at the date)."""
    return proportional(score, gross, net, weight=1.0 / vol.reindex_like(score).replace(0.0, np.nan))


def rank_normalised(score: pd.DataFrame, gross: float = 1.0, net: float = 0.0) -> pd.DataFrame:
    """Weight proportional to the centred percentile rank: the median asset gets zero, the best and worst the extremes."""
    r = score.rank(axis=1, pct=True)
    centred = r.sub(r.mean(axis=1), axis=0) if net < gross - 1e-12 else (r - 0.5).clip(lower=0.0)
    return pd.DataFrame(_scale_sides(_wide(centred.fillna(0.0)), gross, net), index=score.index, columns=score.columns)


def threshold(score: pd.DataFrame, level: float = 0.0, gross: float = 1.0, net: float = 0.0) -> pd.DataFrame:
    """Long assets whose score exceeds ``+level``, short those below ``-level``, equally weighted within each side. A strict threshold gives an empty book on days with no conviction."""
    raw = np.where(score > level, 1.0, np.where(score < -level, -1.0, 0.0))
    if net >= gross - 1e-12:
        raw = np.where(raw > 0, 1.0, 0.0)
    return pd.DataFrame(_scale_sides(raw, gross, net), index=score.index, columns=score.columns)


def regime_filter(weights: pd.DataFrame, on: pd.Series) -> pd.DataFrame:
    """Zero the book on the dates where ``on`` is False (missing counts as off): the regime says the signal does not work now."""
    flag = on.reindex(weights.index).fillna(False).astype(bool).to_numpy().astype(float)
    return weights.mul(flag, axis=0)


def cap_weights(w: pd.DataFrame, max_weight: float, iterations: int = 25) -> pd.DataFrame:
    """Limit every position to ``max_weight`` (absolute) and give the excess to the uncapped positions of the same side in proportion to their size, repeating until nothing exceeds the cap. A date on
    which every position on a side is already at the cap keeps the smaller total."""
    v = _wide(w).copy()
    for sign in (1.0, -1.0):
        side = np.where(sign * v > 0, sign * v, 0.0)
        target = side.sum(axis=1)
        for _ in range(iterations):
            over = side > max_weight + 1e-12
            if not over.any():
                break
            side = np.minimum(side, max_weight)
            free = (side > 0) & (side < max_weight - 1e-12)
            room = target - side.sum(axis=1)
            fs = np.where(free, side, 0.0).sum(axis=1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                add = np.where(free & (fs > 0), side / fs * room[:, None], 0.0)
            side = side + np.clip(add, 0.0, None)
        v = np.where(sign * v > 0, sign * side, v)
    return pd.DataFrame(v, index=w.index, columns=w.columns)


def group_caps(w: pd.DataFrame, groups: dict, cap: float) -> pd.DataFrame:
    """Scale down every group (a sector or asset class) whose gross exposure exceeds ``cap``, leaving the others as they are (the total gross falls by what was cut)."""
    out = w.copy()
    members: dict[str, list[str]] = {}
    for a in w.columns:
        members.setdefault(groups.get(a, "other"), []).append(a)
    for cols in members.values():
        gross = w[cols].abs().sum(axis=1)
        scale = (cap / gross.replace(0.0, np.nan)).clip(upper=1.0).fillna(1.0)
        out[cols] = w[cols].mul(scale, axis=0)
    return out


def holding_period(w: pd.DataFrame, days: int) -> pd.DataFrame:
    """The average of the last ``days`` daily target books: each day's book is one of ``days`` tranches that is kept for ``days`` days, so turnover falls by about that factor."""
    return w if days <= 1 else w.rolling(days, min_periods=1).mean()


def extra_lag(w: pd.DataFrame, days: int) -> pd.DataFrame:
    """Delay the book by ``days`` more days (the first rows are empty)."""
    return w if days <= 0 else w.shift(days).fillna(0.0)


def _floor_std(std: pd.DataFrame, like: pd.DataFrame) -> pd.DataFrame:
    """The predictive standard deviations aligned to ``like``, never below a quarter of the day's median (a near-zero spread would give one asset all the weight)."""
    sd = std.reindex_like(like).where(lambda d: d > 0)
    return sd.clip(lower=0.25 * sd.median(axis=1), axis=0)


def shrink_by_uncertainty(score: pd.DataFrame, std: pd.DataFrame, kappa: float) -> pd.DataFrame:
    """The demeaned score moved toward zero by ``kappa`` times the asset's EXCESS uncertainty (its standard deviation above the day's median): an asset the model is unusually unsure about loses
    conviction, one of typical uncertainty keeps it. (A model's own spread is usually many times its scores, so shrinking by the whole standard deviation would remove every view.)"""
    mu = score.sub(score.mean(axis=1), axis=0)
    excess = std.sub(std.median(axis=1), axis=0).clip(lower=0.0)
    return np.sign(mu) * (mu.abs() - float(kappa) * excess).clip(lower=0.0)


def uncertainty_weighted(score: pd.DataFrame, std: pd.DataFrame, gross: float = 1.0, net: float = 0.0, kappa: float = 0.0, power: float = 1.0) -> pd.DataFrame:
    """Weight proportional to the shrunk score divided by ``std ** power``: conviction per unit of the model's own uncertainty."""
    sd = _floor_std(std, score)
    raw = (shrink_by_uncertainty(score, sd, kappa) * sd ** (-float(power))).fillna(0.0)
    return pd.DataFrame(_scale_sides(_wide(raw), gross, net), index=score.index, columns=score.columns)


def uncertainty_mean_variance(score: pd.DataFrame, std: pd.DataFrame, returns: pd.DataFrame, gross: float = 1.0, net: float = 0.0, kappa: float = 0.0, power: float = 1.0, lookback: int = 126,
                              rebalance: int = 5) -> pd.DataFrame:
    """The views (shrunk scores divided by ``std ** power``) through the inverse covariance of the trailing ``lookback`` days of returns (Ledoit-Wolf), re-solved every ``rebalance`` days and held between. The book is made dollar neutral along
    the covariance-weighted direction, scaled to ``gross``, and a non-zero ``net`` is added as the minimum-variance portfolio. Each solve uses the returns up to and including the day."""
    from sklearn.covariance import LedoitWolf

    sd = _floor_std(std, score)
    mu = shrink_by_uncertainty(score, sd, kappa) * sd ** (-float(power))
    R = returns.reindex(columns=score.columns)
    where = R.index.get_indexer(score.index)
    out, last = np.zeros(score.shape), np.zeros(score.shape[1])
    for i in range(len(score)):
        if i % max(int(rebalance), 1) == 0:
            m = mu.iloc[i].to_numpy()
            ok = np.isfinite(m) & (where[i] >= 0)
            window = R.iloc[max(0, where[i] - int(lookback) + 1): where[i] + 1].loc[:, ok]
            keep = window.notna().mean() > 0.8
            last = np.zeros(score.shape[1])
            if ok.sum() >= 3 and keep.sum() >= 3 and len(window) >= 40 and np.abs(m[ok]).sum() > 0:
                idx = np.flatnonzero(ok)[keep.to_numpy()]
                cov = LedoitWolf().fit(window.loc[:, keep].fillna(0.0)).covariance_ * 252.0 + 1e-8 * np.eye(len(idx))
                inv_mu, inv_one = np.linalg.solve(cov, m[idx]), np.linalg.solve(cov, np.ones(len(idx)))
                w = inv_mu - inv_mu.sum() / inv_one.sum() * inv_one                          # dollar neutral along the covariance-weighted direction
                if np.abs(w).sum() > 0:
                    w = w / np.abs(w).sum() * gross + net * inv_one / inv_one.sum()
                    last[idx] = w
        out[i] = last
    return pd.DataFrame(out, index=score.index, columns=score.columns)


def target_weights(score: pd.DataFrame, gross: float = 1.0) -> pd.DataFrame:
    """The scores read as weights (a policy's output): missing values are flat and a book above ``gross`` is scaled down to it."""
    w = score.fillna(0.0)
    total = w.abs().sum(axis=1)
    return w.mul(np.where(total > gross, gross / total.replace(0.0, np.nan), 1.0), axis=0).fillna(0.0)


def build_weights(score: pd.DataFrame, cfg: dict, vol: pd.DataFrame | None = None, confidence: pd.DataFrame | None = None, regime_on: pd.Series | None = None, groups: dict | None = None,
                  investable: pd.DataFrame | None = None, uncertainty: pd.DataFrame | None = None, returns: pd.DataFrame | None = None) -> pd.DataFrame:
    """Target weights for a panel of scores according to the ``signal`` section of an experiment configuration (method and its settings, ``max_weight``, ``sector_cap``, ``holding_period``)."""
    method = cfg["method"]
    if method not in METHODS:
        raise ValueError(f"unknown signal method '{method}'; choose from {list(METHODS)}")
    gross, net = float(cfg.get("gross", 1.0)), float(cfg.get("net", 0.0))
    s = score.where(investable.reindex_like(score).fillna(False)) if investable is not None else score
    if method == "top_n_long_short":
        w = top_n_long_short(s, int(cfg.get("n", 3)), gross, net)
    elif method == "long_only_top_quantile":
        w = long_only_top_quantile(s, float(cfg.get("quantile", 0.3)), gross)
    elif method == "proportional":
        w = proportional(s, gross, net)
    elif method == "vol_scaled":
        if vol is None:
            raise ValueError("the vol_scaled method needs trailing volatilities")
        w = vol_scaled(s, vol, gross, net)
    elif method == "rank_normalised":
        w = rank_normalised(s, gross, net)
    elif method == "threshold":
        w = threshold(s, float(cfg.get("threshold", 0.0)), gross, net)
    elif method == "regime_filtered":
        if regime_on is None:
            raise ValueError("the regime_filtered method needs a regime series (set dataset.regime_detector)")
        w = regime_filter(top_n_long_short(s, int(cfg.get("n", 3)), gross, net), regime_on)
    elif method in UNCERTAINTY_METHODS:
        if uncertainty is None:
            raise ValueError(f"the {method} method needs a model that reports its own uncertainty (quantile LightGBM, NGBoost, the Temporal Fusion Transformer or a model ensemble)")
        if method == "uncertainty_weighted":
            w = uncertainty_weighted(s, uncertainty, gross, net, float(cfg.get("kappa", 0.0)), float(cfg.get("uncertainty_power", 1.0)))
        else:
            if returns is None:
                raise ValueError("the uncertainty_mean_variance method needs the assets' returns")
            w = uncertainty_mean_variance(s, uncertainty, returns, gross, net, float(cfg.get("kappa", 0.0)), float(cfg.get("uncertainty_power", 1.0)), int(cfg.get("mv_lookback", 126)),
                                          int(cfg.get("mv_rebalance", 5)))
    elif method == "target_weights":
        w = target_weights(s, gross)
    else:
        if confidence is None:
            raise ValueError("the confidence_weighted method needs prediction confidences")
        w = proportional(s, gross, net, weight=confidence.reindex_like(s).clip(lower=0.0))
    if cfg.get("max_weight"):
        w = cap_weights(w, float(cfg["max_weight"]))
    if cfg.get("sector_cap") and groups:
        w = group_caps(w, groups, float(cfg["sector_cap"]))
    return holding_period(w, int(cfg.get("holding_period", 1)))
