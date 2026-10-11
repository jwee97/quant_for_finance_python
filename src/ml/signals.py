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

Controls: ``cap_weights`` (a maximum per position, redistributing the excess), ``group_caps`` (a maximum gross per sector or asset class), ``holding_period`` (the average of the last ``H`` daily books, i.e. staggered
tranches that each stay for ``H`` days) and ``extra_lag`` (a delay on top of the engine's own).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METHODS = ("top_n_long_short", "long_only_top_quantile", "proportional", "vol_scaled", "rank_normalised", "threshold", "regime_filtered", "confidence_weighted")


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


def build_weights(score: pd.DataFrame, cfg: dict, vol: pd.DataFrame | None = None, confidence: pd.DataFrame | None = None, regime_on: pd.Series | None = None, groups: dict | None = None,
                  investable: pd.DataFrame | None = None) -> pd.DataFrame:
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
    else:
        if confidence is None:
            raise ValueError("the confidence_weighted method needs prediction confidences")
        w = proportional(s, gross, net, weight=confidence.reindex_like(s).clip(lower=0.0))
    if cfg.get("max_weight"):
        w = cap_weights(w, float(cfg["max_weight"]))
    if cfg.get("sector_cap") and groups:
        w = group_caps(w, groups, float(cfg["sector_cap"]))
    return holding_period(w, int(cfg.get("holding_period", 1)))
