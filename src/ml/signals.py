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
``cost_aware_mean_variance`` the book that maximises expected return less risk, trading cost and turnover from the book actually held (``allocation.solve``), for a model whose output is an expected return in
                           return units; the covariance is fitted as of each decision date (``risk_models.fit_asof``) and scaled to the forecast horizon, and every solve's status is reported
``calibrated_sizing``      the score calibrated walk-forward to a return, sized by edge, its uncertainty, risk and signal stability; flat when no edge has been shown (``sizing.py``)
``target_weights``         the scores ARE the target weights (a reinforcement-learning policy's output), only clipped to the gross exposure

Overlays on any of the above (optional ``signal`` settings, applied last, for a model whose output is a return forecast): ``trade_filter_k`` keeps a trade only if its net edge is at least that many forecast
standard errors (``tradefilter.py``); ``no_trade_margin`` holds the book unless moving earns its cost with that safety margin (``notrade.py``).

Controls: ``cap_weights`` (a maximum per position, redistributing the excess), ``group_caps`` (a maximum gross per sector or asset class), ``holding_period`` (the average of the last ``H`` daily books, i.e. staggered
tranches that each stay for ``H`` days) and ``extra_lag`` (a delay on top of the engine's own).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

UNCERTAINTY_METHODS = ("uncertainty_weighted", "uncertainty_mean_variance")
ALLOCATION_METHODS = ("cost_aware_mean_variance",)
SIZING_METHODS = ("calibrated_sizing",)
METHODS = ("top_n_long_short", "long_only_top_quantile", "proportional", "vol_scaled", "rank_normalised", "threshold", "regime_filtered", "confidence_weighted", *UNCERTAINTY_METHODS, *ALLOCATION_METHODS,
           *SIZING_METHODS, "target_weights")
OVERLAY_KEYS = ("trade_filter_k", "no_trade_margin", "regime_exposure")              # applied in this order, after the book is built; the first two need a forecast in return units, the last is a risk control that needs none
DERIVED_KEYS = ("cost_bps", "window_days", "label_offset", "rebalance", "lag_total")   # set by config.validate from the other sections, never by a request
STATEFUL_METHODS = (*ALLOCATION_METHODS, *SIZING_METHODS)                              # rules whose book depends on the book held or on a history of settled outcomes


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


def cost_aware_mean_variance(score: pd.DataFrame, returns: pd.DataFrame, investable: pd.DataFrame | None, cfg: dict, groups: dict | None = None,
                             dollar_volume: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """The cost-aware allocation (``allocation.solve``) at every ``opt_every``-th decision date, held between. ``score`` is read as the expected return over ``window_days`` (the dataset's horizon) and the daily
    covariance of each date's fit is scaled to the same window. The book held before each solve is the previous solution, so the trading cost and the turnover penalty act on real trades. Returns the weights and a
    log of every solve: how many were optimal, infeasible or failed, how many were no-trade, and what was done when a solve gave no book (the previous book is held and the event is counted, never hidden).

    **Capital and liquidity inside the choice.** With ``capital`` set, the optimiser is told how large the account is and what each name trades: ``max_participation`` caps each trade at that share of the
    name's average daily dollar volume, and ``impact`` adds the square-root market-impact cost to the objective, so a large account chooses a different book, not the same book filled more slowly.
    The average volume and the volatility are those of the ``adv_window`` sessions up to and including the decision date: nothing later. ``dollar_volume`` (date by asset) is then required: asking for a
    liquidity-aware book without volume is refused, never ignored."""
    from . import allocation as alloc
    from . import risk_models as rm

    net = float(cfg.get("net", 0.0))
    capital = None if cfg.get("capital") in (None, "") else float(cfg["capital"])
    participation = None if cfg.get("max_participation") in (None, "") else float(cfg["max_participation"])
    impact = None if cfg.get("impact") in (None, "") else float(cfg["impact"])
    adv_window = int(cfg.get("adv_window", 20))
    if capital is None and (participation is not None or impact is not None):
        raise ValueError("a participation limit or a market-impact cost needs signal.capital: the size of the account decides how large a trade is next to the market's volume")
    if capital is not None:
        if capital <= 0 or adv_window < 2:
            raise ValueError("signal.capital must be positive and signal.adv_window at least 2 sessions")
        if participation is None and impact is None:
            raise ValueError("signal.capital does nothing on its own: set signal.max_participation, signal.impact or both")
        if dollar_volume is None:
            raise ValueError("a liquidity-aware allocation needs the traded volume of the universe, and this dataset has none: the request is refused, not answered as if liquidity were unlimited")
        dollar_volume = dollar_volume.reindex(index=score.index, columns=score.columns)
    spec = alloc.AllocationSpec(risk_aversion=float(cfg.get("risk_aversion", 5.0)), cost=float(cfg["cost_bps"]) / 1e4, turnover_penalty=float(cfg.get("turnover_penalty", 0.0)),
                                gross_max=float(cfg.get("gross", 1.0)), position_max=float(cfg.get("max_weight") or 0.35), net_min=net, net_max=net, group_max=cfg.get("sector_cap") or None,
                                fallback=str(cfg.get("alloc_fallback", "none")), impact=impact, max_participation=participation)
    cov_spec = rm.CovSpec(str(cfg.get("cov_method", "lw_constant_correlation")), window=int(cfg.get("cov_window", 252)))
    days = int(cfg["window_days"])
    every = int(cfg.get("opt_every") or days)
    out = np.zeros(score.shape)
    last = pd.Series(0.0, index=score.columns)
    log = {"method": "cost_aware_mean_variance", "spec": spec.content(), "covariance": cov_spec.content(), "window_days": days, "opt_every": every, "solves": 0, "optimal": 0, "infeasible": 0, "failed": 0,
           "fallback_used": 0, "held_after_failure": 0, "no_trade": 0, "skipped_too_few_assets": 0, "skipped_no_covariance": 0, "repaired_covariances": 0,
           "note": "a solve with no book holds the previous book and is counted; a configured fallback is labelled in the solve and counted under fallback_used",
           "liquidity": None if capital is None else {"capital": capital, "max_participation": participation, "impact": impact, "adv_window": adv_window, "names_without_volume": 0, "solves_at_a_trade_limit": 0,
                                                      "as_of": "the sessions up to and including the decision date"}}
    for i, date in enumerate(score.index):
        if i % every == 0:
            row = score.iloc[i].dropna()
            if len(row) < 4:
                last = pd.Series(0.0, index=score.columns)
                log["skipped_too_few_assets"] += int(len(row) > 0)                              # a date with no forecast at all (before the model's first prediction) is not a skipped solve
            else:
                try:
                    fit = rm.fit_asof(returns, date, cov_spec, investable)
                except ValueError:
                    fit = None
                names = [] if fit is None else [a for a in row.index if a in fit.matrix.index]
                if len(names) < 4:
                    last = pd.Series(0.0, index=score.columns)
                    log["skipped_no_covariance"] += 1
                else:
                    mu_w, cov_w = alloc.window_inputs(row[names], fit.matrix.loc[names, names], days)
                    log["repaired_covariances"] += int(fit.repaired)
                    liquidity = {}
                    if capital is not None:
                        adv = dollar_volume.iloc[max(0, i - adv_window + 1):i + 1][names].mean(skipna=True)             # what each name traded, on average, up to today
                        sigma = returns.reindex(index=score.index, columns=score.columns).iloc[max(0, i - adv_window + 1):i + 1][names].std(ddof=1)
                        log["liquidity"]["names_without_volume"] += int((~(adv > 0)).sum())                              # such a name cannot be traded at this date: its limit is zero
                        liquidity = {"capital": capital, "adv": adv.fillna(0.0), "sigma": sigma}
                    sol = alloc.solve(mu_w, cov_w, last.reindex(names).fillna(0.0), spec, groups=groups, **liquidity)
                    if capital is not None and participation is not None and sol.trades is not None:
                        limit = (participation * liquidity["adv"] / capital).reindex(sol.trades.index)
                        log["liquidity"]["solves_at_a_trade_limit"] += int((sol.trades.abs() >= limit - 1e-9).any() and bool((limit < 10.0).any()))
                    log["solves"] += 1
                    log[sol.status if sol.status in ("optimal", "infeasible", "failed") else "failed"] += 1
                    log["fallback_used"] += int(sol.fallback_used is not None)
                    if sol.weights is None:
                        log["held_after_failure"] += 1
                    else:
                        log["no_trade"] += int(sol.no_trade)
                        last = sol.weights.reindex(score.columns).fillna(0.0)
        out[i] = last.to_numpy()
    return pd.DataFrame(out, index=score.index, columns=score.columns), log


def target_weights(score: pd.DataFrame, gross: float = 1.0) -> pd.DataFrame:
    """The scores read as weights (a policy's output): missing values are flat and a book above ``gross`` is scaled down to it."""
    w = score.fillna(0.0)
    total = w.abs().sum(axis=1)
    return w.mul(np.where(total > gross, gross / total.replace(0.0, np.nan), 1.0), axis=0).fillna(0.0)


def build_weights(score: pd.DataFrame, cfg: dict, vol: pd.DataFrame | None = None, confidence: pd.DataFrame | None = None, regime_on: pd.Series | None = None, groups: dict | None = None,
                  investable: pd.DataFrame | None = None, uncertainty: pd.DataFrame | None = None, returns: pd.DataFrame | None = None, dollar_volume: pd.DataFrame | None = None) -> pd.DataFrame:
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
    elif method == "cost_aware_mean_variance":
        if returns is None:
            raise ValueError("the cost_aware_mean_variance method needs the assets' returns")
        w, alloc_log = cost_aware_mean_variance(s, returns, investable, cfg, groups, dollar_volume)
    elif method == "calibrated_sizing":
        if returns is None:
            raise ValueError("the calibrated_sizing method needs the assets' returns")
        from . import sizing

        w, size_log = sizing.apply(s, returns, investable, cfg)
    elif method == "target_weights":
        w = target_weights(s, gross)
    else:
        if confidence is None:
            raise ValueError("the confidence_weighted method needs prediction confidences")
        w = proportional(s, gross, net, weight=confidence.reindex_like(s).clip(lower=0.0))
    if cfg.get("max_weight") and method not in ALLOCATION_METHODS:                  # the optimiser enforces its own position and group limits: capping afterwards would move its solution
        w = cap_weights(w, float(cfg["max_weight"]))
    if cfg.get("sector_cap") and groups and method not in ALLOCATION_METHODS:
        w = group_caps(w, groups, float(cfg["sector_cap"]))
    out = holding_period(w, int(cfg.get("holding_period", 1)))
    overlays = {}
    if cfg.get("trade_filter_k") is not None or cfg.get("no_trade_margin") is not None:
        if returns is None:
            raise ValueError("the trade filter and the no-trade region need the assets' returns")
        if cfg.get("trade_filter_k") is not None:
            from . import tradefilter

            out, overlays["trade_filter"] = tradefilter.apply(out, s, returns, investable, cfg)
        if cfg.get("no_trade_margin") is not None:
            from . import notrade

            out, overlays["no_trade"] = notrade.apply(out, s, returns, investable, cfg)
    if cfg.get("regime_exposure") is not None:                                         # last: a risk cut on the whole book is not vetoed by a rule that weighs expected return against cost
        if returns is None:
            raise ValueError("the regime exposure needs the assets' returns")
        from . import regime_exposure

        out, overlays["regime_exposure"] = regime_exposure.apply(out, returns, investable, cfg)
    if method in ALLOCATION_METHODS:
        out.attrs["allocation"] = alloc_log
    if method in SIZING_METHODS:
        out.attrs["sizing"] = size_log
    if overlays:
        out.attrs["overlays"] = overlays
    return out
