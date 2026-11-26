"""Explaining a portfolio's realised return: factor attribution and Brinson-Fachler attribution of what the Portfolio Lab's optimisers actually earned.

Two different questions, never mixed:

**Factor attribution (ex post, realised).** On each day the portfolio's gross return is split into what the *ex-ante* exposures to each factor earned that day, and what was left over in each asset:

    gross_t = sum_k (b_k' w_t) f_kt  +  sum_i w_it e_it            e_it = r_it - sum_k b_ik f_kt

with ``w_t`` the weights held at the start of day ``t`` (drifted from the last rebalance), ``b`` the exposures, ``f`` the realised factor returns and ``e`` the asset-specific residual. The exposures of the
rebalance at row ``p`` are estimated on the window that ends the day *before* ``p`` (the same window the optimiser saw), and held until the next rebalance, so nothing on or after the day being explained enters them.
Net of costs: ``net_t = gross_t - cost_t``; the Lab models no financing (long only, fully invested), so that line is reported as zero and says so. The residual of the reconciliation is
``authoritative_net_t - (sum of the lines)`` and is shown as its own line: it is never spread over the factors.

Linking over time: each day's contributions are scaled by the portfolio's value at the start of the day (``V_{t-1}``), so the cumulative contributions add up to ``V_T - 1`` exactly at every date.

**Ex-ante exposures** (what the portfolio was positioned to do: ``b' w`` at each rebalance, and the share of forecast variance from the factors) are reported separately from the **ex-post contributions** (what happened).

Two factor models, both from :mod:`src.riskengine.factor_model`: ``market`` (one factor, the equal-weighted return of the universe, betas from the past window) and ``statistical`` (principal components of the past window; the
realised factor return of day t is the projection of that day's returns on the fixed loadings).

**Brinson-Fachler attribution (ex post, relative to a benchmark)**, with :func:`src.backtest.attribution.brinson_fachler` on every day's earning weights: allocation, selection and interaction by group, which sum to the
day's gross active return exactly; costs enter as their own line; the whole is Carino-linked (:func:`src.backtest.attribution.link_effects`) on the NET returns, so the linked lines (allocation, selection, interaction, costs,
and the reconciliation residual) sum to the cumulative net active return exactly. Cash (weights summing to less than one), assets with no group (a group called ``Unclassified``, with a warning), assets the benchmark
does not hold (zero weight) and a classification that changes over time (a list of ``(effective date, mapping)`` pairs applied point in time) are handled explicitly; a benchmark whose weights do not sum to one is refused.

**Currency**: :func:`currency_decomposition` splits a base-currency return into local, currency and cross terms that add up to it exactly. The Lab's data are in one currency, so the page does not offer it.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..backtest.attribution import brinson_fachler, link_effects
from ..riskengine import factor_model as fm

TOLERANCE = 1e-9                     # the reconciliation of a replayed daily return with the authoritative one, and of cumulative sums with a cumulative return
CASH = "CASH"
UNCLASSIFIED = "Unclassified"
ANN = 252.0


class AttributionError(ValueError):
    """The attribution cannot be made (and why)."""


# ------------------------------------------------------------------------------------------------------------------------ replaying the book
def replay(data: pd.DataFrame, targets: pd.DataFrame, step: int, cost_bps: float) -> dict:
    """The Portfolio Lab's accounting replayed from its target weights: drift between rebalances, the cost of the whole trade charged on the rebalance day. Returns the gross and net daily returns, the
    cost per day, and the weights that EARN each day (held at the start of it). Identical in arithmetic to ``portfolio.lab.compare``, which is what makes the reconciliation against it meaningful."""
    cost = cost_bps * 1e-4
    R = data.to_numpy(dtype=float)
    n = R.shape[1]
    pos = data.index.get_indexer(pd.DatetimeIndex(targets.index))
    if (pos < 0).any():
        raise AttributionError("a rebalance date is not in the return history")
    gross, net, costs = {}, {}, {}
    earning = {}
    held = np.zeros(n)
    traded_by = {}
    for p, (date, row) in zip(pos, targets.iterrows()):
        w = row.to_numpy(dtype=float)
        traded = float(np.abs(w - held).sum())
        traded_by[data.index[p]] = traded
        cur = w.copy()
        for t in range(p, min(p + step, len(data))):
            gr = float(cur @ R[t])
            day = data.index[t]
            earning[day] = cur.copy()
            gross[day] = gr
            costs[day] = cost * traded if t == p else 0.0
            net[day] = gr - costs[day]
            cur = cur * (1.0 + R[t]) / (1.0 + gr)
        held = cur
    idx = pd.DatetimeIndex(list(gross))
    return {"gross": pd.Series(gross), "net": pd.Series(net), "cost": pd.Series(costs), "earning": pd.DataFrame(earning, index=data.columns).T.loc[idx], "traded": pd.Series(traded_by)}


# ------------------------------------------------------------------------------------------------------------------------ factor attribution
def _models(data: pd.DataFrame, positions: list[int], lookback: int, kind: str, n_factors: int):
    """One factor model per rebalance, each from the window that ends the day before the rebalance."""
    market = data.mean(axis=1).rename("market")
    out = {}
    for p in positions:
        win = data.iloc[p - lookback:p]
        if kind == "market":
            out[p] = fm.fit_time_series(win, market.iloc[p - lookback:p].to_frame(), min_obs=min(60, lookback // 2))
        elif kind == "statistical":
            out[p] = fm.fit_statistical(win, n_factors=n_factors, min_obs=min(60, lookback // 2))
        else:
            raise AttributionError("kind must be market or statistical")
    return out, market


def factor_attribution(data: pd.DataFrame, targets: pd.DataFrame, authoritative_net: pd.Series, lookback: int, step: int, cost_bps: float, kind: str = "market", n_factors: int = 2) -> dict:
    """Daily and cumulative factor attribution of one method (see the module docstring). ``authoritative_net`` is the net daily return the Lab reported; the lines are reconciled against it."""
    if kind not in ("market", "statistical"):
        raise AttributionError("kind must be market or statistical")
    rp = replay(data, targets, step, cost_bps)
    pos = [int(p) for p in data.index.get_indexer(pd.DatetimeIndex(targets.index))]
    models, market = _models(data, pos, lookback, kind, n_factors)
    K = next(iter(models.values())).exposures.shape[1]
    factors = list(next(iter(models.values())).exposures.columns)
    days = rp["gross"].index
    fac = pd.DataFrame(0.0, index=days, columns=factors)
    spec_by_asset = pd.DataFrame(0.0, index=days, columns=data.columns)
    ex_ante = []
    for i, p in enumerate(pos):
        m = models[p]
        end = min(p + step, len(data))
        B = m.exposures.reindex(data.columns).fillna(0.0).to_numpy(dtype=float)               # assets x factors, from the past window only
        seg = data.index[p:end]
        W = rp["earning"].loc[seg].to_numpy(dtype=float)
        Rm = data.loc[seg].to_numpy(dtype=float)
        if kind == "market":
            F = market.loc[seg].to_numpy(dtype=float).reshape(-1, 1)
        else:
            F = Rm @ B                                                                      # the projection of the day's returns on the (orthonormal) loadings fitted in the past
        E = Rm - F @ B.T
        fac.loc[seg] = (W @ B) * F
        spec_by_asset.loc[seg] = W * E
        w0 = targets.iloc[i].reindex(data.columns).to_numpy(dtype=float)
        b0 = B.T @ w0
        Sf = m.factor_cov.to_numpy(dtype=float)
        fvar = float(b0 @ Sf @ b0)
        svar = float((w0 ** 2 * m.specific_var.reindex(data.columns).fillna(0.0).to_numpy(dtype=float)).sum())
        ex_ante.append({"date": str(data.index[p].date()), "exposures": {f: float(x) for f, x in zip(factors, b0)}, "factor_risk_share": fvar / (fvar + svar) if fvar + svar > 0 else None,
                        "ex_ante_volatility": math.sqrt(max(fvar + svar, 0.0) * ANN)})
    specific = spec_by_asset.sum(axis=1)
    cost = rp["cost"].reindex(days)
    model_net = fac.sum(axis=1) + specific - cost
    auth = authoritative_net.reindex(days)
    if auth.isna().any():
        raise AttributionError("the authoritative return series does not cover every attributed day")
    residual = auth - model_net
    # contributions scaled by the value at the start of each day add up to V_T - 1 at every date (wealth-weighted linking)
    V = (1.0 + auth).cumprod()
    V_prev = V.shift(1).fillna(1.0)
    lines = pd.concat([fac, specific.rename("specific"), (-cost).rename("costs"), pd.Series(0.0, index=days, name="financing"), residual.rename("residual")], axis=1)
    cum = lines.mul(V_prev, axis=0).cumsum()
    by_asset_cum = spec_by_asset.mul(V_prev, axis=0).cumsum()
    total_cum = float(V.iloc[-1] - 1.0)
    daily_gap = float(residual.abs().max())
    cum_gap = float(cum.iloc[-1].sum() - total_cum)
    return {"kind": kind, "factors": factors, "days": [str(d.date()) for d in days], "daily": {c: lines[c].tolist() for c in lines.columns}, "cumulative": {c: cum[c].tolist() for c in cum.columns},
            "specific_by_asset_cumulative": {a: float(by_asset_cum[a].iloc[-1]) for a in by_asset_cum.columns}, "totals": {c: float(cum[c].iloc[-1]) for c in cum.columns}, "ex_ante": ex_ante,
            "reconciliation": {"authoritative_cumulative_return": total_cum, "sum_of_lines": float(cum.iloc[-1].sum()), "cumulative_gap": cum_gap, "max_daily_residual": daily_gap, "tolerance": TOLERANCE,
                               "status": "reconciled" if daily_gap <= TOLERANCE and abs(cum_gap) <= TOLERANCE else "does_not_reconcile",
                               "replay_vs_authoritative_max_daily": float((rp["net"].reindex(days) - auth).abs().max())}}


# ------------------------------------------------------------------------------------------------------------------------ Brinson-Fachler
def _versions(sector_of) -> list[tuple[pd.Timestamp, dict]]:
    if isinstance(sector_of, dict):
        return [(pd.Timestamp.min, {str(k): str(v) for k, v in sector_of.items()})]
    out = sorted(((pd.Timestamp(d), {str(k): str(v) for k, v in m.items()}) for d, m in sector_of), key=lambda x: x[0])
    if not out:
        raise AttributionError("an empty classification")
    return out


def brinson(data: pd.DataFrame, portfolio_earning: pd.DataFrame, benchmark_earning: pd.DataFrame, sector_of, portfolio_cost: pd.Series, benchmark_cost: pd.Series, portfolio_net: pd.Series,
            benchmark_net: pd.Series) -> dict:
    """Daily Brinson-Fachler effects by group, Carino-linked on the net returns (see the module docstring)."""
    days = portfolio_earning.index
    versions = _versions(sector_of)
    warnings: list[str] = []
    bench_missing = [c for c in data.columns if benchmark_earning[c].abs().max() <= 1e-12]
    if bench_missing:
        warnings.append(f"the benchmark holds nothing in {', '.join(map(str, bench_missing))}: their benchmark weight is zero")
    ret = data.loc[days]
    wp, wb = portfolio_earning.reindex(columns=data.columns).fillna(0.0), benchmark_earning.reindex(columns=data.columns).fillna(0.0)
    for name, w in (("benchmark", wb), ("portfolio", wp)):
        short = 1.0 - w.sum(axis=1)
        if (short < -1e-9).any():
            raise AttributionError(f"{name} weights sum to more than one on some day: attribution of a levered book is not supported")
        if name == "benchmark" and (short.abs() > 1e-9).any():
            raise AttributionError("the benchmark's weights do not sum to one: a benchmark with cash or missing weights must be completed before it is attributed against")
        if (short > 1e-9).any():
            warnings.append(f"the {name} holds cash on {int((short > 1e-9).sum())} days: it is a group of its own with a zero return")
    cash = 1.0 - wp.sum(axis=1)
    if (cash > 1e-9).any():
        wp = wp.assign(**{CASH: cash.clip(lower=0.0)})
        wb = wb.assign(**{CASH: 0.0})
        ret = ret.assign(**{CASH: 0.0})
    uncovered = [d for d in days if d < versions[0][0]]
    if uncovered:
        raise AttributionError(f"no classification is in force on {uncovered[0].date()}: the first one starts on {versions[0][0].date()}")
    effects = {"allocation": [], "selection": [], "interaction": []}
    sec_last: dict = {}
    rp_parts, rb_parts = [], []
    unclassified = set()
    for k, (eff, mapping) in enumerate(versions):
        nxt = versions[k + 1][0] if k + 1 < len(versions) else pd.Timestamp.max
        seg = [d for d in days if eff <= d < nxt]
        if not seg:
            continue
        sec = {c: mapping.get(str(c), UNCLASSIFIED) for c in ret.columns if c != CASH}
        unclassified |= {c for c, s in sec.items() if s == UNCLASSIFIED}
        if CASH in ret.columns:
            sec[CASH] = "Cash"
        sec_last = dict(sec)
        res = brinson_fachler(wp.loc[seg], wb.loc[seg], ret.loc[seg], sec)
        for name in effects:
            effects[name].append(getattr(res, name))
        rp_parts.append(res.portfolio_return)
        rb_parts.append(res.benchmark_return)
    if unclassified:
        warnings.append(f"no group is given for {', '.join(sorted(map(str, unclassified)))}: they are attributed as '{UNCLASSIFIED}'")
    if len(versions) > 1:
        warnings.append(f"the classification changes {len(versions) - 1} time(s); each day uses the one in force that day")
    eff = {n: pd.concat(parts).fillna(0.0).sort_index() for n, parts in effects.items()}
    groups = sorted(set().union(*[set(e.columns) for e in eff.values()]))
    if len(groups) < 2:
        raise AttributionError("Brinson attribution needs at least two groups")
    for n in eff:
        eff[n] = eff[n].reindex(columns=groups).fillna(0.0)
    rp, rb = pd.concat(rp_parts).sort_index(), pd.concat(rb_parts).sort_index()
    daily_total = sum(e.sum(axis=1) for e in eff.values())
    identity_error = float((daily_total - (rp - rb)).abs().max())
    cost_diff = (portfolio_cost - benchmark_cost).reindex(days)
    net_active = (portfolio_net - benchmark_net).reindex(days)
    residual = net_active - (daily_total - cost_diff)
    cols = {f"allocation:{g}": eff["allocation"][g] for g in groups} | {f"selection:{g}": eff["selection"][g] for g in groups} | {f"interaction:{g}": eff["interaction"][g] for g in groups}
    frame = pd.DataFrame(cols)
    frame["costs"] = -cost_diff
    frame["residual"] = residual
    linked = link_effects(frame, portfolio_net.reindex(days), benchmark_net.reindex(days))
    cum_net_active = float((1 + portfolio_net.reindex(days)).prod() - (1 + benchmark_net.reindex(days)).prod())
    by_group = {g: {"allocation": float(linked[f"allocation:{g}"]), "selection": float(linked[f"selection:{g}"]), "interaction": float(linked[f"interaction:{g}"]),
                    "average_portfolio_weight": float(wp[[c for c in wp.columns if sec_last.get(c, UNCLASSIFIED) == g]].sum(axis=1).mean()),
                    "average_benchmark_weight": float(wb[[c for c in wb.columns if sec_last.get(c, UNCLASSIFIED) == g]].sum(axis=1).mean())} for g in groups}
    totals = {"allocation": float(sum(linked[f"allocation:{g}"] for g in groups)), "selection": float(sum(linked[f"selection:{g}"] for g in groups)),
              "interaction": float(sum(linked[f"interaction:{g}"] for g in groups)), "costs": float(linked["costs"]), "residual": float(linked["residual"])}
    gap = float(sum(totals.values()) - cum_net_active)
    # the cumulative path of each effect, linked with the same Carino factors on the prefix ending at each date, sampled so the page stays light
    sample = list(range(0, len(days), max(1, len(days) // 120)))
    if sample[-1] != len(days) - 1:
        sample.append(len(days) - 1)
    path = {n: [] for n in ("allocation", "selection", "interaction", "costs", "residual")}
    for j in sample:
        pre = frame.iloc[:j + 1]
        l = link_effects(pre, portfolio_net.reindex(days).iloc[:j + 1], benchmark_net.reindex(days).iloc[:j + 1])
        path["allocation"].append(float(sum(l[f"allocation:{g}"] for g in groups)))
        path["selection"].append(float(sum(l[f"selection:{g}"] for g in groups)))
        path["interaction"].append(float(sum(l[f"interaction:{g}"] for g in groups)))
        path["costs"].append(float(l["costs"]))
        path["residual"].append(float(l["residual"]))
    return {"convention": "Brinson-Fachler (allocation = (Wp-Wb)(Rb_group-Rb), selection = Wb(Rp_group-Rb_group), interaction = (Wp-Wb)(Rp_group-Rb_group)), daily, Carino-linked on net returns",
            "groups": groups, "by_group": {g: {k: v for k, v in d.items() if v is not None} for g, d in by_group.items()}, "totals": totals, "cumulative_net_active_return": cum_net_active,
            "dates": [str(days[j].date()) for j in sample], "path": path,
            "reconciliation": {"daily_identity_error": identity_error, "max_daily_residual": float(residual.abs().max()), "cumulative_gap": gap, "tolerance": TOLERANCE,
                               "status": "reconciled" if identity_error <= TOLERANCE and float(residual.abs().max()) <= TOLERANCE and abs(gap) <= TOLERANCE else "does_not_reconcile"},
            "cumulative_returns": {"portfolio_net": float((1 + portfolio_net.reindex(days)).prod() - 1), "benchmark_net": float((1 + benchmark_net.reindex(days)).prod() - 1)}, "warnings": warnings}


# ------------------------------------------------------------------------------------------------------------------------ currency
def currency_decomposition(weights: pd.DataFrame, local_returns: pd.DataFrame, fx_returns: pd.DataFrame) -> pd.DataFrame:
    """A base-currency return ``(1 + local)(1 + fx) - 1 = local + fx + local*fx``, weighted: the local, currency and cross contributions per day, which add up to the base-currency portfolio return exactly.
    ``fx_returns`` is the change in the base-currency value of one unit of each asset's currency (zero for assets already in the base currency)."""
    w = weights.reindex(index=local_returns.index, columns=local_returns.columns)
    fx = fx_returns.reindex(index=local_returns.index, columns=local_returns.columns)
    if w.isna().any().any() or fx.isna().any().any():
        raise AttributionError("weights and FX returns must cover every asset and day")
    out = pd.DataFrame({"local": (w * local_returns).sum(axis=1), "currency": (w * fx).sum(axis=1), "cross": (w * local_returns * fx).sum(axis=1)})
    out["base"] = out.sum(axis=1)
    return out


# ------------------------------------------------------------------------------------------------------------------------ for a Portfolio Lab result
def classification_from(tickers: list[str], sectors: dict | list | None) -> tuple[dict | list | None, str | None]:
    """The classification to use and, when Brinson cannot be made, the reason. ``sectors`` is a mapping or a list of ``(effective date, mapping)``."""
    if not sectors:
        return None, "no classification was given: Brinson attribution needs every asset assigned to a group"
    versions = _versions(sectors)
    covered = max(sum(1 for t in tickers if t in m) for _, m in versions)
    if covered == 0:
        return None, "none of the chosen tickers has a group"
    groups = {m.get(t, UNCLASSIFIED) for _, m in versions for t in tickers}
    if len(groups) < 2:
        return None, "Brinson attribution needs at least two groups; every asset is in one"
    return sectors, None


def attribute_lab_result(res: dict, data: pd.DataFrame, methods: list[str] | None = None, kind: str = "market", n_factors: int = 2, sectors=None, benchmark: str = "equal_weight") -> dict:
    """Factor attribution of every method of a ``portfolio.lab.compare`` result and Brinson attribution against ``benchmark`` (one of its methods) where a classification exists. ``data`` is the
    return frame the comparison used (rows with a full set of returns)."""
    cfg = res["config"]
    data = data.dropna(how="any")
    keys = methods or list(res["weights"])
    out = {"methods": {}, "benchmark": benchmark, "brinson_available": False, "brinson_reason": None, "kind": kind,
           "methodology": ["Exposures of a rebalance are estimated on the window that ends the day before it and held until the next rebalance (point in time).",
                           "Contributions are ex post: ex-ante exposures times the realised factor returns, plus the asset-specific residual, less costs; financing is not modelled by the Lab (long only, fully invested).",
                           "The reconciliation residual is the authoritative net return minus the sum of the lines; it is shown, never spread over the factors.",
                           "Factor lines are wealth-weighted (each day's contribution scaled by the value at the start of the day); Brinson lines are Carino-linked on net returns. The two are not mixed."]}
    equity = res["equity"]
    net = equity.pct_change()
    net.iloc[0] = equity.iloc[0] - 1.0
    for key in keys:
        t = res["weights"][key]
        try:
            out["methods"][key] = {"factor": factor_attribution(data, t, net[key], cfg["lookback"], cfg["step"], cfg["cost_bps"], kind, n_factors)}
        except (AttributionError, fm.FactorModelError) as error:
            out["methods"][key] = {"error": str(error)}
    chosen, reason = classification_from(list(data.columns), sectors)
    if chosen is not None and benchmark not in res["weights"]:
        chosen, reason = None, f"the benchmark method {benchmark!r} was not run"
    if chosen is None:
        out["brinson_reason"] = reason
        return out
    bench = replay(data, res["weights"][benchmark], cfg["step"], cfg["cost_bps"])
    out["brinson_available"] = True
    for key in keys:
        if key == benchmark:
            continue
        try:
            p = replay(data, res["weights"][key], cfg["step"], cfg["cost_bps"])
            days = p["gross"].index
            out["methods"][key]["brinson"] = brinson(data, p["earning"], bench["earning"].loc[days], chosen, p["cost"], bench["cost"].reindex(days), net[key].reindex(days), net[benchmark].reindex(days))
        except AttributionError as error:
            out["methods"][key]["brinson"] = {"error": str(error)}
    return out


def compact(attr: dict, points: int = 160) -> dict:
    """The attribution as the page and a saved study need it: the daily lines are replaced by their totals and a sampled cumulative path (the reconciliation was computed on every day before sampling)."""
    out = {k: v for k, v in attr.items() if k != "methods"}
    out["methods"] = {}
    for key, m in attr["methods"].items():
        mm = dict(m)
        f = mm.get("factor")
        if f:
            n = len(f["days"])
            idx = list(range(0, n, max(1, n // points)))
            if idx[-1] != n - 1:
                idx.append(n - 1)
            mm["factor"] = {**{k: v for k, v in f.items() if k not in ("daily", "cumulative", "days")}, "days": [f["days"][i] for i in idx], "cumulative": {c: [v[i] for i in idx] for c, v in f["cumulative"].items()}}
        out["methods"][key] = mm
    return out
