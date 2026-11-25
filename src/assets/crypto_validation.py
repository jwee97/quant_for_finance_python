"""Historical crypto strategy validation on a 24/7 calendar, observed data only.

Every return here is a calendar-day return and every annualised figure uses 365 (never 252): the platform's equity-style pipeline drops weekends and annualises by 252, which is wrong for a market
that never closes. Positions are decided at the close of day ``t`` from data up to ``t`` and earn the return of day ``t+1``. A coin whose history is simulated is refused: this module backtests
observations or says it cannot.

Families (``FAMILIES``)
  directional      ``tsmom`` (sign of the past return), ``ma_cross`` (moving-average crossover), ``donchian`` (channel breakout)
  relative value   ``pair_spread`` (BTC/ETH log ratio against a rolling hedge ratio, mean reverting, with an Engle-Granger cointegration test reported beside it), ``xs_momentum`` (needs three or
                   more observed coins)
  derivatives      ``funding_carry`` (long spot, short perpetual when trailing funding is high), ``basis_convergence`` (carry when the perpetual is rich to its own history),
                   ``funding_dispersion`` (short the high-funding venue's perpetual, long the low one; needs two observed venues)
  risk-managed     ``vol_target`` wraps any of them: exposure scaled by target volatility over trailing realised volatility, capped
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..derivatives.validation import bootstrap, walk_forward
from .crypto_research import PERIODS, calendar_audit, is_observed, unavailable

FAMILIES = {
    "tsmom": {"family": "directional", "label": "Time-series momentum", "needs": "spot", "params": {"lookback": 90}, "grid": {"lookback": [30, 60, 90, 120, 180]}},
    "ma_cross": {"family": "directional", "label": "Moving-average crossover", "needs": "spot", "params": {"fast": 20, "slow": 100}, "grid": {"fast": [10, 20, 40], "slow": [50, 100, 200]}},
    "donchian": {"family": "directional", "label": "Channel breakout", "needs": "spot", "params": {"window": 55}, "grid": {"window": [20, 40, 55, 90]}},
    "pair_spread": {"family": "relative value", "label": "BTC/ETH spread reversion", "needs": "two spots", "params": {"window": 60, "entry": 1.5, "exit": 0.25}, "grid": {"window": [30, 60, 90], "entry": [1.0, 1.5, 2.0]}},
    "xs_momentum": {"family": "relative value", "label": "Cross-sectional momentum", "needs": "three or more spots", "params": {"lookback": 30}, "grid": {"lookback": [14, 30, 60, 90]}},
    "funding_carry": {"family": "derivatives", "label": "Funding carry (long spot, short perpetual)", "needs": "spot, perpetual, funding", "params": {"window": 30, "threshold": 0.05}, "grid": {"window": [14, 30, 60], "threshold": [0.0, 0.05, 0.10]}},
    "basis_convergence": {"family": "derivatives", "label": "Basis convergence", "needs": "spot, perpetual", "params": {"window": 60, "entry_z": 1.0}, "grid": {"window": [30, 60, 90], "entry_z": [0.5, 1.0, 1.5]}},
    "funding_dispersion": {"family": "derivatives", "label": "Funding dispersion between venues", "needs": "funding on two venues", "params": {"window": 14, "min_gap": 0.03}, "grid": {"window": [7, 14, 30], "min_gap": [0.02, 0.03, 0.05]}},
}
CARRY = ("funding_carry", "basis_convergence", "funding_dispersion")


def _calendar(s: pd.Series | None, index: pd.DatetimeIndex) -> pd.Series | None:
    if s is None:
        return None
    s = pd.Series(s).astype(float)
    s.index = pd.DatetimeIndex(s.index).normalize()
    s = s[~s.index.duplicated(keep="last")]
    return s.reindex(index).ffill(limit=2)


def build_panel(series: dict, observed_only: bool = True) -> dict:
    """Align every observed coin on the full daily calendar between their common start and the newest mark (gaps up to two days are carried forward, longer ones stay missing)."""
    use, refused = {}, {}
    for sym, s in series.items():
        if observed_only and not is_observed(s.source):
            refused[sym] = f"{s.source} data is refused in observed-only mode"
        else:
            use[sym] = s
    if not use:
        return {"state": "unavailable", **unavailable("no observed coin to test", ["download a coin's history or use BTC/ETH from the platform"]), "refused": refused}
    start = max(pd.Timestamp(s.spot.dropna().index[0]).normalize() for s in use.values())
    end = min(pd.Timestamp(s.spot.dropna().index[-1]).normalize() for s in use.values())
    idx = pd.date_range(start, end, freq="D")
    spot = pd.DataFrame({k: _calendar(s.spot, idx) for k, s in use.items()})
    perp = pd.DataFrame({k: _calendar(s.perp, idx) for k, s in use.items() if s.perp is not None})
    fund = pd.DataFrame({k: _calendar(s.funding, idx) for k, s in use.items() if s.funding is not None})
    return {"state": "ok", "spot": spot, "perp": perp, "funding": fund, "series": use, "refused": refused, "calendar": calendar_audit(idx)}


# ------------------------------------------------------------------------------------------------------------------ signals
def _vol_scale(ret: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    return ret.rolling(window, min_periods=window).std(ddof=1) * math.sqrt(PERIODS)


def signal(name: str, p: dict, params: dict, venue_funding: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """``(weights, leg_returns_override, info)``: target weights decided at each close, and for carry strategies the daily return of the unit position (spot return plus funding minus perpetual return)."""
    spot, perp, fund = p["spot"], p["perp"], p["funding"]
    ret = spot.pct_change()
    info: dict = {}
    if name == "tsmom":
        sig = np.sign(spot.pct_change(int(params["lookback"])))
        w = sig.div(_vol_scale(ret).clip(lower=0.05)).div(sig.abs().sum(axis=1).clip(lower=1), axis=0)
        return w, None, info
    if name == "ma_cross":
        f, s_ = int(params["fast"]), int(params["slow"])
        if f >= s_:
            raise ValueError("fast must be shorter than slow")
        sig = np.sign(spot.rolling(f).mean() - spot.rolling(s_).mean())
        return sig.div(sig.abs().sum(axis=1).clip(lower=1), axis=0).where(spot.rolling(s_).mean().notna()), None, info
    if name == "donchian":
        n = int(params["window"])
        hi, lo = spot.rolling(n).max().shift(1), spot.rolling(n).min().shift(1)
        pos = pd.DataFrame(0.0, index=spot.index, columns=spot.columns)
        cur = pd.Series(0.0, index=spot.columns)
        for t in range(len(spot)):
            up, dn = spot.iloc[t] > hi.iloc[t], spot.iloc[t] < lo.iloc[t]
            cur[up.fillna(False)], cur[dn.fillna(False)] = 1.0, -1.0
            pos.iloc[t] = cur
        return pos.div(pos.abs().sum(axis=1).clip(lower=1), axis=0).where(hi.notna()), None, info
    if name == "pair_spread":
        if spot.shape[1] < 2 or not {"BTC", "ETH"} <= set(spot.columns):
            raise ValueError("the BTC/ETH spread needs observed BTC and ETH")
        lb, le = np.log(spot["BTC"]), np.log(spot["ETH"])
        w_ = int(params["window"])
        beta = le.rolling(w_).cov(lb) / lb.rolling(w_).var()
        spread = le - beta * lb
        z = (spread - spread.rolling(w_).mean()) / spread.rolling(w_).std(ddof=1)
        pos = pd.Series(0.0, index=spot.index)
        cur = 0.0
        for t, zt in enumerate(z):
            if np.isnan(zt):
                pos.iloc[t] = np.nan
                continue
            if cur == 0.0 and zt > float(params["entry"]):
                cur = -1.0                                                           # ETH rich to BTC: short the spread
            elif cur == 0.0 and zt < -float(params["entry"]):
                cur = 1.0
            elif cur != 0.0 and abs(zt) < float(params["exit"]):
                cur = 0.0
            pos.iloc[t] = cur
        hr = beta.clip(0.2, 3.0)
        w = pd.DataFrame({"ETH": pos / (1.0 + hr), "BTC": -pos * hr / (1.0 + hr)})
        info["cointegration"] = coint_test(le.dropna(), lb.dropna())
        return w.reindex(columns=spot.columns, fill_value=0.0), None, info
    if name == "xs_momentum":
        if spot.shape[1] < 3:
            raise ValueError("cross-sectional momentum needs three or more observed coins")
        rk = spot.pct_change(int(params["lookback"])).rank(axis=1, pct=True)
        w = (rk > 2 / 3).astype(float) - (rk <= 1 / 3).astype(float)
        w = w.where(rk.notna())
        return w.div(w.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0), None, info
    if name in ("funding_carry", "basis_convergence"):
        coins = [c for c in spot.columns if c in perp.columns and c in fund.columns]
        if not coins:
            raise ValueError(f"{name} needs spot, perpetual and funding on an observed coin")
        carry_ret = pd.DataFrame({c: fund[c] + spot[c].pct_change() - perp[c].pct_change() for c in coins})
        if name == "funding_carry":
            trail = fund[coins].rolling(int(params["window"])).mean() * PERIODS
            on = (trail > float(params["threshold"]))
        else:
            basis = perp[coins] / spot[coins] - 1.0
            z = (basis - basis.rolling(int(params["window"])).mean()) / basis.rolling(int(params["window"])).std(ddof=1)
            on = pd.DataFrame(False, index=z.index, columns=coins)
            cur = pd.Series(False, index=coins)
            for t in range(len(z)):
                cur = (cur | (z.iloc[t] > float(params["entry_z"]))) & ~(z.iloc[t] < 0)
                on.iloc[t] = cur
            on = on.where(z.notna())
        w = on.astype(float).where(on.notna()).div(len(coins))
        info["legs"] = 2
        return w, carry_ret, info
    if name == "funding_dispersion":
        vf = {k: v for k, v in (venue_funding or {}).items() if v is not None}
        if len(vf) < 2:
            raise ValueError("funding dispersion needs the funding of two observed venues")
        f = pd.DataFrame(vf).reindex(spot.index).ffill(limit=2)
        trail = f.rolling(int(params["window"])).mean() * PERIODS
        gap = trail.max(axis=1) - trail.min(axis=1)
        on = (gap > float(params["min_gap"])).where(gap.notna())
        cols = list(trail.columns)
        arr, farr = trail.to_numpy(float), f.to_numpy(float)
        earned = np.full(len(f), np.nan)
        for i in range(1, len(f)):
            row = arr[i - 1]                                                                  # the venues chosen at yesterday's close earn today's funding
            if np.isnan(row).any():
                continue
            earned[i] = farr[i, int(np.argmax(row))] - farr[i, int(np.argmin(row))]
        earned = pd.Series(earned, index=f.index)
        carry = pd.DataFrame({"spread": earned})
        info["legs"] = 2
        info["note"] = "the two perpetuals are assumed to trade at one price, so their price moves cancel; basis differences between venues are not modelled"
        return pd.DataFrame({"spread": on.astype(float).where(on.notna())}), carry, info
    raise ValueError(f"unknown strategy '{name}'")


def coint_test(a: pd.Series, b: pd.Series) -> dict:
    """Engle-Granger test of cointegration of two log prices over the whole sample. Reported beside the spread strategy; a p-value above 5% means the spread may not mean-revert at all."""
    try:
        from statsmodels.tsa.stattools import coint

        both = pd.concat([a, b], axis=1).dropna()
        stat, pvalue, _ = coint(both.iloc[:, 0], both.iloc[:, 1])
        return {"available": True, "statistic": float(stat), "p_value": float(pvalue), "cointegrated_at_5pct": bool(pvalue < 0.05), "n": int(len(both))}
    except Exception as exc:                                                                         # noqa: BLE001 - a missing optional dependency or a degenerate sample
        return {"available": False, "reason": str(exc)}


# ------------------------------------------------------------------------------------------------------------------ the run
def _metrics(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 10:
        return {"n_days": int(len(r))}
    eq = (1 + r).cumprod()
    sd = float(r.std(ddof=1))
    dn = r[r < 0].std(ddof=1)
    dd = eq / eq.cummax() - 1.0
    cagr = float(eq.iloc[-1] ** (PERIODS / len(r)) - 1.0)
    return {"n_days": int(len(r)), "ann_return": float(r.mean() * PERIODS), "cagr": cagr, "ann_vol": float(sd * math.sqrt(PERIODS)), "sharpe": float(r.mean() / sd * math.sqrt(PERIODS)) if sd > 0 else None,
            "sortino": float(r.mean() / dn * math.sqrt(PERIODS)) if dn and dn > 0 else None, "max_drawdown": float(dd.min()), "calmar": float(cagr / -dd.min()) if dd.min() < 0 else None,
            "hit_rate": float((r > 0).mean()), "total_return": float(eq.iloc[-1] - 1.0)}


def _run_returns(p: dict, name: str, params: dict, cost_bps: float, target_vol: float | None, max_leverage: float, allow_short: bool, venue_funding: dict | None = None) -> tuple[pd.Series, pd.Series, pd.Series, dict]:
    w, carry, info = signal(name, p, params, venue_funding)
    spot_ret = p["spot"].pct_change()
    if not allow_short:
        w = w.clip(lower=0.0)
    if carry is not None:
        gross_unit = carry.reindex(columns=w.columns) if set(w.columns) <= set(carry.columns) else carry
        pos = w.shift(1)
        gross = (pos * gross_unit).sum(axis=1, min_count=1)
        legs = 2.0
    else:
        pos = w.shift(1)
        long_r = spot_ret.reindex(columns=w.columns)
        short_r = -long_r + p["funding"].reindex(columns=w.columns) if len(p["funding"].columns) else -long_r        # a short is held through the perpetual: it earns funding
        gross = (pos.clip(lower=0) * long_r + pos.clip(upper=0).abs() * short_r).sum(axis=1, min_count=1)
        legs = 1.0
    scale = pd.Series(1.0, index=gross.index)
    if target_vol:
        rv = gross.rolling(30, min_periods=30).std(ddof=1) * math.sqrt(PERIODS)
        scale = (float(target_vol) / rv).clip(upper=max_leverage).shift(1)
        scale = scale.where(rv.shift(1).notna())
        gross = gross * scale
    else:
        gross_exposure = pos.abs().sum(axis=1)
        over = gross_exposure > max_leverage
        scale = pd.Series(np.where(over, max_leverage / gross_exposure.replace(0, np.nan), 1.0), index=gross.index)
        gross = gross * scale
    turn = (pos.fillna(0.0).mul(scale, axis=0).diff().abs().sum(axis=1)) * legs
    cost = turn * cost_bps / 1e4
    return (gross - cost).dropna(), gross.dropna(), turn, info


def run(series: dict, strategy: str, params: dict | None = None, fee_bps: float = 5.0, slippage_bps: float = 2.0, target_vol: float | None = None, max_leverage: float = 1.0,
        allow_short: bool = True, observed_only: bool = True, venue_funding: dict | None = None, walk: bool = True, train: int = 180, test: int = 60, bootstrap_n: int = 300) -> dict:
    """One strategy over the observed history: net and gross metrics (365), benchmarks, turnover and cost drag, the calendar audit, and (``walk``) a walk-forward that chooses among the strategy's
    parameter grid on each training window and trades the choice out of sample."""
    if strategy not in FAMILIES:
        raise ValueError(f"unknown strategy '{strategy}'; choose from {sorted(FAMILIES)}")
    p = build_panel(series, observed_only)
    if p["state"] != "ok":
        return p
    spec = FAMILIES[strategy]
    prm = {**spec["params"], **(params or {})}
    cost = float(fee_bps) + float(slippage_bps)
    try:
        net, gross, turn, info = _run_returns(p, strategy, prm, cost, target_vol, max_leverage, allow_short, venue_funding)
    except ValueError as exc:
        return unavailable(str(exc), [spec["needs"]])
    if len(net) < 60:
        return unavailable(f"only {len(net)} days of returns after the signal warm-up: too few to judge")
    spot_ret = p["spot"].pct_change().loc[net.index]
    benchmarks = {f"hold {c}": _metrics(spot_ret[c]) for c in spot_ret.columns}
    if spot_ret.shape[1] > 1:
        benchmarks["equal weight (daily rebalanced)"] = _metrics(spot_ret.mean(axis=1))
    benchmarks["cash"] = {"n_days": int(len(net)), "ann_return": 0.0, "sharpe": None}
    out = {"state": "ok", "strategy": strategy, "label": spec["label"], "family": spec["family"], "params": prm, "assets": list(p["spot"].columns), "start": str(net.index[0].date()), "end": str(net.index[-1].date()),
           "calendar": p["calendar"], "annualisation": PERIODS, "net": _metrics(net), "gross": _metrics(gross),
           "costs": {"fee_bps": float(fee_bps), "slippage_bps": float(slippage_bps), "turnover_per_year": float(turn.reindex(net.index).mean() * PERIODS), "cost_drag_per_year": float(turn.reindex(net.index).mean() * PERIODS * cost / 1e4)},
           "constraints": {"max_leverage": float(max_leverage), "target_vol": target_vol, "allow_short": bool(allow_short)}, "benchmarks": benchmarks, "info": info,
           "series": {"dates": [str(d.date()) for d in net.index], "equity": [round(float(x), 5) for x in (1 + net).cumprod()],
                      "drawdown": [round(float(x), 5) for x in ((1 + net).cumprod() / (1 + net).cumprod().cummax() - 1.0)]},
           "bootstrap": bootstrap(net, n=bootstrap_n, periods=PERIODS), "refused": p["refused"],
           "provenance": {c: p["series"][c].source for c in p["series"]},
           "assumptions": ["24/7 daily marks (every calendar day), annualised by 365; the signal at the close of a day trades the next day's return.",
                           f"Costs: {fee_bps:g} bps fee + {slippage_bps:g} bps slippage per one-way trade on the traded exposure" + (" (times two legs for a carry position)." if strategy in CARRY else "."),
                           "A short is held through the perpetual and earns the funding the series records (negative funding is paid); there is no borrowing cost or margin financing.",
                           "Carry returns are the linear approximation funding + spot return - perpetual return; the inverse (Deribit) perpetual's convexity is not modelled."]}
    if walk:
        grid = _grid(spec["grid"])
        rets, bad = {}, []
        for g in grid:
            try:
                rets[_key(g)] = _run_returns(p, strategy, {**prm, **g}, cost, target_vol, max_leverage, allow_short, venue_funding)[0]
            except ValueError as exc:
                bad.append(str(exc))
        try:
            out["walk_forward"] = {**walk_forward(rets, train=train, test=test, periods=PERIODS), "n_settings": len(rets), "n_trials_note": f"{len(rets)} settings were tried: the in-sample best is optimistic by selection"}
        except ValueError as exc:
            out["walk_forward"] = {"available": False, "reason": str(exc)}
    out["verdict"] = verdict(out)
    return out


def _grid(grid: dict) -> list[dict]:
    keys = list(grid)
    out = [{}]
    for k in keys:
        out = [{**o, k: v} for o in out for v in grid[k]]
    return [g for g in out if not ("fast" in g and "slow" in g and g["fast"] >= g["slow"])]


def _key(g: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in g.items())


def verdict(out: dict) -> dict:
    """Plain statements the numbers support, with no recommendation."""
    notes = []
    net, wf = out["net"], out.get("walk_forward") or {}
    if net.get("sharpe") is not None and out["gross"].get("sharpe") is not None and out["gross"]["sharpe"] - net["sharpe"] > 0.3:
        notes.append("costs remove a large part of the gross Sharpe ratio")
    if out["bootstrap"].get("available") and out["bootstrap"].get("sharpe_interval_contains_zero"):
        notes.append("the Sharpe ratio is not distinguishable from zero over this sample")
    if wf.get("oos_sharpe") is not None and wf.get("gap") is not None and wf["gap"] > 0.3:
        notes.append("out-of-sample Sharpe is much lower than the in-sample best: the parameters are fitted")
    best_bench = max((b.get("sharpe") for b in out["benchmarks"].values() if b.get("sharpe") is not None), default=None)
    if net.get("sharpe") is not None and best_bench is not None and net["sharpe"] < best_bench:
        notes.append("a passive benchmark had a higher Sharpe ratio")
    if (out.get("info") or {}).get("cointegration", {}).get("cointegrated_at_5pct") is False:
        notes.append("BTC and ETH are not cointegrated at 5% over the sample: the spread may not revert")
    return {"notes": notes, "reading": "These are historical observations of one path; they do not forecast." if notes else "No flag fired; this still describes one historical path."}
