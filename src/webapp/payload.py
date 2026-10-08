"""A ``PipelineResult`` as plain JSON for the charts: series, monthly grids, tables, flags and a few headline statistics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

ANN = 252.0


def clean(value):
    """Recursively turn numpy / pandas scalars into JSON types; NaN and infinity become null."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return str(value.date())
    if isinstance(value, pd.Interval):
        return f"{value.left:.2f} to {value.right:.2f}"
    if value is pd.NaT or value is pd.NA:
        return None
    return value


def _series(values: pd.Series, digits: int = 5) -> list:
    return [None if not math.isfinite(v) else round(float(v), digits) for v in values.to_numpy(dtype=float)]


def records(table, limit: int = 60) -> list[dict]:
    """A table (frame or series) as a list of row dictionaries, index first."""
    frame = table if isinstance(table, pd.DataFrame) else table.to_frame()
    frame = frame.head(limit).reset_index(drop=isinstance(frame.index, pd.RangeIndex))
    frame.columns = [str(c) for c in frame.columns]
    return clean(frame.to_dict(orient="records"))


def _monthly(net: pd.Series) -> list[dict]:
    grouped = (1.0 + net).groupby([net.index.year, net.index.month]).prod() - 1.0
    return [{"year": int(y), "month": int(m), "value": round(float(v), 5)} for (y, m), v in grouped.items()]


def _annual(net: pd.Series) -> list[dict]:
    out = []
    for year, r in net.groupby(net.index.year):
        out.append({"year": int(year), "value": round(float((1.0 + r).prod() - 1.0), 5), "days": int(len(r)), "partial": bool(len(r) < 240)})
    return out


def _headline(net: pd.Series, gross: pd.Series) -> dict:
    """Headline numbers over the traded window (the same arithmetic the page uses for its tiles)."""
    sd = net.std(ddof=1)
    equity = (1.0 + net).cumprod()
    gsd = gross.std(ddof=1)
    return {"sharpe": ANN ** 0.5 * net.mean() / sd if sd > 0 else float("nan"), "gross_sharpe": ANN ** 0.5 * gross.mean() / gsd if gsd > 0 else float("nan"),
            "cagr": float(equity.iloc[-1] ** (ANN / len(net)) - 1.0), "ann_vol": float(sd * ANN ** 0.5), "max_drawdown": float((equity / equity.cummax() - 1.0).min()),
            "n_days": int(len(net)), "start": str(net.index[0].date())}


def _stats(net: pd.Series) -> dict:
    daily = net.dropna()
    monthly = (1.0 + daily).groupby([daily.index.year, daily.index.month]).prod() - 1.0
    return {"hit_rate_daily": float((daily > 0).mean()), "hit_rate_monthly": float((monthly > 0).mean()), "best_day": float(daily.max()), "worst_day": float(daily.min()),
            "best_month": float(monthly.max()), "worst_month": float(monthly.min()), "var_95_daily": float(-daily.quantile(0.05)), "skew": float(daily.skew()),
            "kurtosis": float(daily.kurt()), "years": float(len(daily) / ANN)}


def _longest_run(values: np.ndarray, positive: bool) -> int:
    """The longest run of strictly positive (or strictly negative) entries; a zero ends a run."""
    best = current = 0
    for v in values:
        if (v > 0) if positive else (v < 0):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _profit_factor(pnl: pd.Series) -> float | None:
    """Total won over total lost; ``None`` when nothing was lost (the ratio is then unbounded)."""
    lost = -float(pnl[pnl < 0].sum())
    return float(pnl[pnl > 0].sum()) / lost if lost > 0 else None


def _extreme(pnl: pd.Series, ret: pd.Series, best: bool, label=str) -> dict | None:
    if pnl.empty:
        return None
    key = pnl.idxmax() if best else pnl.idxmin()
    return {"period": label(key), "pnl": float(pnl.loc[key]), "ret": float(ret.loc[key])}


def _drawdown_summary(value: pd.Series) -> dict:
    """The deepest fall from a peak in dollars and percent, when it began and bottomed, when (if ever) the peak was regained, and the longest time spent below a peak."""
    peak = value.cummax()
    fall = value - peak
    if float(fall.min()) >= 0.0:
        return {"max_dollars": 0.0, "max_pct": 0.0, "peak_date": None, "trough_date": None, "recovered_on": None, "longest_calendar_days": 0}
    trough = fall.idxmin()
    peak_date = value.loc[:trough].idxmax()
    after = value.loc[trough:].iloc[1:]
    regained = after[after >= value.loc[peak_date]]
    below = (value < peak - 1e-9).to_numpy()
    longest, run_start = 0, None
    dates = value.index
    for i, flag in enumerate(below):
        if flag and run_start is None:
            run_start = i - 1 if i > 0 else 0                 # the day of the peak that was fallen from
        if (not flag or i == len(below) - 1) and run_start is not None:
            longest = max(longest, int((dates[i] - dates[run_start]).days))
            run_start = None
    return {"max_dollars": float(fall.min()), "max_pct": float((value / peak - 1.0).min()), "peak_date": str(peak_date.date()), "trough_date": str(trough.date()),
            "recovered_on": str(regained.index[0].date()) if len(regained) else None, "longest_calendar_days": longest}


def earnings_section(net: pd.Series, gross: pd.Series, costs: pd.Series, capital: float, benchmarks: dict, weights: pd.DataFrame | None = None) -> dict:
    """What the strategy would have earned in dollars on ``capital``: the account value day by day (``value = capital x (1 + r)`` compounded), profit by year and month, the best and
    worst periods, how often it made money, profit factor, the deepest loss from a peak, and what it paid in trading costs."""
    value = capital * (1.0 + net).cumprod()
    before = value.shift(1).fillna(capital)
    pnl = value - before                                              # dollars made on each day
    paid = (before * costs.reindex(net.index).fillna(0.0))           # dollars paid in costs on each day
    years = len(net) / ANN
    total = float(value.iloc[-1] - capital)
    monthly_value = value.resample("ME").last()
    monthly_before = monthly_value.shift(1).fillna(capital)
    monthly_pnl = monthly_value - monthly_before
    monthly_ret = monthly_value / monthly_before - 1.0
    yearly_value = value.resample("YE").last()
    yearly_before = yearly_value.shift(1).fillna(capital)
    yearly_pnl = yearly_value - yearly_before
    yearly_ret = yearly_value / yearly_before - 1.0
    days_in_year = net.groupby(net.index.year).size()
    full = pd.concat([pd.Series([capital], index=[value.index[0] - pd.Timedelta(days=1)]), value])          # the account before its first day counts as a peak
    annual = []
    for stamp, year_pnl in yearly_pnl.items():
        base = float(yearly_before.loc[stamp])
        seq = pd.Series([base] + value[value.index.year == stamp.year].tolist())
        annual.append({"year": int(stamp.year), "start_value": base, "end_value": float(yearly_value.loc[stamp]), "pnl": float(year_pnl), "ret": float(yearly_ret.loc[stamp]),
                       "max_drawdown": float((seq / seq.cummax() - 1.0).min()), "days": int(days_in_year.get(stamp.year, 0)), "partial": bool(days_in_year.get(stamp.year, 0) < 240)})
    bench = {}
    for name, series in benchmarks.items():
        grown = capital * (1.0 + series.reindex(net.index).fillna(0.0)).cumprod()
        bench[name] = {"end_value": float(grown.iloc[-1]), "pnl": float(grown.iloc[-1] - capital), "ret": float(grown.iloc[-1] / capital - 1.0)}
    month_label = lambda key: f"{key.year}-{key.month:02d}"                                                  # noqa: E731
    exposure = None
    if weights is not None and len(weights):
        gross_exposure = weights.abs().sum(axis=1).reindex(net.index).fillna(0.0)
        exposure = {"average": float(gross_exposure.mean()), "peak": float(gross_exposure.max()), "share_of_days_above_one": float((gross_exposure > 1.0 + 1e-6).mean())}
    return clean({
        "exposure": exposure,
        "capital": float(capital), "end_value": float(value.iloc[-1]), "net_profit": total, "total_return": float(value.iloc[-1] / capital - 1.0), "years": float(years),
        "profit_per_year": total / years if years > 0 else None, "profit_per_month": total / (years * 12.0) if years > 0 else None, "profit_per_day": float(pnl.mean()),
        "gross_profit": float(pnl.sum() + paid.sum()), "costs_paid": float(paid.sum()), "costs_share_of_gross": float(paid.sum() / (pnl.sum() + paid.sum())) if (pnl.sum() + paid.sum()) > 0 else None,
        "profit_factor_daily": _profit_factor(pnl), "profit_factor_monthly": _profit_factor(monthly_pnl),
        "winning_days": int((pnl > 0).sum()), "losing_days": int((pnl < 0).sum()), "flat_days": int((pnl == 0).sum()),
        "average_winning_day": float(pnl[pnl > 0].mean()) if (pnl > 0).any() else None, "average_losing_day": float(pnl[pnl < 0].mean()) if (pnl < 0).any() else None,
        "winning_months": int((monthly_pnl > 0).sum()), "losing_months": int((monthly_pnl < 0).sum()), "months": int(len(monthly_pnl)),
        "average_winning_month": float(monthly_pnl[monthly_pnl > 0].mean()) if (monthly_pnl > 0).any() else None,
        "average_losing_month": float(monthly_pnl[monthly_pnl < 0].mean()) if (monthly_pnl < 0).any() else None,
        "winning_years": int((yearly_pnl > 0).sum()), "losing_years": int((yearly_pnl < 0).sum()), "full_years": int((days_in_year >= 240).sum()),
        "best_day": _extreme(pnl, net, True, lambda k: str(k.date())), "worst_day": _extreme(pnl, net, False, lambda k: str(k.date())),
        "best_month": _extreme(monthly_pnl, monthly_ret, True, month_label), "worst_month": _extreme(monthly_pnl, monthly_ret, False, month_label),
        "best_year": _extreme(yearly_pnl, yearly_ret, True, lambda k: str(k.year)), "worst_year": _extreme(yearly_pnl, yearly_ret, False, lambda k: str(k.year)),
        "longest_winning_streak_days": _longest_run(pnl.to_numpy(), True), "longest_losing_streak_days": _longest_run(pnl.to_numpy(), False),
        "longest_winning_streak_months": _longest_run(monthly_pnl.to_numpy(), True), "longest_losing_streak_months": _longest_run(monthly_pnl.to_numpy(), False),
        "drawdown": _drawdown_summary(full),
        "annual": annual, "monthly": [{"year": int(k.year), "month": int(k.month), "pnl": float(v), "ret": float(monthly_ret.loc[k]), "end_value": float(monthly_value.loc[k])}
                                      for k, v in monthly_pnl.items()],
        "benchmarks": bench,
    })


def _episodes(sign: np.ndarray) -> list[tuple[int, int]]:
    """``(first, after_last)`` positions of each run of the same non-zero sign; ``after_last`` is the first day the position is gone or reversed (``len`` when still open)."""
    n = len(sign)
    if n == 0:
        return []
    cuts = np.flatnonzero(sign[1:] != sign[:-1]) + 1
    starts, ends = np.r_[0, cuts], np.r_[cuts, n]
    return [(int(a), int(b)) for a, b in zip(starts, ends) if sign[a] != 0]


def trades_section(weights: pd.DataFrame, traded: pd.DataFrame, returns: pd.DataFrame, prices: pd.DataFrame, value: pd.Series, capital: float, listed: int = 200) -> dict:
    """The buys and sells behind the returns. A *round trip* is one stretch in which an asset is held on the same side (long or short): it opens when the position appears and closes when
    it goes to zero or flips. Its profit is the sum, over the days it was held, of the position times that day's return (before trading costs), in dollars on the account value of the
    day. ``orders`` are the individual changes of position the rebalancing made."""
    eps = 1e-4
    w = weights.fillna(0.0)
    r = returns.reindex(index=w.index, columns=w.columns).fillna(0.0)
    contribution = (w.shift(1).fillna(0.0) * r)                         # fraction of the account earned by each asset on each day
    before = value.reindex(w.index).shift(1).fillna(capital)
    dollars = contribution.mul(before, axis=0)
    px = prices.reindex(index=w.index, columns=w.columns)
    rows = []
    for asset in w.columns:
        series = w[asset].to_numpy(dtype=float)
        sign = np.where(np.abs(series) > eps, np.sign(series), 0.0).astype(int)
        cash = dollars[asset].to_numpy(dtype=float)
        price = px[asset].to_numpy(dtype=float)
        for first, after in _episodes(sign):
            is_open = after >= len(sign)
            last = len(sign) - 1 if is_open else after
            entry_price, exit_price = price[first], price[last]
            move = sign[first] * (exit_price / entry_price - 1.0) if np.isfinite(entry_price) and np.isfinite(exit_price) and entry_price > 0 else None
            held_days = last - first
            rows.append({"ticker": str(asset), "side": "long" if sign[first] > 0 else "short", "open": bool(is_open), "entered": str(w.index[first].date()), "exited": None if is_open else str(w.index[last].date()),
                         "days": int(held_days), "entry_price": float(entry_price) if np.isfinite(entry_price) else None, "exit_price": float(exit_price) if np.isfinite(exit_price) else None,
                         "size": float(abs(series[first])), "entry_value": float(abs(series[first]) * value.reindex(w.index).iloc[first]) if first < len(value) else None,
                         "move": None if move is None else float(move), "pnl": float(cash[first + 1:last + 1].sum()) if held_days > 0 else 0.0})
    trips = pd.DataFrame(rows)
    closed = trips[~trips["open"]] if len(trips) else trips
    pnl = closed["pnl"] if len(closed) else pd.Series(dtype=float)
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    gross_exposure = w.abs().sum(axis=1)
    orders = traded.reindex(index=w.index, columns=w.columns).fillna(0.0)
    account = value.reindex(w.index)
    rows_idx, cols_idx = np.nonzero(np.abs(orders.to_numpy(dtype=float)) >= 0.0025)             # ignore changes below a quarter of a percent of the account: drift, not decisions
    order_rows = []
    for i, j in list(zip(rows_idx, cols_idx))[-100:][::-1]:
        x, price = float(orders.iat[i, j]), px.iat[i, j]
        order_rows.append({"date": str(w.index[i].date()), "ticker": str(w.columns[j]), "side": "buy" if x > 0 else "sell", "size": abs(x), "amount": abs(x) * float(account.iloc[i]),
                           "price": float(price) if np.isfinite(price) else None})
    stats = {
        "round_trips": int(len(closed)), "open_positions": int(len(trips) - len(closed)), "winners": int(len(wins)), "losers": int(len(losses)),
        "win_rate": float(len(wins) / len(closed)) if len(closed) else None,
        "average_win": float(wins.mean()) if len(wins) else None, "average_loss": float(losses.mean()) if len(losses) else None,
        "payoff_ratio": float(wins.mean() / -losses.mean()) if len(wins) and len(losses) else None, "profit_factor": _profit_factor(pnl) if len(closed) else None,
        "expectancy": float(pnl.mean()) if len(closed) else None, "average_days_held": float(closed["days"].mean()) if len(closed) else None,
        "longest_days_held": int(closed["days"].max()) if len(closed) else None, "shortest_days_held": int(closed["days"].min()) if len(closed) else None,
        "best": None if not len(closed) else {k: closed.loc[closed["pnl"].idxmax(), k] for k in ("ticker", "side", "entered", "exited", "pnl")},
        "worst": None if not len(closed) else {k: closed.loc[closed["pnl"].idxmin(), k] for k in ("ticker", "side", "entered", "exited", "pnl")},
        "long_trips": int((closed["side"] == "long").sum()) if len(closed) else 0, "short_trips": int((closed["side"] == "short").sum()) if len(closed) else 0,
        "long_pnl": float(closed.loc[closed["side"] == "long", "pnl"].sum()) if len(closed) else 0.0, "short_pnl": float(closed.loc[closed["side"] == "short", "pnl"].sum()) if len(closed) else 0.0,
        "days_in_market": float((gross_exposure > eps).mean()), "average_exposure": float(gross_exposure.mean()), "orders": int((orders.abs() >= 0.0025).to_numpy().sum()),
        "buys": int((orders >= 0.0025).to_numpy().sum()), "sells": int((orders <= -0.0025).to_numpy().sum()),
    }
    shown = pd.concat([trips[trips["open"]], closed.sort_values("exited", ascending=False)]).head(listed) if len(trips) else trips
    return clean({"stats": stats, "round_trips": shown.to_dict("records") if len(shown) else [], "listed": int(len(shown)), "total": int(len(trips)), "orders": order_rows})


def _with_entry_day(result, start) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The held weights and the trades from the day BEFORE ``start``: the first day's return is earned by the position taken at the previous close, which belongs in the log."""
    i = result.weights.index.get_loc(start)
    first = max(int(i) - 1, 0)
    return result.weights.iloc[first:], result.trades.iloc[first:]


def result_payload(result, bundle, flags: list[dict], universe: dict, spec: dict, yaml_text: str, seconds: float, capital: float = 100_000.0) -> dict:
    net = result.net_returns.loc[result.start:].dropna()
    active = net[net.abs() > 0]
    if len(active):                                   # a book that waits for enough assets earns exactly zero until it starts: do not chart or score the wait
        net = net.loc[active.index[0]:]
    start = net.index[0]
    gross = result.gross_returns.reindex(net.index).fillna(0.0)
    equity = (1.0 + net).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    rolling = net.rolling(int(ANN)).apply(lambda x: ANN ** 0.5 * x.mean() / x.std(ddof=1) if x.std(ddof=1) > 0 else np.nan, raw=True)
    benches, bench_returns = {}, {}
    for name, r in result.benchmarks.items():
        b = r.reindex(net.index)
        if b.notna().sum() > 20:
            benches[name] = _series((1.0 + b.fillna(0.0)).cumprod(), 4)
            bench_returns[name] = b
    weights = result.weights.loc[start:]
    held = weights.shift(1).reindex(net.index)
    gross_exposure = weights.abs().sum(axis=1).resample("W").last().dropna()
    net_exposure = weights.sum(axis=1).reindex(gross_exposure.index)
    average = pd.DataFrame({"average_weight": weights.mean(), "average_abs_weight": weights.abs().mean(), "latest_weight": weights.iloc[-1]})
    tables = {}
    for key, table in result.tables.items():
        if isinstance(table, (pd.DataFrame, pd.Series)) and len(table):
            tables[key] = records(table)
    return clean({
        "name": result.spec.name, "universe": universe, "spec": spec, "yaml": yaml_text, "seconds": round(seconds, 1),
        "metrics": {**{k: v for k, v in result.metrics.items() if not isinstance(v, str) or k in ("start", "end", "capacity_note")}, **_headline(net, gross)},
        "validation": {"deflated_sharpe_probability": result.validation.get("deflated_sharpe_probability"), "n_trials": result.validation.get("n_trials"),
                       "causality": result.validation.get("causality")},
        "flags": flags, "stats": _stats(net),
        "dates": [str(d.date()) for d in net.index], "equity": _series(equity, 4), "benchmarks": benches, "drawdown": _series(drawdown, 4), "rolling_sharpe": _series(rolling, 3),
        "exposure": {"dates": [str(d.date()) for d in gross_exposure.index], "gross": _series(gross_exposure, 3), "net": _series(net_exposure, 3)},
        "monthly": _monthly(net), "annual": _annual(net),
        "weights": records(average.sort_values("average_abs_weight", ascending=False), 40),
        "contribution": records(result.tables["attribution_by_asset"], 40) if "attribution_by_asset" in result.tables else [],
        "held_days": int((held.abs().sum(axis=1) > 0).sum()),
        "tables": tables,
        "capital": float(capital), "earnings": earnings_section(net, gross, result.costs, capital, bench_returns, weights),
        "trades": trades_section(*_with_entry_day(result, start), bundle.returns, bundle.prices, capital * equity, capital),
    })
