"""The ML Strategy Explorer's endpoints: what a model predicted, what the decision policy did with it, what was executed, and what it earned.

Read-only: everything is read from the artifacts a ``decision_policy`` experiment saved (decisions, executions, round trips, closes, positions, returns, predictions). Nothing is recomputed, so what the
page shows is what the run recorded. Three layers stay separate in every response: the *prediction* (a score and its rank on a date), the *decision* (BUY / SELL / HOLD ... with its reason, taken from
that day's information only) and the *execution* (the trade the engine made ``lag`` days later, at that day's close, at the configured cost).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..assets import common
from .api import ApiError
from .payload import clean

LIMIT = 500


def _h():
    from . import ml_api

    return ml_api


def _iso(x) -> str | None:
    return None if x is None or pd.isna(x) else str(pd.Timestamp(x).date())


def _num(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else float(x)


class StrategyRoutes:
    def strategy_post_routes(self) -> dict:
        return {"/api/ml/strategy/controls": self.strategy_controls}

    def strategy_get_routes(self) -> dict:
        return {"/api/ml/strategy/runs": self.strategy_runs, "/api/ml/strategy": self.strategy, "/api/ml/strategy/decisions": self.strategy_decisions, "/api/ml/strategy/trades": self.strategy_trades, "/api/ml/strategy/replay": self.strategy_replay,
                "/api/ml/strategy/window": self.strategy_window, "/api/ml/strategy/how": self.strategy_how}

    # ------------------------------------------------------------------------------------------------------------ loading
    def _strategy_files(self, exp_id: str) -> dict:
        d = self.registry.experiment_dir(exp_id)
        if not (d / "decisions.parquet").exists() and not (d / "executions.parquet").exists():
            raise ApiError("this experiment was saved before trades were recorded for every position rule, so it has no trades to show; run it again (a re-backtest of its saved predictions is enough)")
        out = {}
        for name in ("decisions", "executions", "round_trips", "prices", "investable", "asset_returns", "positions", "returns", "predictions"):
            f = d / f"{name}.parquet"
            out[name] = pd.read_parquet(f) if f.exists() else None
        return out

    # ------------------------------------------------------------------------------------------------------------ the runs that have decisions
    def strategy_runs(self, q: dict) -> dict:
        """``GET /api/ml/strategy/runs``: the saved experiments that recorded their trades (every run since the ledger covers all position rules; one with the decision policy also has a decision log), newest first,
        with what to compare at a glance."""
        h = _h()
        h._only(q, set(), "/api/ml/strategy/runs")
        rows = []
        with h._user_errors():
            for rec in self.registry.list_experiments(300):
                folder = self.registry.experiment_dir(rec["id"])
                if not ((folder / "decisions.parquet").exists() or (folder / "executions.parquet").exists()):
                    continue
                try:
                    rep = self.registry.report(rec["id"])
                except Exception:                                                    # a run whose report cannot be read is left out of the list, not an error for the page
                    continue
                t, oos = rep.get("trading") or {}, (rep.get("metrics") or {}).get("oos") or {}
                a = t.get("analytics") or {}
                rows.append({"id": rec["id"], "name": rep.get("name"), "created": rec.get("created"), "model": (rep.get("config") or {}).get("model", {}).get("kind"),
                             "sharpe": _num(oos.get("sharpe")), "cagr": _num(oos.get("cagr")), "n_round_trips": a.get("n_round_trips"), "win_rate": _num(a.get("win_rate")), "ann_turnover": _num(a.get("ann_turnover"))})
        return clean(common.envelope("ml_strategy_runs", {"rows": rows[:100]}, {"registry": "platform"}, [], None, []))

    # ------------------------------------------------------------------------------------------------------------ overview + chart
    def strategy(self, q: dict) -> dict:
        """``GET /api/ml/strategy?id=<experiment>[&asset=<ticker>]``: the policy and analytics of the run, the assets it traded, and for one asset the close, the position held, every executed trade (a marker with
        its reason) and the equity, drawdown and exposure of the whole book against the benchmark."""
        h = _h()
        h._only(q, {"id", "asset"}, "/api/ml/strategy")
        with h._user_errors():
            exp_id = h._id(q)
            report = self.registry.report(exp_id)
            f = self._strategy_files(exp_id)
        trading = report.get("trading") or {}
        ex, trips, prices, pos, ret, dec = f["executions"], f["round_trips"], f["prices"], f["positions"], f["returns"], f["decisions"]
        if prices is not None and pos is not None and len(pos):
            prices = prices.loc[pos.index[0]:]                                              # the file holds the whole history (a control replays on it); the chart starts where the run did
            until = trading.get("scored_until")
            if until:                                                                       # ...and ends where the model's decisions end (plus the lag they need to execute): after that no book is held or traded, which is not the same as a book of cash
                stop = prices.index.searchsorted(pd.Timestamp(until)) + int(trading.get("lag_days") or 1) + 1
                prices = prices.iloc[:stop]
                pos = pos.loc[:prices.index[-1]].copy()
                pos.loc[pos.index > pd.Timestamp(until)] = np.nan                       # the rows after the last score are an empty book for want of scores: the last decision's book stays in force for the lag
                pos = pos.ffill().fillna(0.0)
        traded = (ex.groupby("asset").size().sort_values(ascending=False) if ex is not None and len(ex) else pd.Series(dtype=int))
        asset = str(q.get("asset") or (traded.index[0] if len(traded) else (prices.columns[0] if prices is not None else "")))
        if prices is not None and asset not in prices.columns:
            raise ApiError(f"asset must be one of {list(prices.columns)}")
        rows = []
        for a in (prices.columns if prices is not None else traded.index):
            t = trips[trips["asset"] == a] if trips is not None and len(trips) else pd.DataFrame()
            rows.append({"asset": a, "executions": int(traded.get(a, 0)), "round_trips": int(len(t)),
                         "net_contribution": _num(t["net_contribution"].sum()) if len(t) else 0.0, "win_rate": _num((t["position_return"] > 0).mean()) if len(t) else None})
        chart = {"asset": asset, "dates": [], "close": [], "weight": [], "markers": [], "summary": {}}
        mine = trips[trips["asset"] == asset] if trips is not None and len(trips) else pd.DataFrame()
        if prices is not None:
            px = prices[asset].dropna()
            w = pos[asset].reindex(px.index).fillna(0.0) if pos is not None and asset in pos.columns else pd.Series(0.0, index=px.index)
            chart.update({"dates": [_iso(d) for d in px.index], "close": [_num(x) for x in px.to_numpy()], "weight": [_num(x) for x in w.to_numpy()]})
            if ex is not None and len(ex):
                closed = mine[mine["status"] == "closed"] if len(mine) else mine
                for _, r in ex[ex["asset"] == asset].iterrows():
                    trip = closed[pd.to_datetime(closed["close_date"]) == pd.Timestamp(r["date"])] if len(closed) else closed       # a trade that ends a position carries that position's result
                    chart["markers"].append({"date": _iso(r["date"]), "price": _num(px.get(pd.Timestamp(r["date"]))), "pnl_return": _num(trip["position_return"].iloc[0]) if len(trip) else None,
                                             "pnl_contribution": _num(trip["net_contribution"].iloc[0]) if len(trip) else None, "kind": r["kind"], "side": r["side"], "weight_change": _num(r["trade_weight"]),
                                             "weight_after": _num(r["weight_after"]), "action": r.get("action"), "reason": r.get("reason"), "decision_date": _iso(r.get("decision_date")), "cost": _num(r["cost"]),
                                             "rank_pct": _num(r.get("rank_pct")), "score": _num(r.get("score"))})
        book = {"dates": [], "net": [], "benchmark": [], "drawdown": [], "gross": [], "cash": [], "positions": [], "long": [], "short": []}
        if ret is not None:
            cum, bcum = (1 + ret["net"].fillna(0)).cumprod(), (1 + ret["benchmark"].fillna(0)).cumprod()
            book.update({"dates": [_iso(d) for d in ret.index], "net": [_num(x) for x in cum.to_numpy()], "benchmark": [_num(x) for x in bcum.to_numpy()],
                         "drawdown": [_num(x) for x in (cum / cum.cummax() - 1).to_numpy()]})
            if pos is not None:
                p = pos.reindex(ret.index).fillna(0.0)
                book["gross"], book["positions"] = [_num(x) for x in p.abs().sum(axis=1)], [int(x) for x in (p.abs() > 1e-9).sum(axis=1)]
                book["long"], book["short"] = [_num(x) for x in p.clip(lower=0).sum(axis=1)], [_num(x) for x in p.clip(upper=0).sum(axis=1)]
                book["cash"] = [_num(max(0.0, 1.0 - x)) for x in p.abs().sum(axis=1)]
        if chart["close"]:                                                                      # the numbers above the chart: where the asset and the book stand now
            last_w = next((w for w in reversed(chart["weight"]) if w is not None), None)
            opened = mine[mine["status"] != "closed"] if len(mine) else mine
            chart["summary"] = {"last_date": chart["dates"][-1], "last_price": chart["close"][-1], "first_price": next((c for c in chart["close"] if c is not None), None), "position": last_w,
                                "realized": _num(mine.loc[mine["status"] == "closed", "net_contribution"].sum()) if len(mine) else 0.0, "unrealized": _num(opened["net_contribution"].sum()) if len(opened) else 0.0,
                                "book_gross": next((g for g in reversed(book["gross"]) if g is not None), None), "book_return": (book["net"][-1] - 1.0) if book["net"] else None,
                                "benchmark_return": (book["benchmark"][-1] - 1.0) if book["benchmark"] else None}
        if ex is not None and len(ex):
            decision_dates = sorted({_iso(d) for d in (dec["date"] if dec is not None and len(dec) else ex["decision_date"].dropna())})
        else:
            decision_dates = []
        out = {"id": exp_id, "name": report.get("name"), "summary": trading, "simulation": report.get("simulation"), "assets": rows, "chart": chart, "book": book, "decision_dates": decision_dates,
               "layers": {"prediction": "the model's score and its cross-sectional rank on the decision date", "decision": "the policy's action from that day's information only",
                          "execution": f"the trade the engine made {trading.get('lag_days', 1)} day(s) later at that day's close, charged {trading.get('cost_bps')} bp one way"}}
        return clean(common.envelope("ml_strategy", out, {"registry": "platform"}, [], None, [trading["note"]] if trading.get("note") else []))

    # ------------------------------------------------------------------------------------------------------------ decision log
    def strategy_decisions(self, q: dict) -> dict:
        """``GET /api/ml/strategy/decisions?id=[&asset=][&action=BUY,SELL..][&q=text][&from=][&to=][&limit=100][&offset=0]``: the searchable decision log, newest last, each row with the prediction, the rule that
        fired, the book before and after, and (for an exit) the unrealised return at the decision."""
        h = _h()
        h._only(q, {"id", "asset", "action", "q", "from", "to", "limit", "offset", "hold"}, "/api/ml/strategy/decisions")
        with h._user_errors():
            exp_id = h._id(q)
            d = self._strategy_files(exp_id)["decisions"]
            if d is None:
                d = pd.DataFrame(columns=["date", "asset", "action"])
            d = d.copy()
            if q.get("asset"):
                d = d[d["asset"] == str(q["asset"])]
            if q.get("action"):
                want = {a.strip().upper() for a in str(q["action"]).split(",") if a.strip()}
                d = d[d["action"].isin(want)]
            elif str(q.get("hold") or "") not in ("1", "true"):
                d = d[d["action"] != "HOLD"]                                                # holds are many and dull: they are there on request
            if q.get("q"):
                needle = str(q["q"]).lower()[:80]
                d = d[d["reason"].str.lower().str.contains(needle, regex=False) | d["asset"].str.lower().str.contains(needle, regex=False) | d["action"].str.lower().str.contains(needle, regex=False)]
            if q.get("from"):
                d = d[d["date"] >= pd.Timestamp(h._date("from", q["from"]))]
            if q.get("to"):
                d = d[d["date"] <= pd.Timestamp(h._date("to", q["to"]))]
            limit, offset = h._int(q, "limit", 100, 1, LIMIT), h._int(q, "offset", 0, 0, 10_000_000)
            page = d.iloc[offset: offset + limit]
        rows = [{"date": _iso(r["date"]), "asset": r["asset"], "action": r["action"], "kind": r.get("kind"), "reason": r["reason"], "score": _num(r["score"]), "rank_pct": _num(r["rank_pct"]),
                 "confidence": _num(r["confidence"]), "prev_weight": _num(r["prev_weight"]), "target_weight": _num(r["target_weight"]), "price": _num(r["price"]), "entry_price": _num(r["entry_price"]),
                 "holding_days": int(r["holding_days"]), "unrealised_return": _num(r.get("unrealised_ret"))} for _, r in page.iterrows()]
        counts = {k: int(v) for k, v in d["action"].value_counts().items()} if len(d) else {}
        return clean(common.envelope("ml_strategy_decisions", {"id": exp_id, "rows": rows, "n": int(len(d)), "offset": offset, "limit": limit, "counts": counts},
                                     {"registry": "platform"}, [], None, ["A decision is made from the close of its date; it is executed later (see the trades). HOLD rows are listed only with hold=1."]))

    # ------------------------------------------------------------------------------------------------------------ trades
    def strategy_trades(self, q: dict) -> dict:
        """``GET /api/ml/strategy/trades?id=[&kind=executions|round_trips][&asset=][&limit=100][&offset=0]``: the executed trades (each linked to its decision, or marked as a drift correction) or the round
        trips (open to close, with the position's own return, its contribution to the book, its costs and why it was closed)."""
        h = _h()
        h._only(q, {"id", "kind", "asset", "limit", "offset"}, "/api/ml/strategy/trades")
        with h._user_errors():
            exp_id = h._id(q)
            f = self._strategy_files(exp_id)
            kind = str(q.get("kind") or "round_trips")
            if kind not in ("executions", "round_trips"):
                raise ApiError("kind must be executions or round_trips")
            t = f[kind]
            t = pd.DataFrame() if t is None else t.copy()
            if len(t) and q.get("asset"):
                t = t[t["asset"] == str(q["asset"])]
            limit, offset = h._int(q, "limit", 100, 1, LIMIT), h._int(q, "offset", 0, 0, 10_000_000)
            page = t.iloc[offset: offset + limit]
        rows = []
        for _, r in page.iterrows():
            row = {k: (_iso(v) if isinstance(v, pd.Timestamp) or k.endswith("_date") else (_num(v) if isinstance(v, (float, np.floating)) else (int(v) if isinstance(v, (int, np.integer)) else v)))
                   for k, v in r.items()}
            rows.append(row)
        return clean(common.envelope("ml_strategy_trades", {"id": exp_id, "table": kind, "rows": rows, "n": int(len(t)), "offset": offset, "limit": limit}, {"registry": "platform"}, [], None, []))

    # ------------------------------------------------------------------------------------------------------------ random-signal control
    def strategy_controls(self, body: dict) -> dict:
        """``POST /api/ml/strategy/controls`` (a job): ``{id, draws}``. Replays the saved run's position rule, costs and rank persistence with random scores and reports where the model's Sharpe ratio sits among them,
        beside a beta-matched benchmark. The replay is checked against the saved run first."""
        h = _h()
        h._only(body, {"id", "draws", "seed"}, "/api/ml/strategy/controls")
        with h._user_errors():
            exp_id = h._id(body)
            f = self._strategy_files(exp_id)
            report = self.registry.report(exp_id)
        draws = int(h._num("draws", body.get("draws", 40), 5, 200, True))
        seed = int(h._num("seed", body.get("seed", 0), 0, 1_000_000, True))

        def work(progress):
            from ..ml import controls

            preds, prices, ret = f["predictions"], f["prices"], f["returns"]
            if preds is None or prices is None:
                raise ApiError("this run has no saved predictions or prices to replay")
            scores = preds["score"].unstack().reindex(columns=prices.columns)
            returns = f["asset_returns"] if f["asset_returns"] is not None else prices.pct_change(fill_method=None)
            cfg = report["config"]
            lag = int(cfg["dataset"]["lag"])                                                  # the engine adds backtest.extra_lag itself
            inv = f["investable"]
            out = controls.run(cfg, scores.reindex(returns.index), returns, None if ret is None else ret["net"], investable=inv, n=draws, seed=seed, lag=lag, progress=progress)
            out["id"] = exp_id
            (self.registry.experiment_dir(exp_id) / "controls.json").write_text(json.dumps(clean(out), allow_nan=False))            # the Research Evidence page reads it
            return clean(out)

        return self.app.submit_task("Random-signal control", work, kind="lab")

    # ------------------------------------------------------------------------------------------------------------ how the model decides
    def strategy_how(self, q: dict) -> dict:
        """``GET /api/ml/strategy/how?id=``: in words, how this run turns data into a buy or a sell. What it is asked to predict (a future return, never the price level), the indicators it reads (the feature
        groups and the ones the model leaned on most, each with what it measures), the model, how its score becomes positions, when a trade happens and what keeps the test honest."""
        h = _h()
        h._only(q, {"id"}, "/api/ml/strategy/how")
        with h._user_errors():
            exp_id = h._id(q)
            rep = self.registry.report(exp_id)
        from ..ml import features as fx
        from ..ml.models import model_catalog

        cfg = rep.get("config") or {}
        ds, sg, sp, bt = cfg.get("dataset") or {}, cfg.get("signal") or {}, cfg.get("split") or {}, cfg.get("backtest") or {}
        horizon = int(ds.get("horizon") or 21)
        target = str(ds.get("target") or "")
        base = {"fwd_ret": f"the stock's own return over the next {horizon} trading days", "excess_ret": f"the stock's return over the next {horizon} trading days minus the average return of its universe (so it learns which names will beat the group, not whether the market goes up)",
                "rank": f"the stock's rank among its peers by return over the next {horizon} trading days", "direction": f"whether the stock's return over the next {horizon} trading days is positive",
                "topbottom": f"whether the stock lands in the top or the bottom of the day's forward returns over {horizon} trading days"}
        kind = next((k for k in base if target.startswith(k)), None)
        predicts = base[kind] if kind else f"the target '{target}'"
        names = list((rep.get("features") or {}).get("names") or [])
        by_group: dict[str, list[str]] = {}
        for n in names:
            spec = fx.FEATURES.get(n)
            by_group.setdefault(spec.group if spec else "other", []).append(n)
        GROUP_TEXT = {"price": "trailing returns and where the price sits against its moving averages and its highs and lows", "momentum": "the classic momentum signals (12-1, 6-1, 3-1 month), RSI, MACD, breakouts and trend strength",
                      "mean_reversion": "short-term reversal, z-scores, Bollinger position and autocorrelation", "volatility": "realised, downside and range-based volatility, skew, beta and drawdown",
                      "volume": "traded dollar volume, volume surges and illiquidity", "relative": "strength against the rest of the universe and its asset class", "macro": "levels and changes of macro series",
                      "futures": "carry, curve slope and basis momentum", "options": "implied volatility and its premium over realised volatility", "crypto": "funding, perpetual basis and stablecoin supply", "fracdiff": "fractionally differenced prices"}
        groups = [{"group": g, "n": len(v), "what": GROUP_TEXT.get(g, ""), "examples": [{"name": n, "meaning": (fx.FEATURES[n].description if n in fx.FEATURES else "")} for n in v[:6]]} for g, v in by_group.items()]
        imp = (rep.get("importance") or {}).get("table") or []
        leaning = [{"name": r["feature"], "share": _num(r.get("share")), "group": fx.FEATURES[r["feature"]].group if r["feature"] in fx.FEATURES else None,
                    "meaning": fx.FEATURES[r["feature"]].description if r["feature"] in fx.FEATURES else ""} for r in imp[:12]]
        kind_id = (cfg.get("model") or {}).get("kind")
        info = next((m for m in model_catalog() if m["kind"] == kind_id), {})
        method = str(sg.get("method"))
        rule = h.SIGNAL_TEXT.get(method, method)
        lag = int((rep.get("trading") or {}).get("lag_days") or ds.get("lag") or 1)
        steps = [f"At the close of each day the model reads {len(names)} indicators for every stock, each computed only from data up to that close.",
                 f"It outputs one score per stock: its estimate of {predicts}.",
                 f"The position rule ('{method}') turns the day's scores into a target book: {rule[:1].lower() + rule[1:]}.",
                 f"The book is rebalanced {str(bt.get('rebalance', 'on a schedule')).replace('_', ' ')}; a trade decided at the close of day t is executed {lag} trading day(s) later and charged {bt.get('cost_bps', '?')} bp plus {bt.get('slippage_bps', '?')} bp slippage one way.",
                 "A buy is a rise of a stock's target weight and a sell a fall of it; the trade list shows each one with the score and rank behind it."]
        guards = ["It never sees the price level as a target: a price trends, and any model can fit a trending series perfectly and forecast nothing. It is asked for a future return, which is what a trade earns.",
                  "Every indicator is a function of data up to its own date (tested by replacing the future with noise and requiring the indicators to be unchanged).",
                  f"The model is re-fitted every {sp.get('refit_every', 21)} days on past data only, with the training rows whose outcome windows overlap a test day removed (the purge), so a score on day t never used day t's outcome.",
                  "The result is judged out of sample, after costs, against the equal-weight benchmark of the same stocks; it is not judged on how well it fits the past."]
        out = {"id": exp_id, "name": rep.get("name"), "predicts": predicts, "target": target, "horizon": horizon, "n_features": len(names), "groups": groups, "leaning_on": leaning, "has_importance": bool(leaning),
               "model": {"kind": kind_id, "label": info.get("label") or kind_id, "family": info.get("family"), "text": info.get("description"), "refit_every": sp.get("refit_every"), "window": sp.get("window")},
               "rule": {"method": method, "text": rule, "rebalance": bt.get("rebalance"), "lag_days": lag}, "steps": steps, "guards": guards}
        return clean(common.envelope("ml_strategy_how", out, {"registry": "platform"}, [], None, []))

    # ------------------------------------------------------------------------------------------------------------ a period
    def strategy_window(self, q: dict) -> dict:
        """``GET /api/ml/strategy/window?id=[&days=5][&from=&to=]``: a forward test of a period, not of one date. The run's own book over the last ``days`` trading days (or ``from``..``to``): what it earned
        against the equal-weight benchmark, which positions made or lost it, the trades inside it, and what the model said about each asset at the start of the period against what then happened.

        Nothing is re-fitted. The book is the one the run recorded, so a day the model had trained on is not a forward day: ``coverage`` counts the window's days by segment and says so. A development run is
        scored only to the end of its validation segment (the test and forward outcomes are protected), so its window cannot reach past that."""
        h = _h()
        h._only(q, {"id", "days", "from", "to"}, "/api/ml/strategy/window")
        with h._user_errors():
            exp_id = h._id(q)
            f = self._strategy_files(exp_id)
            report = self.registry.report(exp_id)
            ret, pos, preds = f["returns"], f["positions"], f["predictions"]
            if ret is None or pos is None or not len(ret):
                raise ApiError("this experiment has no recorded return stream to test a period on")
            lag = int((report.get("trading") or {}).get("lag_days", 1))
            days = h._int(q, "days", 5, 1, 504)
            if q.get("from") or q.get("to"):
                a = pd.Timestamp(h._date("from", q["from"])) if q.get("from") else ret.index[0]
                b = pd.Timestamp(h._date("to", q["to"])) if q.get("to") else ret.index[-1]
                if b < a:
                    raise ApiError("'to' must not be before 'from'")
                win = ret.loc[a:b]
            else:
                win = ret.iloc[-days:]
            if not len(win):
                raise ApiError(f"the run has no days between {_iso(q.get('from'))} and {_iso(q.get('to'))}; it covers {_iso(ret.index[0])} to {_iso(ret.index[-1])}")
            net, gross, bench = win["net"].fillna(0.0), win["gross"].fillna(0.0) if "gross" in win else win["net"].fillna(0.0), win["benchmark"].fillna(0.0) if "benchmark" in win else pd.Series(0.0, index=win.index)
            comp = lambda r: float((1.0 + r).prod() - 1.0)                                                  # noqa: E731
            curve_n, curve_b = (1.0 + net).cumprod(), (1.0 + bench).cumprod()
            metrics = {"net_return": comp(net), "gross_return": comp(gross), "benchmark_return": comp(bench), "active_return": comp(net) - comp(bench), "cost": comp(gross) - comp(net),
                       "days": int(len(win)), "up_days": int((net > 0).sum()), "best_day": _num(net.max()), "worst_day": _num(net.min()), "max_drawdown": _num((curve_n / curve_n.cummax() - 1.0).min()),
                       "turnover": _num(win["turnover"].sum()) if "turnover" in win else None, "beat_benchmark_days": int((net > bench).sum())}
            segs = {}
            if preds is not None and "segment" in preds.columns:                                           # which part of the run each day of the window belongs to
                by_day = preds["segment"].groupby(level=0).first()
                got = by_day.reindex(win.index)
                segs = {str(k): int(v) for k, v in got.value_counts().items()}
                segs["not_scored"] = int(got.isna().sum())
            oos = sum(v for k, v in segs.items() if k in ("validation", "test", "forward"))
            coverage = {"segments": segs, "out_of_sample_days": oos, "days": int(len(win)), "train_days": int(segs.get("train", 0)), "not_scored_days": int(segs.get("not_scored", 0))}
            notes = []
            if coverage["train_days"]:
                notes.append(f"{coverage['train_days']} of the {len(win)} days lie in the training segment: the model had learned from them, so they say nothing about forward skill.")
            if coverage["not_scored_days"] > lag and segs:
                notes.append(f"{coverage['not_scored_days']} day(s) were not scored by the model (a development run stops at the end of validation; the test and forward outcomes are protected).")
            if len(win) < 20:
                notes.append(f"{len(win)} days is a very short test: a week of returns is mostly noise, and it cannot separate skill from luck. Read it as a check that the book behaved as designed, not as evidence of an edge.")
            ar = f["asset_returns"] if f["asset_returns"] is not None else (f["prices"].pct_change() if f["prices"] is not None else None)
            assets, read = [], None
            if ar is not None:
                held = pos.shift(lag).reindex(win.index).fillna(0.0)
                r = ar.reindex(index=win.index, columns=pos.columns).fillna(0.0)
                contrib = (held * r).sum()
                cumr = (1.0 + r).prod() - 1.0
                sc0 = None
                if preds is not None:
                    p = preds.reset_index()
                    d0, a0 = p.columns[0], p.columns[1]
                    before = p[p[d0] < win.index[0]][d0]
                    day0 = before.max() if len(before) else p[d0].min()                                    # the last score known before the window opened
                    sc0 = p[p[d0] == day0].set_index(a0)["score"]
                for a in pos.columns:
                    if abs(held[a]).max() < 1e-9 and abs(cumr[a]) < 1e-12:
                        continue
                    assets.append({"asset": a, "avg_weight": _num(held[a].mean()), "asset_return": _num(cumr[a]), "contribution": _num(contrib[a]), "score_at_start": _num(sc0.get(a)) if sc0 is not None else None,
                                   "rank_at_start": _num(sc0.rank(pct=True).get(a)) if sc0 is not None else None})
                assets.sort(key=lambda x: -abs(x["contribution"] or 0.0))
                if sc0 is not None and len(sc0.dropna()) >= 4:
                    both = pd.concat([sc0, cumr], axis=1, join="inner").dropna()
                    both.columns = ["score", "ret"]
                    top = both["score"] >= both["score"].median()
                    read = {"rank_correlation": _num(both["score"].rank().corr(both["ret"].rank())), "top_half_return": _num(both.loc[top, "ret"].mean()), "bottom_half_return": _num(both.loc[~top, "ret"].mean()),
                            "n_assets": int(len(both)), "note": "the model's score at the start of the window against what each asset then did: a rank correlation across only a handful of assets over a few days is noise on its own."}
            ex = f["executions"]
            trades = []
            if ex is not None and len(ex):
                inside = ex[(ex["date"] >= win.index[0]) & (ex["date"] <= win.index[-1])]
                trades = [{"date": _iso(r["date"]), "asset": r["asset"], "action": r.get("action"), "weight_change": _num(r["trade_weight"]), "weight_after": _num(r["weight_after"]),
                           "decision_date": _iso(r.get("decision_date")), "reason": r.get("reason")} for _, r in inside.head(200).iterrows()]
        out = {"id": exp_id, "name": report.get("name"), "window": {"from": _iso(win.index[0]), "to": _iso(win.index[-1]), "days": int(len(win)), "run_from": _iso(ret.index[0]), "run_to": _iso(ret.index[-1])},
               "metrics": metrics, "coverage": coverage, "curve": {"dates": [_iso(d) for d in win.index], "net": [_num(x) for x in curve_n], "benchmark": [_num(x) for x in curve_b]},
               "assets": assets[:30], "read": read, "trades": trades}
        return clean(common.envelope("ml_strategy_window", out, {"registry": "platform"}, notes, None, []))

    # ------------------------------------------------------------------------------------------------------------ replay
    def strategy_replay(self, q: dict) -> dict:
        """``GET /api/ml/strategy/replay?id=&date=``: what was known on a decision date and what the policy did with it: every asset's score and rank that day, the book before and after, the decisions taken, and
        the executions that followed. Nothing dated after ``date`` is used to build the 'known' side."""
        h = _h()
        h._only(q, {"id", "date"}, "/api/ml/strategy/replay")
        with h._user_errors():
            exp_id = h._id(q)
            f = self._strategy_files(exp_id)
            report = self.registry.report(exp_id)
            day = pd.Timestamp(h._date("date", q.get("date") or ""))
            dec, ex, pos, preds, prices = f["decisions"], f["executions"], f["positions"], f["predictions"], f["prices"]
            if prices is not None and pos is not None and len(pos):
                prices = prices.loc[pos.index[0]:]
            if pos is None or not len(pos):
                raise ApiError("this experiment has no recorded positions")
            idx = pos.index
            at = idx[idx.searchsorted(day, side="right") - 1] if day >= idx[0] else idx[0]               # the last recorded day on or before the date asked for
            lag = int((report.get("trading") or {}).get("lag_days", 1))
            scores = None
            if preds is not None:
                p = preds.reset_index()
                lvl0, lvl1 = p.columns[0], p.columns[1]
                s = p[p[lvl0] == at]
                if len(s):
                    scores = s.set_index(lvl1)["score"]
            before = pos.shift(1).loc[at].fillna(0.0) if len(pos) else None
            after = pos.loc[at].fillna(0.0)
            did = ex[ex["decision_date"] == at] if ex is not None and len(ex) else pd.DataFrame()
            if dec is not None:
                made = dec[dec["date"] == at]
            else:                                                                           # a rule with no decision log: what it decided that day is the trades it led to, with their reasons
                made = did[["asset", "action", "reason"]].copy() if len(did) else pd.DataFrame(columns=["asset", "action", "reason"])
            table = []
            for a in pos.columns:
                sc = _num(scores.get(a)) if scores is not None else None
                m = made[made["asset"] == a]
                table.append({"asset": a, "score": sc, "rank_pct": _num(scores.rank(pct=True).get(a)) if scores is not None else None, "close": _num(prices.loc[at, a]) if prices is not None and at in prices.index else None,
                              "weight_held_before": _num(before.get(a)), "book_after": _num(after.get(a)), "decision": None if not len(m) else " + ".join(m["action"]),
                              "reason": None if not len(m) else " -> ".join(m["reason"])})
            table.sort(key=lambda r: -(r["score"] if r["score"] is not None else -1e18))
        out = {"id": exp_id, "date": _iso(at), "asked": _iso(day), "lag_days": lag, "assets": table,
               "executed_later": [{"date": _iso(r["date"]), "asset": r["asset"], "kind": r["kind"], "weight_change": _num(r["trade_weight"]), "cost": _num(r["cost"])} for _, r in did.iterrows()],
               "known": "scores, ranks and closes up to and including this date, and the book held going into it; the executions on the right happen later and are not known at the decision"}
        return clean(common.envelope("ml_strategy_replay", out, {"registry": "platform"}, [], None, ["The features behind each score are not stored with the run; the replay shows the score and its rank, not the inputs."]))
