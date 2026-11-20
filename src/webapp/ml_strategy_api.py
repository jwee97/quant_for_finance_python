"""The ML Strategy Explorer's endpoints: what a model predicted, what the decision policy did with it, what was executed, and what it earned.

Read-only: everything is read from the artifacts a ``decision_policy`` experiment saved (decisions, executions, round trips, closes, positions, returns, predictions). Nothing is recomputed, so what the
page shows is what the run recorded. Three layers stay separate in every response: the *prediction* (a score and its rank on a date), the *decision* (BUY / SELL / HOLD ... with its reason, taken from
that day's information only) and the *execution* (the trade the engine made ``lag`` days later, at that day's close, at the configured cost).
"""

from __future__ import annotations

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
        return {"/api/ml/strategy/runs": self.strategy_runs, "/api/ml/strategy": self.strategy, "/api/ml/strategy/decisions": self.strategy_decisions, "/api/ml/strategy/trades": self.strategy_trades, "/api/ml/strategy/replay": self.strategy_replay}

    # ------------------------------------------------------------------------------------------------------------ loading
    def _strategy_files(self, exp_id: str) -> dict:
        d = self.registry.experiment_dir(exp_id)
        if not (d / "decisions.parquet").exists() and not (d / "executions.parquet").exists():
            raise ApiError("this experiment did not use the decision_policy position rule, so it has no decisions to show; run it with signal.method = decision_policy (and backtest.rebalance = on_change)")
        out = {}
        for name in ("decisions", "executions", "round_trips", "prices", "investable", "asset_returns", "positions", "returns", "predictions"):
            f = d / f"{name}.parquet"
            out[name] = pd.read_parquet(f) if f.exists() else None
        return out

    # ------------------------------------------------------------------------------------------------------------ the runs that have decisions
    def strategy_runs(self, q: dict) -> dict:
        """``GET /api/ml/strategy/runs``: the saved experiments that used the decision policy (they have a decision log), newest first, with what to compare at a glance."""
        h = _h()
        h._only(q, set(), "/api/ml/strategy/runs")
        rows = []
        with h._user_errors():
            for rec in self.registry.list_experiments(300):
                if not (self.registry.experiment_dir(rec["id"]) / "decisions.parquet").exists():
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
        ex, trips, prices, pos, ret = f["executions"], f["round_trips"], f["prices"], f["positions"], f["returns"]
        if prices is not None and pos is not None and len(pos):
            prices = prices.loc[pos.index[0]:]                                              # the file holds the whole history (a control replays on it); the chart starts where the run did
        traded = (ex.groupby("asset").size().sort_values(ascending=False) if ex is not None and len(ex) else pd.Series(dtype=int))
        asset = str(q.get("asset") or (traded.index[0] if len(traded) else (prices.columns[0] if prices is not None else "")))
        if prices is not None and asset not in prices.columns:
            raise ApiError(f"asset must be one of {list(prices.columns)}")
        rows = []
        for a in (prices.columns if prices is not None else traded.index):
            t = trips[trips["asset"] == a] if trips is not None and len(trips) else pd.DataFrame()
            rows.append({"asset": a, "executions": int(traded.get(a, 0)), "round_trips": int(len(t)),
                         "net_contribution": _num(t["net_contribution"].sum()) if len(t) else 0.0, "win_rate": _num((t["position_return"] > 0).mean()) if len(t) else None})
        chart = {"asset": asset, "dates": [], "close": [], "weight": [], "markers": []}
        if prices is not None:
            px = prices[asset].dropna()
            w = pos[asset].reindex(px.index).fillna(0.0) if pos is not None and asset in pos.columns else pd.Series(0.0, index=px.index)
            chart.update({"dates": [_iso(d) for d in px.index], "close": [_num(x) for x in px.to_numpy()], "weight": [_num(x) for x in w.to_numpy()]})
            if ex is not None and len(ex):
                for _, r in ex[ex["asset"] == asset].iterrows():
                    chart["markers"].append({"date": _iso(r["date"]), "price": _num(px.get(pd.Timestamp(r["date"]))), "kind": r["kind"], "side": r["side"], "weight_change": _num(r["trade_weight"]),
                                             "weight_after": _num(r["weight_after"]), "action": r.get("action"), "reason": r.get("reason"), "decision_date": _iso(r.get("decision_date")), "cost": _num(r["cost"]),
                                             "rank_pct": _num(r.get("rank_pct")), "score": _num(r.get("score"))})
        book = {"dates": [], "net": [], "benchmark": [], "drawdown": [], "gross": [], "cash": [], "positions": []}
        if ret is not None:
            cum, bcum = (1 + ret["net"].fillna(0)).cumprod(), (1 + ret["benchmark"].fillna(0)).cumprod()
            book.update({"dates": [_iso(d) for d in ret.index], "net": [_num(x) for x in cum.to_numpy()], "benchmark": [_num(x) for x in bcum.to_numpy()],
                         "drawdown": [_num(x) for x in (cum / cum.cummax() - 1).to_numpy()]})
            if pos is not None:
                p = pos.reindex(ret.index).fillna(0.0)
                book["gross"], book["positions"] = [_num(x) for x in p.abs().sum(axis=1)], [int(x) for x in (p.abs() > 1e-9).sum(axis=1)]
                book["cash"] = [_num(max(0.0, 1.0 - x)) for x in p.abs().sum(axis=1)]
        out = {"id": exp_id, "name": report.get("name"), "summary": trading, "assets": rows, "chart": chart, "book": book,
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
            return clean(out)

        return self.app.submit_task("Random-signal control", work, kind="lab")

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
            made = dec[dec["date"] == at] if dec is not None else pd.DataFrame()
            did = ex[ex["decision_date"] == at] if ex is not None and len(ex) else pd.DataFrame()
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
