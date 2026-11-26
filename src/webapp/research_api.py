"""The research workspace's endpoints: sessions with a shared context, saved results, the Market Regime Lab and the Research Comparison.

``GET  /api/research/sessions``            every session (a short summary), and what the store accepts
``GET  /api/research/session?id=``         one session with its context, notes, comparison groups and the summaries of its results
``GET  /api/research/artifact?id=``        one saved result in full (its series included)
``GET  /api/research/compat?session=&artifact=``   whether the session's current context still describes a saved result, field by field
``GET  /api/research/overview``            what the Overview page shows: counts, recent results, jobs, data freshness, the latest regime and market snapshot, each labelled with where it comes from
``POST /api/research/session/create|update|delete|duplicate``    sessions; ``/note/add|delete``; ``/group/save|delete``
``POST /api/research/artifact/save``       keep a finished backtest job (``{"session", "job"}``) or a saved machine-learning experiment (``{"session", "experiment"}``)
``POST /api/research/artifact/delete``
``POST /api/research/compare``             compare saved results over the days they share (``{"artifacts": [...]}`` or ``{"session", "group"}``)
``POST /api/research/export``              the same comparison as ``csv`` or ``markdown`` text
``GET  /api/regime/plan``                  the regime models the lab offers and their settings
``POST /api/regime/run``                   estimate regimes on a market proxy as a job; with ``session`` the result is kept
``POST /api/regime/by-result``             how a saved result's returns behaved in each regime of a saved regime result

The numbers of a saved result come from the server's own records (the job's result file, the registry), never from the page. The files live in ``data/user/research``.
"""

from __future__ import annotations

import time

import pandas as pd

from ..framework import DETECTORS
from ..framework.experiments import git_commit
from ..framework.regimes import market_proxy
from ..research import ContextError, ResearchStore, StoreError, compatibility, validate_context
from ..research import compare as cmp
from ..research import extract
from ..research import regime_lab as rl
from ..research.context import ASSET_CLASSES
from .api import ApiError
from .payload import clean
from .universe import UniverseError, normalise

REGIME_MODELS = {"hmm": "Hidden Markov model on the market's daily return (refitted on past data only)", "vol_state": "Volatility state: where today's volatility ranks in its own history, with a drawdown test for crises",
                 "gmm": "Gaussian mixture on the market's daily return (each day classified on its own)"}


def _id(body: dict, key: str = "id") -> str:
    value = str(body.get(key) or "")
    if not value:
        raise ApiError(f"{key} is required")
    return value


class ResearchApi:
    def __init__(self, app, ml):
        self.app, self.ml = app, ml
        self.store = ResearchStore(app.root / "data" / "user" / "research")
        self.code = None

    # ------------------------------------------------------------------------------------------------------ routes
    def get_routes(self) -> dict:
        return {"/api/research/sessions": self.sessions, "/api/research/session": self.session, "/api/research/artifact": self.artifact, "/api/research/compat": self.compat, "/api/research/overview": self.overview,
                "/api/regime/plan": self.regime_plan}

    def post_routes(self) -> dict:
        return {"/api/research/session/create": self.session_create, "/api/research/session/update": self.session_update, "/api/research/session/delete": self.session_delete,
                "/api/research/session/duplicate": self.session_duplicate, "/api/research/note/add": self.note_add, "/api/research/note/delete": self.note_delete,
                "/api/research/group/save": self.group_save, "/api/research/group/delete": self.group_delete, "/api/research/config/save": self.config_save, "/api/research/config/delete": self.config_delete, "/api/research/artifact/save": self.artifact_save,
                "/api/research/artifact/delete": self.artifact_delete, "/api/research/compare": self.compare, "/api/research/export": self.export, "/api/regime/run": self.regime_run,
                "/api/regime/by-result": self.regime_by_result}

    # ------------------------------------------------------------------------------------------------------ helpers
    def _sources(self) -> set:
        return set(self.app.builder.sources)

    def _ctx(self, raw) -> dict:
        try:
            return validate_context(raw, sources=self._sources() | {"platform"})
        except ContextError as error:
            raise ApiError(str(error)) from None

    def _store(self, fn, *args, **kw):
        try:
            return fn(*args, **kw)
        except (StoreError, ContextError) as error:
            raise ApiError(str(error)) from None

    def _code_version(self) -> str:
        if self.code is None:
            self.code = git_commit(self.app.root)
        return self.code

    # ------------------------------------------------------------------------------------------------------ sessions
    def sessions(self, q: dict) -> dict:
        from ..research.store import MAX_ARTIFACTS, MAX_SESSIONS, SCHEMA

        return clean({"sessions": self.store.list_sessions(), "schema": SCHEMA, "limits": {"sessions": MAX_SESSIONS, "artifacts_per_session": MAX_ARTIFACTS}, "asset_classes": list(ASSET_CLASSES),
                      "sources": sorted(self._sources())})

    def session(self, q: dict) -> dict:
        rec = self._store(self.store.get_session, _id(q))
        return clean({"session": rec, "artifacts": self._store(self.store.list_artifacts, rec["id"])})

    def session_create(self, body: dict) -> dict:
        ctx = self._ctx(body.get("context"))
        return clean({"session": self._store(self.store.create_session, str(body.get("name") or ""), ctx)})

    def session_update(self, body: dict) -> dict:
        ctx = None
        if body.get("context") is not None:
            self._ctx(body["context"])                                                       # validated here for the message; the store validates again and removes the fields sent as null
            ctx = body["context"]
        rec = self._store(self.store.update_session, _id(body), name=body.get("name"), context=ctx, replace_context=bool(body.get("replace")))
        return clean({"session": rec})

    def session_delete(self, body: dict) -> dict:
        self._store(self.store.delete_session, _id(body), keep_artifacts=bool(body.get("keep_results")))
        return {"deleted": _id(body)}

    def session_duplicate(self, body: dict) -> dict:
        return clean({"session": self._store(self.store.duplicate_session, _id(body))})

    def note_add(self, body: dict) -> dict:
        return {"note": self._store(self.store.add_note, _id(body, "session"), body.get("text"))}

    def note_delete(self, body: dict) -> dict:
        self._store(self.store.delete_note, _id(body, "session"), _id(body, "note"))
        return {"deleted": _id(body, "note")}

    def group_save(self, body: dict) -> dict:
        return {"group": self._store(self.store.save_group, _id(body, "session"), str(body.get("name") or ""), list(body.get("artifacts") or []), group_id=body.get("id") or None,
                                     benchmark=str(body.get("benchmark") or ""), start=str(body.get("start") or ""), end=str(body.get("end") or ""))}

    def group_delete(self, body: dict) -> dict:
        self._store(self.store.delete_group, _id(body, "session"), _id(body, "group"))
        return {"deleted": _id(body, "group")}

    def config_save(self, body: dict) -> dict:
        return clean({"config": self._store(self.store.add_config, _id(body, "session"), str(body.get("name") or ""), body.get("body"))})

    def config_delete(self, body: dict) -> dict:
        self._store(self.store.delete_config, _id(body, "session"), _id(body, "config"))
        return {"deleted": _id(body, "config")}

    # ------------------------------------------------------------------------------------------------------ results
    def artifact(self, q: dict) -> dict:
        return clean(self._store(self.store.get_artifact, _id(q)))

    def compat(self, q: dict) -> dict:
        rec = self._store(self.store.get_session, _id(q, "session"))
        art = self._store(self.store.get_artifact, _id(q, "artifact"), series=False)
        issues = compatibility(rec["context"], art["context"])
        return {"artifact": art["id"], "issues": issues, "compatible": not any(i["severity"] == "mismatch" for i in issues),
                "advice": "" if not issues else "The result is kept as it was made. Run it again under the current context to get numbers that describe it, or compare it only with results made under the same assumptions."}

    def artifact_save(self, body: dict) -> dict:
        session = _id(body, "session")
        self._store(self.store.get_session, session)
        if body.get("job"):
            job = self.app.job(str(body["job"]))
            if job.get("status") != "done" or not isinstance(job.get("result"), dict):
                raise ApiError("only a finished run can be saved: this one is " + str(job.get("status")))
            if job.get("kind") == "regime":
                art = self._regime_artifact(job["result"], str(body["job"]))
            elif job.get("kind") == "backtest":
                art = self._store(extract.backtest_artifact, job["result"], job_id=str(body["job"]), code_version=self._code_version())
            elif job.get("kind") == "lab" and "table" in job["result"] and "equity" in job["result"]:
                if not body.get("method"):
                    raise ApiError("say which method of the study to keep")
                art = self._store(extract.portfolio_lab_artifact, job["result"], str(body["method"]), job_id=str(body["job"]), code_version=self._code_version())
            elif job.get("kind") == "lab" and "comparison" in job["result"] and "combination" in job["result"]:
                art = self._store(extract.factor_study_artifact, job["result"], job_id=str(body["job"]), code_version=self._code_version())
            elif job.get("kind") == "forecast" and "evaluation" in job["result"]:
                art = self._store(extract.forecast_artifact, job["result"], job_id=str(body["job"]), code_version=self._code_version())
            elif job["result"].get("kind") in extract.STRATEGY_LABS:
                art = self._store(extract.strategy_artifact, job["result"], job_id=str(body["job"]), code_version=self._code_version())
            else:
                raise ApiError(f"a {job.get('kind')} run cannot be saved here yet: save a backtest, a strategy backtest from the Options, Futures or Crypto lab, a machine-learning experiment or a regime estimate")
        elif body.get("experiment"):
            exp_id = str(body["experiment"])
            try:
                reg = self.ml.registry
                report = reg.report(exp_id)
                ret = pd.read_parquet(reg.root / "experiments" / exp_id / "returns.parquet")
            except Exception as error:
                raise ApiError(f"experiment {exp_id}: {error}") from None
            art = self._store(extract.ml_artifact, report, ret, code_version=self._code_version())
        else:
            raise ApiError("say which run to keep: job or experiment")
        if body.get("title"):
            art["title"] = str(body["title"])[:120]
        saved = self._store(self.store.add_artifact, session, art)
        return clean({"artifact": {k: v for k, v in saved.items() if k != "series"}})

    def artifact_delete(self, body: dict) -> dict:
        self._store(self.store.delete_artifact, _id(body))
        return {"deleted": _id(body)}

    # ------------------------------------------------------------------------------------------------------ comparison
    def _compare(self, body: dict) -> tuple[dict, dict | None, dict | None]:
        session = group = None
        ids = list(body.get("artifacts") or [])
        bench, start, end = str(body.get("benchmark") or ""), str(body.get("start") or ""), str(body.get("end") or "")
        if body.get("session") and body.get("group"):
            session = self._store(self.store.get_session, str(body["session"]))
            group = next((g for g in session["groups"] if g["id"] == body["group"]), None)
            if group is None:
                raise ApiError("unknown comparison group")
            ids, bench, start, end = group["artifacts"], group.get("benchmark") or bench, group.get("start") or start, group.get("end") or end
        if not 1 <= len(ids) <= 12:
            raise ApiError("choose between 1 and 12 results to compare")
        arts = [self._store(self.store.get_artifact, str(i)) for i in ids]
        rf = body.get("risk_free")
        try:
            rf = None if rf in (None, "") else float(rf)
            cleaned = validate_context({"start": start, "end": end})
        except (TypeError, ValueError, ContextError) as error:
            raise ApiError(str(error)) from None
        out = cmp.compare(arts, benchmark=bench, start=cleaned.get("start", ""), end=cleaned.get("end", ""), risk_free=rf)
        return out, session, group

    def compare(self, body: dict) -> dict:
        out, _, _ = self._compare(body)
        return clean(out)

    def export(self, body: dict) -> dict:
        out, session, group = self._compare(body)
        fmt = str(body.get("format") or "markdown")
        if fmt == "csv":
            return {"filename": "research_comparison.csv", "text": cmp.to_csv(out), "format": "csv"}
        if fmt == "markdown":
            return {"filename": "research_comparison.md", "text": cmp.to_markdown(out, session=session, group=group), "format": "markdown"}
        raise ApiError("format must be csv or markdown")

    # ------------------------------------------------------------------------------------------------------ regimes
    def regime_plan(self, q: dict) -> dict:
        return {"models": [{"name": k, "description": v} for k, v in REGIME_MODELS.items()], "defaults": {"model": "hmm", "n_states": 2, "min_train": 750, "refit_every": 126, "min_days": rl.MIN_DAYS},
                "notes": ["Filtered estimates use only data up to each day and a model fitted before it: a rule could have traded on them.",
                          "The retrospective estimate uses the whole sample and is shown only to compare: it cannot be traded on."]}

    def regime_run(self, body: dict) -> dict:
        model = str(body.get("model") or "hmm")
        if model not in REGIME_MODELS:
            raise ApiError(f"model must be one of {', '.join(REGIME_MODELS)}")
        try:
            tickers = normalise(body.get("tickers") or body.get("assets") or self.app.builder.default_tickers())
            self.app.builder.preflight(tickers, body.get("start"), body.get("source"))
        except UniverseError as error:
            raise ApiError(str(error)) from None
        try:
            n_states, min_train, refit = int(body.get("n_states") or 2), int(body.get("min_train") or 750), int(body.get("refit_every") or 126)
            min_days = int(body.get("min_days") or rl.MIN_DAYS)
        except (TypeError, ValueError):
            raise ApiError("n_states, min_train, refit_every and min_days must be whole numbers") from None
        if model == "hmm" and n_states not in (2, 3):
            raise ApiError("choose 2 or 3 states")
        if not 250 <= min_train <= 5000 or not 21 <= refit <= 504 or not 20 <= min_days <= 500:
            raise ApiError("min_train must be 250 to 5000 days, refit_every 21 to 504 and min_days 20 to 500")
        session = str(body.get("session") or "")
        if session:
            self._store(self.store.get_session, session)
        start, source, refresh = body.get("start") or None, body.get("source"), bool(body.get("refresh"))

        def work(progress):
            progress("building the universe (downloading any new tickers)")
            bundle, universe = self.app.builder.resolve(tickers, body.get("classes"), start, refresh, source, progress)
            proxy = market_proxy(bundle).dropna()
            proxy = proxy[proxy.index >= proxy.index[(proxy != 0).to_numpy().argmax()]] if len(proxy) else proxy
            progress("estimating the filtered regimes (each day uses only the past)")
            smoothed = None
            if model == "hmm":
                filtered = rl.filtered_hmm(proxy, n_states, min_train, refit)
                progress("estimating the retrospective regimes for comparison")
                smoothed = rl.smoothed_hmm(proxy, n_states)
            else:
                det = DETECTORS.create(model)
                probs = det.detect(bundle).probabilities.dropna(how="all")
                probs = probs.loc[:, [c for c in probs.columns if probs[c].notna().any()]]
                filtered = {"probabilities": probs.reindex(proxy.index), "fits": None, "transmat": None, "names": list(probs.columns), "settings": {"model": model}}
            result = rl.analyse(proxy, filtered, smoothed, min_days=min_days)
            result.update(model=model, model_text=REGIME_MODELS[model], universe=universe, proxy=rl.market_proxy_stats(proxy),
                          retrospective_note="The retrospective estimate uses the whole sample and cannot be traded on." if smoothed is not None else "This model has no retrospective counterpart: it is causal by construction.",
                          code_version=self._code_version())
            result["_proxy"] = {"dates": [str(d.date()) for d in proxy.index], "returns": [round(float(v), 6) for v in proxy.to_numpy()]}
            if session:
                art = self._regime_artifact(result, "")
                result["artifact"] = {k: v for k, v in self._store(self.store.add_artifact, session, art).items() if k != "series"}
            return result

        return self.app.submit_task("Regime lab: " + ", ".join(tickers[:4]) + ("…" if len(tickers) > 4 else ""), work, kind="regime")

    def _regime_artifact(self, result: dict, job_id: str) -> dict:
        tl, proxy = result.get("timeline") or {}, result.get("_proxy") or {}
        if not tl.get("dates") or not proxy.get("dates"):
            raise ApiError("this run has no regime timeline to keep")
        p = pd.Series(proxy["returns"], index=pd.to_datetime(proxy["dates"]))
        eq = (1.0 + p).cumprod()
        dates = pd.to_datetime(tl["dates"])
        eq = eq.reindex(dates).ffill()
        names = result["names"]
        extra = {"label": [names[i] for i in tl["label"]], **{f"p_{n}": v for n, v in tl["probabilities"].items()}}
        uni = result.get("universe") or {}
        ctx = {"assets": list(uni.get("tickers", [])), "start": tl["dates"][0], "end": tl["dates"][-1], "frequency": "daily", "data_source": str(uni.get("provider") or "platform"),
               "dataset_version": extract.fingerprint(uni.get("tickers"), uni.get("first"), uni.get("last"), uni.get("days"), uni.get("source"))}
        stab = result.get("stability") or {}
        metrics = {k: v for k, v in {"mean_confidence": stab.get("mean_confidence"), "uncertain_share": stab.get("uncertain_share"), "switches_per_year": stab.get("switches_per_year"),
                                     "agreement_with_retrospective": stab.get("agreement_with_retrospective")}.items() if v is not None}
        return {"kind": "regime", "title": f"{result.get('model', 'regime')} regimes ({', '.join(names)})", "evidence": "walk_forward", "data_kind": extract.data_kind(str(uni.get("source", "")), uni.get("provider")),
                "context": ctx, "config": {"model": result.get("model"), "settings": result.get("settings"), "names": names, "durations": result.get("durations"), "transitions": result.get("transitions")},
                "metrics": metrics, "series": {"dates": tl["dates"], "equity": [None if v != v else round(float(v), 5) for v in eq.to_numpy()], "extra": extra},
                "refs": {"job": job_id, "code_version": result.get("code_version", ""), "dataset": ctx["dataset_version"]},
                "notes": ["Filtered (causal) regime labels of an equal-weight market proxy: each day uses only data up to that day. The equity curve is the proxy, not a strategy."]}

    def regime_by_result(self, body: dict) -> dict:
        """The returns of a saved result in each state of a saved regime estimate (``lag=1``: the label known at the close against the next day's return)."""
        reg = self._store(self.store.get_artifact, _id(body, "regime"))
        res = self._store(self.store.get_artifact, _id(body, "result"))
        if reg["kind"] != "regime":
            raise ApiError("the regime argument must be a saved regime estimate")
        r = cmp._returns(res)
        if r is None:
            raise ApiError("this result has no return series")
        labels = pd.Series(reg["series"]["extra"]["label"], index=pd.to_datetime(reg["series"]["dates"]), dtype=object)
        try:
            lag, min_days = int(body.get("lag", 1)), int(body.get("min_days") or rl.MIN_DAYS)
            out = rl.performance_by_regime(r, labels, lag=lag, min_days=min_days)
        except ValueError as error:
            raise ApiError(str(error)) from None
        overlap = r.index.intersection(labels.dropna().index)
        out.update(result=res["id"], regime=reg["id"], result_evidence=res["evidence"], result_title=res["title"], regime_title=reg["title"],
                   window={"start": str(overlap[0].date()), "end": str(overlap[-1].date())} if len(overlap) else None,
                   label_basis="filtered (causal)", note="The label is the regime known at the close of the previous day; the return is the result's return on the day after.")
        if res["evidence"] == "historical_simulation":
            out["caution"] = "The result is a historical simulation of the whole sample: its return in a regime is descriptive, not evidence that the strategy works in that regime."
        return clean(out)

    # ------------------------------------------------------------------------------------------------------ overview
    def overview(self, q: dict) -> dict:
        """What the Overview page draws. Every block says where it comes from (``source``) and, when there is nothing to show, says so (``available: false``) instead of inventing something."""
        sessions = self.store.list_sessions()
        everything = self.store.list_artifacts()                                           # read once: a result's file holds its series, so each listing costs a read of every file
        recent = everything[:8]
        jobs = self.app.jobs_list()
        out: dict = {"generated": time.time(), "sessions": {"n": len(sessions), "recent": sessions[:5], "source": "your saved research sessions"},
                     "artifacts": {"n": len(everything), "recent": [{k: a[k] for k in ("id", "session_id", "kind", "title", "created", "evidence", "data_kind", "metrics")} for a in recent],
                                   "source": "results you saved from the labs"}}
        rows = jobs["jobs"]
        out["jobs"] = {"active": jobs["active"], "recent": rows[:8], "counts": {s: sum(1 for r in rows if r["status"] == s) for s in ("queued", "running", "done", "error", "interrupted", "cancelled")}, "source": "this app's job log"}
        out["ml"] = self._ml_block()
        out["data"] = self._data_block()
        out["market"] = self._market_block()
        reg = next((a for a in everything if a["kind"] == "regime"), None)
        out["regime"] = {"available": False, "reason": "no regime estimate has been saved: run the Market Regime Lab and save it to a session"} if reg is None else self._regime_block(reg)
        bt = next((a for a in everything if a["kind"] == "backtest"), None)
        out["portfolio_risk"] = ({"available": False, "reason": "no backtest has been saved to a session"} if bt is None else
                                 {"available": True, "title": bt["title"], "artifact": bt["id"], "session": bt["session_id"], "evidence": bt["evidence"], "data_kind": bt["data_kind"],
                                  "metrics": {k: bt["metrics"].get(k) for k in ("ann_vol", "max_drawdown", "sharpe", "cagr")}, "source": "your latest saved backtest (a historical simulation)"})
        opt = next((a for a in everything if a["kind"] == "options"), None)
        out["options"] = ({"available": False, "reason": "no option strategy has been saved to a session: run a strategy backtest in the Options Lab and choose Save to session"} if opt is None else
                          {"available": True, "title": opt["title"], "artifact": opt["id"], "session": opt["session_id"], "evidence": opt["evidence"], "data_kind": opt["data_kind"],
                           "metrics": {k: opt["metrics"].get(k) for k in ("sharpe", "cagr", "max_drawdown", "ann_vol")}, "source": "your latest saved option strategy backtest, on a simulated option market"})
        return clean(out)

    def _regime_block(self, reg: dict) -> dict:
        art = self.store.get_artifact(reg["id"])
        ex = art["series"]["extra"]
        names = art["config"].get("names", [])
        last = {n: ex.get(f"p_{n}", [None])[-1] for n in names}
        return {"available": True, "artifact": art["id"], "session": art["session_id"], "as_of": art["series"]["dates"][-1], "state": ex["label"][-1], "probabilities": last, "evidence": art["evidence"],
                "basis": "filtered (uses data up to that date)", "source": "your latest saved regime estimate", "title": art["title"]}

    def _ml_block(self) -> dict:
        try:
            rows = self.ml.registry.list_experiments(8)
        except Exception as error:
            return {"available": False, "reason": f"the experiment registry could not be read: {error}"}
        if not rows:
            return {"available": False, "reason": "no machine-learning experiment has been run yet"}
        out = []
        for r in rows:
            m = (self.ml.registry.get_experiment(r["id"]).get("metrics") or {})
            oos = m.get("oos") or m.get("validation") or {}
            out.append({"id": r["id"], "name": r["name"], "model_kind": r.get("model_kind"), "created": r.get("created"), "sharpe": oos.get("sharpe"), "stage": r.get("stage")})
        return {"available": True, "recent": out, "source": "the experiment registry (walk-forward out-of-sample results)"}

    def _data_block(self) -> dict:
        b = self.app.builder
        default = b.default_bundle()
        last = default.index[-1]
        age = (pd.Timestamp.now().normalize() - last.normalize()).days
        sources = [{"name": s.name, "label": s.label, "available": bool(s.status().get("available")), "problem": s.status().get("problem", "")} for s in b.sources.values()]
        return {"available": True, "platform": {"first": str(default.index[0].date()), "last": str(last.date()), "age_days": int(age), "n_assets": len(default.assets), "stale": age > 7,
                                                 "note": "A versioned snapshot kept with the repository; it is not updated by the app."}, "sources": sources, "source": "the platform dataset and the configured data sources"}

    def _market_block(self) -> dict:
        b = self.app.builder
        default = b.default_bundle()
        px = default.prices.ffill()
        last = px.index[-1]
        rows = []
        for t in list(default.assets)[:12]:
            s = px[t].dropna()
            if len(s) < 260:
                continue
            rows.append({"ticker": t, "asset_class": default.asset_class.get(t, "unknown"), "last": float(s.iloc[-1]), "d1": float(s.iloc[-1] / s.iloc[-2] - 1), "m1": float(s.iloc[-1] / s.iloc[-22] - 1), "y1": float(s.iloc[-1] / s.iloc[-253] - 1)})
        return {"available": bool(rows), "as_of": str(last.date()), "rows": rows, "kind": "historical", "source": "the platform dataset: a historical snapshot, not a live feed",
                "reason": "" if rows else "the platform dataset has too little history"}

