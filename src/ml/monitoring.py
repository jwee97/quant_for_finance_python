"""Forward testing and drift monitoring: following a chosen model into the future, one day at a time.

The daily loop (``ForwardTracker.step``) is the one a live deployment would run, and it uses only what is known at the close of the day it is run for:

1. load the experiment's configuration and its final model (the one fitted to all labelled data when the experiment ran; it is NOT refitted here, so the track record is of a frozen model)
2. rebuild the features from the data up to that day
3. predict the scores of the last ``holding_period`` days, turn them into the target portfolio by the experiment's own signal rules
4. compare the target with what is currently held, giving the trades (the orders a paper account would place) and the turnover
5. save the signal; and settle the signals whose forward window has completed, with the realised return of their book and the rank IC of their scores against what each asset then did

``summary`` then reports the forward track record next to the backtest it should resemble, the live information coefficient against the out-of-sample one, and drift: the population stability index of
each feature and of the scores between the period the model was fitted on and the latest weeks. A PSI above 0.25 is the conventional line for "the inputs have changed enough that the model is working on a
different distribution".
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import execution as xe
from . import governance as gv
from . import prospective as pr
from . import provenance as pv
from .bundle import InferenceBundle
from .contracts import OutputContractError, check_signal_use
from .config import DEFAULTS, experiment_config, merge
from .datasets import build_dataset, resolve_bundle
from .evaluation import rank_ic_wide, sharpe_of
from .experiments import confidence_from_scores, regime_series, trailing_vol
from .models import Model, task_for
from .registry import Registry, RegistryError
from .signals import OVERLAY_KEYS, STATEFUL_METHODS, build_weights
from .targets import forward_return, is_classification

PSI_ALERT = 0.25
PSI_WATCH = 0.10


def psi(expected, actual, bins: int = 10) -> float:
    """Population stability index of ``actual`` against ``expected``: ``sum (a_i - e_i) ln(a_i / e_i)`` over bins whose edges are the deciles of ``expected`` (empty bins floored at 0.5 observations). Below 0.1
    the distributions match, 0.1 to 0.25 is a shift worth watching, above 0.25 is a different distribution."""
    e = np.asarray(expected, dtype=float)
    a = np.asarray(actual, dtype=float)
    e, a = e[np.isfinite(e)], a[np.isfinite(a)]
    if len(e) < 50 or len(a) < 20:
        return float("nan")
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    pe = np.histogram(e, edges)[0] / len(e)
    pa = np.histogram(a, edges)[0] / len(a)
    pe, pa = np.maximum(pe, 0.5 / len(e)), np.maximum(pa, 0.5 / len(a))
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def feature_drift(X: pd.DataFrame, recent_days: int = 63, reference_days: int = 756) -> pd.DataFrame:
    """PSI of every feature between the ``reference_days`` dates just before the latest ``recent_days`` and those latest dates, sorted from the most drifted."""
    dates = X.index.get_level_values(0)
    uniq = dates.unique().sort_values()
    if len(uniq) < recent_days + 100:
        raise ValueError("not enough history to compare recent features with a reference period")
    recent = uniq[-recent_days:]
    ref = uniq[-(recent_days + reference_days):-recent_days]
    rows = {c: psi(X.loc[dates.isin(ref), c].to_numpy(), X.loc[dates.isin(recent), c].to_numpy()) for c in X.columns}
    out = pd.Series(rows, name="psi").dropna().sort_values(ascending=False)
    return out.to_frame().assign(status=lambda f: np.where(f["psi"] > PSI_ALERT, "alert", np.where(f["psi"] > PSI_WATCH, "watch", "ok")))


def is_rebalance_day(date: pd.Timestamp, frequency: str) -> bool:
    """Whether trades happen at the close of ``date`` under the experiment's rebalance frequency, judged from the calendar (a live loop cannot see tomorrow): every day; Fridays; the last business day of the month."""
    if frequency == "daily":
        return True
    if frequency == "weekly":
        return date.weekday() == 4
    return (date + pd.offsets.BDay(1)).month != date.month


def forward_record(signals: list[dict], name: str, experiment_id: str, horizon: int, seen_through: str | None = None) -> dict:
    """What the recorded forward signals say, from the registry's rows alone (no data is read): how many, how many have settled, and the realised track record of those that have. A signal is
    ``backfilled`` when it was recorded more than a day after its date (replayed history, not a live forecast)."""
    done = [s for s in signals if s["realised"] is not None]
    out = {"experiment_id": experiment_id, "name": name, "n_signals": len(signals), "n_settled": len(done), "first_signal": signals[0]["signal_date"] if signals else None,
           "last_signal": signals[-1]["signal_date"] if signals else None, "horizon": horizon, "alerts": [],
           "n_backfilled": sum((pd.Timestamp(s["created"]).normalize() - pd.Timestamp(s["signal_date"])).days > 1 for s in signals),
           "n_replay": sum(pd.Timestamp(s["signal_date"]) < pd.Timestamp(seen_through) for s in signals) if seen_through else None, "model_seen_through": seen_through}
    if done:
        r = pd.Series([s["realised"] for s in done], index=pd.to_datetime([s["signal_date"] for s in done]))
        # The realised value of one signal is the return of its book over ITS horizon, and horizons overlap: these numbers describe the forecasts, they are not a daily P&L. No equity curve is built from them
        # (dividing a multi-day return by the horizon and compounding it is wrong); the strategy's equity is built from the committed holdings and the daily prices by the ledger (``ForwardTracker.summary``).
        out.update({"mean_realised": float(r.mean()), "hit_rate": float((r > 0).mean()), "horizon_outcomes": {"n": int(len(r)), "mean": float(r.mean()), "hit_rate": float((r > 0).mean()),
                                                                                                   "note": "return of each signal's book over its own horizon; overlapping, so not independent and not a daily P&L"}})
    if out["n_replay"]:
        out["alerts"].append(f"{out['n_replay']} of {len(signals)} signals are for dates the saved model had already seen (its labels reach {seen_through}): they are replays, not forecasts, and say nothing about live performance.")
    if out["n_backfilled"]:
        out["alerts"].append(f"{out['n_backfilled']} of {len(signals)} signals were recorded after their date (replayed history, not live forecasts).")
    return out


def ledger_view(registry: Registry, experiment_id: str) -> dict:
    """What the prospective ledger holds for an experiment's current strategy version, from the registry alone (no data is loaded): the version and its operating state, the commitments (creation time,
    classification and why, cutoffs, settled versions), the counts of prospective and replay evidence kept apart, and the forecast skill of the settled commitments."""
    v = gv.current_version(registry, experiment_id)
    if v is None:
        return {"strategy_version": None, "note": "no commitment has been written for this experiment yet"}
    vid = v["version_id"]
    rows = pr._rows(registry, vid)                                                                # noqa: SLF001
    sett = pr._settlements(registry, [r["commitment_id"] for r in rows])                          # noqa: SLF001
    return {"strategy_version": vid, "state": gv.state(registry, vid), "counts": pr.counts(registry, vid),
            "commitments": [{"commitment_id": r["commitment_id"], "decision_ts": r["decision_ts"], "created_at": r["created_at"], "classification": r["classification"], "why": r["why"], "data_cutoff": r["data_cutoff"],
                             "fit_cutoff": r["fit_cutoff"], "n_forecasts": len(json.loads(r["forecasts"])), "settled_versions": len(sett[r["commitment_id"]])} for r in rows[-200:]],
            "evaluation": {c: pr.evaluate(registry, vid, classification=c) for c in ("prospective", "replay")}}


class ForwardTracker:
    """Follow one saved experiment forward in time."""

    def __init__(self, registry: Registry, experiment_id: str, config=None, builder=None):
        self.registry, self.id, self.config, self.builder = registry, experiment_id, config, builder
        self.record = registry.get_experiment(experiment_id)
        d = Path(registry.root) / "experiments" / experiment_id
        saved = yaml.safe_load((d / "config.yaml").read_text())
        self.cfg = experiment_config(merge(DEFAULTS, saved), name=saved.get("name"))
        try:                                                                                              # an experiment saved before output contracts existed may carry a rule that misreads its model's output
            check_signal_use(self.cfg["model"]["kind"], task_for(is_classification(self.cfg["dataset"]["target"])), self.cfg["dataset"]["target"], self.cfg["signal"])
        except OutputContractError as exc:
            raise ValueError(f"this experiment cannot be forward-tested: {exc}") from None
        model_path = d / "model.pkl"
        if not model_path.exists():
            raise RegistryError(f"experiment '{experiment_id}' has no saved model: it was run with final_model=False")
        self.model = Model.from_bytes(model_path.read_bytes())
        bundle_path = d / "model.qfb"
        self.bundle = InferenceBundle.load(bundle_path) if bundle_path.exists() else None                 # experiments saved before the bundle existed have none: their schema is checked by the model alone
        self.seen_through = (getattr(self.model, "fit_info_", None) or {}).get("label_end_max")           # the last day the final model's labels reached: it has seen the market through then
        self._version: dict | None = None                                                                 # the strategy version whose commitments this tracker writes (registered on first use)

    # ------------------------------------------------------------------------------------------------------------ one day
    def _dataset(self, bundle, as_of):
        if bundle is None:
            bundle, info = resolve_bundle(self.config, self.cfg["dataset"]["universe"], self.builder)
        else:
            info = {"source": "user"}
        if as_of is not None:
            from .datasets import _slice_bundle

            bundle = _slice_bundle(bundle, slice(None, pd.Timestamp(as_of)))
        return build_dataset(bundle, self.cfg["dataset"], self.cfg["name"], source=info), bundle

    def step(self, bundle=None, as_of=None, capital: float = 100_000.0, _now: float | None = None) -> dict:
        """Produce and save the signal for the last date in the data (or ``as_of``). Returns the signal: scores, target weights, current weights, trades and the paper orders for ``capital``, and the ledger's answer
        (``commitment``: committed as prospective or as a replay, unchanged, refused by a risk limit or a halt, or rejected as a conflict). ``_now`` is a test hook; the API never passes it."""
        sig_cfg = self.cfg["signal"]
        if sig_cfg["method"] in STATEFUL_METHODS or any(sig_cfg.get(k) is not None for k in OVERLAY_KEYS):
            raise ValueError(f"forward testing does not support signal.method '{sig_cfg['method']}' with the cost-aware optimiser, calibrated sizing, a no-trade region or a trade filter: these rules depend on the book actually held "
                             "and on a history of settled outcomes, and a step scores only the last few dates. Forward-test an experiment that uses a rank-based rule, or re-backtest this one")
        ds, bundle = self._dataset(bundle, as_of)
        date = ds.dates[-1]
        h = max(int(sig_cfg.get("holding_period", 1)), 1)
        dates = ds.dates[-h:]
        rows = ds.X[ds.X.index.get_level_values(0).isin(dates)]
        if self.bundle is not None:
            rows, _ = self.bundle.conform(rows)                                                             # a drifted schema is refused here, with the difference named, instead of being scored
        replay = bool(self.seen_through and date < pd.Timestamp(self.seen_through))                     # a model fitted at the close of ``date`` may decide that close: same-close ordering
        if self.model.needs_dataset:
            self.model.bind(ds)
        scores = pd.Series(self.model.predict(rows), index=rows.index).unstack("asset").reindex(index=dates, columns=ds.assets)
        prob = None
        if self.model.task == "classification":
            prob = pd.Series(self.model.predict_proba(rows), index=rows.index).unstack("asset").reindex(index=dates, columns=ds.assets)
        vol = trailing_vol(ds.returns).reindex(dates)
        dist = self.model.predict_extras(rows).get("distribution")
        unc = None if dist is None else dist["std"].unstack("asset").reindex(index=dates, columns=ds.assets)
        weights = build_weights(scores, sig_cfg, vol, confidence_from_scores(scores, prob), regime_series(ds, sig_cfg.get("regime_on_states", [0])), ds.group or ds.asset_class, ds.investable, unc,
                                ds.returns.loc[:date])
        target = weights.iloc[-1]
        past = self.registry.forward_signals(self.id)
        held = pd.Series(past[-1]["weights"], dtype=float).reindex(ds.assets).fillna(0.0) if past else pd.Series(0.0, index=ds.assets)
        rebalance = is_rebalance_day(date, self.cfg["backtest"]["rebalance"])
        new_held = target if (rebalance or not past) else held
        score_row = scores.iloc[-1]
        # The commitment is written (or refused) BEFORE anything is recorded as held: a book that breaches a documented limit, or a version that is halted, never becomes a signal.
        commitment = self._commit(ds, date, score_row, new_held, held, replay, _now)
        if commitment["status"] in ("refused", "rejected_conflict") and not any(str(date.date()) == s["signal_date"] for s in past):
            new_held = held
        trades = (new_held - held)
        turnover = float(trades.abs().sum())
        orders = [{"asset": a, "side": "buy" if d > 0 else "sell", "weight_change": float(d), "notional": float(abs(d) * capital)} for a, d in trades.items() if abs(d) > 1e-4]
        if any(str(date.date()) == s["signal_date"] for s in past) or commitment["status"] in ("refused", "rejected_conflict"):
            sid = None                                                                     # the signal for this date exists (do not log it twice), or the ledger refused the book or the key holds other content
        else:
            sid = self.registry.log_forward_signal(self.id, date, {a: float(w) for a, w in new_held.items() if abs(w) > 1e-8}, {a: float(v) for a, v in score_row.dropna().items()}, turnover)
        return {"experiment_id": self.id, "date": str(date.date()), "signal_id": sid, "rebalance_day": bool(rebalance), "scores": {a: float(v) for a, v in score_row.dropna().items()},
                "target": {a: float(w) for a, w in target.items() if abs(w) > 1e-8}, "held_before": {a: float(w) for a, w in held.items() if abs(w) > 1e-8},
                "weights": {a: float(w) for a, w in new_held.items() if abs(w) > 1e-8}, "turnover": turnover, "orders": orders, "capital": float(capital),
                "replay": replay, "model_seen_through": self.seen_through, "commitment": commitment,
                "note": (f"the model was fitted with labels that end on {self.seen_through}: a signal for {date.date()} is a replay of history it has seen, not a forecast" if replay else None)}

    # ------------------------------------------------------------------------------------------------------------ the prospective ledger
    def _ledger_version(self) -> str:
        """The strategy version whose commitments this tracker writes: the latest registered for the experiment, or a new one (no risk limits) when none exists."""
        if self._version is None:
            self._version = gv.current_version(self.registry, self.id) or gv.register_version(self.registry, self.id, actor="forward-test")
        return self._version["version_id"]

    def _commit(self, ds, date, score_row: pd.Series, book: pd.Series, previous: pd.Series, replay: bool, clock: float | None) -> dict:
        """Write the decision of ``date`` to the immutable ledger. The creation time is the server's; whether it is prospective or a replay is decided by the ledger from the cutoffs, not stated here."""
        vid = self._ledger_version()
        model_sha = hashlib.sha256((Path(self.registry.root) / "experiments" / self.id / "model.pkl").read_bytes()).hexdigest()[:32]
        lag = max(int(self.cfg["dataset"].get("lag", 1)) + int(self.cfg["backtest"].get("extra_lag", 0)), 1)
        arts = {"model": model_sha, "data": pv.dataset_manifest(ds)["data_hash"][:32], "code": pv.source_tree_hash()["sha256"][:32]}
        universe = hashlib.sha256(",".join(sorted(map(str, ds.assets))).encode()).hexdigest()[:12]
        try:
            res = gv.commit_forecast(self.registry, vid, decision_ts=date, universe_version=universe, horizon=int(self.cfg["dataset"]["horizon"]), forecasts={a: float(v) for a, v in score_row.dropna().items()},
                                     weights={a: float(w) for a, w in book.items() if abs(w) > 1e-8}, previous_weights={a: float(w) for a, w in previous.items() if abs(w) > 1e-8}, data_cutoff=ds.dates[-1],
                                     fit_cutoff=self.seen_through, artifact_hashes=arts, model_created=self.record.get("created"), label_start=date + pd.offsets.BDay(lag), actor="forward-test", now=clock)
        except (gv.LimitBreach, gv.StrategyHalted) as exc:
            return {"status": "refused", "reason": str(exc), "strategy_version": vid}
        except pr.CommitmentConflict as exc:
            return {"status": "rejected_conflict", "reason": str(exc), "strategy_version": vid}
        return {**res, "strategy_version": vid}

    def _settle_ledger(self, ds, fwd: pd.DataFrame) -> list[dict]:
        """Append the outcomes of every commitment whose label window has completed in the data. A settled commitment whose outcomes now differ (the data was revised) gets a NEW version with a reason."""
        existing = gv.current_version(self.registry, self.id)
        if existing is None:
            return []
        out = []
        for r in pr._rows(self.registry, existing["version_id"]):                                  # noqa: SLF001
            d = pd.Timestamp(r["decision_ts"])
            if d not in fwd.index or fwd.loc[d].isna().all():
                continue
            outcomes = {a: float(v) for a, v in fwd.loc[d].dropna().items()}
            res = pr.settle(self.registry, r["commitment_id"], outcomes, vintage=str(ds.dates[-1].date()), reason="the realised returns differ from the earlier settlement: the data was revised", actor="forward-test")
            if res["status"] == "settled":
                out.append({"commitment_id": r["commitment_id"], "decision_ts": r["decision_ts"], "version": res["version"], "classification": r["classification"]})
        return out

    def ledger(self, ds=None, volume=None) -> dict:
        """The ledger's view of this experiment's current strategy version (``ledger_view``) and, when ``ds`` is given, the actual daily equity of each stream built from the committed holdings and the daily
        prices."""
        out = ledger_view(self.registry, self.id)
        if ds is None or out.get("strategy_version") is None:
            return out
        vid = out["strategy_version"]
        lag = max(int(self.cfg["dataset"].get("lag", 1)) + int(self.cfg["backtest"].get("extra_lag", 0)), 1)
        out["equity"] = {}
        for cls in ("prospective", "replay"):
            if not out["counts"][cls]["committed"]:
                continue
            eq = pr.equity(self.registry, vid, ds.prices, xe.ExecutionSpec(delay=lag), volume=volume, classification=cls)
            if eq.get("equity") is None:
                out["equity"][cls] = {"eligibility": eq["eligibility"], "reason": eq["reason"]}
                continue
            r = eq["returns"].dropna()
            ok = len(r) >= 20
            out["equity"][cls] = {"eligibility": "applied", "n_days": int(len(r)), "n_decisions": eq["n_decisions"], "total_return": float(eq["equity"].iloc[-1] / eq["equity"].iloc[0] - 1.0) if len(eq["equity"]) > 1 else 0.0,
                                  "sharpe": sharpe_of(r) if ok else None, "reason": "" if ok else f"{len(r)} daily returns: too few for a ratio", "dates": [str(d.date()) for d in eq["equity"].index], "values": [float(v) for v in eq["equity"]],
                                  "reconciliation": eq["reconciliation"], "limitations": eq["limitations"], "note": eq["note"]}
        return out

    # ------------------------------------------------------------------------------------------------------------ settling
    def settle(self, bundle=None, as_of=None) -> list[dict]:
        """Fill in the realised return of every signal whose window (decision date + lag + horizon) has completed in the data: the return of that signal's book over the window less the cost of its trades, and
        the rank IC of its scores against the assets' own returns over the same window."""
        ds, _ = self._dataset(bundle, as_of)
        fwd = forward_return(ds.prices, ds.horizon, ds.lag).reindex(index=ds.dates, columns=ds.assets)
        cost = (float(self.cfg["backtest"]["cost_bps"]) + float(self.cfg["backtest"]["slippage_bps"])) / 1e4
        settled = []
        for s in self.registry.forward_signals(self.id):
            if s["realised"] is not None:
                continue
            d = pd.Timestamp(s["signal_date"])
            if d not in fwd.index:
                continue
            r = fwd.loc[d]
            if r.isna().all():
                continue                                                                        # the window has not completed
            w = pd.Series(s["weights"], dtype=float).reindex(ds.assets).fillna(0.0)
            realised = float((w * r.fillna(0.0)).sum() - cost * float(s["turnover"]))
            self.registry.set_realised(s["id"], realised, ds.dates[-1])
            sc = pd.Series(s["scores"], dtype=float).reindex(ds.assets)
            ic = rank_ic_wide(sc.to_frame().T, r.to_frame().T, min_assets=3).iloc[0] if sc.notna().sum() >= 3 else float("nan")
            settled.append({"signal_date": s["signal_date"], "realised": realised, "ic": None if not np.isfinite(ic) else float(ic)})
        self._settle_ledger(ds, fwd)                                                              # the ledger keeps its own append-only, versioned settlements of the same windows
        return settled

    # ------------------------------------------------------------------------------------------------------------ the record
    def summary(self, bundle=None, as_of=None, monitor: bool = False, volume=None) -> dict:
        """The forward track record, the live IC against the backtest's, drift, and the ledger's view (commitments, prospective and replay evidence apart, actual daily equity from the committed holdings). With
        ``monitor`` the check is also written to the governance audit (and can halt the version on a breach of a documented limit); a plain read writes nothing."""
        sigs = self.registry.forward_signals(self.id)
        report = self.registry.report(self.id)
        out = forward_record(sigs, self.cfg["name"], self.id, int(self.cfg["dataset"]["horizon"]), self.seen_through)
        backtest_ic = report["ic"]["all"].get("mean")
        ds, _ = self._dataset(bundle, as_of)
        fwd = forward_return(ds.prices, ds.horizon, ds.lag).reindex(index=ds.dates, columns=ds.assets)
        live = []
        for s in sigs:
            d = pd.Timestamp(s["signal_date"])
            if d in fwd.index and fwd.loc[d].notna().sum() >= 3 and len(s["scores"]) >= 3:
                sc = pd.Series(s["scores"], dtype=float).reindex(ds.assets)
                live.append(float(rank_ic_wide(sc.to_frame().T, fwd.loc[d].to_frame().T, 3).iloc[0]))
        live = [x for x in live if np.isfinite(x)]
        out["live_ic"] = {"mean": float(np.mean(live)) if live else None, "n": len(live), "backtest_mean": backtest_ic}
        if len(live) >= 20 and backtest_ic is not None:
            se = float(np.std(live, ddof=1) / math.sqrt(len(live) / max(ds.horizon, 1)))
            z = (float(np.mean(live)) - backtest_ic) / se if se > 0 else float("nan")
            out["live_ic"]["z_vs_backtest"] = None if not np.isfinite(z) else float(z)
            if np.isfinite(z) and z < -2.0:
                out["alerts"].append(f"The live information coefficient ({np.mean(live):.3f}) is {abs(z):.1f} standard errors below the backtest's ({backtest_ic:.3f}): the edge may be decaying.")
        drift = feature_drift(ds.X)
        out["drift"] = {"top": [{"feature": f, "psi": float(r_["psi"]), "status": r_["status"]} for f, r_ in drift.head(10).iterrows()], "n_alert": int((drift["psi"] > PSI_ALERT).sum()), "n_watch": int(((drift["psi"] > PSI_WATCH) & (drift["psi"] <= PSI_ALERT)).sum())}
        if out["drift"]["n_alert"]:
            out["alerts"].append(f"{out['drift']['n_alert']} feature(s) have drifted (PSI above {PSI_ALERT}): the model is being asked about inputs unlike those it was fitted on.")
        try:
            oos = pd.read_parquet(Path(self.registry.root) / "experiments" / self.id / "predictions.parquet")["score"]
            recent = pd.Series(self.model.predict(ds.X[ds.X.index.get_level_values(0).isin(ds.dates[-63:])]))
            out["drift"]["prediction_psi"] = psi(oos.to_numpy(), recent.to_numpy())
        except (FileNotFoundError, KeyError, ValueError):
            out["drift"]["prediction_psi"] = None
        out["ledger"] = self.ledger(ds, volume)
        if out["ledger"].get("strategy_version"):
            if out["ledger"]["counts"]["replay"]["committed"]:
                out["alerts"].append(f"{out['ledger']['counts']['replay']['committed']} commitment(s) are replays: they are listed and never count as prospective evidence.")
            if monitor:
                eq = (out["ledger"].get("equity") or {}).get("prospective") or {}
                rets = pd.Series(eq["values"], index=pd.to_datetime(eq["dates"])).pct_change().dropna() if eq.get("values") else None
                last = out["ledger"]["commitments"][-1]["decision_ts"] if out["ledger"]["commitments"] else None
                out["monitoring"] = gv.monitor(self.registry, out["ledger"]["strategy_version"], drift=out["drift"], live_ic=out["live_ic"], returns=rets, last_decision=last)
        return out
