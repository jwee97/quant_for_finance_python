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

import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .config import DEFAULTS, experiment_config, merge
from .datasets import build_dataset, resolve_bundle
from .evaluation import rank_ic_wide, sharpe_of
from .experiments import confidence_from_scores, regime_series, trailing_vol
from .models import Model
from .registry import Registry, RegistryError
from .signals import build_weights
from .targets import forward_return

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


class ForwardTracker:
    """Follow one saved experiment forward in time."""

    def __init__(self, registry: Registry, experiment_id: str, config=None, builder=None):
        self.registry, self.id, self.config, self.builder = registry, experiment_id, config, builder
        self.record = registry.get_experiment(experiment_id)
        d = Path(registry.root) / "experiments" / experiment_id
        saved = yaml.safe_load((d / "config.yaml").read_text())
        self.cfg = experiment_config(merge(DEFAULTS, saved), name=saved.get("name"))
        model_path = d / "model.pkl"
        if not model_path.exists():
            raise RegistryError(f"experiment '{experiment_id}' has no saved model: it was run with final_model=False")
        self.model = Model.from_bytes(model_path.read_bytes())

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

    def step(self, bundle=None, as_of=None, capital: float = 100_000.0) -> dict:
        """Produce and save the signal for the last date in the data (or ``as_of``). Returns the signal: scores, target weights, current weights, trades and the paper orders for ``capital``."""
        ds, bundle = self._dataset(bundle, as_of)
        date = ds.dates[-1]
        sig_cfg = self.cfg["signal"]
        h = max(int(sig_cfg.get("holding_period", 1)), 1)
        dates = ds.dates[-h:]
        rows = ds.X[ds.X.index.get_level_values(0).isin(dates)]
        scores = pd.Series(self.model.predict(rows), index=rows.index).unstack("asset").reindex(index=dates, columns=ds.assets)
        prob = None
        if self.model.task == "classification":
            prob = pd.Series(self.model.predict_proba(rows), index=rows.index).unstack("asset").reindex(index=dates, columns=ds.assets)
        vol = trailing_vol(ds.returns).reindex(dates)
        weights = build_weights(scores, sig_cfg, vol, confidence_from_scores(scores, prob), regime_series(ds, sig_cfg.get("regime_on_states", [0])), ds.group or ds.asset_class, ds.investable)
        target = weights.iloc[-1]
        past = self.registry.forward_signals(self.id)
        held = pd.Series(past[-1]["weights"], dtype=float).reindex(ds.assets).fillna(0.0) if past else pd.Series(0.0, index=ds.assets)
        rebalance = is_rebalance_day(date, self.cfg["backtest"]["rebalance"])
        new_held = target if (rebalance or not past) else held
        trades = (new_held - held)
        turnover = float(trades.abs().sum())
        orders = [{"asset": a, "side": "buy" if d > 0 else "sell", "weight_change": float(d), "notional": float(abs(d) * capital)} for a, d in trades.items() if abs(d) > 1e-4]
        if any(str(date.date()) == s["signal_date"] for s in past):
            sid = None                                                                     # the signal for this date exists: do not log it twice
        else:
            sid = self.registry.log_forward_signal(self.id, date, {a: float(w) for a, w in new_held.items() if abs(w) > 1e-8}, {a: float(v) for a, v in scores.iloc[-1].dropna().items()}, turnover)
        return {"experiment_id": self.id, "date": str(date.date()), "signal_id": sid, "rebalance_day": bool(rebalance), "scores": {a: float(v) for a, v in scores.iloc[-1].dropna().items()},
                "target": {a: float(w) for a, w in target.items() if abs(w) > 1e-8}, "held_before": {a: float(w) for a, w in held.items() if abs(w) > 1e-8},
                "weights": {a: float(w) for a, w in new_held.items() if abs(w) > 1e-8}, "turnover": turnover, "orders": orders, "capital": float(capital)}

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
        return settled

    # ------------------------------------------------------------------------------------------------------------ the record
    def summary(self, bundle=None, as_of=None) -> dict:
        """The forward track record, the live IC against the backtest's, and drift."""
        sigs = self.registry.forward_signals(self.id)
        report = self.registry.report(self.id)
        done = [s for s in sigs if s["realised"] is not None]
        out = {"experiment_id": self.id, "name": self.cfg["name"], "n_signals": len(sigs), "n_settled": len(done), "first_signal": sigs[0]["signal_date"] if sigs else None, "last_signal": sigs[-1]["signal_date"] if sigs else None,
               "horizon": self.cfg["dataset"]["horizon"], "alerts": []}
        if done:
            r = pd.Series([s["realised"] for s in done], index=pd.to_datetime([s["signal_date"] for s in done]))
            per_day = r / max(int(self.cfg["dataset"]["horizon"]), 1)
            out.update({"mean_realised": float(r.mean()), "hit_rate": float((r > 0).mean()), "cumulative": float((1.0 + per_day).prod() - 1.0), "sharpe": sharpe_of(per_day),
                        "series": {"dates": [str(d.date()) for d in r.index], "realised": [float(v) for v in r], "cumulative": [float(v) for v in (1.0 + per_day).cumprod() - 1.0]}})
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
        return out
