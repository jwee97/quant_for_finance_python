"""The prospective forecast ledger (spec v3, Release 5, P1-24, and section 8.3).

A forecast counts as prospective evidence only if it was committed *before* its outcome could be known **and** the data and the fitted model it used were already available when it was committed. A
backtest replayed at a past ``as_of`` date is a replay, however it is labelled, and never counts. The ledger enforces that by construction:

* **commitments are immutable.** The unique key is ``strategy version | decision timestamp | universe version | horizon``. Committing the same content again is a no-op (idempotent, audited); different
  content under the same key is rejected and audited; nothing is updated or deleted (triggers refuse it);
* **the creation time is the server's**, taken when the commitment is written; a caller cannot back-date it. The record keeps the creation time, the data-availability cutoff, the fit cutoff and the
  hashes of the model, the data and the code, and classifies the commitment ``prospective`` or ``replay``: prospective needs the data cutoff and the fit cutoff at or before the decision, the model
  artifact to exist before the commitment, and the commitment to be made before the first return of the label could be observed;
* **settlements are appended, never written over.** A first settlement records the outcomes as observed; a revision is a new version with a correction reason, and both evaluations are kept;
* **the two streams stay apart.** Forecast skill (a rank IC per settled decision) is evaluated on forecasts; portfolio P&L is built from the committed target weights, executed by the platform's own
  ledger on the actual prices (``equity``), never from dividing a multi-day forecast return by its horizon and compounding it, and never by averaging segment Sharpe ratios.

Elapsed time cannot be manufactured: ``counts`` reports how many distinct decision days have settled prospectively, and the gate asks the adopted policy for a minimum.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Mapping

import pandas as pd

from . import execution as xe
from . import provenance as pv

VERSION = "prospective-v1"
from .r5_schema import PROSPECTIVE as TABLES  # noqa: E402  (the DDL lives with the other Release 5 migrations)


class CommitmentConflict(ValueError):
    """Different content under an existing immutable key."""


class LedgerError(ValueError):
    pass


def _ts(x) -> pd.Timestamp:
    return pd.Timestamp(x)


def commit_key(strategy_version: str, decision_ts, universe_version: str, horizon: int) -> str:
    return f"{strategy_version}|{_ts(decision_ts).isoformat()}|{universe_version}|{int(horizon)}"


def _content_hash(forecasts: Mapping, weights: Mapping | None, data_cutoff, fit_cutoff, artifact_hashes: Mapping) -> str:
    body = {"forecasts": {k: float(v) for k, v in sorted(forecasts.items())}, "weights": None if weights is None else {k: float(v) for k, v in sorted(weights.items())},
            "data_cutoff": None if data_cutoff is None else str(_ts(data_cutoff)), "fit_cutoff": None if fit_cutoff is None else str(_ts(fit_cutoff)),
            "artifacts": {k: v for k, v in sorted(artifact_hashes.items()) if k != "code"}}              # the platform's code hash is recorded with the commitment but is not its content: the same forecast made by a newer build is the same forecast
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:32]


def classify(created: pd.Timestamp, decision_ts: pd.Timestamp, horizon_label_start: pd.Timestamp, data_cutoff, fit_cutoff, model_created) -> tuple[str, str]:
    """``('prospective' | 'replay', why)``. Prospective needs every one of: the commitment made before the first return of the label could be observed; the data cutoff and the fit cutoff at or before the decision;
    the model artifact in existence before the commitment. Anything else is a replay."""
    problems = []
    if created >= horizon_label_start:
        problems.append(f"it was written at {created} which is not before the label's first return day {horizon_label_start}: the outcome was already (partly) observable")
    if data_cutoff is None or _ts(data_cutoff) > decision_ts:
        problems.append("the data-availability cutoff is missing or later than the decision")
    if fit_cutoff is None or _ts(fit_cutoff) > decision_ts:
        problems.append("the fit cutoff is missing or later than the decision")
    if model_created is None or _ts(model_created) > created:
        problems.append("the model artifact did not exist before the commitment (or its creation time is not recorded)")
    return ("replay", "; ".join(problems)) if problems else ("prospective", "committed before the outcome was observable, with data and a fitted model that already existed")


def commit(registry, *, strategy_version: str, decision_ts, universe_version: str, horizon: int, forecasts: Mapping[str, float], weights: Mapping[str, float] | None = None, data_cutoff=None,
           fit_cutoff=None, artifact_hashes: Mapping[str, str] | None = None, model_created=None, label_start=None, actor: str, now: float | None = None) -> dict:
    """Write a commitment. ``label_start`` is the first day whose return the label contains (the session after the fill); it defaults to the day after ``decision_ts``. ``now`` exists for tests only: the
    API never takes it."""
    if not forecasts:
        raise LedgerError("a commitment needs forecasts")
    if not actor:
        raise LedgerError("a commitment records who made it")
    dts = _ts(decision_ts)
    key = commit_key(strategy_version, dts, universe_version, horizon)
    arts = dict(artifact_hashes or {})
    h = _content_hash(forecasts, weights, data_cutoff, fit_cutoff, arts)
    t = time.time() if now is None else float(now)
    created = pd.Timestamp(t, unit="s")
    first_ret = _ts(label_start) if label_start is not None else dts + pd.Timedelta(days=1)
    cls, why = classify(created, dts, first_ret, data_cutoff, fit_cutoff, model_created)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t))
    conflict = None
    with registry.transaction() as c:
        row = c.execute("SELECT commitment_id, content_hash FROM commitments WHERE commit_key=?", (key,)).fetchone()
        if row is not None:
            same = row["content_hash"] == h
            c.execute("INSERT INTO commitment_audit (commit_key, event, content_hash, existing_hash, actor, detail, at) VALUES (?,?,?,?,?,?,?)",
                      (key, "duplicate_identical" if same else "rejected_conflict", h, row["content_hash"], actor, "" if same else "different content under an immutable key", stamp))
            if same:
                return {"commitment_id": row["commitment_id"], "commit_key": key, "status": "unchanged", "content_hash": h}
            conflict = row["content_hash"]                                                         # raised AFTER the transaction commits: a raise inside it would roll the audit row back
        else:
            cid = "cm-" + hashlib.sha256(key.encode()).hexdigest()[:16]
            c.execute("INSERT INTO commitments (commitment_id, commit_key, content_hash, strategy_version, decision_ts, universe_version, horizon, created_at, created_epoch, data_cutoff, fit_cutoff, "
                      "model_created, artifact_hashes, classification, why, forecasts, weights, actor) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                      (cid, key, h, strategy_version, str(dts), universe_version, int(horizon), stamp, t, None if data_cutoff is None else str(_ts(data_cutoff)),
                       None if fit_cutoff is None else str(_ts(fit_cutoff)), None if model_created is None else str(_ts(model_created)), json.dumps(arts, sort_keys=True), cls, why,
                       json.dumps({k: float(v) for k, v in forecasts.items()}), None if weights is None else json.dumps({k: float(v) for k, v in weights.items()}), actor))
            c.execute("INSERT INTO commitment_audit (commit_key, event, content_hash, actor, detail, at) VALUES (?,?,?,?,?,?)", (key, "committed", h, actor, f"{cls}: {why}", stamp))
    if conflict is not None:
        raise CommitmentConflict(f"commitment key '{key}' already holds different content ({conflict}); a commitment is never changed - use a new decision timestamp or strategy version")
    return {"commitment_id": cid, "commit_key": key, "status": "committed", "classification": cls, "why": why, "content_hash": h, "created_at": stamp}


def settle(registry, commitment_id: str, outcomes: Mapping[str, float], *, vintage: str = "first", reason: str | None = None, actor: str, now: float | None = None) -> dict:
    """Append a settlement version. Version 1 records the outcomes as first observed. A later version needs a correction ``reason`` and different outcomes; the earlier versions stay."""
    if not outcomes:
        raise LedgerError("a settlement needs outcomes")
    oh = hashlib.sha256(json.dumps({k: float(v) for k, v in sorted(outcomes.items())}, sort_keys=True).encode()).hexdigest()[:16]
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time() if now is None else now))
    with registry.transaction() as c:
        if c.execute("SELECT 1 FROM commitments WHERE commitment_id=?", (commitment_id,)).fetchone() is None:
            raise LedgerError(f"unknown commitment {commitment_id}")
        last = c.execute("SELECT version, outcome_hash FROM settlements WHERE commitment_id=? ORDER BY version DESC LIMIT 1", (commitment_id,)).fetchone()
        if last is None:
            version, prev = 1, None
        else:
            if last["outcome_hash"] == oh:
                return {"commitment_id": commitment_id, "version": last["version"], "status": "unchanged"}
            if not (reason and reason.strip()):
                raise LedgerError("a revision of a settlement needs a correction reason: the original is kept, and this is a new version")
            version, prev = last["version"] + 1, last["version"]
        c.execute("INSERT INTO settlements (commitment_id, version, outcomes, outcome_hash, vintage, correction_reason, settled_at, actor) VALUES (?,?,?,?,?,?,?,?)",
                  (commitment_id, version, json.dumps({k: float(v) for k, v in outcomes.items()}), oh, vintage, reason if version > 1 else None, stamp, actor))
    return {"commitment_id": commitment_id, "version": version, "previous_version": prev, "status": "settled", "outcome_hash": oh}


def _rows(registry, strategy_version: str, classification: str | None = None) -> list[dict]:
    q = "SELECT * FROM commitments WHERE strategy_version=?" + (" AND classification=?" if classification else "") + " ORDER BY decision_ts"
    with registry._conn() as c:                                                                  # noqa: SLF001
        return [dict(r) for r in c.execute(q, (strategy_version, classification) if classification else (strategy_version,))]


def _settlements(registry, ids: list[str]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {i: [] for i in ids}
    if not ids:
        return out
    with registry._conn() as c:                                                                  # noqa: SLF001
        for r in c.execute(f"SELECT * FROM settlements WHERE commitment_id IN ({','.join('?' * len(ids))}) ORDER BY commitment_id, version", ids):
            out[r["commitment_id"]].append(dict(r))
    return out


def counts(registry, strategy_version: str) -> dict:
    """How much genuine prospective evidence exists: committed, settled, and distinct settled decision days, for prospective and (separately, never added) replay commitments."""
    out: dict = {}
    for cls in ("prospective", "replay"):
        rows = _rows(registry, strategy_version, cls)
        sett = _settlements(registry, [r["commitment_id"] for r in rows])
        done = [r for r in rows if sett[r["commitment_id"]]]
        out[cls] = {"committed": len(rows), "settled": len(done), "distinct_settled_days": len({pd.Timestamp(r["decision_ts"]).normalize() for r in done})}
    out["note"] = "replays are listed so that they are visible; they never count toward prospective evidence or Gate C"
    return out


def _ic(forecasts: dict, outcomes: dict) -> float:
    keys = sorted(set(forecasts) & set(outcomes))
    if len(keys) < 3:
        return float("nan")
    f, o = pd.Series([forecasts[k] for k in keys]).rank(), pd.Series([outcomes[k] for k in keys]).rank()
    return float(f.corr(o)) if f.std() > 0 and o.std() > 0 else float("nan")


def evaluate(registry, strategy_version: str, *, classification: str = "prospective") -> dict:
    """Forecast skill on settled commitments: the rank IC of each decision for the outcomes AS FIRST OBSERVED and, where a revision exists, as revised, kept apart. These are forecast diagnostics - the
    overlapping outcomes of a multi-day horizon are not independent daily portfolios and no P&L is derived from them."""
    rows = _rows(registry, strategy_version, classification)
    sett = _settlements(registry, [r["commitment_id"] for r in rows])
    first, latest, revised = {}, {}, 0
    for r in rows:
        s = sett[r["commitment_id"]]
        if not s:
            continue
        f = json.loads(r["forecasts"])
        first[r["decision_ts"]] = _ic(f, json.loads(s[0]["outcomes"]))
        latest[r["decision_ts"]] = _ic(f, json.loads(s[-1]["outcomes"]))
        revised += int(len(s) > 1)
    a, b = pd.Series(first, dtype=float).dropna(), pd.Series(latest, dtype=float).dropna()
    return {"version": VERSION, "classification": classification, "strategy_version": strategy_version, "n_settled": int(len(first)), "n_revised": revised,
            "ic_first_observed": {"mean": float(a.mean()) if len(a) else None, "n": int(len(a))}, "ic_latest_version": {"mean": float(b.mean()) if len(b) else None, "n": int(len(b))},
            "ic_by_decision_first": {k: float(v) for k, v in a.items()},
            "note": "rank IC per settled decision; horizons overlap, so these are not independent observations and the dependence is the reader's to account for (see inference.hac_mean with horizon)"}


def equity(registry, strategy_version: str, prices: pd.DataFrame, spec: "xe.ExecutionSpec | None" = None, *, volume: pd.DataFrame | None = None, classification: str = "prospective") -> dict:
    """The actual daily equity path of the committed target weights, executed by the platform's ledger on ``prices`` (a decision at one close is filled at a later close), with the reconciliation. This
    is the strategy stream; the forecast stream is ``evaluate``. Empty if there are no committed weights."""
    rows = [r for r in _rows(registry, strategy_version, classification) if r["weights"]]
    if not rows:
        return {"eligibility": "insufficient", "reason": "no committed weights for this strategy version", "equity": None}
    w = pd.DataFrame({pd.Timestamp(r["decision_ts"]).normalize(): json.loads(r["weights"]) for r in rows}).T.reindex(columns=prices.columns).fillna(0.0).sort_index()
    common = w.index.intersection(prices.index)
    full = pd.DataFrame(float("nan"), index=prices.index, columns=prices.columns)
    full.loc[common] = w.loc[common]
    if len(common) == 0:
        return {"eligibility": "insufficient", "reason": "none of the committed decision dates is a date of the prices given", "equity": None}
    res = xe.execute_weights(full, prices, spec or xe.ExecutionSpec(), volume=volume, data_status="prospective")
    return {"eligibility": "applied", "equity": res.equity, "returns": res.returns, "reconciliation": res.reconciliation, "limitations": res.limitations, "n_decisions": int(len(common)), "decisions_without_prices": int(len(w) - len(common)),
            "note": "built from committed holdings, fills and cash; the forecast horizon plays no part in it"}


def provenance_of(strategy_version: str, model_bytes: bytes | None, data_hash: str, code_hash: str | None = None) -> dict:
    """The artifact hashes to put in a commitment."""
    return {"strategy_version": strategy_version, "model": hashlib.sha256(model_bytes).hexdigest()[:32] if model_bytes is not None else "", "data": data_hash, "code": code_hash or pv.source_tree_hash()["sha256"]}


__all__ = ["commit", "settle", "counts", "evaluate", "equity", "classify", "commit_key", "provenance_of", "CommitmentConflict", "LedgerError", "TABLES"]
