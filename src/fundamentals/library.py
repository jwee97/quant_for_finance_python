"""The alpha library and the graveyard: what has been tried, what survived, what each survivor's state is, and whether it still works.

* **The graveyard** keeps every idea that failed, with the stage it failed at and why, and is never emptied. Its purpose is the opposite of tidiness: the same formula under the same conditions is
  never tested twice, and the number of ideas tried (which every later significance statement depends on) is never forgotten.
* **The library** keeps the survivors with their evidence, under a status:

    ``Candidate``  passed every screen, including a segment of dates that was never used to choose it. Not yet trusted.
    ``Validated``  a candidate whose out-of-sample result also survives the correction for the number of ideas tried (or one a person approved, with the reason recorded).
    ``Production`` a validated alpha someone has decided to use.
    ``Decaying``   a validated or production alpha whose recent behaviour has weakened (set by the monitor).
    ``Retired``    no longer used. ``Rejected``: a candidate someone turned down.

  Status changes follow a fixed table of allowed moves and every one is logged with its reason. Metadata only is stored (the expression, the conditions, a few dozen numbers): the panels an
  experiment produces are recomputed from the database when needed, never kept, so the database does not grow with every idea tried.
* **The monitor** recomputes an alpha on the latest data and compares its recent information coefficient with the one it earned when it was found.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from . import expr as X
from . import transforms
from .errors import FundamentalsError
from .evaluate import PERIODS_PER_YEAR, ic_stats, rank_ic
from .jsonutil import clean
from .research import FREQ_DAYS, ResearchContext, UniverseSpec, horizon_days
from .schema import now_iso
from .store import FundamentalsStore

STATUSES = ("Candidate", "Validated", "Production", "Decaying", "Retired", "Rejected")
TRANSITIONS = {
    "Candidate": ("Validated", "Rejected", "Retired"),
    "Validated": ("Production", "Decaying", "Retired"),
    "Production": ("Decaying", "Retired", "Validated"),
    "Decaying": ("Production", "Validated", "Retired"),
    "Retired": (),
    "Rejected": (),
}
MONITOR_STATES = ("Healthy", "Weakening", "Decaying", "Broken")
VALIDATE_MIN_DSR = 0.90
VALIDATE_MAX_Q = 0.10
VALIDATE_MIN_OOS_T = 1.0


def settings_hash(settings: dict) -> str:
    """The identity of the conditions a formula was tested under: universe, grid, horizon, transformations, costs and the date blocks. The direction found for a formula is a result of the test,
    not a condition of it, so it is left out."""
    conditions = {k: v for k, v in settings.items() if k != "direction"}
    return hashlib.sha256(json.dumps(clean(conditions), sort_keys=True, default=str, separators=(",", ":")).encode()).hexdigest()[:12]


def alpha_id_for(expression_hash: str, settings_digest: str) -> str:
    return f"A{expression_hash[:8]}-{settings_digest[:4]}"


class AlphaLibrary:
    def __init__(self, store: FundamentalsStore):
        self.store = store

    # ------------------------------------------------------------------------------------------------------------ what has been seen
    def seen(self, expression_hash: str, settings_digest: str) -> dict | None:
        """Where this formula under these conditions already is: ``{'where': 'library'|'graveyard', ...}`` or ``None``."""
        with self.store.db.transaction() as conn:
            row = conn.execute("SELECT alpha_id, status FROM alpha_library WHERE expression_hash = ? AND settings_hash = ?", (expression_hash, settings_digest)).fetchone()
            if row:
                return {"where": "library", "alpha_id": row[0], "status": row[1]}
            row = conn.execute("SELECT reason_rejected, stage, date_tested FROM alpha_graveyard WHERE expression_hash = ? AND settings_hash = ?", (expression_hash, settings_digest)).fetchone()
        return {"where": "graveyard", "reason": row[0], "stage": row[1], "date_tested": row[2]} if row else None

    def same_idea(self, expression: dict, settings_digest: str | None = None) -> list[dict]:
        """Library entries and graveyard rows that are the same idea as ``expression`` in another spelling (``rank(x)``, ``zscore(x)`` and ``neg(x)`` are ``x``), under any conditions or the given ones."""
        key = X.signal_key(expression)
        found = []
        with self.store.db.transaction() as conn:
            rows = conn.execute("SELECT alpha_id, expression, settings_hash, status FROM alpha_library").fetchall()
            gone = conn.execute("SELECT expression, settings_hash, reason_rejected, stage FROM alpha_graveyard").fetchall()
        for r in rows:
            if settings_digest in (None, r[2]) and _safe_key(r[1]) == key:
                found.append({"where": "library", "alpha_id": r[0], "status": r[3], "expression": X.to_string(X.loads(r[1]))})
        for r in gone:
            if settings_digest in (None, r[1]) and _safe_key(r[0]) == key:
                found.append({"where": "graveyard", "reason": r[2], "stage": r[3], "expression": X.to_string(X.loads(r[0]))})
        return found

    # ------------------------------------------------------------------------------------------------------------ writing
    def add_graveyard(self, expression: dict, settings: dict, reason: str, stage: str, metrics: dict | None = None, run_id: str | None = None) -> bool:
        """Remember a failure. Returns False if the same formula under the same conditions was already there (nothing is overwritten)."""
        digest, ehash = settings_hash(settings), X.expression_hash(expression)
        m = metrics or {}
        with self.store.db.transaction() as conn:
            cur = conn.execute("INSERT OR IGNORE INTO alpha_graveyard (expression_hash, settings_hash, expression, settings, reason_rejected, stage, ic, oos_ic, sharpe, date_tested, run_id) "
                               "VALUES (?,?,?,?,?,?,?,?,?,?,?)", (ehash, digest, X.dumps(expression), json.dumps(clean(settings)), str(reason)[:300], stage, _num(m.get("ic")), _num(m.get("oos_ic")),
                                                                  _num(m.get("sharpe")), now_iso(), run_id))
        return cur.rowcount > 0

    def add_candidate(self, expression: dict, settings: dict, metrics: dict, name: str | None = None, description: str = "", status: str = "Candidate", reason: str = "") -> str:
        """Add a survivor (or return the id it already has). ``metrics`` carries the numbers the table columns mirror (``ic``, ``icir``, ``sharpe`` ...) and anything else worth keeping."""
        if status not in STATUSES:
            raise FundamentalsError(f"status must be one of {STATUSES}")
        digest, ehash = settings_hash(settings), X.expression_hash(expression)
        alpha_id = alpha_id_for(ehash, digest)
        stamp = now_iso()
        text = X.to_string(X.canonical(expression))
        with self.store.db.transaction() as conn:
            exists = conn.execute("SELECT 1 FROM alpha_library WHERE alpha_id = ?", (alpha_id,)).fetchone()
            if exists:
                return alpha_id
            conn.execute(
                "INSERT INTO alpha_library (alpha_id, name, expression, expression_hash, settings_hash, settings, description, created_at, updated_at, universe, rebalance_frequency, forward_horizon, "
                "ic, icir, sharpe, sortino, max_drawdown, turnover, oos_ic, oos_sharpe, complexity, fitness, q_value, deflated_sharpe, status, status_reason, metrics) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (alpha_id, name or text, X.dumps(expression), ehash, digest, json.dumps(clean(settings)), description, stamp, stamp, json.dumps(clean(settings.get("universe", {}))), settings.get("freq"),
                 horizon_days(settings.get("horizon", "1m")), _num(metrics.get("ic")), _num(metrics.get("icir")), _num(metrics.get("sharpe")), _num(metrics.get("sortino")),
                 _num(metrics.get("max_drawdown")), _num(metrics.get("turnover")), _num(metrics.get("oos_ic")), _num(metrics.get("oos_sharpe")), _num(metrics.get("complexity")),
                 _num(metrics.get("fitness")), _num(metrics.get("q_value")), _num(metrics.get("deflated_sharpe")), status, reason or "passed every screen", json.dumps(clean(metrics))))
            conn.execute("INSERT INTO alpha_status_log (alpha_id, at, from_status, to_status, reason, forced) VALUES (?,?,?,?,?,0)", (alpha_id, stamp, None, status, reason or "added"))
        return alpha_id

    def set_status(self, alpha_id: str, new_status: str, reason: str, force: bool = False) -> dict:
        """Move an alpha to ``new_status`` if the table of allowed moves permits it. ``force`` allows any move away from Retired or Rejected only if the reason says why; it is recorded as forced."""
        if new_status not in STATUSES:
            raise FundamentalsError(f"status must be one of {STATUSES}")
        if not str(reason).strip():
            raise FundamentalsError("a change of status needs a reason")
        row = self.get(alpha_id)
        current = row["status"]
        if new_status == current:
            return row
        if new_status not in TRANSITIONS[current] and not force:
            raise FundamentalsError(f"{alpha_id} is {current}: it cannot go straight to {new_status} (allowed: {', '.join(TRANSITIONS[current]) or 'nothing'})")
        with self.store.db.transaction() as conn:
            conn.execute("UPDATE alpha_library SET status = ?, status_reason = ?, updated_at = ? WHERE alpha_id = ?", (new_status, str(reason)[:300], now_iso(), alpha_id))
            conn.execute("INSERT INTO alpha_status_log (alpha_id, at, from_status, to_status, reason, forced) VALUES (?,?,?,?,?,?)",
                         (alpha_id, now_iso(), current, new_status, str(reason)[:300], 1 if new_status not in TRANSITIONS[current] else 0))
        return self.get(alpha_id)

    def validate(self, alpha_id: str, force_reason: str | None = None) -> dict:
        """Candidate -> Validated if the evidence holds: out-of-sample information coefficient of the right sign and at least ``VALIDATE_MIN_OOS_T``, and either a false-discovery q-value of at most
        ``VALIDATE_MAX_Q`` or a deflated Sharpe probability of at least ``VALIDATE_MIN_DSR``. Otherwise it says which test failed. A person may approve it anyway with a written reason, recorded as forced."""
        row = self.get(alpha_id)
        m = row["metrics"]
        problems = []
        if (m.get("oos_t") is None) or m["oos_t"] < VALIDATE_MIN_OOS_T:
            problems.append(f"out-of-sample t-statistic {m.get('oos_t')} is below {VALIDATE_MIN_OOS_T}")
        q, dsr = row.get("q_value"), row.get("deflated_sharpe")
        if not ((q is not None and q <= VALIDATE_MAX_Q) or (dsr is not None and dsr >= VALIDATE_MIN_DSR)):
            problems.append(f"after allowing for the {m.get('n_tested', 'unknown number of')} ideas tried, neither the false-discovery q-value ({q}) nor the deflated Sharpe probability ({dsr}) is convincing")
        if problems and not force_reason:
            raise FundamentalsError("not validated: " + "; ".join(problems))
        reason = "evidence holds after the correction for the number of ideas tried" if not problems else f"approved by a person despite: {'; '.join(problems)}. Reason: {force_reason}"
        with self.store.db.transaction() as conn:
            conn.execute("UPDATE alpha_library SET status = 'Validated', status_reason = ?, updated_at = ? WHERE alpha_id = ?", (reason[:300], now_iso(), alpha_id))
            conn.execute("INSERT INTO alpha_status_log (alpha_id, at, from_status, to_status, reason, forced) VALUES (?,?,?,?,?,?)",
                         (alpha_id, now_iso(), row["status"], "Validated", reason[:300], 1 if problems else 0))
        return self.get(alpha_id)

    # ------------------------------------------------------------------------------------------------------------ reading
    def get(self, alpha_id: str) -> dict:
        with self.store.db.transaction() as conn:
            row = conn.execute("SELECT * FROM alpha_library WHERE alpha_id = ?", (alpha_id,)).fetchone()
        if row is None:
            raise FundamentalsError(f"no alpha called {alpha_id!r} in the library")
        return _row(row)

    def list(self, status: str | None = None, order_by: str = "fitness", limit: int = 200) -> list[dict]:
        if order_by not in ("fitness", "ic", "icir", "sharpe", "oos_ic", "oos_sharpe", "created_at", "turnover", "complexity"):
            raise FundamentalsError("cannot order the library by that")
        sql = "SELECT * FROM alpha_library" + (" WHERE status = ?" if status else "") + f" ORDER BY {order_by} DESC, alpha_id LIMIT ?"
        with self.store.db.transaction() as conn:
            rows = conn.execute(sql, ([status] if status else []) + [int(limit)]).fetchall()
        return [_row(r) for r in rows]

    def graveyard(self, limit: int = 200, stage: str | None = None) -> list[dict]:
        sql = "SELECT * FROM alpha_graveyard" + (" WHERE stage = ?" if stage else "") + " ORDER BY id DESC LIMIT ?"
        with self.store.db.transaction() as conn:
            rows = conn.execute(sql, ([stage] if stage else []) + [int(limit)]).fetchall()
        return [{**dict(r), "text": _text(r["expression"])} for r in rows]

    def counts(self) -> dict:
        with self.store.db.transaction() as conn:
            by_status = {r[0]: int(r[1]) for r in conn.execute("SELECT status, COUNT(*) FROM alpha_library GROUP BY status")}
            dead = int(conn.execute("SELECT COUNT(*) FROM alpha_graveyard").fetchone()[0])
            by_stage = {r[0]: int(r[1]) for r in conn.execute("SELECT COALESCE(stage, '?'), COUNT(*) FROM alpha_graveyard GROUP BY stage")}
        return {"library": by_status, "graveyard": dead, "graveyard_by_stage": by_stage, "tried": sum(by_status.values()) + dead}

    def status_log(self, alpha_id: str) -> list[dict]:
        with self.store.db.transaction() as conn:
            rows = conn.execute("SELECT at, from_status, to_status, reason, forced FROM alpha_status_log WHERE alpha_id = ? ORDER BY id", (alpha_id,)).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------------------------------------------------ the monitor
    def monitor_history(self, alpha_id: str, limit: int = 60) -> list[dict]:
        with self.store.db.transaction() as conn:
            rows = conn.execute("SELECT * FROM alpha_monitor WHERE alpha_id = ? ORDER BY id DESC LIMIT ?", (alpha_id, int(limit))).fetchall()
        return [{**dict(r), "detail": json.loads(r["detail"]) if r["detail"] else {}} for r in rows]

    def monitor(self, alpha_id: str, ctx: ResearchContext | None = None, recent: int | None = None) -> dict:
        """Recompute the alpha on the data now in the database and judge it: ``Healthy``, ``Weakening``, ``Decaying`` or ``Broken``.

        The recent information coefficient (the last ``recent`` dates, a year by default) is compared with the one the alpha earned in its discovery sample. ``Decaying`` means the recent IC has
        turned zero or negative over a year, or has fallen below a quarter of the original over two; ``Broken`` means it has reversed with evidence (a t-statistic of 2 against the original
        direction), or the alpha can no longer be computed for most of the universe. A validated or production alpha judged Decaying or Broken is moved to ``Decaying`` automatically.
        """
        row = self.get(alpha_id)
        s = row["settings"]
        expression = X.loads(row["expression"])
        try:
            ctx = ctx or ResearchContext(self.store, UniverseSpec.from_dict(s.get("universe")), freq=s.get("freq", "M"), entry_lag=int(s.get("entry_lag", 1)))
            signal = build_signal(ctx, expression, s)
        except FundamentalsError as error:
            return self._record_monitor(row, "Broken", {"error": str(error)[:200]}, None)
        h, step, ppy = horizon_days(s.get("horizon", "1m")), FREQ_DAYS[ctx.freq], PERIODS_PER_YEAR[ctx.freq]
        ic = rank_ic(signal, ctx.forward(s.get("horizon", "1m")).reindex(signal.index), int(s.get("min_assets", 20)))
        sign = float(row["metrics"].get("direction", 1.0)) or 1.0
        usable_ends = ic.index[-1] if len(ic) else None
        n_recent = recent or (12 if ctx.freq == "M" else (52 if ctx.freq == "W" else 4))
        if len(ic) < 2 * n_recent:
            return self._record_monitor(row, "Weakening", {"note": f"only {len(ic)} dates of IC: too few to compare a recent window with the past"}, usable_ends)
        recent_ic, past_ic = ic.iloc[-n_recent:], ic.iloc[:-n_recent]
        recent_two_years = ic.iloc[-2 * n_recent:]
        original = float(row["ic"] or row["metrics"].get("ic") or past_ic.mean()) * sign if row["ic"] is not None else float(past_ic.mean()) * sign
        original = abs(original) if original else float(abs(past_ic.mean()))
        r_stats = ic_stats(recent_ic * sign, h, step, ppy)
        two = float((recent_two_years * sign).mean())
        coverage_now = float(signal.iloc[-n_recent:].where(ctx.investable().reindex_like(signal).iloc[-n_recent:]).notna().sum(axis=1).mean())
        coverage_then = float(signal.iloc[:-n_recent].where(ctx.investable().reindex_like(signal).iloc[:-n_recent]).notna().sum(axis=1).mean())
        mean_recent = float((recent_ic * sign).mean())
        if (r_stats.get("t_stat") is not None and np.isfinite(r_stats["t_stat"]) and r_stats["t_stat"] <= -2.0) or (coverage_then > 0 and coverage_now < 0.3 * coverage_then):
            state = "Broken"
        elif mean_recent <= 0 or (original > 0 and two < 0.25 * original):
            state = "Decaying"
        elif original > 0 and mean_recent < 0.5 * original:
            state = "Weakening"
        else:
            state = "Healthy"
        detail = {"recent_ic": mean_recent, "recent_t": r_stats.get("t_stat"), "original_ic": original, "two_year_ic": two, "recent_window": n_recent, "coverage_now": coverage_now,
                  "coverage_before": coverage_then, "ratio": (mean_recent / original) if original else None}
        return self._record_monitor(row, state, detail, usable_ends, recent_ic=mean_recent, historical=original, coverage=coverage_now)

    def _record_monitor(self, row: dict, state: str, detail: dict, as_of, recent_ic: float | None = None, historical: float | None = None, coverage: float | None = None) -> dict:
        with self.store.db.transaction() as conn:
            conn.execute("INSERT INTO alpha_monitor (alpha_id, checked_at, as_of, status, recent_ic, historical_ic, recent_sharpe, recent_spread, coverage, turnover, detail) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                         (row["alpha_id"], now_iso(), None if as_of is None else pd.Timestamp(as_of).strftime("%Y-%m-%d"), state, _num(recent_ic), _num(historical), None, None, _num(coverage), None,
                          json.dumps(clean(detail))))
        moved = None
        if state in ("Decaying", "Broken") and row["status"] in ("Validated", "Production"):
            moved = self.set_status(row["alpha_id"], "Decaying", f"the monitor judged it {state}: " + ", ".join(f"{k}={v}" for k, v in clean(detail).items() if k in ("recent_ic", "original_ic", "recent_t", "error")))["status"]
        return {"alpha_id": row["alpha_id"], "monitor": state, "detail": clean(detail), "status": moved or row["status"], "as_of": None if as_of is None else pd.Timestamp(as_of).strftime("%Y-%m-%d")}


# ------------------------------------------------------------------------------------------------------------------ helpers
def build_signal(ctx: ResearchContext, expression: dict, settings: dict) -> pd.DataFrame:
    """The signal panel an expression and its stored conditions describe: the formula, the pipeline of transformations, then the sign found in the discovery sample."""
    raw = X.evaluate_panel(expression, ctx)
    out = transforms.apply(raw, settings.get("steps", [{"op": "winsorize", "pct": 1.0}, {"op": "zscore"}]), ctx)
    return out * (1.0 if float(settings.get("direction", 1)) >= 0 else -1.0)


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _safe_key(text: str) -> str | None:
    try:
        return X.signal_key(X.loads(text))
    except FundamentalsError:
        return None


def _text(expression_json: str) -> str:
    try:
        return X.to_string(X.loads(expression_json))
    except FundamentalsError:
        return expression_json


def _row(r) -> dict:
    d = dict(r)
    d["settings"] = json.loads(d["settings"]) if d.get("settings") else {}
    d["metrics"] = json.loads(d["metrics"]) if d.get("metrics") else {}
    d["text"] = _text(d["expression"])
    return d


__all__ = ["AlphaLibrary", "STATUSES", "TRANSITIONS", "MONITOR_STATES", "settings_hash", "alpha_id_for", "build_signal"]
