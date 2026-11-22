"""Layers 1-3 of an import, as one workflow: *the scanner discovers, the person confirms, the application stores and analyses only what was approved*.

``ReviewMixin`` is mixed into :class:`~.pipeline.ImportService`. It owns

* the archive of observations (``raw_observations``: every figure the scanner found, mapped or not; never deleted by a later reading);
* the person's decisions (``CONFIRMED``, ``CORRECTED_AND_CONFIRMED``, ``EXCLUDED``, ``UNRESOLVED``, else ``UNCONFIRMED``), kept apart from the scanner's confidence, with a history of every change;
* the plan of what an approval will publish and the construction of the *publishable draft* from confirmed observations only (the rest of the machinery, checks, restatement decisions, provenance, is
  unchanged and runs on that draft);
* the research-ready store of confirmed figures that are not a line of the canonical taxonomy (``custom_observations``).

Rules (each has a test):
1. Scanner confidence never confirms. A missing confidence never confirms. Editing never confirms.
2. A confirmed observation whose value, currency, unit, period, scope or classification is changed goes back to ``UNCONFIRMED``.
3. An inferred period must be accepted by a person before its figure can be confirmed.
4. An observation in another currency than the report's is never merged into the report's figures; it is stored under its own currency.
5. Two different confirmed figures for the same line and period are a conflict a person must resolve.
"""

from __future__ import annotations

import copy
import json
import re
import uuid
from datetime import date

from .. import canonical, manual
from ..errors import FundamentalsError
from ..schema import now_iso
from . import currency as currency_module
from . import observations as obs_module
from . import periods as period_module

UNIT_FACTOR = {"units": 1.0, "thousands": 1e3, "millions": 1e6, "billions": 1e9, "percent": 1.0}
SCAN_FIELDS = ("grp", "label", "value_text", "value", "is_percent", "page", "table_ref", "row_ref", "col_ref", "bbox", "column_header", "column_role", "period_start", "period_end", "period_type",
               "period_guessed", "fiscal_period", "currency", "unit", "scope", "statement", "table_title", "metric_guess", "selected", "guess_reason", "method", "extractor_version", "mapping_version",
               "taxonomy_version")
USER_FIELDS = ("value", "currency", "unit", "period_start", "period_end", "period_type", "fiscal_period", "scope", "metric")
COLUMNS = ["obs_id", "item_id", "obs_key", *SCAN_FIELDS, "extraction_id", "origin", "status", "user_value", "user_currency", "user_unit", "user_period_start", "user_period_end", "user_period_type",
           "user_fiscal_period", "user_scope", "user_metric", "custom_label", "note", "confirmed_at", "history", "published", "orphaned", "created_at", "updated_at"]
SORTS = {"label": "label COLLATE NOCASE", "value": "ABS(COALESCE(user_value, value))", "page": "page", "period": "COALESCE(user_period_end, period_end)", "status": "status", "group": "grp"}
DERIVED = ("income_statement", "balance_sheet", "cash_flow", "equity", "segment", "notes", "kpi", "summary", "other")


def factor(unit: str | None) -> float:
    return UNIT_FACTOR.get((unit or "units").lower(), 1.0)


def _row(r, cols=COLUMNS) -> dict:
    d = dict(zip(cols, r))
    d["history"] = json.loads(d["history"] or "[]")
    d["bbox"] = json.loads(d["bbox"]) if d.get("bbox") else None
    for k in ("is_percent", "period_guessed", "selected", "published", "orphaned"):
        d[k] = bool(d.get(k))
    d["effective"] = effective(d)
    return d


def effective(o: dict) -> dict:
    """What the observation means after a person's corrections: the figure, currency, unit, period, scope and line that would be published."""
    metric = o.get("user_metric") or o.get("metric_guess")
    return {"value": o["user_value"] if o.get("user_value") is not None else o["value"], "currency": o.get("user_currency") or o.get("currency"), "unit": o.get("user_unit") or o.get("unit"),
            "period_start": o.get("user_period_start") or o.get("period_start"), "period_end": o.get("user_period_end") or o.get("period_end"),
            "period_type": o.get("user_period_type") or o.get("period_type"), "fiscal_period": o.get("user_fiscal_period") or o.get("fiscal_period"), "scope": o.get("user_scope") or o.get("scope"),
            "metric": metric, "canonical": metric if metric in canonical.METRICS else None, "label": o.get("custom_label") or o.get("label")}


def _changed(o: dict) -> bool:
    return any(o.get(f"user_{k}") is not None and o.get(f"user_{k}") != (o.get(k) if k != "metric" else o.get("metric_guess")) for k in USER_FIELDS)


def _cumulative_label(e: dict) -> str | None:
    """Which cumulative period a year-to-date figure is: 3 months (Q1), 6 (H1) or 9 (Q3YTD), from its start and end dates."""
    if not e.get("period_start") or not e.get("period_end"):
        return None
    d0, d1 = date.fromisoformat(e["period_start"]), date.fromisoformat(e["period_end"])
    months = round((d1 - d0).days / 30.44)
    return {3: "Q1", 6: "H1", 9: "Q3YTD"}.get(months)


def custom_code(label: str) -> str:
    """A stable code for a figure that is not a canonical line: ``custom:`` and the label in lower case words."""
    return "custom:" + (re.sub(r"[^a-z0-9]+", "_", (label or "").lower()).strip("_")[:60] or "unnamed")


class ReviewMixin:
    # ------------------------------------------------------------------------------------------------------------ the archive (layer 1)
    def _store_observations(self, item_id: int, draft: dict, extraction_id: int | None = None) -> int:
        """Write the scanner's observations. A later reading adds the figures it finds and refreshes the unreviewed ones; **a figure a person has decided about is never replaced or removed**."""
        found = draft.pop("raw_observations", None)
        if not found:
            found = obs_module.from_draft(draft)
        now = now_iso()
        keys = set()
        with self.store.db.transaction() as c:
            have = {r[0]: (r[1], r[2]) for r in c.execute("SELECT obs_key, obs_id, status FROM raw_observations WHERE item_id = ? AND origin = 'scanner'", (int(item_id),)).fetchall()}
            for o in found:
                keys.add(o["obs_key"])
                vals = [o.get(f) if f not in ("grp", "table_ref", "row_ref", "col_ref") else o.get({"grp": "group", "table_ref": "table", "row_ref": "row", "col_ref": "col"}[f]) for f in SCAN_FIELDS]
                vals = [json.dumps(v) if f == "bbox" and v is not None else (int(v) if isinstance(v, bool) else v) for f, v in zip(SCAN_FIELDS, vals)]
                if o["obs_key"] in have:
                    oid, status = have[o["obs_key"]]
                    if status == "UNCONFIRMED":
                        sets = ", ".join(f"{f} = ?" for f in SCAN_FIELDS)
                        c.execute(f"UPDATE raw_observations SET {sets}, orphaned = 0, extraction_id = ?, updated_at = ? WHERE obs_id = ?", [*vals, extraction_id, now, oid])
                    continue
                c.execute(f"INSERT INTO raw_observations (item_id, obs_key, {', '.join(SCAN_FIELDS)}, extraction_id, origin, created_at, updated_at) VALUES (?, ?, {', '.join('?' * len(SCAN_FIELDS))}, ?, 'scanner', ?, ?)",
                          [int(item_id), o["obs_key"], *vals, extraction_id, now, now])
            for key, (oid, status) in have.items():
                if key in keys:
                    continue
                if status == "UNCONFIRMED":
                    c.execute("DELETE FROM raw_observations WHERE obs_id = ?", (oid,))
                else:
                    c.execute("UPDATE raw_observations SET orphaned = 1 WHERE obs_id = ?", (oid,))
        draft["observation_count"] = len(found)
        return len(found)

    def _history(self, c, oid: int, entry: dict) -> None:
        h = json.loads(c.execute("SELECT history FROM raw_observations WHERE obs_id = ?", (oid,)).fetchone()[0] or "[]")
        h.append({"at": now_iso(), **entry})
        c.execute("UPDATE raw_observations SET history = ?, updated_at = ? WHERE obs_id = ?", (json.dumps(h), now_iso(), oid))

    # ------------------------------------------------------------------------------------------------------------ reading them
    def _ensure(self, item_id: int) -> None:
        """An item that was not read from tables (a structured filing, a workbook) or was read before this archive existed gets its observations from its draft, once."""
        with self.store.db.transaction() as c:
            n = c.execute("SELECT COUNT(*) FROM raw_observations WHERE item_id = ?", (int(item_id),)).fetchone()[0]
        if n:
            return
        it = self.item(item_id, full=True)
        draft = it["draft"] or it["reviewed"]                                                            # the scanner's reading, not a person's edited copy
        if draft:
            self._store_observations(item_id, copy.deepcopy(draft))

    def observations(self, item_id: int, group: str | None = None, status: str | None = None, q: str | None = None, unmapped: bool = False, sort: str = "page", descending: bool = False,
                     limit: int = 200, offset: int = 0, column_role: str | None = None) -> dict:
        """A page of an item's observations with the counts the review screens show (by group and by decision)."""
        self.item(item_id)
        self._ensure(item_id)
        where, args = ["item_id = ?"], [int(item_id)]
        if group and group != "all":
            if group == "unmapped":
                unmapped = True
            else:
                where.append("grp = ?")
                args.append(group)
        if status and status != "all":
            where.append("status = ?")
            args.append(status)
        if unmapped:
            where.append("(metric_guess IS NULL AND user_metric IS NULL)")
        if column_role:
            where.append("column_role = ?")
            args.append(column_role)
        if q:
            where.append("(label LIKE ? OR custom_label LIKE ? OR value_text LIKE ? OR column_header LIKE ?)")
            args += [f"%{q}%"] * 4
        order = SORTS.get(sort, "page") + (" DESC" if descending else "")
        with self.store.db.transaction() as c:
            total = c.execute(f"SELECT COUNT(*) FROM raw_observations WHERE {' AND '.join(where)}", args).fetchone()[0]
            rows = c.execute(f"SELECT {', '.join(COLUMNS)} FROM raw_observations WHERE {' AND '.join(where)} ORDER BY {order}, obs_id LIMIT ? OFFSET ?", [*args, int(limit), int(offset)]).fetchall()
        return {"total": total, "rows": [_row(r) for r in rows], **self.observation_counts(item_id)}

    def observation_counts(self, item_id: int) -> dict:
        with self.store.db.transaction() as c:
            by_group = dict(c.execute("SELECT grp, COUNT(*) FROM raw_observations WHERE item_id = ? GROUP BY grp", (int(item_id),)).fetchall())
            by_status = dict(c.execute("SELECT status, COUNT(*) FROM raw_observations WHERE item_id = ? GROUP BY status", (int(item_id),)).fetchall())
            unmapped = c.execute("SELECT COUNT(*) FROM raw_observations WHERE item_id = ? AND metric_guess IS NULL AND user_metric IS NULL", (int(item_id),)).fetchone()[0]
            corrected = c.execute("SELECT COUNT(*) FROM raw_observations WHERE item_id = ? AND status = 'CORRECTED_AND_CONFIRMED'", (int(item_id),)).fetchone()[0]
        by_group["unmapped"] = unmapped
        return {"by_group": by_group, "by_status": {s: by_status.get(s, 0) for s in obs_module.STATUSES}, "corrected": corrected}

    def observation(self, obs_id: int) -> dict:
        with self.store.db.transaction() as c:
            r = c.execute(f"SELECT {', '.join(COLUMNS)} FROM raw_observations WHERE obs_id = ?", (int(obs_id),)).fetchone()
        if r is None:
            raise FundamentalsError("no such observation")
        return _row(r)

    # ------------------------------------------------------------------------------------------------------------ the person's decisions (layer 2)
    def decide(self, item_id: int, actions: list[dict]) -> dict:
        """Apply a list of decisions. Each: ``{obs_id | ids, action, ...}`` with ``action`` one of ``confirm``, ``exclude``, ``unresolve``, ``reset``, ``correct`` (fields ``value``, ``currency``, ``unit``,
        ``period_end``, ``period_start``, ``period_type``, ``fiscal_period``, ``scope``, ``metric`` (a canonical line or any name), ``custom_label``, ``note``; ``confirm: true`` confirms in the same step),
        ``add`` (a figure the scanner missed: ``label``, ``value``, ``period_end``, ... ) or ``confirm_suggested`` (every figure the scanner chose for a line of the draft: an explicit, bulk decision).
        Returns the counts. Nothing is stored in research until :meth:`approve`."""
        it = self.item(item_id)
        if it["status"] in ("saved", "duplicate", "rejected"):
            raise FundamentalsError(f"this item is {it['status']}: its observations cannot be changed")
        self._ensure(item_id)
        touched = 0
        for a in actions or []:
            kind = a.get("action")
            if kind == "add":
                self._add_observation(item_id, a)
                touched += 1
                continue
            ids = a.get("ids") if a.get("ids") is not None else [a.get("obs_id")]
            if kind == "confirm_suggested":
                with self.store.db.transaction() as c:
                    ids = [r[0] for r in c.execute("SELECT obs_id FROM raw_observations WHERE item_id = ? AND selected = 1 AND status = 'UNCONFIRMED'", (int(item_id),)).fetchall()]
                kind = "confirm"
            if kind not in ("confirm", "exclude", "unresolve", "reset", "correct"):
                raise FundamentalsError(f"unknown action {kind!r}: confirm, exclude, unresolve, reset, correct, add or confirm_suggested")
            for oid in ids:
                self._decide_one(item_id, int(oid), kind, a)
                touched += 1
        self._refresh_after_decisions(item_id)
        return {"changed": touched, **self.observation_counts(item_id)}

    def _decide_one(self, item_id: int, oid: int, kind: str, a: dict) -> None:
        o = self.observation(oid)
        if o["item_id"] != int(item_id):
            raise FundamentalsError("that observation belongs to another item")
        before = {"status": o["status"], **o["effective"]}
        sets: dict = {}
        if kind == "correct":
            for k in USER_FIELDS:
                if k in a and a[k] not in (None, ""):
                    v = a[k]
                    if k == "value":
                        try:
                            v = manual.parse_number(v) if isinstance(v, str) else float(v)
                        except Exception:
                            raise FundamentalsError(f"{a[k]!r} is not a number") from None
                    elif k == "currency":
                        v = str(v).strip().upper()
                        if not manual.CURRENCY.match(v):
                            raise FundamentalsError("currency is a three-letter code such as USD or KRW")
                    elif k == "unit":
                        v = str(v).strip().lower()
                        if v not in UNIT_FACTOR:
                            try:
                                float(v)                                                         # a custom multiplier ("1000", "0.001")
                            except ValueError:
                                raise FundamentalsError("unit is units, thousands, millions, billions, percent or a number (a custom multiplier)") from None
                    elif k in ("period_end", "period_start"):
                        try:
                            date.fromisoformat(str(v))
                        except ValueError:
                            raise FundamentalsError(f"{k} is a date such as 2020-12-31") from None
                    sets[f"user_{k}"] = v
            if a.get("custom_label"):
                sets["custom_label"] = str(a["custom_label"]).strip()[:120]
            if a.get("note") is not None:
                sets["note"] = str(a["note"])[:300]
            if not sets:
                raise FundamentalsError("correct: nothing to change")
            if o["status"] in obs_module.PUBLISHABLE:
                sets["status"], sets["confirmed_at"] = "UNCONFIRMED", None                         # an edit is not a confirmation, and it withdraws one
        elif kind == "exclude":
            sets["status"] = "EXCLUDED"
            sets["note"] = str(a.get("note") or o.get("note") or "")[:300]
        elif kind == "unresolve":
            sets["status"] = "UNRESOLVED"
            sets["note"] = str(a.get("note") or o.get("note") or "")[:300]
        elif kind == "reset":
            sets.update({f"user_{k}": None for k in USER_FIELDS})
            sets.update({"status": "UNCONFIRMED", "custom_label": None, "confirmed_at": None})
        if kind in ("confirm",) or (kind == "correct" and a.get("confirm")):
            merged = {**o, **sets}
            eff = effective(merged)
            if eff["value"] is None or eff["period_end"] is None:
                raise FundamentalsError(f"“{o['label']}”: set the period before confirming it")
            if o["period_guessed"] and not (merged.get("user_period_end") or a.get("accept_period")):
                raise FundamentalsError(f"“{o['label']}”: its period was inferred ({eff['period_end'] or 'unknown'}); correct it or accept it explicitly before confirming")
            sets["status"] = "CORRECTED_AND_CONFIRMED" if _changed(merged) else "CONFIRMED"
            sets["confirmed_at"] = now_iso()
            if a.get("accept_period") and not merged.get("user_period_end"):
                sets["user_period_end"] = eff["period_end"]
        with self.store.db.transaction() as c:
            if sets:
                c.execute(f"UPDATE raw_observations SET {', '.join(f'{k} = ?' for k in sets)}, updated_at = ? WHERE obs_id = ?", [*sets.values(), now_iso(), oid])
            after = {**o, **sets}
            self._history(c, oid, {"action": kind if not (kind == "correct" and a.get("confirm")) else "correct+confirm", "before": before, "after": {"status": after["status"], **effective(after)},
                                   "note": a.get("note")})

    def _add_observation(self, item_id: int, a: dict) -> int:
        label = str(a.get("label") or "").strip()
        if not label:
            raise FundamentalsError("add: a figure needs a label")
        try:
            value = manual.parse_number(a["value"]) if isinstance(a.get("value"), str) else float(a["value"])
        except Exception:
            raise FundamentalsError("add: value is a number") from None
        end = a.get("period_end")
        if end:
            try:
                date.fromisoformat(str(end))
            except ValueError:
                raise FundamentalsError("add: period_end is a date such as 2020-12-31") from None
        ccy = (str(a.get("currency") or "").upper() or None)
        if ccy and not manual.CURRENCY.match(ccy):
            raise FundamentalsError("add: currency is a three-letter code")
        key = "m:" + uuid.uuid4().hex[:12]
        now = now_iso()
        metric = a.get("metric") if a.get("metric") in canonical.METRICS else None
        with self.store.db.transaction() as c:
            c.execute(f"INSERT INTO raw_observations (item_id, obs_key, grp, label, value_text, value, period_end, period_start, period_type, period_guessed, fiscal_period, currency, unit, scope, metric_guess, "
                      f"selected, method, origin, custom_label, column_role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, 0, 'manual', 'manual', ?, 'CURRENT_PERIOD', ?, ?)",
                      (int(item_id), key, a.get("group") if a.get("group") in obs_module.GROUPS else "other", label, str(a["value"]), value, end, a.get("period_start"), a.get("period_type"),
                       a.get("fiscal_period"), ccy, a.get("unit") or "units", a.get("scope"), metric, None if metric else label, now, now))
            oid = c.execute("SELECT obs_id FROM raw_observations WHERE item_id = ? AND obs_key = ?", (int(item_id), key)).fetchone()[0]
            self._history(c, oid, {"action": "added by a person", "label": label, "value": value})
        if a.get("confirm"):
            self._decide_one(item_id, oid, "confirm", {})
        return oid

    def _refresh_after_decisions(self, item_id: int) -> None:
        it = self.item(item_id, full=True)
        draft = it["reviewed"] or it["draft"]
        if draft is None:
            return
        batch = self._batch(it["batch_id"])
        pub, _custom, _skipped = self.publishable(item_id)
        self._set(item_id, issues=self._issues(pub, batch) if pub["periods"] else self._issues(draft, batch) + [{"level": "warning", "kind": "nothing_confirmed", "message": "No observation has been confirmed yet: nothing would be stored."}])

    # ------------------------------------------------------------------------------------------------------------ what an approval publishes
    def publishable(self, item_id: int) -> tuple[dict, list[dict], list[dict]]:
        """``(draft, custom, skipped)``: the draft rebuilt from **confirmed** observations only; the confirmed figures that are not canonical lines or not in the report's currency; and the confirmed
        observations that cannot be published, each with the reason."""
        it = self.item(item_id, full=True)
        base = it["reviewed"] or it["draft"]
        if base is None:
            raise FundamentalsError("this item has no extracted draft")
        self._ensure(item_id)
        draft = copy.deepcopy(base)
        draft.pop("raw_observations", None)
        if "conversion" in draft:                                                                       # the observations hold the figures as reported: a conversion is applied again to what is published
            currency_module.restore(draft)
        base_ccy = draft.get("currency")
        base_unit = (draft.get("units") or {}).get("money", "units")
        ye_month = (draft.get("fiscal_year_end") or [None])[0]
        filing = next((p.get("filing_date") for p in draft["periods"] if p.get("filing_date")), None)
        old_sources = {(p["period_end"], c): s for p in draft["periods"] for c, s in (p.get("sources") or {}).items()}
        for p in draft["periods"]:
            p["values"], p["sources"] = {}, {}
        with self.store.db.transaction() as c:
            rows = c.execute(f"SELECT {', '.join(COLUMNS)} FROM raw_observations WHERE item_id = ? AND status IN ('CONFIRMED', 'CORRECTED_AND_CONFIRMED') ORDER BY obs_id", (int(item_id),)).fetchall()
        custom, skipped, claimed = [], [], {}
        for r in rows:
            o = _row(r)
            e = o["effective"]
            where = f"“{o['label']}” ({e['period_end']})"
            if not e["period_end"]:
                skipped.append({"obs_id": o["obs_id"], "reason": f"{where}: no period"})
                continue
            if e["canonical"] and e["period_type"] in ("YTD", "HALF_YEAR_YTD") and canonical.METRICS[e["canonical"]].kind in ("flow", "per_share") and canonical.METRICS[e["canonical"]].statement in ("income", "cashflow"):
                label = _cumulative_label(e)
                if label is None:
                    skipped.append({"obs_id": o["obs_id"], "reason": f"{where}: a year-to-date figure needs its start date (or the number of months) to be stored: set the period start"})
                else:
                    custom.append({**o, "why_custom": f"year-to-date ({label}) figure: kept as reported and used to derive the standalone quarter", "ytd": label})
                continue
            if e["unit"] == "percent" or o["is_percent"] or not e["canonical"] or (e["currency"] and base_ccy and e["currency"] != base_ccy):
                reason = "percentage or ratio" if (e["unit"] == "percent" or o["is_percent"]) else "not a canonical line" if not e["canonical"] else f"{e['currency']} differs from the report's {base_ccy}: stored under its own currency"
                custom.append({**o, "why_custom": reason})
                continue
            m = canonical.METRICS[e["canonical"]]
            money = m.kind not in ("per_share", "shares")
            value = e["value"] * factor(e["unit"]) / factor(base_unit) if money else e["value"]
            period = next((p for p in draft["periods"] if p["period_end"] == e["period_end"] and (not e["fiscal_period"] or p["fiscal_period"] == e["fiscal_period"])
                           and (e["period_type"] in (None, "INSTANT") or p.get("period_type") in (None, e["period_type"]))), None)
            if period is None:
                d0 = date.fromisoformat(e["period_end"])
                ptype = e["period_type"] if e["period_type"] not in (None, "INSTANT") else "ANNUAL"
                fp = e["fiscal_period"] or ("FY" if ptype == "ANNUAL" else f"Q{period_module.fiscal_quarter_of(d0, ye_month)}" if ptype == "QUARTER" else "H1")
                period = {"fiscal_period": fp, "period_end": e["period_end"], "fiscal_year": period_module.fiscal_year_of(d0, ye_month), "filing_date": filing, "currency": base_ccy, "values": {}, "sources": {},
                          "guessed": False, "period_type": ptype, "start_date": e["period_start"], "is_forecast": False}
                draft["periods"].append(period)
            slot = (period["fiscal_period"], period["period_end"], e["canonical"])
            if slot in claimed and abs(claimed[slot][0] - value) > 1e-9 * max(1.0, abs(value)):
                skipped.append({"obs_id": o["obs_id"], "conflict": True, "reason": f"{where}: another confirmed figure for {m.label} in {period['fiscal_period']} differs ({claimed[slot][0]:,.6g} against {value:,.6g}); exclude one"})
                continue
            claimed[slot] = (value, o["obs_id"])
            old = old_sources.get((period["period_end"], e["canonical"])) or {}
            period["values"][e["canonical"]] = value
            period["guessed"] = False
            overlay = {"reported": e["value"], "raw_value": (str(int(e["value"])) if float(e["value"]).is_integer() else str(e["value"])), "unit": o["user_unit"] or old.get("unit") or e["unit"], "currency": e["currency"], "page": o["page"], "label": o["label"],
                                                 "method": o["method"], "statement": o["statement"], "scope": e["scope"], "table": o["table_ref"], "row": o["row_ref"], "col": o["col_ref"], "bbox": o["bbox"],
                                                 "column_role": o["column_role"], "column_header": o["column_header"], "mapping_rule": old.get("mapping_rule"), "checks": old.get("checks"), "confidence": old.get("confidence"), "confirmed": True,
                                                 "edited": o["status"] == "CORRECTED_AND_CONFIRMED", "obs_id": o["obs_id"], "confirmed_at": o["confirmed_at"], "unit_multiplier": factor(e["unit"])}
            period["sources"][e["canonical"]] = {**{"tag": None}, **old, **{k: v for k, v in overlay.items() if v is not None}, "confirmed": True, "edited": o["status"] == "CORRECTED_AND_CONFIRMED"}
        draft["periods"] = sorted([p for p in draft["periods"] if p["values"]], key=lambda p: (p["period_end"], p["fiscal_period"]))
        return draft, custom, skipped

    def plan(self, item_id: int) -> dict:
        """The final-approval screen: counts, what would be committed (canonical lines and custom figures), what cannot be, and the warnings still open."""
        pub, custom, skipped = self.publishable(item_id)
        counts = self.observation_counts(item_id)
        it = self.item(item_id)
        lines = [{"period": f"{p['fiscal_period']} to {p['period_end']}", "metric": c, "label": canonical.METRICS[c].label, "value": v, "currency": p.get("currency"), "unit": pub["units"]["money"],
                  "corrected": bool(p["sources"][c].get("edited")), "obs_id": p["sources"][c].get("obs_id")} for p in pub["periods"] for c, v in p["values"].items()]
        issues = it.get("issues") or []
        return {"counts": {"extracted": sum(counts["by_status"].values()), **{k.lower(): v for k, v in counts["by_status"].items()}}, "will_commit": lines,
                "will_commit_custom": [{"obs_id": c["obs_id"], "label": c["effective"]["label"], "value": c["effective"]["value"], "currency": c["effective"]["currency"], "unit": c["effective"]["unit"],
                                        "period_end": c["effective"]["period_end"], "why": c["why_custom"]} for c in custom],
                "cannot_commit": skipped, "warnings": [i for i in issues if i.get("level") in ("warning", "error")], "approvable": bool(lines or custom) and not any(s.get("conflict") for s in skipped)}

    # ------------------------------------------------------------------------------------------------------------ layer 3 for custom figures
    def _publish_custom(self, item_id: int, security_id: int, custom: list[dict], filing_date: str | None) -> int:
        it = self.item(item_id)
        n = 0
        ytd = [o for o in custom if o.get("ytd")]
        custom = [o for o in custom if not o.get("ytd")]
        if ytd:
            from .. import quarters
            for o in ytd:
                e = o["effective"]
                quarters.save_cumulative(self.store, security_id, e["canonical"], quarters.fiscal_year_of_start(e["period_start"]), o["ytd"], e["period_end"], e["value"] * factor(e["unit"]) / factor("units"), e["currency"], filing_date,
                                         source=it["source_name"] or "import", note=f"import item {item_id}, page {o['page']}")
                n += 1
            quarters.derive(self.store, security_id, write=True)
        with self.store.db.transaction() as c:
            for o in custom:
                e = o["effective"]
                label = e["label"]
                code = e["metric"] if e["metric"] and (e["metric"] in canonical.METRICS or str(e["metric"]).startswith("custom:")) else custom_code(label)
                same = c.execute("SELECT id, value FROM custom_observations WHERE security_id = ? AND metric_code = ? AND period_end = ? AND COALESCE(currency, '') = ? AND COALESCE(scope, '') = ? "
                                 "ORDER BY vintage DESC LIMIT 1", (int(security_id), code, e["period_end"], e["currency"] or "", e["scope"] or "")).fetchone()
                if same is not None and abs(same[1] - e["value"]) < 1e-12 * max(1.0, abs(e["value"])):
                    continue                                                                          # the same figure again (a re-import): nothing to add
                vintage = 0 if same is None else c.execute("SELECT MAX(vintage) + 1 FROM custom_observations WHERE security_id = ? AND metric_code = ? AND period_end = ?", (int(security_id), code, e["period_end"])).fetchone()[0]
                c.execute("INSERT INTO custom_observations (security_id, obs_id, import_item_id, content_hash, metric_code, label, original_label, value, original_value, currency, unit, unit_multiplier, period_start, "
                          "period_end, period_type, fiscal_period, scope, statement, page, table_ref, column_header, extraction_method, extractor_version, taxonomy_version, filing_date, pit_quality, vintage, "
                          "confirmed_at, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                          (int(security_id), o["obs_id"], int(item_id), it["content_hash"], code, label, o["label"], e["value"], o["value"], e["currency"], e["unit"], factor(e["unit"]), e["period_start"],
                           e["period_end"], e["period_type"], e["fiscal_period"], e["scope"], o["statement"], o["page"], o["table_ref"], o["column_header"], o["method"], o["extractor_version"],
                           o["taxonomy_version"], filing_date, "filed" if filing_date else "estimated", vintage, o["confirmed_at"] or now_iso(), now_iso()))
                n += 1
        return n

    def _mark_published(self, item_id: int) -> None:
        with self.store.db.transaction() as c:
            c.execute("UPDATE raw_observations SET published = 1 WHERE item_id = ? AND status IN ('CONFIRMED', 'CORRECTED_AND_CONFIRMED')", (int(item_id),))

    # ------------------------------------------------------------------------------------------------------------ the older review edits, expressed as decisions
    def _relabel_currency(self, item_id: int, old: str | None, new: str | None) -> None:
        """A person states the currency the report's figures are in: the observations that carried the previous one (or none) are corrected to it, recorded in their history."""
        if not new or new == old:
            return
        self._ensure(item_id)
        with self.store.db.transaction() as c:
            ids = [r[0] for r in c.execute("SELECT obs_id FROM raw_observations WHERE item_id = ? AND COALESCE(user_currency, currency, '') IN (?, '') AND status NOT IN ('EXCLUDED')", (int(item_id), old or "")).fetchall()]
        for oid in ids:
            self._decide_one(item_id, oid, "correct", {"currency": new, "note": f"currency stated by a person: {new} (was {old or 'unknown'})"})

    def _sync_edits(self, item_id: int, draft: dict, edited: list, moved: list) -> None:
        if not edited and not moved:
            return
        self._ensure(item_id)
        for (old_fp, old_end), new in moved:
            with self.store.db.transaction() as c:
                ids = [r[0] for r in c.execute("SELECT obs_id FROM raw_observations WHERE item_id = ? AND COALESCE(user_period_end, period_end) = ?", (int(item_id), old_end)).fetchall()]
            for oid in ids:
                if new is None:
                    self._decide_one(item_id, oid, "exclude", {"note": "its period was dropped by a person"})
                elif new[1] != old_end or new[0] != old_fp:
                    self._decide_one(item_id, oid, "correct", {"period_end": new[1], "fiscal_period": new[0], "note": f"period corrected from {old_fp} {old_end}"})
        base_unit = draft["units"]["money"]
        for fp, end, code, number in edited:
            with self.store.db.transaction() as c:
                row = c.execute("SELECT obs_id FROM raw_observations WHERE item_id = ? AND metric_guess = ? AND selected = 1 AND COALESCE(user_period_end, period_end) = ? ORDER BY obs_id LIMIT 1",
                                (int(item_id), code, end)).fetchone()
            if row is None:
                m = canonical.METRICS[code]
                self._add_observation(item_id, {"label": m.label, "value": number, "period_end": end, "fiscal_period": fp, "currency": draft.get("currency"), "unit": base_unit if m.kind not in ("per_share", "shares") else "units",
                                                "metric": code, "group": {"income": "income_statement", "balance": "balance_sheet", "cashflow": "cash_flow"}.get(m.statement, "other")})
            else:
                self._decide_one(item_id, row[0], "correct", {"value": number, "unit": base_unit if canonical.METRICS[code].kind not in ("per_share", "shares") else "units", "note": "value typed by a person"})

    def _accept_periods(self, item_id: int) -> None:
        """A person accepts the dates the importer inferred: the observations remember it, so they can be confirmed."""
        self._ensure(item_id)
        with self.store.db.transaction() as c:
            ids = [r[0] for r in c.execute("SELECT obs_id FROM raw_observations WHERE item_id = ? AND period_guessed = 1 AND period_end IS NOT NULL AND user_period_end IS NULL", (int(item_id),)).fetchall()]
            for oid in ids:
                c.execute("UPDATE raw_observations SET user_period_end = period_end WHERE obs_id = ?", (oid,))
                self._history(c, oid, {"action": "inferred period accepted by a person"})

    def _confirm_lines(self, item_id: int, draft: dict, which) -> None:
        self._ensure(item_id)
        codes = None if which == "all" else set(which)
        with self.store.db.transaction() as c:
            rows = c.execute("SELECT obs_id, metric_guess, user_metric FROM raw_observations WHERE item_id = ? AND selected = 1 AND status = 'UNCONFIRMED'", (int(item_id),)).fetchall()
        for oid, guess, user in rows:
            if codes is None or (user or guess) in codes:
                self._decide_one(item_id, oid, "confirm", {})
        edited = [r[0] for r in self._rows(item_id, "status = 'UNCONFIRMED' AND user_value IS NOT NULL AND origin = 'manual' OR (status = 'UNCONFIRMED' AND user_value IS NOT NULL)")]
        for oid in edited:
            o = self.observation(oid)
            if codes is None or (o["effective"]["metric"] in codes):
                self._decide_one(item_id, oid, "confirm", {})

    def _rows(self, item_id: int, cond: str) -> list:
        with self.store.db.transaction() as c:
            return c.execute(f"SELECT obs_id FROM raw_observations WHERE item_id = ? AND ({cond})", (int(item_id),)).fetchall()

    def _relabel_unit(self, item_id: int, old: str | None, new: str | None) -> None:
        """A person states the unit the report's amounts are in: the observations read in the previous one are corrected to it (percentages keep theirs), each recorded in its history."""
        if not new or new == old:
            return
        self._ensure(item_id)
        with self.store.db.transaction() as c:
            ids = [r[0] for r in c.execute("SELECT obs_id FROM raw_observations WHERE item_id = ? AND is_percent = 0 AND COALESCE(user_unit, unit, 'units') = ? AND status != 'EXCLUDED'", (int(item_id), old or "units")).fetchall()]
        for oid in ids:
            self._decide_one(item_id, oid, "correct", {"unit": new, "note": f"unit stated by a person: {new} (was {old})"})

    META = ("company", "ticker", "exchange", "filing_type", "accounting_standard", "fiscal_year", "fiscal_period", "period_start", "period_end", "publication_date", "scope")

    def _apply_meta(self, item_id: int, draft: dict, meta: dict) -> None:
        """Step 1 of the wizard: the filing's details as a person states them (company, ticker, exchange, filing type, accounting standard, fiscal year and period, period dates, publication date, consolidation scope).
        They are kept with the draft; the publication date becomes the date the figures are public from; a stated scope is given to the observations that had none."""
        if not isinstance(meta, dict):
            raise FundamentalsError("meta must be a dictionary")
        clean = {k: str(v).strip()[:120] for k, v in meta.items() if k in self.META and v not in (None, "")}
        for k in ("publication_date", "period_start", "period_end"):
            if k in clean:
                try:
                    date.fromisoformat(clean[k])
                except ValueError:
                    raise FundamentalsError(f"{k} is a date such as 2020-12-31") from None
        if clean.get("scope") and clean["scope"] not in ("consolidated", "parent", "segment", "unspecified"):
            raise FundamentalsError("scope is consolidated, parent, segment or unspecified")
        draft.setdefault("meta", {}).update(clean)
        if "accounting_standard" in clean:
            draft.setdefault("standard", {})["standard"] = clean["accounting_standard"]
            draft["standard"]["how"] = "stated by a person"
        if "publication_date" in clean:
            for p in draft["periods"]:
                p["filing_date"] = clean["publication_date"]
        if "fiscal_year" in clean and clean["fiscal_year"].isdigit():
            for p in draft["periods"]:
                p["fiscal_year"] = int(clean["fiscal_year"])
        if clean.get("scope"):
            self._ensure(item_id)
            with self.store.db.transaction() as c:
                ids = [r[0] for r in c.execute("SELECT obs_id FROM raw_observations WHERE item_id = ? AND COALESCE(user_scope, scope, 'unspecified') = 'unspecified'", (int(item_id),)).fetchall()]
            for oid in ids:
                self._decide_one(item_id, oid, "correct", {"scope": clean["scope"], "note": "consolidation scope stated by a person"})
        draft.setdefault("corrections", []).append({"kind": "meta", "meta": clean, "at": now_iso()})
