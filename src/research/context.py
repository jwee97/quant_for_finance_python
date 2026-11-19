"""The research context: what the modules of a session share, validated.

A context is a plain dict of a few typed fields. Every field is optional: a person may start in any module and fill in only what that module needs. What is given is checked (a ticker looks like a
ticker, a start date precedes the end date, a rate is a plausible rate), unknown fields are refused instead of silently kept, and the result is a new dict that holds only what was understood.

Sharing a context never changes a result that was already produced. A saved artifact keeps a *snapshot* of the context it was made under; :func:`compatibility` then says, field by field, whether
the context the person has now still describes that result, and what differs. The page shows that as a warning with an offer to run again; it never edits the old result to fit.
"""

from __future__ import annotations

import datetime as dt
import math
import re

TICKER = re.compile(r"^[A-Z0-9^][A-Z0-9.^=\-]{0,11}$")
MAX_ASSETS = 40
ASSET_CLASSES = ("equity", "rates", "fixed_income", "credit", "commodity", "real_estate", "crypto", "futures", "options", "mixed", "unknown")
FREQUENCIES = ("daily",)                                                  # the engines work on daily bars; weekly and monthly data are not offered because nothing would honour them
VOL_SOURCES = ("model", "real_chain", "user_file")
FIELDS = ("name", "assets", "asset_class", "start", "end", "frequency", "data_source", "dataset_version", "risk_free", "benchmark", "costs", "strategy", "ml", "regime", "options", "selected_run", "notes")
SNAPSHOT_FIELDS = ("assets", "start", "end", "frequency", "data_source", "dataset_version", "risk_free", "benchmark", "costs")      # what an artifact remembers about how it was made


class ContextError(ValueError):
    """The context is not valid; the message says which field and why, in words a person can act on."""


def _date(value, field: str) -> str:
    if value in (None, ""):
        return ""
    try:
        day = dt.date.fromisoformat(str(value)[:10])
    except ValueError:
        raise ContextError(f"{field} must be a date written YYYY-MM-DD") from None
    if not dt.date(1900, 1, 1) <= day <= dt.date.today() + dt.timedelta(days=1):
        raise ContextError(f"{field} {day} is outside the range 1900 to today")
    return day.isoformat()


def _number(value, field: str, lo: float, hi: float, unit: str = "") -> float | None:
    if value in (None, ""):
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise ContextError(f"{field} must be a number") from None
    if not math.isfinite(x) or not lo <= x <= hi:
        raise ContextError(f"{field} must be between {lo:g} and {hi:g}{unit}")
    return x


def _small_dict(value, field: str, limit: int = 4000) -> dict:
    import json

    if value in (None, {}):
        return {}
    if not isinstance(value, dict):
        raise ContextError(f"{field} must be an object")
    try:
        text = json.dumps(value, allow_nan=False, default=str)
    except (TypeError, ValueError):
        raise ContextError(f"{field} holds a value that cannot be saved (not a finite number or text)") from None
    if len(text) > limit:
        raise ContextError(f"{field} is too large to keep in a context ({len(text)} characters; the limit is {limit})")
    return value


def validate_context(raw, *, sources=None, classes=ASSET_CLASSES) -> dict:
    """The context in ``raw`` checked and normalised. Raises :class:`ContextError`. ``sources`` (names of the data sources the app knows) is checked when given."""
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ContextError("a research context must be an object")
    unknown = sorted(set(raw) - set(FIELDS))
    if unknown:
        raise ContextError(f"unknown context field {unknown[0]!r}; the fields are {', '.join(FIELDS)}")
    out: dict = {}
    if "name" in raw:
        out["name"] = str(raw["name"] or "").strip()[:80]
    if "assets" in raw:
        assets = raw["assets"] or []
        if isinstance(assets, str):
            assets = re.split(r"[,\s]+", assets.strip())
        if not isinstance(assets, (list, tuple)):
            raise ContextError("assets must be a list of tickers")
        clean = []
        for item in assets:
            ticker = str(item).strip().upper()
            if not ticker:
                continue
            if not TICKER.match(ticker):
                raise ContextError(f"{item!r} is not a valid ticker")
            if ticker not in clean:
                clean.append(ticker)
        if len(clean) > MAX_ASSETS:
            raise ContextError(f"at most {MAX_ASSETS} assets in a context")
        out["assets"] = clean
    if "asset_class" in raw:
        cls = str(raw["asset_class"] or "")
        if cls and cls not in classes:
            raise ContextError(f"unknown asset class {cls!r}; choose from {', '.join(classes)}")
        out["asset_class"] = cls
    if "start" in raw:
        out["start"] = _date(raw["start"], "start")
    if "end" in raw:
        out["end"] = _date(raw["end"], "end")
    if out.get("start") and out.get("end") and out["start"] >= out["end"]:
        raise ContextError(f"the start {out['start']} must come before the end {out['end']}")
    if "frequency" in raw:
        freq = str(raw["frequency"] or "daily")
        if freq not in FREQUENCIES:
            raise ContextError(f"frequency {freq!r} is not supported: the engines work on daily data")
        out["frequency"] = freq
    if "data_source" in raw:
        src = str(raw["data_source"] or "")
        if src and sources is not None and src not in sources:
            raise ContextError(f"unknown data source {src!r}; choose from {', '.join(sorted(sources))}")
        out["data_source"] = src
    if "dataset_version" in raw:
        out["dataset_version"] = str(raw["dataset_version"] or "")[:64]
    if "risk_free" in raw:
        out["risk_free"] = _number(raw["risk_free"], "the risk-free rate", -0.02, 0.25)
    if "benchmark" in raw:
        bench = str(raw["benchmark"] or "").strip().upper()
        if bench and not TICKER.match(bench):
            raise ContextError(f"{raw['benchmark']!r} is not a valid benchmark ticker")
        out["benchmark"] = bench
    if "costs" in raw:
        c = _small_dict(raw["costs"], "costs")
        extra = sorted(set(c) - {"cost_bps", "borrow_pct", "fee_per_order"})
        if extra:
            raise ContextError(f"unknown cost field {extra[0]!r}; use cost_bps, borrow_pct or fee_per_order")
        out["costs"] = {k: v for k, v in {"cost_bps": _number(c.get("cost_bps"), "the cost in basis points", 0, 500), "borrow_pct": _number(c.get("borrow_pct"), "the borrow fee in per cent", 0, 50),
                                          "fee_per_order": _number(c.get("fee_per_order"), "the fee per order", 0, 1000)}.items() if v is not None}
    for key in ("strategy", "ml", "regime"):
        if key in raw:
            out[key] = _small_dict(raw[key], key)
    if "options" in raw:
        o = _small_dict(raw["options"], "options")
        res = {}
        if "rate" in o:
            res["rate"] = _number(o["rate"], "the option rate", -0.02, 0.25)
        if "div_yield" in o:
            res["div_yield"] = _number(o["div_yield"], "the dividend yield", 0, 0.25)
        if "vol_source" in o:
            if o["vol_source"] not in VOL_SOURCES:
                raise ContextError(f"vol_source must be one of {', '.join(VOL_SOURCES)}")
            res["vol_source"] = o["vol_source"]
        out["options"] = {k: v for k, v in res.items() if v is not None}
    if "selected_run" in raw:
        run = str(raw["selected_run"] or "")
        if run and not re.fullmatch(r"[0-9a-f]{12}", run):
            raise ContextError("selected_run must be the id of a saved result")
        out["selected_run"] = run
    if "notes" in raw:
        out["notes"] = str(raw["notes"] or "")[:4000]
    return out


def snapshot(context: dict) -> dict:
    """What a result remembers of the context it was made under (only the fields that change what the numbers mean)."""
    return {k: context[k] for k in SNAPSHOT_FIELDS if k in context and context[k] not in (None, "", [], {})}


def _coverage(snap: dict, ctx: dict) -> str | None:
    """Whether the dates the person has now lie inside the dates the result covers; ``None`` if so, else a sentence."""
    a, b, c, d = snap.get("start"), snap.get("end"), ctx.get("start"), ctx.get("end")
    if (c and a and c < a) or (d and b and d > b):
        return f"the result covers {a or 'the first day'} to {b or 'the last day'}, the context asks for {c or 'the first day'} to {d or 'the last day'}"
    return None


def compatibility(context: dict, snap: dict) -> list[dict]:
    """How the current ``context`` differs from the ``snap`` of a saved result, as a list of ``{field, severity, message}`` (empty: the result still describes the context).

    ``severity`` is ``"mismatch"`` when the numbers would be different if the result were made again under the context (other assets, other dates, another data source, other costs) and ``"note"`` when
    the difference is real but does not change the numbers (a benchmark chosen only for the comparison)."""
    issues: list[dict] = []

    def add(field, severity, message):
        issues.append({"field": field, "severity": severity, "message": message})

    if "assets" in context and "assets" in snap and set(context["assets"]) != set(snap["assets"]):
        missing, extra = sorted(set(context["assets"]) - set(snap["assets"])), sorted(set(snap["assets"]) - set(context["assets"]))
        add("assets", "mismatch", "the result was made on " + (", ".join(snap["assets"]) or "no assets") + (f"; the context adds {', '.join(missing)}" if missing else "") + (f" and drops {', '.join(extra)}" if extra else ""))
    window = _coverage(snap, context)
    if window:
        add("dates", "mismatch", window)
    elif (context.get("start") and snap.get("start") and context["start"] != snap["start"]) or (context.get("end") and snap.get("end") and context["end"] != snap["end"]):
        add("dates", "note", f"the context uses a shorter window ({context.get('start') or snap.get('start')} to {context.get('end') or snap.get('end')}) than the result ({snap.get('start')} to {snap.get('end')}); the result is not cut")
    for field, label in (("data_source", "data source"), ("dataset_version", "dataset version"), ("frequency", "frequency")):
        if context.get(field) and snap.get(field) and context[field] != snap[field]:
            add(field, "mismatch", f"the {label} was {snap[field]!r} and is now {context[field]!r}")
    if context.get("costs") is not None and snap.get("costs") is not None and context["costs"] != snap["costs"]:
        add("costs", "mismatch", f"the costs were {snap['costs']} and are now {context['costs']}")
    if context.get("risk_free") is not None and snap.get("risk_free") is not None and abs(context["risk_free"] - snap["risk_free"]) > 1e-9:
        add("risk_free", "mismatch", f"the risk-free rate was {snap['risk_free']:.2%} and is now {context['risk_free']:.2%}; ratios built on excess returns differ")
    if context.get("benchmark") and snap.get("benchmark") and context["benchmark"] != snap["benchmark"]:
        add("benchmark", "note", f"the benchmark was {snap['benchmark']} and is now {context['benchmark']}")
    return issues
