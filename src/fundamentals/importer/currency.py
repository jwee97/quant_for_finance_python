"""Reporting currency at import: detect-convert-or-block.

A report is read in the currency it states ("¥", "HK$'000", "RMB", an XBRL unit). A company's statements must stay in one currency, or every ratio and every study that reads them compares
unlike amounts. So an import whose currency differs from the one already held is **blocked** (see :mod:`reconcile`) until the person either converts it or says the change is real.

**Converting.** The reviewer picks the currency to store in. Exchange rates come from the ``fx_rates`` table (units of a currency per US dollar); a rate that is missing for the dates needed is
fetched from Yahoo Finance (``<CCY>USD=X``, the same source and table the prices use) and kept for next time. A currency is converted through the dollar. A flow (revenue, net income, cash flow, earnings
per share) is converted at the **average** rate over its period, a stock (assets, debt, equity) at the rate on the **period end**, a count of shares not at all. If no rate can be had the
conversion is refused with the reason, and nothing is stored in a guessed one.

Every converted figure keeps what the report said: its provenance names the original currency, the rate and whether it was an average or a closing rate. The first reading is kept in the draft
(``pre_conversion``), so choosing "as reported" again restores it.
"""

from __future__ import annotations

import calendar
from datetime import date, timedelta

import pandas as pd

from .. import canonical
from ..errors import FundamentalsError

TOLERANCE_DAYS = 10                                                                     # a closing rate older than this is not used (weekends and holidays are the usual gap)
PERIOD_MONTHS = {"FY": 12, "H1": 6, "H2": 6, "Q1": 3, "Q2": 3, "Q3": 3, "Q4": 3}
SKIP_KINDS = ("shares",)


def _months_back(end: date, months: int) -> date:
    y, m = divmod(end.year * 12 + end.month - 1 - months, 12)
    m += 1
    return date(y, m, min(end.day, calendar.monthrange(y, m)[1])) + timedelta(days=1)


class Rates:
    """Currency per US dollar by date, from the database and, for what is missing, from Yahoo Finance."""

    def __init__(self, store, fetch=None):
        self.store, self.fetch = store, fetch
        self.failed: dict[str, str] = {}
        self._series: dict[str, pd.Series] = {}
        self.fetched: list[str] = []

    def per_usd(self, ccy: str) -> pd.Series | None:
        ccy = ccy.upper()
        if ccy == "USD":
            return None
        if ccy not in self._series:
            frame = self.store.fx()
            frame = frame[(frame["base"] == "USD") & (frame["quote"] == ccy)]
            self._series[ccy] = pd.Series(frame["rate"].to_numpy(float), index=pd.DatetimeIndex(frame["date"])).sort_index()
        return self._series[ccy]

    def ensure(self, currencies, start: date, end: date) -> None:
        """Fetch the rates of any currency that the database does not cover from ``start`` to ``end``."""
        need = []
        for c in sorted({c.upper() for c in currencies} - {"USD"}):
            s = self.per_usd(c)
            if s is None or s.empty or s.index.min() > pd.Timestamp(start + timedelta(days=TOLERANCE_DAYS)) or s.index.max() < pd.Timestamp(min(end, date.today()) - timedelta(days=TOLERANCE_DAYS + 5)):
                need.append(c)
        if not need:
            return
        from .. import prices
        fetch = self.fetch or prices.yahoo_download
        try:
            report = prices.load_fx(self.store, (start - timedelta(days=15)).isoformat(), fetch, currencies=need)
        except Exception as error:                                                       # no network, a change at the source
            for c in need:
                self.failed[c] = f"{type(error).__name__}: {str(error)[:100]}"
            return
        self.failed.update(report.get("failed", {}))
        self.fetched += report.get("loaded", [])
        for c in need:
            self._series.pop(c, None)

    def _pair(self, frm: str, to: str) -> pd.Series | None:
        """Units of ``to`` per one unit of ``frm``, daily; None when either side has no rate."""
        a, b = self.per_usd(frm), self.per_usd(to)                                        # frm per USD, to per USD
        if frm.upper() == to.upper():
            return None
        if frm.upper() == "USD":
            return b
        if a is None or a.empty:
            return None
        if to.upper() == "USD":
            return 1.0 / a
        if b is None or b.empty:
            return None
        both = pd.concat([a, b], axis=1, keys=["a", "b"]).ffill().dropna()
        return both["b"] / both["a"]

    def closing(self, frm: str, to: str, day: date) -> float | None:
        s = self._pair(frm, to)
        if s is None or s.empty:
            return None
        upto = s[: pd.Timestamp(day)]
        if upto.empty or pd.Timestamp(day) - upto.index[-1] > pd.Timedelta(days=TOLERANCE_DAYS):
            return None
        return float(upto.iloc[-1])

    def average(self, frm: str, to: str, start: date, end: date) -> float | None:
        s = self._pair(frm, to)
        if s is None or s.empty:
            return None
        window = s[pd.Timestamp(start): pd.Timestamp(end)]
        if len(window) < 5:                                                              # too few days to call an average: use the closing rate
            return None
        return float(window.mean())


def _cur(draft: dict, p: dict) -> str | None:
    c = p.get("currency") or draft.get("currency")
    return str(c).upper() if c else None


def restore(draft: dict) -> None:
    """Put the draft back as it was read, undoing a conversion."""
    pre = draft.pop("pre_conversion", None)
    draft.pop("conversion", None)
    if not pre:
        return
    for p, o in zip(draft["periods"], pre["periods"]):
        p["values"] = dict(o["values"])
        p["currency"] = o["currency"]
        for code, unit in o["units"].items():
            if code in (p.get("sources") or {}):
                p["sources"][code]["unit"] = unit
                p["sources"][code].pop("original_currency", None)
                p["sources"][code].pop("fx_rate", None)
                p["sources"][code].pop("fx_basis", None)
    draft["currency"] = pre["currency"]


def apply(draft: dict, target: str | None, rates: Rates) -> list[dict]:
    """Convert the draft in place to ``target`` (or restore it when ``target`` is empty). Returns issues: an error when a rate could not be had, an info line saying what was converted."""
    if not target:
        if "conversion" in draft:
            restore(draft)
        return []
    target = target.upper()
    done = draft.get("conversion")
    if done and done.get("to") == target:
        return []
    if done:
        restore(draft)
    periods = draft.get("periods", [])
    sources = {c for p in periods if (c := _cur(draft, p)) and c != target}
    if not sources:
        return []
    ends = [date.fromisoformat(p["period_end"]) for p in periods]
    rates.ensure(sources | {target}, min(_months_back(e, 12) for e in ends), max(ends))
    issues: list[dict] = []
    plan = []
    for p in periods:
        frm = _cur(draft, p)
        if not frm or frm == target:
            plan.append(None)
            continue
        end = date.fromisoformat(p["period_end"])
        start = _months_back(end, PERIOD_MONTHS.get(p["fiscal_period"], 12))
        label = f"{p['fiscal_period']} to {p['period_end']}"
        close = rates.closing(frm, target, end)
        avg = rates.average(frm, target, start, end) or close
        if close is None or avg is None:
            why = rates.failed.get(frm) or rates.failed.get(target) or "no rate for that date"
            issues.append({"level": "error", "message": f"no {frm}/{target} exchange rate for {label} ({why}); load rates in the Data manager (Prices) or choose to keep the report as reported.", "period": label, "field": "currency", "kind": "currency"})
            plan.append(None)
            continue
        plan.append((frm, close, avg, label))
    if issues:
        return issues
    pre = {"currency": draft.get("currency"), "periods": []}
    note = []
    for p, step in zip(periods, plan):
        pre["periods"].append({"values": dict(p["values"]), "currency": p.get("currency"), "units": {c: (s or {}).get("unit") for c, s in (p.get("sources") or {}).items()}})
        if step is None:
            continue
        frm, close, avg, label = step
        for code in list(p["values"]):
            kind = canonical.METRICS[code].kind
            if kind in SKIP_KINDS:
                continue
            basis, rate = ("closing", close) if kind == "stock" else ("average", avg)
            src = p.setdefault("sources", {}).setdefault(code, {"tag": None, "method": draft.get("method"), "unit": (draft.get("units") or {}).get("money")})
            src.setdefault("reported", p["values"][code])                                # what the report said, in its own currency
            p["values"][code] = p["values"][code] * rate
            src["original_currency"], src["fx_rate"], src["fx_basis"] = frm, rate, basis
            src["unit"] = f"{src.get('unit') or 'units'} {frm}, converted to {target} at {rate:.6g} ({basis} rate)"
        p["currency"] = target
        note.append({"period": label, "from": frm, "closing": close, "average": avg})
    draft["pre_conversion"] = pre
    draft["currency"] = target
    draft["conversion"] = {"to": target, "source": "fx_rates (Yahoo Finance for what was missing)", "periods": note, "fetched": list(rates.fetched)}
    issues.append({"level": "info", "message": f"converted from {', '.join(sorted(sources))} to {target}: flows at the average rate of their period, balances at the period-end rate (see the draft's conversion section).",
                   "period": None, "field": "currency", "kind": "currency_converted"})
    return issues


__all__ = ["Rates", "apply", "restore"]
