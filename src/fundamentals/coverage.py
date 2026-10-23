"""Which fiscal years of a company the database holds, which it does not, and what asking for the missing ones would cost.

OpenFilings charges per request, not per year: a series costs 10 credits whether it brings back three years or thirty, and what comes back is whatever the provider holds for that company within the
window asked. So the unit of planning here is the *request*: one request for a window of fiscal years, and the years of that window that are already held are left alone when the answer is stored.

Every year in the grid is in one of five states:

* ``held``: a statement for that fiscal year is in the database. It is never asked for again by this view.
* ``missing``: not held, and no request has covered it yet (a gap between two held years, or a year before the earliest one).
* ``empty``: a request covered it when it was due and the provider sent nothing for it. Asking again costs the same as any request and has probably the same answer, so it is left out unless asked for.
  (For a year newer than anything held this lasts a month: a filing that was late to appear may be there by then, and the year is ``due`` again.)
* ``due``: later than the latest held year, and its filing should exist by now (the year ended, plus the usual filing lag, has passed).
* ``notyet``: later than the latest held year and its filing cannot exist yet.

Nothing here talks to a provider or reads a database: the callers pass in what is held and what was asked, so every rule can be tested with a calendar and a list.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

DEPTH = 25                                          # years back from this one that the grid reaches (the length of a first fetch), more if the company has older statements
FIRST_YEAR = 1990
MAX_PICK = 60
RETRY_NEWEST_DAYS = 30                              # a year newer than anything held, found empty, is worth asking about again after this long
STATES = ("held", "missing", "empty", "due", "notyet")
ANNUAL_WORDS = ("FY", "ANNUAL", "A", "YEAR", "FULLYEAR", "FISCALYEAR")          # the ways a provider may label a full fiscal year
QUARTER_LABELS = ("Q1", "Q2", "Q3", "Q4", "H1", "H2")


def label_of(raw) -> str:
    """A period label as the application writes it: ``FY``, ``Q1`` ... ``Q4``, ``H1``, ``H2``, or the provider's own word in capitals when it is none of those."""
    text = str(raw or "").strip().upper().replace(" ", "").replace("-", "").replace("_", "")
    if not text or text in ANNUAL_WORDS:
        return "FY"
    return text[:12]


def entries_of(raw) -> list[dict]:
    """The list of periods inside a stored answer: OpenFilings sends ``{"periods": [...]}``, the simulated world the list itself."""
    items = raw.get("periods") if isinstance(raw, dict) else raw
    return [e for e in items if isinstance(e, dict)] if isinstance(items, list) else []


def _year(entry: dict) -> int | None:
    fy = entry.get("fiscal_year")
    if isinstance(fy, int) and not isinstance(fy, bool):
        return fy
    end = str(entry.get("period_end") or "")[:4]
    return int(end) if end.isdigit() else None


def _echo(raw) -> dict:
    """The window a provider says it ran: OpenFilings answers with a ``query`` object that holds the ``fiscal_year_from`` and ``fiscal_year_to`` it understood."""
    query = raw.get("query") if isinstance(raw, dict) and isinstance(raw.get("query"), dict) else {}
    return {k: query[f"fiscal_year_{k}"] for k in ("from", "to") if isinstance(query.get(f"fiscal_year_{k}"), int) and not isinstance(query.get(f"fiscal_year_{k}"), bool)}


def summarise_answer(raw) -> dict:
    """What a provider's answer held, whatever this application made of it: ``entries`` (all the periods in it), ``labels`` (how many of each kind), ``annual`` (full years), the first and last
    fiscal year among those, and ``echoed``, the window the provider says it ran (when it says)."""
    labels: dict[str, int] = {}
    years = []
    for entry in entries_of(raw):
        label = label_of(entry.get("fiscal_period"))
        labels[label] = labels.get(label, 0) + 1
        year = _year(entry) if label == "FY" else None
        if year is not None:
            years.append(year)
    return {"entries": sum(labels.values()), "labels": labels, "annual": labels.get("FY", 0), "first_year": min(years) if years else None, "last_year": max(years) if years else None, "echoed": _echo(raw)}


def asked_years(request: dict | None) -> tuple[int | None, int | None]:
    """The fiscal years a request asked for, from the parameters it was kept with: OpenFilings names them ``fiscal_year_from`` and ``fiscal_year_to``, the simulated world ``start_year`` and ``end_year``."""
    request = request if isinstance(request, dict) else {}

    def year(*names):
        for name in names:
            value = request.get(name)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
        return None
    return year("fiscal_year_from", "start_year"), year("fiscal_year_to", "end_year")


def derive_log(answers: list[dict]) -> list[dict]:
    """Log rows for answers kept from before the log existed (``store.raw_answers``): what each asked for and what it held, so that an old database can say why a company has the years it has. The
    cost of such a request was not recorded and is left unknown."""
    out = []
    for a in answers:
        summary = summarise_answer(a.get("payload"))
        asked_from, asked_to = asked_years(a.get("parameters"))
        out.append({"at": str(a.get("retrieved_at") or "")[:10], "asked_from": asked_from, "asked_to": asked_to, "entries": summary["entries"], "kept": summary["annual"], "new": None,
                    "first_year": summary["first_year"], "last_year": summary["last_year"], "cost": None, "reason": "", "derived": True,
                    "detail": {"labels": summary["labels"], "echoed": summary["echoed"], "message": _note(a.get("payload"))}})
    return out


def _note(raw) -> str:
    """The free-text note a provider puts in an answer (OpenFilings' ``message``), cut to length."""
    return str(raw.get("message") or "")[:300] if isinstance(raw, dict) else ""


def _shift(day: date, years: int) -> date:
    try:
        return day.replace(year=day.year + years)
    except ValueError:                                                          # 29 February
        return day.replace(year=day.year + years, day=28)


def grid(held: list[dict], log: list[dict], today: date, lag_days: int, depth: int = DEPTH) -> dict:
    """The fiscal years from ``depth`` years back (or the earliest year asked for or held) to this one, each in one of the five states.

    ``held``: the annual statements in the database (``fiscal_year``, ``period_end``, ``filing_date``). ``log``: the requests made (``at``, ``asked_from``, ``asked_to``; ``None`` is open).
    ``lag_days`` is how long after a year ends its statement is normally filed. A year after the latest held one is judged against that year's own year end moved on by whole years.
    """
    first_filing: dict[int, dict] = {}
    for row in sorted(held, key=lambda r: str(r["period_end"])):
        first_filing.setdefault(int(row["fiscal_year"]), row)
    have = sorted(first_filing)
    earliest, latest = (have[0], have[-1]) if have else (None, None)
    anchor = date.fromisoformat(str(first_filing[latest]["period_end"])[:10]) if have else None
    asked = [r for r in log if r.get("at")]
    lo = max(min([today.year - depth] + have + [int(r["asked_from"]) for r in asked if r.get("asked_from")]), FIRST_YEAR)
    hi = max(today.year, latest or 0)

    def expected_end(y: int) -> date:
        return _shift(anchor, y - latest) if anchor is not None else date(y, 12, 31)

    def due_on(y: int, day: date) -> bool:
        return expected_end(y) + timedelta(days=lag_days) <= day

    def asked_on(y: int) -> str | None:
        hits = [str(r["at"])[:10] for r in asked if (r.get("asked_from") is None or y >= int(r["asked_from"])) and (r.get("asked_to") is None or y <= int(r["asked_to"]))
                and due_on(y, date.fromisoformat(str(r["at"])[:10]))]                    # a request made before the year's filing was due says nothing about it
        if not hits:
            return None
        newest = max(hits)
        if (latest is None or y > latest) and (today - date.fromisoformat(newest)).days >= RETRY_NEWEST_DAYS:
            return None                                                         # long enough ago that the filing may have appeared since
        return newest

    cells = []
    for y in range(lo, hi + 1):
        if y in first_filing:
            row = first_filing[y]
            cells.append({"year": y, "state": "held", "period_end": str(row["period_end"])[:10], "filed": str(row.get("filing_date") or "")[:10] or None, "asked_on": None, "expected_filing": None})
            continue
        when = asked_on(y)
        if when:
            state = "empty"
        elif latest is not None and y < latest:
            state = "missing"
        elif due_on(y, today):
            state = "due" if latest is not None else "missing"
        else:
            state = "notyet"
        cells.append({"year": y, "state": state, "period_end": None, "filed": None, "asked_on": when,
                      "expected_filing": (expected_end(y) + timedelta(days=lag_days)).isoformat() if state in ("due", "notyet") else None})
    counts = {s: 0 for s in STATES}
    for c in cells:
        counts[c["state"]] += 1
    return {"first": lo, "last": hi, "cells": cells, "counts": counts, "earliest_held": earliest, "latest_held": latest}


def amount(n: int, unit: str) -> str:
    """``10 credits`` and ``1 credit``: a unit written in the plural loses its s for one."""
    return f"{n} {unit[:-1] if n == 1 and unit.endswith('s') else unit}"


def describe_window(window: dict | None) -> str:
    """``FY2005 to FY2012``, ``FY2020 onwards`` or ``FY2020`` for the years a request covers."""
    if not window or window.get("from") is None:
        return "every year"
    first, last = window["from"], window.get("to")
    if last is None:
        return f"FY{first} onwards"
    return f"FY{first}" if last == first else f"FY{first} to FY{last}"


def plan(cells: list[dict], years, ask_again: bool, latest_held: int | None, per_request: int) -> dict:
    """What asking for ``years`` would do. Held years are never asked for. A year a request already found empty, or one whose filing cannot exist yet, is left out unless ``ask_again``. The rest
    is one request: from the earliest of them to the latest, left open at the end when the latest is newer than anything held, so that a filing newer than the one picked is brought along.

    One request, because the price is per request: three separate gaps cost 10 credits together, not 30. Years inside that window that are already held come back in the answer and are ignored; a year
    inside it that is not held, picked or not, is stored, since it has been paid for.
    """
    by_year = {c["year"]: c for c in cells}
    picked = sorted({int(y) for y in years})
    held, empty, notyet, outside, wanted = [], [], [], [], []
    for y in picked:
        cell = by_year.get(y)
        if cell is None:
            outside.append(y)
        elif cell["state"] == "held":
            held.append(y)
        elif cell["state"] == "empty" and not ask_again:
            empty.append(y)
        elif cell["state"] == "notyet" and not ask_again:
            notyet.append(y)
        else:
            wanted.append(y)
    window = None
    if wanted:
        window = {"from": wanted[0], "to": None if latest_held is None or wanted[-1] > latest_held else wanted[-1]}
    asking_again = [y for y in wanted if by_year[y]["state"] == "empty"]
    return {"picked": picked, "to_fetch": wanted, "already_held": held, "left_out_empty": empty, "left_out_notyet": notyet, "outside": outside, "asking_again": asking_again, "window": window,
            "requests": 1 if wanted else 0, "cost": per_request if wanted else 0}


def nothing_to_fetch(p: dict) -> str:
    """Why a plan with no years in it asks for nothing."""
    if not p["picked"]:
        return "pick at least one year"
    reasons = []
    if p["already_held"]:
        reasons.append(f"you already hold {years_text(p['already_held'])}")
    if p["left_out_empty"]:
        reasons.append(f"{years_text(p['left_out_empty'])} {'was' if len(p['left_out_empty']) == 1 else 'were'} asked for before and the provider sent nothing (ask again only if you want to pay for another try)")
    if p["left_out_notyet"]:
        reasons.append(f"{years_text(p['left_out_notyet'])} cannot have been filed yet")
    if p["outside"]:
        reasons.append(f"{years_text(p['outside'])} {'is' if len(p['outside']) == 1 else 'are'} outside the years shown")
    return "nothing to ask for: " + "; ".join(reasons)


def years_text(years: list[int]) -> str:
    """``FY2019``, ``FY2019 and FY2021`` or ``FY2015 to FY2018, FY2020 and FY2022`` for a list of years: a run of years is joined."""
    ys = sorted(set(years))
    runs = [[ys[0], ys[0]]]
    for y in ys[1:]:
        if y == runs[-1][1] + 1:
            runs[-1][1] = y
        else:
            runs.append([y, y])
    parts = [f"FY{a}" if a == b else f"FY{a} to FY{b}" for a, b in runs]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def plan_text(p: dict, unit: str, remaining: int | None) -> str:
    """The plan in a sentence: what is asked, what it costs, and what is left alone."""
    if not p["to_fetch"]:
        return nothing_to_fetch(p)
    left = "" if remaining is None else f" ({remaining:,} left today)"
    text = f"One request for {describe_window(p['window'])}: {amount(p['cost'], unit)}{left}. Any year in that window not held yet is stored; years you hold are left alone."
    if p["already_held"]:
        text += f" {years_text(p['already_held'])} {'is' if len(p['already_held']) == 1 else 'are'} held and not asked for."
    if p["asking_again"]:
        text += f" It includes {years_text(p['asking_again'])}, which the provider sent nothing for last time: the same price may buy nothing again."
    if p["left_out_empty"]:
        text += f" Left out because the provider sent nothing for them last time: {years_text(p['left_out_empty'])}."
    if p["left_out_notyet"]:
        text += f" Left out because they cannot have been filed yet: {years_text(p['left_out_notyet'])}."
    return text


def reading(row: dict | None, provider: str, unit: str, price_note: str | None = None, early_note: str | None = None) -> str:
    """The last request in plain words, and what it says about why the years that are held are the years that are held. ``price_note`` is the provider's own sentence about what a request buys
    (OpenFilings charges per request, not per year) and ``early_note`` what it says of years before the earliest it sent; both default to OpenFilings'."""
    if not row:
        return ""
    asked = describe_window({"from": row.get("asked_from"), "to": row.get("asked_to")})
    cost = row.get("cost")
    price = "" if cost is None else f" and cost {amount(cost, unit)}"
    start = f"On {str(row.get('at') or '')[:10]} the request to {provider} for {asked} brought back"
    entries, kept = int(row.get("entries") or 0), int(row.get("kept") or 0)
    note = str((row.get("detail") or {}).get("message") or "").strip()
    said = f" {provider} added a note to its answer: \u201c{note}\u201d" + ("" if note.endswith((".", "!", "?")) else ".") if note else ""
    if entries == 0:
        return f"{start} no periods at all{price}.{said}"
    if kept == 0:
        labels = ", ".join(f"{n} {k}" for k, n in sorted((row.get("detail") or {}).get("labels", {}).items()))
        return f"{start} {entries} {'entry' if entries == 1 else 'entries'}{f' ({labels})' if labels else ''}, none of them an annual statement this application reads{price}.{said}"
    first, last = row.get("first_year"), row.get("last_year")
    span = f"FY{first}" if first == last else f"FY{first} to FY{last}"
    price_note = price_note if price_note is not None else "Credits pay for the request, not for the years: the same price would have brought back a longer or a shorter series."
    text = f"{start} {kept} annual {'statement' if kept == 1 else 'statements'} ({span}){price}. {price_note}"
    unused = entries - kept
    if unused > 0 and not row.get("derived"):
        other = ", ".join(f"{n} {k}" for k, n in sorted(((row.get("detail") or {}).get("labels") or {}).items()) if k != "FY")
        text += (f" It sent {entries} entries in all; {unused} {'was' if unused == 1 else 'were'} not used because {'it was' if unused == 1 else 'they were'} not a full year with figures" + (f" ({other})" if other else "") + ".")
    text += said
    echo = (row.get("detail") or {}).get("echoed") or {}
    if (echo.get("from") is not None and row.get("asked_from") is not None and echo["from"] != row["asked_from"]) or (echo.get("to") is not None and row.get("asked_to") is not None and echo["to"] != row["asked_to"]):
        text += f" {provider} recorded the request it ran as {describe_window({'from': echo.get('from'), 'to': echo.get('to')})}, not {asked}: it changed the window before answering."
    if row.get("asked_from") and first and first > row["asked_from"]:
        gap = f"FY{row['asked_from']}" + (f" to FY{first - 1}" if first - 1 > row["asked_from"] else "")
        early_note = early_note if early_note is not None else ("either it holds no statements that early for this company or it limits how long a series it returns (its documentation says neither). "
                                                                "Asking for just those years tells which, for the same price.")
        text += f" It sent nothing for {gap}: {early_note}"
    return text


def report_text(report: dict, provider: str) -> str:
    """What a request for chosen years did, in words: what was added, what the provider sent nothing for, and what it cost."""
    added = [a["year"] for a in report.get("added", [])]
    price = f" {amount(report['cost'], report['unit'])} spent."
    if not added:
        return (f"{provider} brought back nothing new for {describe_window(report.get('window'))}." + (f" It sent {report['entries']} {'entry' if report['entries'] == 1 else 'entries'}, none of them an annual statement not already held."
                                                                                                          if report.get("entries") else " Its answer held no periods.") + price)
    text = f"Added {years_text(added)} ({len(added)} annual {'statement' if len(added) == 1 else 'statements'})." + price
    if report.get("nothing_for"):
        text += f" {provider} sent nothing for {years_text(report['nothing_for'])}: it may hold nothing that early for this company, so those years will not be offered again unless you ask."
    if report.get("left_alone"):
        text += f" {report['left_alone']} {'year' if report['left_alone'] == 1 else 'years'} in its answer {'was' if report['left_alone'] == 1 else 'were'} already held and left alone."
    return text


def parse_years(text: str) -> list[int]:
    """``2005-2012,2015`` as the years 2005 to 2012 and 2015 (the way a command line names them)."""
    years: set[int] = set()
    for part in re.split(r"[,\s]+", str(text or "").strip()):
        if not part:
            continue
        found = re.fullmatch(r"(\d{4})(?:-(\d{4}))?", part)
        if not found:
            raise ValueError(f"'{part}' is not a year or a range of years such as 2005-2012")
        first, last = int(found.group(1)), int(found.group(2) or found.group(1))
        if last < first or first < FIRST_YEAR or last > 2100:
            raise ValueError(f"'{part}' is not a range of years between {FIRST_YEAR} and 2100")
        years.update(range(first, last + 1))
        if len(years) > MAX_PICK:
            raise ValueError(f"pick at most {MAX_PICK} years at a time")
    if not years:
        raise ValueError("pick at least one year, such as 2005-2012,2015")
    return sorted(years)
