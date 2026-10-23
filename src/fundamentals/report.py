"""Plain-text reports of what the Fundamentals tab shows, for the terminal. Each function takes the dictionary the page is drawn from (so the two never disagree) and returns a string.

A key never appears: the status carries the name of each slot (``OPENFILINGS_API_KEY_1``), what it has spent and whether it is the one in use, nothing else. A missing number is ``n/a``, never zero."""

from __future__ import annotations

import math

from .evaluate import EVIDENCE_WORDS, evidence_level


def _a(word: str) -> str:
    """The word with its article: *an exploratory*, *a standard*."""
    return ("an " if word[:1].lower() in "aeiou" else "a ") + word


def _bad(x) -> bool:
    return x is None or (isinstance(x, float) and not math.isfinite(x))


def num(x, d: int = 2) -> str:
    return "n/a" if _bad(x) else f"{x:,.{d}f}"


def pct(x, d: int = 1) -> str:
    return "n/a" if _bad(x) else f"{x * 100:.{d}f}%"


def sgn(x, d: int = 3) -> str:
    return "n/a" if _bad(x) else f"{x:+.{d}f}"


def spct(x, d: int = 2) -> str:
    return "n/a" if _bad(x) else f"{x * 100:+.{d}f}%"


def count(x) -> str:
    return "n/a" if _bad(x) else f"{int(x):,}"


def grid(headers: list[str], rows: list[list], right: tuple[int, ...] = ()) -> str:
    """A text table: columns as wide as their longest entry, the numeric ones (``right``) lined up on the right."""
    cells = [[str(h) for h in headers]] + [["" if v is None else str(v) for v in row] for row in rows]
    widths = [max(len(r[i]) for r in cells) for i in range(len(headers))]
    lines = []
    for n, row in enumerate(cells):
        lines.append("  ".join(c.rjust(widths[i]) if i in right else c.ljust(widths[i]) for i, c in enumerate(row)).rstrip())
        if n == 0:
            lines.append("  ".join("-" * w for w in widths))
    return "\n".join(lines)


def _span(ci, fmt) -> str:
    """``a to b`` for a two-number interval, or nothing when it is not known."""
    if not ci or len(ci) != 2 or _bad(ci[0]) or _bad(ci[1]):
        return ""
    return f"{fmt(ci[0])} to {fmt(ci[1])}"


def _interval(ci, fmt) -> str:
    """``, 95% interval a to b`` after a figure, or nothing when the interval is not known."""
    span = _span(ci, fmt)
    return f", 95% interval {span}" if span else ""


def headline(study: dict) -> str:
    """The one-line reading the page puts on a study, from the same two numbers: the t-statistic of the IC and the net Sharpe of the long-short portfolio."""
    return EVIDENCE_WORDS[evidence_level(study["ic"].get("t_stat"), (study["portfolio"].get("net") or {}).get("sharpe"))]


def format_status(s: dict, db: str) -> str:
    c, ready = s["counts"], s["readiness"]
    out = [f"Fundamentals database ({'demo: a simulated world' if db == 'demo' else 'real data'})",
           f"  {count(c['companies'])} companies in {c['countries']} countries, {count(c['with_financials'])} with statements ({count(c['annual_periods'])} annual periods), {count(c['price_rows'])} daily prices, "
           f"{count(c['alphas'])} alphas in the library, {s['database_bytes'] / 1e6:.1f} MB; last update {s['last_update'] or 'never'}",
           "  ready to study: yes" if ready["ready"] else "  ready to study: no. " + " ".join(ready["problems"])]
    for p in s["providers"]:
        if (db == "demo") != (p["name"] == "simulated"):
            continue
        out.append("")
        if p["name"] == "simulated":
            out.append("Simulated world: no provider is called here.")
            continue
        out.append(f"{p['name']}: " + ("key set" if p["keyed"] and p["available"] else "not set up") + (f", plan {p['plan']}" if p.get("plan") else ""))
        if not p["available"]:
            out.append(f"  {p['reason']}")
        out.append(f"  allowance ({p['unit']}): today {count(p['used_today'])} of {count(p['daily_limit'])} (resets {p['resets']['daily']}), this month {count(p['used_month'])} of "
                   f"{count(p['monthly_limit'])} (resets {p['resets']['monthly']}); {p['allowance_source']}")
        out.append(f"  a new company costs about {p['cost_per_company']} {p['unit']}; {p['requests_today']} requests today ({p['successful']} ok, {p['failed']} failed)")
        if p["keys"]:
            out.append(grid(["slot", "variable", "role", "state", "used today", "used this month"], [[k["slot"], k["variable"], k["role"], k["state"], count(k["used_today"]), count(k["used_month"])] for k in p["keys"]], right=(4, 5)))
        out.extend(f"  {n}" for n in p.get("key_notes") or [])
    out += ["", "Keys: " + s["policy"], s["key_setup"]["text"].replace("typed into this page", "typed into the app or this command")]
    return "\n".join(out)


def format_search(found: dict) -> str:
    """What a search of the provider's directory found, how it read the text, what it cost and what to do next (the same answer the Company explorer draws)."""
    spent = ("answered from the saved answer of " + str(found["searched_at"]) + ", for nothing") if found["cached"] else (f"{found['credits']} {found['unit'][:-1] if found['credits'] == 1 else found['unit']} spent"
                                                                                                                       if found["credits"] else "nothing spent")
    left = "" if found["remaining_today"] is None else f"; {found['remaining_today']} left today"
    word, flag = found.get("provider_word", "OpenFilings"), "" if found.get("provider", "openfilings") == "openfilings" else f" --provider {found['provider']}"
    out = [f"{word}, \u201c{found['query']}\u201d read as {found['kind_word']}" + (f" on the {found['market']} market" if found["market"] else "") + f": {found['count']} "
           f"{'match' if found['count'] == 1 else 'matches'} ({spent}{left})"]
    if found["message"]:
        out.append(f"  {found['message']}")
    if found.get("markets_skipped"):
        out.append(f"  {word} did not search these markets this time: " + ", ".join(m.upper() for m in found["markets_skipped"]) + ". A company listed only there will not appear.")
    if not found["results"]:
        for kind in found["alternatives"]:
            out.append(f"  It could also be {'a ticker' if kind == 'symbol' else 'a company name'}: quant fundamentals search{flag} --kind {kind} \"{found['query']}\"" + (f" (one more {found['unit'][:-1]})" if found.get("credits") else ""))
        return "\n".join(out)
    rows = []
    for r in found["results"]:
        status = ("in the database" + (f", {r['periods']} annual periods" if r["periods"] else ", no statements yet")) if r["in_database"] else \
            "no statements in this market" if not r["has_financials"] else "not on your plan" if r["plan_allows"] is False else "not added"
        rows.append([r["name"] or "(no name)", r["symbol"], r["country"] or "", r["isin"] or "", status])
    out += ["", grid(["company", "symbol", "country", "ISIN", "status"], rows)]
    also = [f"{r['symbol']} is also listed as {', '.join(r['also_listed'])}" for r in found["results"] if r["also_listed"]]
    if also:
        out += ["", "Other listings of the same company (add one only if you want its filings in that market): " + "; ".join(also)]
    todo = [r for r in found["results"] if not r["in_database"] and r["has_financials"] and r["plan_allows"] is not False]
    if todo:
        out += ["", f"Add one for nothing with `quant fundamentals add{flag} {todo[0]['symbol']}`; `quant fundamentals sync{flag} {todo[0]['symbol']} --budget {found['cost_to_fetch']}` then fetches its statements "
                    f"(about {found['cost_to_fetch']} {found['unit']})."]
    return "\n".join(out)


def format_plan(plan: dict) -> str:
    p, a = plan["plan"], plan["affordability"]
    out = [f"To fetch: {p['to_fetch']} ({p['fresh']} already fresh), costing {p['cost']} {p['unit']}" + (" (" + ", ".join(f"{n} {r}" for r, n in p["reasons"].items()) + ")" if p["reasons"] else ""),
           f"Left today: {'n/a' if a['remaining_today'] is None else a['remaining_today']} {p['unit']}; {a['fit_today']} of {a['tasks']} fit today" + (f"; {a['days_needed']} days at {a['daily_limit']} a day" if a.get("days_needed") else "")]
    out.extend(p.get("notes") or [])
    if plan["tasks"]:
        out.append(grid(["company", "why", "cost"], [[t["symbol"], t["reason"], t["cost"]] for t in plan["tasks"][:25]], right=(2,)))
        if len(plan["tasks"]) > 25:
            out.append(f"... and {len(plan['tasks']) - 25} more")
    return "\n".join(out)


def format_coverage(c: dict) -> str:
    """The fiscal years of one company: what is held, what is missing, what a provider was asked for and sent nothing, what is due, and what the last request was."""
    from .coverage import amount, years_text
    a = c["annual"]
    by_state: dict[str, list[int]] = {}
    for cell in a["cells"]:
        by_state.setdefault(cell["state"], []).append(cell["year"])
    out = [f"{c['name'] or c['ticker']} ({c['symbol'] or c['ticker']}) from {c['provider_word']}"]
    for state, label in (("held", "held"), ("missing", "missing"), ("empty", "asked for, nothing sent"), ("due", "due (a filing should exist)"), ("notyet", "not due yet")):
        if by_state.get(state):
            out.append(f"  {label:<28}{years_text(by_state[state])}  ({len(by_state[state])})")
    if c["reading"]:
        out.extend(["", c["reading"]])
    q = c["quarterly"]
    out.append("")
    out.append(f"Quarterly and half-year statements held: {q['held']}." + (f" {q['note']}" if q["note"] else ""))
    out.append(f"One request costs {amount(c['request_cost'], c['unit'])}" + ("" if c["remaining_today"] is None else f" ({c['remaining_today']:,} left today)") + ", however many years it brings back." +
               ("" if c["can_fetch"] else f" It cannot be asked now: {c['why_not']}"))
    return "\n".join(out)


def format_year_plan(p: dict) -> str:
    out = [p["text"]]
    if p["to_fetch"] and not p["fits_today"]:
        out.append(f"It does not fit in what is left of today's allowance ({p['remaining_today']} {p['unit']}).")
    return "\n".join(out)


def format_sync(r: dict) -> str:
    out = [f"Fetched {r['fetched']} companies ({r['periods']} periods stored), spent {r['cost']} {r['unit']}; {len(r['failed'])} failed, {len(r['not_found'])} not found, {len(r['pending'])} still being collected"]
    if r.get("stopped"):
        out.append(r["stopped"])
    if r["failed"]:
        out.append(grid(["company", "why"], [[x["symbol"], x["error"]] for x in r["failed"][:20]]))
    if r.get("unmapped"):
        out.append("Figures the provider sent that this app does not map yet (kept in the raw answers): " + ", ".join(r["unmapped"][:20]))
    return "\n".join(out)


def format_prices(r: dict) -> str:
    out = [f"Prices for {r['loaded']} of {r['companies']} companies ({count(r['rows'])} daily rows)"]
    if r.get("fx"):
        out.append(f"Exchange rates: {len(r['fx'].get('loaded', []))} of {len(r['fx'].get('currencies', []))} currencies")
    out.extend(r.get("notes") or [])
    if r["failed"]:
        out.append(grid(["company", "symbol tried", "why"], [[x["ticker"], x.get("symbol") or "", x["reason"]] for x in r["failed"][:30]]))
    if r.get("stopped"):
        out.append(r["stopped"])
    return "\n".join(out)


def format_factors(cat: dict) -> str:
    rows = [[f["name"], f["family"], "higher" if f["direction"] > 0 else "lower", f["label"]] for f in cat["factors"]]
    return grid(["factor", "family", "better when", "what it is"], rows) + f"\n\n{len(rows)} factors. Study one with `quant fundamentals evaluate NAME`, or set them side by side with `quant fundamentals factors --scan`."


def format_scan(sc: dict, top: int = 30) -> str:
    rows = [r for r in sc["rows"] if not r.get("error")]
    out = [f"{sc['tested']} factors under one set of conditions: {sc['universe']['companies']} companies, {sc['universe']['dates']} dates", sc["note"], ""]
    out.append(grid(["factor", "family", "coverage", "mean IC", "t", "ICIR", "dates +", "top-bottom", "net Sharpe", "turnover"],
                    [[r["name"], r["family"], pct(r["coverage"], 0), sgn(r["mean_ic"]), num(r["t_stat"], 1), num(r["icir"]), pct(r["positive_share"], 0), spct(r["spread"]), num(r["net_sharpe"]), pct(r["turnover"], 0)] for r in rows[:top]], right=(2, 3, 4, 5, 6, 7, 8, 9)))
    if len(rows) > top:
        out.append(f"... and {len(rows) - top} more (use --json to keep them all)")
    errors = [r for r in sc["rows"] if r.get("error")]
    if errors:
        out.append("\nCould not be studied: " + "; ".join(f"{r['name']} ({r['error']})" for r in errors[:8]))
    return "\n".join(out)


def format_study(d: dict) -> str:
    ic, q, pf, u = d["ic"], d["quantiles"], d["portfolio"], d["universe"]
    net, gross = pf.get("net") or {}, pf.get("gross") or {}
    out = [f"{d['label']}: {headline(d)}", d["verdict"]]
    out.extend(d.get("warnings") or [])
    out += ["", f"Universe: {u['companies']} companies, {d['n_dates']} {'monthly' if u['freq'] == 'M' else 'weekly' if u['freq'] == 'W' else 'quarterly'} dates from {u['first']} to {u['last']}, "
                f"traded {u['entry_lag_days']} day(s) after the signal; each figure is used {u['pit']['info_lag_days']} day(s) after the day it became public",
            f"Mean rank IC {sgn(ic['mean_ic'])}{_interval(ic.get('ci95'), sgn)} (t = {num(ic['t_stat'], 1)}, allowing for overlapping returns), IC information ratio {num(ic['icir_annualised'])}, positive on {pct(ic['positive_share'], 0)} of dates",
            "Average next-period return by bucket (1 = lowest): " + ", ".join(f"{k}: {spct(v)}" for k, v in q["mean_forward_return"].items()) + f"; top minus bottom {spct(q['spread_top_minus_bottom'])}, "
            f"monotonic {num((q.get('monotonicity') or {}).get('spearman'))}",
            f"Long-short, net of costs and borrowing: Sharpe {num(net.get('sharpe'))}{_interval(net.get('sharpe_ci95'), num)} (gross {num(gross.get('sharpe'))}), {pct(net.get('cagr'))} a year, worst fall {pct(net.get('max_drawdown'))}, turnover {pct(pf.get('avg_turnover'), 0)} a rebalance",
            "", "How fast the signal fades (mean IC against longer returns):", grid(["horizon", "mean IC", "t", "dates"], [[r["horizon"], sgn(r["mean_ic"]), num(r["t_stat"], 1), r["n_dates"]] for r in d["decay"]], right=(1, 2, 3))]
    st = d["stability"]
    years = list(st["by_year"].items())
    out += ["", f"IC by year (positive in {sum(1 for _, v in years if v > 0)} of {len(years)}; the two halves of the sample {'agree' if st['halves_agree'] else 'disagree'} in sign: {sgn(st['first_half_ic'])} then {sgn(st['second_half_ic'])}):",
            "  " + "  ".join(f"{y}: {sgn(v, 2)}" for y, v in years)]
    for key, title in (("country", "By country"), ("sector", "By sector"), ("region", "By region")):
        rows = list((d.get("breakdown") or {}).get(key, {}).items())
        if rows:
            out += ["", title + ":", grid([key, "companies", "mean IC", "t"], [[k, v.get("companies", ""), sgn(v["mean_ic"]), num(v["t_stat"], 1)] for k, v in rows], right=(1, 2, 3))]
    power = d.get("power") or {}
    if power.get("available"):
        out += ["", "Could this study have seen a real effect? " + power["reading"],
                grid(["a true mean IC of", "would show t of about", "chance of |t| above 2", "chance of |t| above 3"], [[num(r["ic"], 2), num(r["expected_t"], 1), pct(r["power_t2"], 0), pct(r["power_t3"], 0)] for r in power["table"]], right=(0, 1, 2, 3))]
    fm = d.get("premium")
    if fm:
        a, b = fm["with_controls"], fm["without_controls"]
        out += ["", "The premium, from a regression on every date (Fama-MacBeth): " + fm["reading"]]
        if "premium" in a and "premium" in b:
            out.append(grid(["", "premium over the window", "95% interval", "t", "dates"],
                            [["holding " + (", ".join(fm["controls"]) or "nothing") + " fixed", spct(a["premium"]), _span(a.get("ci95"), spct), num(a["t_stat"], 1), a["n_dates"]],
                             ["on its own", spct(b["premium"]), _span(b.get("ci95"), spct), num(b["t_stat"], 1), b["n_dates"]]], right=(1, 3, 4)))
    out += ["", "Interesting is not validated. A study measured on the sample it was chosen on is a lead; `quant fundamentals discover` counts every idea tried and tests the best on dates it never saw."]
    return "\n".join(out)


def format_lookahead(a: dict) -> str:
    rows = [["each figure from the day it was public", sgn(a["honest"]["mean_ic"]), num(a["honest"]["t_stat"], 1), a["honest"]["n_dates"]], ["each figure from its period end (naive)", sgn(a["naive"]["mean_ic"]), num(a["naive"]["t_stat"], 1), a["naive"]["n_dates"]]]
    text = a["verdict"] + "\n" + grid(["version", "mean IC", "t", "dates"], rows, right=(1, 2, 3))
    if a.get("by_lag"):
        text += ("\n\nWhat is left if a figure is used only some days after its filing: " + a["lag_verdict"] + "\n"
                 + grid(["days after the filing", "mean IC", "t", "dates"], [[r["lag_days"], sgn(r["mean_ic"]), num(r["t_stat"], 1), r["n_dates"]] for r in a["by_lag"]], right=(0, 1, 2, 3)))
    return text


def format_discovery(d: dict, top: int = 15) -> str:
    c = d.get("counts") or {}
    out = [f"Discovery run {d['run_id']}: {d['status']}, {d['progress']['done']} of {d['progress']['total']} formulas",
           f"Tried {c.get('generated', d['progress']['done'])} ({c.get('skipped_seen', 0)} already known, {c.get('duplicate_idea', 0)} the same idea twice); tested in full {c.get('tested', 0)}; survived the stages {c.get('survivors', 0)}; "
           f"candidates {c.get('candidates', 0)} ({c.get('failed_locked_test', 0)} failed the locked test); looks at the locked dates {c.get('oos_looks', 0)}"
           + (f"; chance of overfitting among the survivors {pct(c['pbo'], 0)}" if not _bad(c.get("pbo")) else ""),
           f"Bars: {d.get('strictness', 'exploratory')} (first screen |t| of {(d.get('limits') or {}).get('t_ic', 1.65):g}, false-discovery q-value of {(d.get('limits') or {}).get('max_q', 0.25):g})"
           + (f". The correction for the number tried counts {c['n_tested_family']} ideas, from {c['runs_in_family']} run{'s' if c['runs_in_family'] != 1 else ''} on this question" if c.get("n_tested_family") else ""),
           "", "Where the formulas fell:", grid(["stage", "passed"], [[r["stage"], r["passed"]] for r in d["funnel"]], right=(1,)), "", "The best so far (nothing here is validated):"]
    if d["leaderboard"]:
        out.append(grid(["formula", "used as", "status", "IC", "val IC", "locked IC", "t", "Sharpe", "turn", "q", "DSR"],
                        [[r["expression"], "lower" if (r.get("direction") or 1) < 0 else "higher", r["status"], sgn(r["ic"]), sgn(r.get("val_ic")), sgn(r.get("oos_ic")), num(r.get("oos_t"), 1), num(r.get("sharpe")), pct(r.get("turnover"), 0),
                          num(r.get("q_value")), num(r.get("deflated_sharpe"))] for r in d["leaderboard"][:top]], right=(3, 4, 5, 6, 7, 8, 9, 10)))
    else:
        out.append("No formula has survived the stages yet.")
    out.append("\n" + d["note"])
    return "\n".join(out)


def format_library(rows: list[dict], counts: dict) -> str:
    out = [f"{len(rows)} shown; library {', '.join(f'{k} {v}' for k, v in sorted(counts['library'].items())) or 'empty'}; {counts['graveyard']} failed ideas in the graveyard; {counts['tried']} ideas tried in all"]
    if rows:
        out.append(grid(["alpha", "formula", "used as", "status", "IC", "locked IC", "Sharpe", "turn", "fitness", "q", "DSR"],
                        [[r["alpha_id"], r["text"], "lower" if ((r.get("metrics") or {}).get("direction") or 1) < 0 else "higher", r["status"], sgn(r["ic"]), sgn(r["oos_ic"]), num(r["sharpe"]), pct(r["turnover"], 0), num(r["fitness"]),
                          num(r["q_value"]), num(r["deflated_sharpe"])] for r in rows], right=(4, 5, 6, 7, 8, 9, 10)))
    return "\n".join(out)


def format_graveyard(rows: list[dict], counts: dict) -> str:
    out = [f"{counts['graveyard']} failed ideas, kept: the record of what was tried is what makes the survivors honest"]
    if rows:
        out.append(grid(["formula", "failed at", "why", "IC", "tested"], [[r["text"], r["stage"], r["reason_rejected"], sgn(r["ic"]), str(r["date_tested"])[:10]] for r in rows], right=(3,)))
    return "\n".join(out)


def format_monitor(m: dict) -> str:
    out = [f"{m['checked']} alpha(s) checked", m["note"]]
    if m["results"]:
        out.append(grid(["formula", "state", "status", "recent IC", "original IC", "share", "recent t"],
                        [[r["text"], r["monitor"], r["status"], sgn(r["detail"].get("recent_ic")), sgn(r["detail"].get("original_ic")), pct(r["detail"].get("ratio"), 0), num(r["detail"].get("recent_t"), 1)] for r in m["results"]], right=(3, 4, 5, 6)))
    return "\n".join(out)


def format_backtest(r: dict) -> str:
    s, b, g, cost, lim = r["summary"], r["benchmark"], r["gross_summary"], r["costs"], r["limits"]
    dsr = r.get("deflated_sharpe")
    out = [f"Backtest: {r['signal']['label']}", f"{s['start']} to {s['end']}, {'long and short' if r['portfolio']['side'] == 'long_short' else 'long only'}, "
           f"{'top ' + str(r['portfolio']['n']) if r['portfolio']['selection'] == 'top_n' else 'top ' + str(r['portfolio']['percentile']) + '%'}, {r['portfolio']['weighting']} weights"]
    out.extend("Watch: " + w for w in r["warnings"])
    out += ["", r["reading"], "",
            f"Net Sharpe {num(s['sharpe'])} (gross {num(g['sharpe'])}, benchmark {num(b['sharpe'])}); return {pct(s['cagr'])} a year (benchmark {pct(b['cagr'])}); volatility {pct(s['ann_vol'])}; worst fall {pct(s['max_drawdown'])}",
            f"Alpha against the universe {pct(s['alpha_ann'])} a year, beta {num(s['beta_to_benchmark'])}, information ratio {num(s['information_ratio'])}",
            f"Turnover {num(cost['annual_turnover'], 1)}x a year; trading {num(cost['trading_bps_a_year'], 0)} bps and borrowing {num(cost['borrow_bps_a_year'], 0)} bps a year; breakeven one-way cost {num(cost['breakeven_bps'], 0)} bps",
            "Deflated Sharpe " + (f"{num(dsr['probability'])} after {dsr['n_trials']} idea(s) tried (0.95 is the bar)" if dsr else "n/a"), "",
            f"Limits: changed the book on {lim['projected_dates']} of {lim['dates']} rebalance dates; not fully invested on {lim['dates_not_fully_invested']}; in cash on {lim['dates_in_cash']}; "
            f"average {num(lim['average_names_long'], 1)} names long, {num(lim['average_names_short'], 1)} short; largest position {pct(lim['max_position_held'])}", "", "One-way cost against net return:",
            grid(["bps", "net return a year", "net Sharpe", "cost a year"], [[x["cost_bps"], pct(x["net_ann_return"]), num(x["net_sharpe"]), pct(x["annual_cost_drag"], 2)] for x in cost["sensitivity"]], right=(0, 1, 2, 3))]
    if r.get("blocks"):
        out += ["", "On the dates the alpha was found on (the locked block is the honest one):",
                grid(["block", "from", "to", "days", "net Sharpe", "return a year"], [[k, v["start"], v["end"], v.get("n_obs", ""), num(v.get("sharpe")), pct(v.get("cagr"))] for k, v in r["blocks"].items()], right=(3, 4, 5))]
    out += ["", f"Latest book ({r['as_of']}):", grid(["ticker", "company", "weight", "sector", "country"], [[h["ticker"], h["name"], spct(h["weight"]), h.get("sector") or "", h.get("country") or ""] for h in r["holdings"][:25]], right=(2,)),
            "", f"Recorded as experiment {r['experiment_id']}."]
    return "\n".join(out)


__all__ = ["format_status", "format_plan", "format_coverage", "format_year_plan", "format_sync", "format_prices", "format_factors", "format_scan", "format_study", "format_lookahead", "format_discovery", "format_library", "format_graveyard",
           "format_monitor", "format_backtest", "headline", "grid"]


def format_preflight(pf: dict) -> str:
    rows = [[num(r["ic"], 2), pct(r["first_screen"], 0), pct(r["validation"], 0), pct(r["locked_test"], 0), pct(r["found"], 0)] for r in pf["table"]]
    b = pf["blocks"]
    return "\n".join([f"What {_a(pf['strictness'])} search of {pf['formulas']} formulas could find", pf["reading"], "",
                      f"Training dates {b['train'][0]} to {b['train'][1]} ({b['dates']['train']}), validation {b['validation'][0]} to {b['validation'][1]} ({b['dates']['validation']}), "
                      f"locked test {b['oos'][0]} to {b['oos'][1]} ({b['dates']['oos']}); {b['embargo']} date(s) left out between blocks",
                      "", "The chance that a real alpha of each strength gets through (before the screens in between):",
                      grid(["true mean IC", "first screen", "validation", "locked test", "all three"], rows, right=(0, 1, 2, 3, 4))])


def format_placebo(r: dict) -> str:
    s = r["summary"]
    rows = [[k, pct(v["share"], 1), pct(v["expected"], 1)] for k, v in s["tail"].items()]
    out = [f"Calibration of {_a(r['strictness'])} search on shuffled returns ({s['runs']} shuffle{'s' if s['runs'] != 1 else ''}, {s['tested']} formulas tested)", r["reading"], "",
           "Where the formulas fell, on average per shuffled run:", grid(["stage", "passed"], [[k, num(v, 1)] for k, v in s["funnel_mean"].items()], right=(1,)), "",
           "Shuffled formulas whose |t| reached each level (a correct t-statistic gives the second column):", grid(["|t| of at least", "found", "expected"], rows, right=(0, 1, 2))]
    return "\n".join(out)
