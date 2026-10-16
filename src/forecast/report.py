"""The result of a forecast and its test as plain text, for ``quant forecast``. The Forecast tab draws the same numbers."""

from __future__ import annotations

import textwrap

from .verdict import ICON


def _grid(header: list[str], rows: list[list[str]]) -> list[str]:
    """A table with the first column to the left and the rest to the right."""
    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) for i in range(len(header))]
    cell = lambda text, i: text.ljust(widths[i]) if i == 0 else text.rjust(widths[i])
    return ["  " + "  ".join(cell(h, i) for i, h in enumerate(header))] + ["  " + "  ".join(cell(c, i) for i, c in enumerate(r)) for r in rows]


def _pct(x, digits: int = 0) -> str:
    return "n/a" if x is None else f"{x * 100:.{digits}f}%"


def _signed(x, digits: int = 1) -> str:
    if x is None:
        return "n/a"
    text = f"{abs(x) * 100:.{digits}f}%"
    return text if float(text[:-1]) == 0 else ("+" if x >= 0 else "-") + text


def format_report(ticker: str, asof: str, last_price: float, horizon: int, evaluation: dict, now: dict, verdict: dict, width: int = 110) -> str:
    """The verdict, the scoreboard of every method at the horizon, and what each method says from the last day, as text."""
    hz = str(int(horizon))
    short = {m["id"]: m.get("short") or m["label"] for m in evaluation["methods"]}
    base = evaluation["baseline"]
    o = evaluation["origins"]
    lines = [f"Forecast for {ticker}, as of {asof}: last price {last_price:,.2f}, {horizon} trading days ahead", ""]
    lines += textwrap.wrap(f"{ICON.get(verdict['level'], '?')} {verdict['headline']}", width)
    for reason in verdict["reasons"]:
        lines += textwrap.wrap(reason, width, initial_indent="  - ", subsequent_indent="    ")

    rows = []
    for m in sorted(evaluation["methods"], key=lambda m: -9.0 if m["id"] == base else -(m["scores"].get(hz, {}).get("skill") or -9.0)):          # the yardstick first, then the best
        s = m["scores"].get(hz)
        if not s or s.get("crps") is None:
            continue
        if m["id"] == base:
            skill, adjusted = "yardstick", ""
        else:
            skill = _signed(s.get("skill")) + (" ✓" if s.get("beats_baseline") else "")
            adjusted = "n/a" if s.get("p_adjusted") is None else f"{s['p_adjusted']:.3f}"
        rows.append([short[m["id"]], skill, adjusted, _pct((s.get("coverage") or {}).get("90")), _pct(s.get("var5")), f"{s['crps'] * 1000:.2f}"])
    if rows:
        lines += ["", f"How each method scored at {horizon} trading days, over {o['count']} forecasts (skill: how far below the random walk's CRPS; ✓: the gain holds after allowing for the number of methods compared)"]
        lines += _grid(["method", "skill", "adjusted p", "90% range held", "below 5% quantile", "CRPS per 1,000"], rows)

    rows = []
    for mid, m in now["methods"].items():
        f, i = m["fan"], int(horizon) - 1
        rows.append([short.get(mid, m["label"]), f"{f['0.05'][i]:,.2f}", f"{f['0.5'][i]:,.2f}", f"{f['0.95'][i]:,.2f}", _pct(m["summary"][hz]["p_up"]), _pct(m["summary"][hz]["p_loss_5"])])
    if rows:
        lines += ["", f"What each method says now: the price in {horizon} trading days"]
        lines += _grid(["method", "5% worst", "median", "95% best", "chance of a rise", "chance of -5% or worse"], rows)
    for mid, why in now.get("errors", {}).items():
        lines += textwrap.wrap(f"{short.get(mid, mid)} could not forecast from the last day: {why}", width, initial_indent="  ! ", subsequent_indent="    ")
    lines += ["", "A forecast is not a strategy and is not advice: how good a forecast is and how much money a rule built on it makes are different questions."]
    return "\n".join(lines)
