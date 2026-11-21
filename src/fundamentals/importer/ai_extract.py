"""AI-assisted extraction: the last resort, off unless asked for, and never trusted without evidence.

It is used only when the deterministic readers (XBRL, tables, the PDF's words, Docling) found too little to use, and only when it is switched on (``FUNDAMENTALS_AI_EXTRACTION=1``) and a client is available
(the Anthropic API by default, with ``ANTHROPIC_API_KEY``; the model is ``FUNDAMENTALS_AI_MODEL``). The document's text goes to the service, so it is opt-in. The model is asked for structured output and for
the sentence of the document that each figure came from; then **every figure is checked against the document**: the number must appear in the text as written (digits, with or without separators or brackets)
and the quoted evidence must be a verbatim part of the text. A figure that fails either test is dropped and listed as rejected: a figure that is not in the document is invented, however plausible. What
remains is marked ``ai-assisted`` and always sent for review.
"""

from __future__ import annotations

import json
import os
import re

from .. import canonical
from ..errors import FundamentalsError

MAX_CHARS = 60_000
SCHEMA_HINT = ('{"unit": "units|thousands|millions|billions", "currency": "ISO code or null", "periods": [{"fiscal_period": "FY|Q1|Q2|Q3|Q4|H1|H2", "period_end": "YYYY-MM-DD", '
               '"values": {"<line>": {"value": <number as printed, negative as negative>, "evidence": "<the exact text of the document that contains the number>"}}}]}')


def enabled(env=None) -> bool:
    return str((env if env is not None else os.environ).get("FUNDAMENTALS_AI_EXTRACTION", "")).strip().lower() in ("1", "true", "yes", "on")


def default_client(env=None):
    """A callable ``prompt -> text`` over the Anthropic API, or None when the package or the key is missing."""
    env = env if env is not None else os.environ
    key = str(env.get("ANTHROPIC_API_KEY", "")).strip()
    if not key:
        return None
    try:
        import anthropic
    except ImportError:
        return None
    model = str(env.get("FUNDAMENTALS_AI_MODEL", "")).strip() or "claude-sonnet-5-5"

    def call(prompt: str) -> str:
        msg = anthropic.Anthropic(api_key=key).messages.create(model=model, max_tokens=4000, messages=[{"role": "user", "content": prompt}])
        return "".join(getattr(b, "text", "") for b in msg.content)
    return call


def _digits(v: float) -> list[str]:
    """The ways a number is printed: with and without thousands separators, with brackets for a negative."""
    a = abs(v)
    forms = {f"{a:,.0f}", f"{a:.0f}", f"{a:,.2f}", f"{a:.2f}", f"{a:,.1f}", f"{a:.1f}", f"{a:g}"}
    return sorted(forms, key=len, reverse=True)


def verify(value, evidence: str, text: str) -> str | None:
    """None when the figure is supported by the text, else the reason it is not."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "the value is not a number"
    squashed = re.sub(r"\s+", " ", text)
    ev = re.sub(r"\s+", " ", str(evidence or "")).strip()
    if not ev or ev not in squashed:
        return "the quoted evidence is not in the document"
    if not any(f in ev for f in _digits(float(value))):
        return "the number is not in the quoted evidence"
    return None


def extract(text: str, client=None, language: str = "en", env=None) -> dict:
    if not enabled(env):
        raise FundamentalsError("AI-assisted extraction is off (set FUNDAMENTALS_AI_EXTRACTION=1 to allow the document's text to be sent to the model)")
    client = client or default_client(env)
    if client is None:
        raise FundamentalsError("AI-assisted extraction needs a client: install the anthropic package and set ANTHROPIC_API_KEY")
    body = text[:MAX_CHARS]
    lines = ", ".join(sorted(canonical.METRICS))
    prompt = (f"Extract the figures of the financial statements in this {language} report. Use only these line names: {lines}. For every figure give the exact text of the document that contains it. "
              f"Do not infer, compute or convert anything; leave a line out if it is not printed. Answer with JSON only, in this shape: {SCHEMA_HINT}\n\nDOCUMENT:\n{body}")
    raw = client(prompt)
    m = re.search(r"\{.*\}", raw or "", re.S)
    try:
        data = json.loads(m.group(0)) if m else None
    except ValueError:
        data = None
    if not isinstance(data, dict) or not isinstance(data.get("periods"), list):
        raise FundamentalsError("the model did not answer in the structure asked for: nothing was read")
    periods, rejected = [], []
    for p in data["periods"][:12]:
        values, sources = {}, {}
        for code, item in (p.get("values") or {}).items():
            if code not in canonical.METRICS or not isinstance(item, dict):
                rejected.append({"line": code, "reason": "not a line this application knows"})
                continue
            why = verify(item.get("value"), item.get("evidence"), body)
            if why:
                rejected.append({"line": code, "value": item.get("value"), "reason": why})
                continue
            values[code] = float(item["value"])
            sources[code] = {"tag": None, "reported": float(item["value"]), "unit": data.get("unit") or "units", "text": str(item["evidence"])[:160], "method": "ai-assisted"}
        if values and p.get("period_end"):
            periods.append({"fiscal_period": str(p.get("fiscal_period") or "FY").upper(), "period_end": p["period_end"], "fiscal_year": int(str(p["period_end"])[:4]), "filing_date": None, "currency": data.get("currency"),
                            "values": values, "sources": sources})
    if not periods:
        raise FundamentalsError(f"no figure the model gave was found in the document ({len(rejected)} rejected): nothing was read")
    return {"periods": periods, "units": {"money": data.get("unit") if data.get("unit") in ("units", "thousands", "millions", "billions") else "units", "shares": "units"}, "currency": data.get("currency"), "company": {},
            "notes": ["Read by an AI model because the deterministic readers found too little. Every figure was checked to be in the document as printed, and the evidence is kept; check each against the page.",
                      f"{len(rejected)} figure(s) the model gave were not supported by the document and were dropped."], "ytd_only": [], "unmapped": [], "rejected": rejected[:40], "method": "ai-assisted",
            "review_required": True, "units_stated": data.get("unit") in ("units", "thousands", "millions", "billions"), "accession": None}
