"""Read-only live verification of the exchange adapters.

Sends the public GET requests listed in ``assets.free_data.capabilities.PROBES`` (no keys, no signed or trading endpoints), writes the matrix to ``docs/generated/exchange_capabilities.json`` and ``docs/exchange_capabilities.md``
and saves a trimmed copy of every verified response to ``tests/fixtures/exchanges/`` for the regression tests. Run it from the repository root: ``python scripts/verify_exchanges.py [VENUE ...]``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assets.free_data import capabilities as cap  # noqa: E402


def trim(x, n=6):
    if isinstance(x, list):
        return [trim(v, n) for v in x[:n]]
    if isinstance(x, dict):
        return {k: trim(v, n) for k, v in x.items()}
    return x


def main(argv):
    only = {a.upper() for a in argv} or None
    matrix = cap.run(only=only)
    fx = ROOT / "tests" / "fixtures" / "exchanges"
    fx.mkdir(parents=True, exist_ok=True)
    for r in matrix["rows"]:
        if r.get("status") in ("verified", "not_implemented") and r.get("sample") is not None:
            (fx / f"{r['venue'].lower()}_{r['feature']}.json").write_text(json.dumps(trim(r["sample"]), indent=1, sort_keys=True))
        r.pop("sample", None)
    out = ROOT / "docs" / "generated"
    out.mkdir(parents=True, exist_ok=True)
    (out / "exchange_capabilities.json").write_text(json.dumps(matrix, indent=1))
    lines = ["# Exchange capability matrix", "", f"Generated {matrix['generated_at']} by `scripts/verify_exchanges.py` from read-only public requests. **Only a live answer sets `verified`.** A mocked test proves a parser, not that an exchange still answers.", "",
             "| Venue | Feature | Status | Tested (UTC) | Detail | Adapter |", "|---|---|---|---|---|---|"]
    for r in matrix["rows"]:
        lines.append(f"| {r['venue']} | {r['feature']} | **{r['status']}** | {r.get('tested_at') or ''} | {str(r.get('detail') or '')[:110].replace('|', '/')} ({r.get('note') or ''}) | {r.get('adapter') or ''} |")
    lines += ["", "Legend: " + "; ".join(f"`{k}` {v}" for k, v in matrix["legend"].items())]
    (ROOT / "docs" / "exchange_capabilities.md").write_text("\n".join(lines) + "\n")
    for r in matrix["rows"]:
        print(f"{r['venue']:9} {r['feature']:26} {r['status']:15} {str(r.get('http_status') or ''):4} {str(r.get('detail') or '')[:70]}")


if __name__ == "__main__":
    main(sys.argv[1:])
