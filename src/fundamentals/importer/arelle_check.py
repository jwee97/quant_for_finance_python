"""Arelle as an independent check of a filing read by this application's own XBRL reader.

Arelle is the XBRL consortium's open-source processor. It resolves the filing's taxonomy (downloading it, then keeping it in a cache directory), so it knows every fact's concept, period, unit and dimensions
as the standard defines them, and it can validate the filing. It is slow (about a minute for a large annual report the first time, the taxonomy download included) and heavy, so it is **opt-in** (the hint
``arelle`` on an import, or ``FUNDAMENTALS_ARELLE=1``), and it runs in a separate process (a crash or a hang does not touch the application) with a timeout.

What the cross-check says: how many facts Arelle found against how many this reader found; for every company-wide numeric fact of this reader, whether Arelle has a fact of the same name and period with the
same value; the facts only one of them has; and Arelle's own validation messages (errors and warnings). A disagreement is an issue on the draft, and a draft is not marked clean while one stands. It was run
against a real ESEF annual report (Hermès 2024: 544 facts each way, every numeric fact equal); a filing with no reachable taxonomy gives Arelle only concept names, which the check reports as such.
"""

from __future__ import annotations

import csv
import importlib.util
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from ..errors import FundamentalsError

COLUMNS = "Name,Value,contextRef,unitRef,Dec,Prec,Lang,EntityScheme,EntityIdentifier,Period,Dimensions"
TOLERANCE = 1e-9


def available() -> bool:
    return importlib.util.find_spec("arelle") is not None


def wanted(hint: dict | None = None, env=None) -> bool:
    env = os.environ if env is None else env
    return bool((hint or {}).get("arelle")) or str(env.get("FUNDAMENTALS_ARELLE", "")).strip().lower() in ("1", "true", "yes", "on")


def run(data: bytes, name: str, cache_dir: str | Path | None = None, online: bool = True, validate: bool = True, timeout: int = 900) -> dict:
    """Run Arelle on a filing and return ``{"facts": [...], "messages": [...]}``; the facts are dicts of name, value, unit, start, end and dimensions."""
    if not available():
        raise FundamentalsError("Arelle is not installed (pip install arelle-release)")
    suffix = Path(name).suffix.lower() or ".xbrl"
    with tempfile.TemporaryDirectory() as tmp:
        src, out, log = Path(tmp) / f"filing{suffix}", Path(tmp) / "facts.csv", Path(tmp) / "log.txt"
        src.write_bytes(data)
        cmd = [sys.executable, "-m", "arelle.CntlrCmdLine", "--file", str(src), "--plugins", "inlineXbrlDocumentSet", "--factListCols", COLUMNS, "--facts", str(out), "--logFile", str(log),
               "--logFormat", "%(levelname)s|[%(messageCode)s] %(message)s", "--internetConnectivity", "online" if online else "offline"]
        if cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            cmd += ["--cacheDirectory", str(cache_dir)]
        if validate:
            cmd.append("--validate")
        try:
            subprocess.run(cmd, check=False, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise FundamentalsError(f"Arelle did not finish within {timeout} seconds") from None
        if not out.exists():
            raise FundamentalsError("Arelle could not read this file as an XBRL filing")
        csv.field_size_limit(10_000_000)
        facts = []
        with open(out, encoding="utf-8-sig", newline="") as fh:
            for r in csv.DictReader(fh):
                value = r.get("Value")
                try:
                    number = float(value) if value not in (None, "") else None
                except ValueError:
                    number = None
                facts.append({"name": r.get("Name"), "value": number, "text": value if number is None else None, "unit": r.get("unitRef") or None, "start": r.get("Start") or None, "end": r.get("End/Instant") or None,
                              "dims": r.get("Dimensions") or ""})
        messages = []
        if log.exists():
            for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
                level, _, rest = line.partition("|")
                if level in ("ERROR", "WARNING", "INCONSISTENCY") and rest and "unrecognizedFactListColumn" not in rest:
                    messages.append({"level": level.lower(), "message": rest[:300]})
        return {"facts": facts, "messages": messages}


def compare(read: dict, arelle: dict) -> dict:
    """This reader's facts (``xbrl.read_facts``) against Arelle's. Only company-wide numeric facts are compared value by value; counts are for every fact."""
    contexts = read["contexts"]
    ours = {}
    for f in read["facts"]:
        c = contexts.get(f["context"])
        if c is None or c["dimensional"] or c["end"] is None:
            continue
        ours.setdefault((f["field"], c["start"].isoformat() if c["start"] else None, c["end"].isoformat()), f["value"])
    theirs = {}
    for f in arelle["facts"]:
        if f["value"] is not None and not f["dims"] and f["end"]:
            theirs.setdefault((f["name"], f["start"], f["end"]), f["value"])
    matched = [k for k in ours if k in theirs and abs(ours[k] - theirs[k]) <= TOLERANCE * max(abs(ours[k]), 1.0)]
    different = [{"fact": k[0], "period": f"{k[1] or ''}..{k[2]}", "this_reader": ours[k], "arelle": theirs[k]} for k in ours if k in theirs and k not in matched]
    only_ours = [k for k in ours if k not in theirs]
    known = ("us-gaap:", "ifrs-full:", "dei:", "jppfs_cor:", "jpigp_cor:", "jpdei_cor:", "jpcrp_cor:")
    only_theirs = [k for k in theirs if k not in ours and k[0].startswith(known)]
    extension = sum(1 for k in theirs if k not in ours and not k[0].startswith(known))                    # the company's own elements: this reader maps only the standard taxonomies
    concept_only = bool(arelle["facts"]) and not any(f["value"] is not None for f in arelle["facts"])
    errors = [m for m in arelle["messages"] if m["level"] in ("error", "inconsistency")]
    return {"engine": "arelle", "facts_arelle": len(arelle["facts"]), "facts_this_reader": len(read["facts"]) + len(read["cover"]), "compared": len(ours), "agree": len(matched), "different": different[:30],
            "only_this_reader": [{"fact": k[0], "period": f"{k[1] or ''}..{k[2]}"} for k in only_ours[:30]], "only_arelle": [{"fact": k[0], "period": f"{k[1] or ''}..{k[2]}"} for k in only_theirs[:30]],
            "validation_errors": len(errors), "validation_warnings": sum(1 for m in arelle["messages"] if m["level"] == "warning"), "messages": arelle["messages"][:20],
            "extension_facts": extension, "no_values": concept_only, "agrees": not different and not concept_only and not only_theirs and not only_ours, "valid": not errors}


def crosscheck(data: bytes, name: str, cache_dir=None, online: bool = True, package: bytes | None = None) -> dict:
    """Read ``data`` with this application's reader and the same filing with Arelle: from the whole report ``package`` (a ZIP) when there is one, because the company's own taxonomy files are in it."""
    from . import xbrl
    return compare(xbrl.read_facts(data), run(package, "package.zip", cache_dir, online) if package else run(data, name, cache_dir, online))
