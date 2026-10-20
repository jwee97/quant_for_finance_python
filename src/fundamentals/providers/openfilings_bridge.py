"""Talk to the OpenFilings library (https://github.com/ryanrodrigues25200525-svg/Openfilings) from a separate process and print JSON.

OpenFilings needs Python 3.11 or later and brings its own dependencies (one of them, PyMuPDF, is AGPL-licensed), so the application does not import it. It runs this script with the interpreter
that has OpenFilings installed (``OPENFILINGS_PYTHON``, or the interpreter running the application when that has it) and reads the JSON on standard output. This file uses only the standard
library and OpenFilings, so that interpreter needs nothing else.

    python openfilings_bridge.py probe
    python openfilings_bridge.py search "Nokia" --source esef --limit 5
    python openfilings_bridge.py financials <company id> --source esef --filings 8

Every answer is one JSON object. A failure is ``{"error": ..., "detail": ...}`` and exit status 2; ``probe`` lists what the installed OpenFilings offers, which is the first thing to send when
something does not line up (the library is young and its interface has changed between releases; this bridge was written against its documented models for version 0.22).
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import sys


def dump(obj):
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if isinstance(obj, (list, tuple, set)):
        return [dump(o) for o in obj]
    if isinstance(obj, dict):
        return {str(k): dump(v) for k, v in obj.items()}
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return obj


async def call(fn, *args, **kwargs):
    """Call ``fn`` with the keyword arguments it accepts (the library's signatures differ a little between versions), awaiting the result if it is awaitable."""
    try:
        params = inspect.signature(fn).parameters
        takes_all = any(p.kind is p.VAR_KEYWORD for p in params.values())
        kwargs = {k: v for k, v in kwargs.items() if takes_all or k in params}
    except (TypeError, ValueError):
        pass
    out = fn(*args, **kwargs)
    return await out if inspect.isawaitable(out) else out


def fail(message: str, detail: str = "", code: int = 2):
    print(json.dumps({"error": message, "detail": detail}))
    sys.exit(code)


def load():
    try:
        import openfilings
    except ImportError as error:
        fail("openfilings is not installed in this interpreter", str(error))
    return openfilings


async def service(openfilings):
    client = getattr(openfilings, "OpenFilings", None)
    if client is None:
        fail("the openfilings package has no OpenFilings class", ", ".join(sorted(n for n in dir(openfilings) if not n.startswith("_"))))
    factory = getattr(client, "from_settings", None)
    return factory() if factory else client()


def public(obj) -> list[str]:
    return sorted(n for n in dir(obj) if not n.startswith("_"))


async def run(args) -> dict:
    openfilings = load()
    cm = await service(openfilings)
    async with cm as svc:
        if args.command == "probe":
            return {"version": getattr(openfilings, "__version__", None), "package": public(openfilings), "service": public(svc)}
        if args.command == "search":
            fn = getattr(svc, "search_companies", None)
            if fn is None:
                fail("this OpenFilings has no search_companies", ", ".join(public(svc)))
            hits = await call(fn, args.query, limit=args.limit, source=args.source)
            return {"companies": dump(hits)}
        if args.command == "financials":
            return await financials(svc, args)
    fail(f"unknown command {args.command!r}")


async def financials(svc, args) -> dict:
    lister = getattr(svc, "list_filings", None) or getattr(svc, "get_filings", None)
    extractor = getattr(svc, "get_filing_financials", None)
    if lister is None or extractor is None:
        fail("this OpenFilings has neither list_filings nor get_filing_financials", ", ".join(public(svc)))
    filings = await call(lister, args.company_id, category=args.category, limit=args.filings, source=args.source)
    filings = list(getattr(filings, "items", filings))
    out = []
    for filing in filings[: args.filings]:
        record = dump(filing)
        try:
            record["financials"] = dump(await call(extractor, record["id"]))
        except Exception as error:                              # one filing that cannot be read does not end the others
            record["financials"] = None
            record["extraction_error"] = f"{type(error).__name__}: {str(error)[:200]}"
        out.append(record)
    return {"company_id": args.company_id, "filings": out}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("probe")
    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("--source", default="all")
    s.add_argument("--limit", type=int, default=10)
    f = sub.add_parser("financials")
    f.add_argument("company_id")
    f.add_argument("--source", default="all")
    f.add_argument("--filings", type=int, default=8)
    f.add_argument("--category", default="accounts")
    args = ap.parse_args(argv)
    try:
        result = asyncio.run(run(args))
    except SystemExit:
        raise
    except Exception as error:
        fail(f"{type(error).__name__}", str(error)[:500])
    print(json.dumps(result, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
