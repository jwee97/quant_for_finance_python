"""Top up the stored Finzdata curves without opening the app, for a daily job.

Finzdata's free plan shows the last 30 days of futures, so a history only grows if it is downloaded again before the window slides past it. The Futures tab does this with its Update button; this script
does the same from a terminal, a cron line or a Replit scheduled deployment, with the same client, the same call ledger (so the day's allowance is shared with the app) and the same stored files
(``data/user/downloads/futures/<ROOT>.csv``, git-ignored: Finzdata's terms forbid publishing its data).

    python experiments/finzdata_update.py                 # every root already updated from Finzdata (the app's list of stored markets)
    python experiments/finzdata_update.py ES CL ZN        # these markets (a first download of a new one asks for everything the plan shows)
    python experiments/finzdata_update.py --status        # what is stored, how old, how many days before the window passes it; no call is made

The key is read from ``FINZDATA_API_KEY`` (or a git-ignored ``.env``) and from nowhere else. The exit code is 0 when every market was updated, 1 when any failed, 2 when there is no key or nothing to do.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.assets.free_data.store import DownloadStore                       # noqa: E402
from src.webapp.data_finzdata import MAX_CALLS_PER_ROOT, FinzdataData, check_root   # noqa: E402
from src.webapp.finzdata import FinzdataClient, FinzdataError                    # noqa: E402


def status_lines(svc: FinzdataData) -> list[str]:
    stored = svc.stored()
    if not stored:
        return ["nothing is stored from Finzdata yet"]
    lines = [f"{'root':<6} {'stored':<23} {'days':>5} {'contracts':>9} {'age':>4} {'left':>5}  gaps"]
    for s in stored:
        left = "-" if s["left"] is None else str(s["left"])
        flag = "  <- update now" if s["urgent"] else ""
        lines.append(f"{s['root']:<6} {s['stored_from']} to {s['stored_to']} {s['days']:>5} {s['contracts']:>9} {s['age']:>4} {left:>5}  {s['gap_count']}{flag}")
    return lines


def main(argv=None, root: Path | None = None, transport=None, today=None, out=print) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("roots", nargs="*", help="Finzdata symbols to update (default: every market already stored)")
    parser.add_argument("--status", action="store_true", help="show what is stored and make no call")
    parser.add_argument("--max-calls", type=int, default=MAX_CALLS_PER_ROOT, help="calls one market may use (default %(default)s)")
    args = parser.parse_args(argv)
    base = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    client = FinzdataClient(root=base, transport=transport, sleep=lambda s: None, pause=0.0) if transport is not None else None             # a transport is only ever a test's
    svc = FinzdataData(base, DownloadStore(base), client, today=today)
    if args.status:
        for line in status_lines(svc):
            out(line)
        return 0
    ok, why = svc.client.available()
    if not ok:
        out(why)
        return 2
    try:
        roots = [check_root(r) for r in args.roots] or list(svc.registry())
    except ValueError as exc:
        out(str(exc))
        return 2
    if not roots:
        out("nothing to update: name the markets (python experiments/finzdata_update.py ES CL) or download one in the Futures tab first")
        return 2
    failed = 0
    for i, r in enumerate(roots):
        try:
            result = svc.update_and_save(r, max_calls=args.max_calls)
            out(f"{r:<6} ok      {result['message']}")
        except FinzdataError as exc:
            failed += 1
            out(f"{r:<6} FAILED  {exc}")
            if exc.fatal:                                                           # the key or the day's allowance: the rest would meet the same wall
                for rest in roots[i + 1:]:
                    failed += 1
                    out(f"{rest:<6} FAILED  not tried: {exc}")
                break
        except Exception as exc:                                                    # noqa: BLE001 - one market failing must not lose the others
            failed += 1
            out(f"{r:<6} FAILED  {type(exc).__name__}: {exc}")
    calls = svc.client.describe()
    out(f"{len(roots) - failed} of {len(roots)} updated; {calls['calls_today']} Finzdata calls made today, {calls['calls_left']} left of {calls['budget']}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
