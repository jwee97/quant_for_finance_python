"""A smoke check of every live data feed the application reads, for a scheduled run: does each public feed still answer, and does the answer still have the shape the parsers expect?

The test suite never touches the network (a feed that changes its fields would not fail it), so this is the one place that does. It fetches a small, fixed set of things from the keyless feeds (Cboe,
Deribit, four exchanges, Yahoo Finance) and from the SEC (a company's facts and record, and the list of tickers), checks sizes and dates rather than values, and says for each one:

``ok``       it answered and looked right;
``refused``  the server would not talk to this address (HTTP 403, 429 or 451: a shared cloud address is often refused; nothing is known about the feed);
``broken``   it answered with something the parsers cannot use, or did not answer: the feed changed or is down;
``skipped``  it needs something that is not set (the SEC wants a contact).

The exit code is 1 if anything is ``broken`` (with ``--strict``, ``refused`` too), so a scheduled workflow fails exactly when a feed changed. iTick is not checked: it needs a key, and the author has
never had a working one.

    python experiments/live_feed_smoke.py [--only cboe,sec] [--json FILE] [--strict]

The SEC contact is read from ``SEC_EDGAR_USER_AGENT``; inside GitHub Actions without it, the address of the repository stands in (the SEC asks for a contact, and a repository page is one).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.assets.free_data import crypto, options                      # noqa: E402
from src.assets.free_data.http import FetchError, Http                 # noqa: E402
from src.fundamentals.errors import FundamentalsError, NotFound, ProviderError, RateLimited   # noqa: E402
from src.fundamentals.providers.sec_edgar import SecEdgarProvider      # noqa: E402

REFUSED_STATUS = (403, 429, 451)
STATUSES = ("ok", "refused", "broken", "skipped")


class Broken(AssertionError):
    """An answer that came but does not look like what the parsers expect."""


class Skipped(Exception):
    """A check that cannot run here, with the reason."""


@dataclass
class Result:
    name: str
    status: str
    detail: str
    seconds: float


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise Broken(message)


def fresh(day, days: int, what: str) -> None:
    """The last observation is not older than ``days`` days (a feed that stopped updating answers, and is wrong)."""
    when = date.fromisoformat(str(day)[:10])
    age = (datetime.now(timezone.utc).date() - when).days
    expect(age <= days, f"{what} ends {when}, {age} days ago")


# ------------------------------------------------------------------------------------------------------------------ the checks
def cboe(ctx) -> str:
    d = options.fetch_cboe_chain(ctx["http"], "SPY")
    expect(d.meta["contracts"] >= 200 and d.meta["spot"] > 0 and d.meta["expiries"] >= 5, f"{d.meta['contracts']} contracts, {d.meta['expiries']} expiries")
    fresh(d.meta["as_of"], 7, "the Cboe quotes")
    return f"SPY: {d.meta['contracts']} contracts over {d.meta['expiries']} expiries, spot {d.meta['spot']:.2f}, as of {d.meta['as_of']}"


def deribit(ctx) -> str:
    d = options.fetch_deribit_chain(ctx["http"], "BTC")
    expect(len(d.frame) >= 50, f"{len(d.frame)} contracts")
    return f"BTC: {len(d.frame)} contracts, as of {d.meta['as_of']}"


def exchanges(ctx) -> str:
    d = crypto.fetch_venues(ctx["http"], "BTC")
    got = d.payload["venues"]
    expect(len(got) >= 2, f"{len(got)} venues gave a book: " + "; ".join(f"{k}: {v}" for k, v in d.payload["failed"].items()))
    return f"BTC order books on {', '.join(v['label'] for v in got)}" + (f" (not: {', '.join(d.payload['failed'])})" if d.payload["failed"] else "")


def crypto_history(ctx) -> str:
    d = crypto.fetch_history(ctx["http"], "BTC", 260)
    expect(len(d.frame) >= 250, f"{len(d.frame)} days")
    fresh(d.meta["as_of"], 4, "the daily history")
    return f"BTC: {len(d.frame)} daily closes to {d.meta['as_of']} from {d.meta['source']}"


def okx_funding(ctx) -> str:
    d = crypto.fetch_funding(ctx["http"], "BTC", 60)
    expect(len(d.frame) >= 40, f"{len(d.frame)} days")
    return f"BTC funding: {len(d.frame)} days from OKX"


def yahoo(ctx) -> str:
    d = options.fetch_underlying_prices(ctx["http"], "SPY", 1.0)
    expect(len(d.frame) >= 200 and float(d.frame["close"].iloc[-1]) > 0, f"{len(d.frame)} daily prices")
    fresh(d.meta["as_of"], 7, "Yahoo Finance's daily prices")
    return f"SPY: {len(d.frame)} daily prices to {d.meta['as_of']}"


def _sec(ctx) -> SecEdgarProvider:
    contact = ctx.get("sec_contact")
    if not contact:
        raise Skipped("no contact for the SEC: set SEC_EDGAR_USER_AGENT to a name and an email address")
    return SecEdgarProvider(user_agent=contact)


def sec_facts(ctx) -> str:
    result = _sec(ctx).get_financials("cik:320193")
    periods = result.periods
    expect(len(periods) >= 15, f"{len(periods)} annual statements")
    expect(result.profile.name.lower().startswith("apple"), f"the company is called {result.profile.name!r}")
    expect(all(p.filing_date and p.filing_date > p.period_end for p in periods), "a statement was filed before its period ended")
    fresh(periods[-1].period_end, 500, "the latest annual statement")
    values = periods[-1].values
    revenue = next((v for k, v in values.items() if k.endswith("Revenues") or "RevenueFromContract" in k), None)
    expect(revenue is not None and revenue > 1e10, f"revenue {revenue}")
    return f"Apple: {len(periods)} annual statements, FY{periods[0].fiscal_year} to FY{periods[-1].fiscal_year}, latest filed {periods[-1].filing_date}"


def sec_record(ctx) -> str:
    profile = _sec(ctx).get_company("cik:320193")
    expect(profile.sector == "Technology" and profile.industry, f"sector {profile.sector!r}, industry {profile.industry!r}")
    return f"Apple: {profile.name}, {profile.sector}, {profile.industry}"


def sec_tickers(ctx) -> str:
    listing = _sec(ctx).tickers(refresh=True)
    expect(len(listing.rows) >= 5000, f"{len(listing.rows)} tickers")
    row = listing.find("AAPL")
    expect(row is not None and row[0] == 320193, f"AAPL is {row}")
    return f"{len(listing.rows)} tickers; AAPL is CIK {row[0]} ({row[3] or 'no exchange'})"


CHECKS = {"cboe": cboe, "deribit": deribit, "exchanges": exchanges, "crypto_history": crypto_history, "okx_funding": okx_funding, "yahoo": yahoo,
          "sec_facts": sec_facts, "sec_record": sec_record, "sec_tickers": sec_tickers}


# ------------------------------------------------------------------------------------------------------------------ running them
def classify(error: BaseException) -> tuple[str, str]:
    """``(status, detail)`` for a check that raised: a refusal to this address is told apart from a feed that changed."""
    if isinstance(error, Skipped):
        return "skipped", str(error)
    if isinstance(error, FetchError):
        return ("refused" if error.status in REFUSED_STATUS or error.blocked else "broken"), str(error)
    if isinstance(error, RateLimited):
        return "refused", str(error)
    if isinstance(error, (NotFound, ProviderError, FundamentalsError)):
        return ("refused" if "HTTP 403" in str(error) else "broken"), str(error)
    if isinstance(error, Broken):
        return "broken", f"the answer does not look right: {error}"
    return "broken", f"{type(error).__name__}: {error}"


def run_checks(checks: dict, ctx: dict, only: list[str] | None = None, clock=time.monotonic) -> list[Result]:
    out = []
    for name, check in checks.items():
        if only and name not in only and name.split("_")[0] not in only:
            continue
        started = clock()
        try:
            status, detail = "ok", check(ctx)
        except Exception as error:                                          # a check that fails for any reason is a result, never a crash of the run
            status, detail = classify(error)
        out.append(Result(name, status, str(detail)[:400], round(clock() - started, 2)))
    return out


def exit_code(results: list[Result], strict: bool = False) -> int:
    bad = {"broken", "refused"} if strict else {"broken"}
    return 1 if any(r.status in bad for r in results) else 0


def render(results: list[Result]) -> str:
    width = max(len(r.name) for r in results) if results else 4
    lines = [f"{r.name:<{width}}  {r.status:<8}  {r.seconds:>5.1f}s  {r.detail}" for r in results]
    counts = {s: sum(1 for r in results if r.status == s) for s in STATUSES}
    return "\n".join(lines + ["", "  ".join(f"{n} {s}" for s, n in counts.items())])


def markdown(results: list[Result]) -> str:
    rows = ["| feed | result | seconds | what it said |", "|---|---|---|---|"] + [f"| {r.name} | {r.status} | {r.seconds:.1f} | {r.detail.replace('|', '/')} |" for r in results]
    return "\n".join(["### Live feeds", ""] + rows)


def annotations(results: list[Result]) -> list[str]:
    """GitHub Actions annotation lines: an error for a broken feed, a warning for a refused one."""
    return [f"::{'error' if r.status == 'broken' else 'warning'} title=live feed {r.name} {r.status}::{r.detail}" for r in results if r.status in ("broken", "refused")]


def main(argv=None, transport=None, checks: dict | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", help="comma-separated checks (or groups: sec, cboe, ...) to run")
    parser.add_argument("--json", help="also write the results to this file")
    parser.add_argument("--strict", action="store_true", help="a refusal fails the run too")
    args = parser.parse_args(argv)
    contact = os.environ.get("SEC_EDGAR_USER_AGENT", "").strip()
    if not contact and os.environ.get("GITHUB_REPOSITORY"):
        contact = f"QuantLab live-feed check (https://github.com/{os.environ['GITHUB_REPOSITORY']})"
    ctx = {"http": Http(transport=transport, timeout=30.0), "sec_contact": contact}
    results = run_checks(checks or CHECKS, ctx, [x.strip() for x in args.only.split(",")] if args.only else None)
    print(render(results))
    if args.json:
        Path(args.json).write_text(json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(markdown(results) + "\n")
    for line in annotations(results):
        print(line)
    return exit_code(results, args.strict)


if __name__ == "__main__":
    sys.exit(main())
