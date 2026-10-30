"""A smoke check of every live data feed the application reads, for a scheduled run: does each public feed still answer, and does the answer still have the shape the parsers expect?

The test suite never touches the network (a feed that changes its fields would not fail it), so this is the one place that does. It fetches a small, fixed set of things from the keyless feeds (Cboe,
Deribit, four exchanges, Yahoo Finance, and the futures curves: Cboe's VIX futures files, Yahoo's contracts, Deribit's expired futures) and from the SEC (a company's facts and record, the list of tickers, and the list of companies that report financial data, which stands in for the tickers
when the SEC refuses a program the ticker file), checks sizes and dates rather than values, and says for each one:

``ok``       it answered and looked right;
``refused``  the server would not talk to this address (HTTP 403, 429 or 451: a shared cloud address is often refused; nothing is known about the feed);
``broken``   it answered with something the parsers cannot use, or did not answer: the feed changed or is down;
``skipped``  it cannot be judged now: it needs something that is not set (the SEC wants a contact), or the options market is closed and Cboe's delayed quotes carry no offers (it answered in the
             shape the parser reads, so nothing is known against the feed).

The exit code is 1 if anything is ``broken`` (with ``--strict``, ``refused`` too), so a scheduled workflow fails exactly when a feed changed. iTick is not checked: it needs a key, and the author has
never had a working one.

    python experiments/live_feed_smoke.py [--only cboe,sec] [--json FILE] [--strict]

The SEC contact is read from ``SEC_EDGAR_USER_AGENT`` and from nowhere else: nothing is invented, so without it the SEC checks are ``skipped``. Finzdata's key is read from ``FINZDATA_API_KEY`` and from
nowhere else, and without it the ``finzdata`` check is ``skipped``; with it the ``finzdata`` check spends about five of the 1,000 calls a day, none of its data is kept, and its answer names the columns the service
really sends (its OpenAPI document leaves them undefined), so that a parser which guessed a name wrongly can be corrected from one run. ``finzdata_reach`` spends about nine more to say what the plan is
and, for the prices, crypto, options, fundamentals and positioning endpoints that no tab reads yet, which columns their rows have; it judges nothing and fails only when the key is refused.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd                                                     # noqa: E402

from src.assets import futures_analytics as fa                         # noqa: E402
from src.assets.free_data import cboe_futures, crypto, finzdata as fz, futures, options   # noqa: E402
from src.assets.free_data.http import FetchError, Http                 # noqa: E402
from src.fundamentals.errors import FundamentalsError, NotFound, ProviderError, RateLimited   # noqa: E402
from src.fundamentals.providers.sec_edgar import SecEdgarProvider      # noqa: E402
from src.webapp.finzdata import FinzdataClient, FinzdataError          # noqa: E402

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


def cboe_vx(ctx) -> str:
    """Cboe's own list of VIX futures (one call) and the file of the nearest contract that has not expired (one more): the list must hold the contracts since 2013 and the file a recent settlement with open interest."""
    http = ctx["http"]
    listed = cboe_futures.contracts(http.get_json(cboe_futures.LIST_URL))
    expect(len(listed) >= 100, f"{len(listed)} monthly VX contracts in Cboe's list")
    today = pd.Timestamp.now().normalize()
    alive = [c for c in listed if c["expiry"] >= today]
    expect(len(alive) >= 4, f"{len(alive)} VX contracts that have not expired")
    nearest = alive[0]
    frame = cboe_futures.parse_contract(http.get_text(cboe_futures.FILE_URL.format(path=nearest["path"])), nearest["expiry"])
    expect(len(frame) >= 20 and float(frame["price"].iloc[-1]) > 0, f"{nearest['contract']}: {len(frame)} days")
    expect(bool(frame["open_interest"].notna().any()), f"{nearest['contract']} has no open interest")
    fresh(frame["date"].max(), 7, f"the {nearest['contract']} file")
    return f"VX: {len(listed)} monthly contracts since {listed[0]['expiry'].year}; {nearest['contract']} (expires {nearest['expiry'].date()}) has {len(frame)} days to {frame['date'].max().date()}"


def yahoo_futures(ctx) -> str:
    d = futures.fetch_yahoo_curve(ctx["http"], "CL", 4, "6mo")
    expect(d.frame["contract"].nunique() >= 3 and len(d.frame) >= 200, f"{d.frame['contract'].nunique()} contracts, {len(d.frame)} rows")
    fresh(d.meta["as_of"], 7, "Yahoo's crude oil contracts")
    return f"CL: {d.frame['contract'].nunique()} contracts, {len(d.frame)} rows to {d.meta['as_of']}"


def deribit_futures(ctx) -> str:
    """The expired Bitcoin futures of the last year and a bit, from Deribit's chart service (about twenty calls): the contracts that have expired must still be there."""
    today = pd.Timestamp.now().normalize()
    d = futures.fetch_deribit_history(ctx["http"], "BTC", first_year=today.year - 1)
    expired = int(d.frame[pd.to_datetime(d.frame["expiry"]) < today]["contract"].nunique())
    expect(expired >= 4 and d.frame["contract"].nunique() >= 6, f"{expired} expired contracts of {d.frame['contract'].nunique()}")
    fresh(d.meta["as_of"], 4, "Deribit's futures")
    return f"BTC: {d.frame['contract'].nunique()} contracts, {expired} of them expired, {len(d.frame)} rows to {d.meta['as_of']}"


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


def sec_filers(ctx) -> str:
    names = _sec(ctx).filers()
    expect(len(names.rows) >= 4000, f"{len(names.rows)} companies in {names.period}")
    hit = next(iter(names.search("apple inc", 1)), None)
    expect(hit is not None and hit[1][0] == 320193, f"Apple Inc. is {hit}")
    return f"{len(names.rows)} companies reported assets in {names.period}; Apple Inc. is CIK {hit[1][0]}"


def _finzdata_client(ctx) -> FinzdataClient:
    key = ctx.get("finzdata_key")
    if not key:
        raise Skipped("no key for Finzdata: set the FINZDATA_API_KEY secret")
    return FinzdataClient(key=key, transport=ctx.get("finzdata_transport"), sleep=ctx.get("finzdata_sleep", time.sleep), pause=ctx.get("finzdata_pause", 0.2))


def finzdata(ctx) -> str:
    """Finzdata's futures: the list of markets, then for the E-mini S&P 500 and crude oil the contract list and the daily bars of every contract (the same path as a download), and that the Futures tab
    can list every contract it got. Judged on counts and dates, not values; the answer names the columns the rows really have."""
    client = _finzdata_client(ctx)
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    roots_answer = client.roots()
    roots = fz.normalise_roots(roots_answer)
    expect(len(roots) >= 100 and {"ES", "CL", "ZN"} <= {r["root"] for r in roots}, f"{len(roots)} markets; ES, CL and ZN must be among them (its page lists 139)")
    seen, said, columns = [], [], {"roots": ",".join(dict.fromkeys(k for r in fz.rows_of(roots_answer, "the markets")[:20] for k in r))}
    for root, floor in (("ES", 8), ("CL", 20)):
        d = fz.fetch_curve(client, root, start=today - pd.Timedelta(days=12), today=today)       # the contract list, then the bars of every contract: the path of a download
        frame, meta = d.frame, d.meta
        unread = next((n for n in meta["notes"] if "could not be read" in n), "")
        expect(meta["contracts_listed"] >= floor, f"{root}: {meta['contracts_listed']} contracts listed (a market with a contract every {'quarter' if root == 'ES' else 'month'} has at least {floor}) {unread}".strip())
        expect(frame["contract"].nunique() >= 2 and len(frame) >= 10, f"{root}: {frame['contract'].nunique()} contracts and {len(frame)} rows of bars in the last 12 days")
        fresh(frame["date"].max(), 8, f"Finzdata's {root} bars")
        expect(bool((frame["price"] > 0).all()) and bool((frame["expiry"] >= frame["date"]).all()), f"{root}: a price that is not positive, or a bar after its contract's expiry")
        with tempfile.TemporaryDirectory() as tmp:                                      # the table is read back the way the Futures tab reads a download, and not kept
            path = Path(tmp) / f"{root}.csv"
            frame.to_csv(path, index=False)
            data = fa.load_curve_file(path, root, source="downloaded")
        every = fa.curve_snapshot(data, None)
        columns["contracts"], columns["daily"] = ",".join(meta["columns"]["contracts"]), ",".join(meta["columns"]["daily"])
        expect(len(every["rows"]) == every["listed"] == int((frame["date"] == frame["date"].max()).sum()), f"{root}: the Futures tab lists {len(every['rows'])} of {every['listed']} contracts on {every['date']}")
        seen.append(f"{root}: {meta['contracts_listed']} listed ({','.join(meta['expiry_basis'])}), {frame['contract'].nunique()} with bars, {every['listed']} on {every['date']}, {meta['mode']}, oi {'yes' if meta['has_open_interest'] else 'no'}")
        said += [n for n in meta["notes"] if "plan" in n.lower() or "30 days" in n][:1]
    return "; ".join([f"columns - markets: {columns['roots']} | contracts: {columns['contracts']} | daily: {columns['daily']}", *seen, *(["plan: " + said[0]] if said else [])])


# What the key can read besides futures, for the tabs Finzdata could feed next. The service's OpenAPI document leaves every row undefined, so the columns are only known by asking; the answer of this check
# is how they are found out. Nothing here is judged: a dataset the plan leaves out, or a request the service does not understand, is told in a line and the run goes on.
REACH = (("prices", "/v1/prices/daily", {"symbol": "SPY", "limit": 3, "format": "json"}),
         ("crypto pairs", "/v1/crypto/pairs", {}),
         ("crypto bars", "/v1/crypto/bars", {"symbol": None, "limit": 3, "format": "json"}),
         ("gex", "/v1/gex/daily", {"ticker": "SPY", "limit": 3, "format": "json"}),
         ("options", "/v1/options/eod", {"ticker": "SPY", "limit": 3, "format": "json"}),
         ("fundamentals", "/v1/fundamentals", {"ticker": "AAPL", "limit": 3, "format": "json"}),
         ("cot", "/v1/cot", {"limit": 3, "format": "json"}))
PLAN_KEY = re.compile(r"plan|tier|limit|quota|remain|window|history|calls|rate|days", re.I)
PRIVATE_KEY = re.compile(r"mail|key|token|secret|user|owner|customer|account|address|^ip", re.I)


def plan_fields(answer) -> str:
    """The plan and the allowance out of ``/v1/me``, and nothing that names a person: numbers and short words under keys that talk about the plan, never an address, a key or an id."""
    flat: dict[str, object] = {}
    for k, v in (answer.items() if isinstance(answer, dict) else []):
        if isinstance(v, dict):
            flat.update({f"{k}.{kk}": vv for kk, vv in v.items()})
        else:
            flat[str(k)] = v
    kept = []
    for k, v in flat.items():
        text = str(v)
        if v is None or isinstance(v, (dict, list)) or not PLAN_KEY.search(k) or PRIVATE_KEY.search(k) or "@" in text or text.startswith("fz_") or len(text) > 40:
            continue
        kept.append(f"{k}={text}")
    return ", ".join(kept[:8]) or "no plan fields recognised"


def dataset_names(answer) -> list[str]:
    if isinstance(answer, dict):
        listed = answer["data"] if isinstance(answer.get("data"), list) else next((v for v in answer.values() if isinstance(v, list)), None)
        if listed is None:
            return [str(k)[:24] for k in answer if k not in ("history_note", "upgrade")][:14]
    else:
        listed = answer if isinstance(answer, list) else []
    return [str((r.get("key") or r.get("id") or r.get("name") or r.get("dataset") or "?") if isinstance(r, dict) else r)[:24] for r in listed][:14]


def _look(client: FinzdataClient, name: str, path: str, params: dict):
    """``(line, rows)`` for one request: the number of rows and their columns, or a sentence on why there are none. A refused key or an used-up day is raised: it ends the check."""
    try:
        answer = client.get(path, params)
    except FinzdataError as exc:
        if exc.kind in ("key", "quota", "budget"):
            raise
        return f"{name}: " + ("not in the plan" if exc.kind == "plan" else f"HTTP {exc.status}" if exc.status else exc.kind), []
    try:
        rows = fz.rows_of(answer, name)
    except fz.ShapeError:
        return f"{name}: a document with keys {','.join(sorted(answer)[:8]) if isinstance(answer, dict) else type(answer).__name__}", []
    if not rows:
        return f"{name}: no rows", []
    return f"{name}: {len(rows)} rows ({','.join(list(dict.fromkeys(k for r in rows[:20] for k in r))[:10])})", rows


def finzdata_reach(ctx) -> str:
    """The plan, the datasets, and for each tab that Finzdata might feed one call that shows the columns of its rows (nothing is kept, nothing is judged)."""
    client = _finzdata_client(ctx)
    me = client.me()
    lines = [f"plan: {plan_fields(me)}", f"datasets: {','.join(dataset_names(client.get('/v1/datasets'))) or 'none listed'}"]
    pair = None
    for name, path, params in REACH:
        if name == "crypto bars" and pair is None:
            lines.append("crypto bars: no pair to ask for")
            continue
        line, rows = _look(client, name, path, {**params, "symbol": pair} if name == "crypto bars" else params)
        lines.append(line)
        if name == "crypto pairs" and rows:
            names = [str(v) for r in rows for k, v in r.items() if str(k).lower() in ("symbol", "pair", "id", "name") and isinstance(v, str)]
            pair = next((n for n in names if "BTC" in n.upper()), names[0] if names else None)
    return "reach (not judged) - " + " | ".join(lines)


CHECKS = {"cboe": cboe, "deribit": deribit, "exchanges": exchanges, "crypto_history": crypto_history, "okx_funding": okx_funding, "yahoo": yahoo, "cboe_vx": cboe_vx, "yahoo_futures": yahoo_futures,
          "deribit_futures": deribit_futures, "finzdata": finzdata, "finzdata_reach": finzdata_reach, "sec_facts": sec_facts, "sec_record": sec_record, "sec_tickers": sec_tickers, "sec_filers": sec_filers}


# ------------------------------------------------------------------------------------------------------------------ running them
def classify(error: BaseException) -> tuple[str, str]:
    """``(status, detail)`` for a check that raised: a refusal to this address is told apart from a feed that changed."""
    if isinstance(error, Skipped):
        return "skipped", str(error)
    if isinstance(error, options.NoLiveOffers):                            # a chain with no offer at all is a closed market, not a feed that changed (the scheduled run of 26 October met it before the open)
        return "skipped", f"the options market is closed, so there is nothing to judge: {error}"
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
        out.append(Result(name, status, str(detail)[:900], round(clock() - started, 2)))
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
    ctx = {"http": Http(transport=transport, timeout=30.0), "sec_contact": contact, "finzdata_key": os.environ.get("FINZDATA_API_KEY", "").strip()}
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
