"""Stable identities: a ticker change keeps the security (and its prices, filings and confirmed figures); a company listed twice is one issuer with two securities.

* ``rename_ticker`` closes the old name's validity and opens the new one; both stay resolvable *as of a date*, so a backtest that asks for the ticker a company had in 2015 gets the right security.
* ``link_listings`` groups securities of one issuer (a primary listing marked); ``one_per_issuer`` keeps one listing per issuer in a universe so a company on two exchanges is not counted twice in a
  cross-section.
* ``resolve`` never guesses: a name that matches several securities as of the date is *ambiguous* and says so.
"""

from __future__ import annotations

from datetime import date as _date, timedelta

from .errors import FundamentalsError


def _day(d) -> str:
    return str(d)[:10] if d else _date.today().isoformat()


def ensure_aliases(store, security_id: int) -> None:
    """The security's current name as an alias open since the start, so a first rename has something to close."""
    with store.db.transaction() as c:
        have = c.execute("SELECT COUNT(*) FROM security_aliases WHERE security_id = ?", (int(security_id),)).fetchone()[0]
        row = c.execute("SELECT ticker, exchange FROM securities WHERE security_id = ?", (int(security_id),)).fetchone()
        if row is None:
            raise FundamentalsError("no such security")
        if not have:
            c.execute("INSERT INTO security_aliases (security_id, ticker, exchange, valid_from, valid_to, note) VALUES (?, ?, ?, '1900-01-01', NULL, 'original name')", (int(security_id), row[0], row[1] or ""))


def rename_ticker(store, security_id: int, new_ticker: str, effective: str, new_exchange: str | None = None, note: str = "") -> None:
    ensure_aliases(store, security_id)
    new_ticker = new_ticker.strip().upper()
    if not new_ticker:
        raise FundamentalsError("the new ticker is empty")
    eff = _day(effective)
    with store.db.transaction() as c:
        cur = c.execute("SELECT ticker, exchange FROM securities WHERE security_id = ?", (int(security_id),)).fetchone()
        exch = cur[1] if new_exchange is None else new_exchange
        clash = c.execute("SELECT security_id FROM securities WHERE ticker = ? AND exchange = ? AND security_id != ?", (new_ticker, exch or "", int(security_id))).fetchone()
        if clash:
            raise FundamentalsError(f"{new_ticker} on {exch or 'no exchange'} is already security {clash[0]}: merge them explicitly, a rename never merges")
        end = (_date.fromisoformat(eff) - timedelta(days=1)).isoformat()
        c.execute("UPDATE security_aliases SET valid_to = ? WHERE security_id = ? AND valid_to IS NULL", (end, int(security_id)))
        c.execute("INSERT OR REPLACE INTO security_aliases (security_id, ticker, exchange, valid_from, valid_to, note) VALUES (?, ?, ?, ?, NULL, ?)", (int(security_id), new_ticker, exch or "", eff, note[:200]))
        c.execute("UPDATE securities SET ticker = ?, exchange = ? WHERE security_id = ?", (new_ticker, exch or "", int(security_id)))


def resolve(store, ticker: str, exchange: str | None = None, as_of: str | None = None) -> dict:
    """The security that carried ``ticker`` on ``as_of`` (today when omitted)."""
    day = _day(as_of)
    sql = "SELECT security_id, ticker, exchange, valid_from, valid_to FROM security_aliases WHERE ticker = ? AND valid_from <= ? AND (valid_to IS NULL OR valid_to >= ?)"
    args: list = [ticker.strip().upper(), day, day]
    if exchange is not None:
        sql += " AND exchange = ?"
        args.append(exchange)
    with store.db.transaction() as c:
        rows = c.execute(sql, args).fetchall()
        if not rows:                                                                                   # a security that never had a rename has no alias rows: its own name applies
            q = "SELECT security_id, ticker, exchange, '1900-01-01', NULL FROM securities WHERE ticker = ? AND security_id NOT IN (SELECT security_id FROM security_aliases)" + (" AND exchange = ?" if exchange is not None else "")
            rows = c.execute(q, args[:1] + ([exchange] if exchange is not None else [])).fetchall()
    ids = sorted({r[0] for r in rows})
    if not ids:
        return {"security_id": None, "status": "not found", "candidates": []}
    if len(ids) > 1:
        return {"security_id": None, "status": "ambiguous", "candidates": ids, "reason": "several securities carried this ticker on that date: give the exchange"}
    return {"security_id": ids[0], "status": "ok", "candidates": ids}


def history(store, security_id: int) -> list[dict]:
    ensure_aliases(store, security_id)
    with store.db.transaction() as c:
        rows = c.execute("SELECT ticker, exchange, valid_from, valid_to, note FROM security_aliases WHERE security_id = ? ORDER BY valid_from", (int(security_id),)).fetchall()
    return [dict(zip(("ticker", "exchange", "valid_from", "valid_to", "note"), r)) for r in rows]


def link_listings(store, security_ids: list[int], issuer_id: str, primary: int | None = None) -> None:
    ids = [int(i) for i in security_ids]
    if len(ids) < 2:
        raise FundamentalsError("an issuer with two or more listings: give at least two securities")
    with store.db.transaction() as c:
        for i in ids:
            c.execute("INSERT OR REPLACE INTO issuer_links (security_id, issuer_id, primary_listing) VALUES (?, ?, ?)", (i, issuer_id, 1 if (primary if primary is not None else ids[0]) == i else 0))


def one_per_issuer(store, security_ids: list[int]) -> list[int]:
    """The universe with one security per issuer: the marked primary listing, else the one with the most stored prices."""
    with store.db.transaction() as c:
        links = {r[0]: (r[1], r[2]) for r in c.execute("SELECT security_id, issuer_id, primary_listing FROM issuer_links").fetchall()}
        counts = dict(c.execute("SELECT security_id, COUNT(*) FROM prices_daily GROUP BY security_id").fetchall())
    best: dict[str, tuple] = {}
    keep = []
    for i in security_ids:
        if i not in links:
            keep.append(i)
            continue
        issuer, prim = links[i]
        score = (prim, counts.get(i, 0), -i)
        if issuer not in best or score > best[issuer][0]:
            best[issuer] = (score, i)
    return sorted(keep + [v[1] for v in best.values()])
