"""HTTP routes of the crypto research workflows (``assets.crypto_research`` and ``assets.crypto_validation``): market overview, derivatives intelligence, carry economics, depth-aware execution,
strategy validation and liquidation risk. Every route follows the data mode of the request (observed-only unless ``data_mode=model``)."""

from __future__ import annotations

import pandas as pd

from ..assets import crypto_analytics as ca
from ..assets import crypto_research as cr
from ..assets import crypto_validation as cv
from .api import ApiError, _choice, _number


class CryptoResearchApi:
    def __init__(self, api):
        self.api = api

    def get_routes(self) -> dict:
        return {"/api/crypto/market": self.market, "/api/crypto/derivatives": self.derivatives, "/api/crypto/carry": self.carry, "/api/crypto/executable": self.executable,
                "/api/crypto/execution": self.execution, "/api/crypto/calendar": self.calendar, "/api/crypto/margin-schedules": self.margin_schedules, "/api/crypto/history": self.history}

    def post_routes(self) -> dict:
        return {"/api/crypto/validate": self.validate, "/api/crypto/liquidation": self.liquidation, "/api/crypto/margin-schedules/refresh": self.margin_refresh, "/api/crypto/history/record": self.history_record, "/api/crypto/replay": self.replay}

    def history(self, q: dict) -> dict:
        """``GET /api/crypto/history?coin=BTC``: what the recorder holds (books by venue with first/last time, rows, gaps; funding, open interest, futures), and when recording began. Nothing exists before that."""
        from ..assets.crypto_store import CryptoStore

        store, coin = CryptoStore(self.api.root), str(q.get("coin") or "BTC").upper()
        books = []
        base = store.dir / "books"
        for venue_dir in sorted(base.glob("*")) if base.is_dir() else []:
            for inst_dir in sorted(venue_dir.glob("*")):
                if coin in inst_dir.name.upper() or (coin == "BTC" and "XBT" in inst_dir.name.upper()):
                    c = store.coverage(venue_dir.name, inst_dir.name)
                    if c["state"] == "ok":
                        books.append({"venue": venue_dir.name, "instrument": inst_dir.name, **c})
        out = {"coin": coin, "books": books, "funding_rows": int(sum(len(store.funding(v.name, i.stem)) for v in (store.dir / "funding").glob("*") for i in v.glob("*.parquet"))) if (store.dir / "funding").is_dir() else 0,
               "bytes": store.size_bytes(), "reading": "Recording begins when the recorder is first run (`python -m src.cli crypto-record`); no order book exists for earlier periods, and a gap in time or sequence is marked, never filled."}
        return self._wrap("crypto_history", out, {"history": "downloaded" if books else "unavailable"})

    @staticmethod
    def _inst(body: dict, key: str) -> tuple:
        raw = str(body.get(key) or "")
        if ":" not in raw:
            raise ApiError(f"{key} must be VENUE:INSTRUMENT, for example OKX-PERP:BTC-USDT-SWAP")
        venue, inst = raw.split(":", 1)
        return venue.strip().upper(), inst.strip().upper()

    def replay(self, body: dict) -> dict:
        """``POST /api/crypto/replay``: replay recorded observations (``assets.crypto_replay``). ``strategy`` is ``cross_exchange`` (``a``, ``b``), ``funding_carry`` (``spot``, ``perp``, ``funding``), ``futures_basis``
        (``coin``) or ``execution`` (``a``, ``side``, ``notional``). Instruments are ``VENUE:INSTRUMENT``. Only recorded books, settlements and observations are used; a gap is reported, never filled."""
        from ..assets import crypto_replay as rp
        from ..assets.crypto_store import CryptoStore

        store, name = CryptoStore(self.api.root), str(body.get("strategy") or "")
        try:
            if name == "cross_exchange":
                out = rp.cross_exchange(store, self._inst(body, "a"), self._inst(body, "b"), _number(body, "notional", 10_000.0, 100.0, 1e8), (_number(body, "fee_a_bps", 10.0, 0.0, 200.0), _number(body, "fee_b_bps", 10.0, 0.0, 200.0)),
                                        _number(body, "max_age_s", 5.0, 0.5, 600.0), _number(body, "max_skew_s", 2.0, 0.1, 600.0), _number(body, "min_edge_bps", 0.0, -50.0, 500.0), _number(body, "leg_delay_s", 1.0, 0.0, 60.0),
                                        _number(body, "inventory_usd", 100_000.0, 100.0, 1e9))
            elif name == "funding_carry":
                out = rp.funding_carry(store, self._inst(body, "spot"), self._inst(body, "perp"), self._inst(body, "funding"), _number(body, "notional", 10_000.0, 100.0, 1e8), _number(body, "entry_annualised", 0.10, -1.0, 5.0),
                                       _number(body, "exit_annualised", 0.0, -1.0, 5.0), int(_number(body, "window_settlements", 9, 1, 200, integer=True)), (_number(body, "fee_spot_bps", 10.0, 0.0, 200.0), _number(body, "fee_perp_bps", 5.0, 0.0, 200.0)),
                                       _number(body, "max_age_s", 10.0, 0.5, 600.0), _number(body, "financing_apr", 0.0, 0.0, 1.0), _number(body, "leverage", 3.0, 1.0, 125.0), _number(body, "mmr", 0.005, 0.0005, 0.5))
            elif name == "futures_basis":
                out = rp.futures_basis(store, str(body.get("venue") or "DERIBIT").upper(), str(body.get("coin") or "BTC").upper(), _number(body, "entry_annualised", 0.05, -1.0, 5.0), _number(body, "fee_bps", 3.5, 0.0, 200.0),
                                       _number(body, "notional", 10_000.0, 100.0, 1e8))
            elif name == "execution":
                v, i = self._inst(body, "a")
                out = rp.execution_history(store, v, i, _choice(body, "side", "buy", ("buy", "sell")), _number(body, "notional", 10_000.0, 100.0, 1e9))
            else:
                raise ApiError("strategy must be cross_exchange, funding_carry, futures_basis or execution")
        except ValueError as exc:
            raise ApiError(str(exc)) from None
        parts = {"replay": "downloaded" if out.get("state") == "ok" else "unavailable"}
        return self._wrap("crypto_replay", out, parts, [], None, ["Historical replay of recorded observations: not a simulation of liquidity that was not observed."])

    def history_record(self, body: dict) -> dict:
        """``POST /api/crypto/history/record {coin}``: one sampling pass of public, read-only requests into the store."""
        from ..assets.crypto_store import CryptoStore, record_once
        from ..assets.free_data.http import Http

        coin = str(body.get("coin") or "BTC").upper()
        try:
            rep = record_once(Http(), CryptoStore(self.api.root), coin)
        except ValueError as exc:
            raise ApiError(str(exc)) from None
        return self._wrap("crypto_record", rep, {"history": "downloaded"})

    def margin_schedules(self, q: dict) -> dict:
        """``GET /api/crypto/margin-schedules``: every stored schedule with its versions, source and retrieval time; which venues have a verified schedule and which are hypothetical."""
        from ..assets.margin_schedules import ScheduleStore

        rows = ScheduleStore(self.api.root).listing()
        return self._wrap("crypto_margin_schedules", {"schedules": rows, "verified_venues": sorted({r["venue"] for r in rows}), "hypothetical_venues": sorted({"BINANCE", "BYBIT", "OKX", "DERIBIT"} - {r["venue"] for r in rows}),
                          "reading": "A venue without a stored verified schedule uses a hypothetical margin model, and no liquidation estimate built on it is claimed to be exchange-accurate."}, {"schedules": "downloaded" if rows else "unavailable"})

    def margin_refresh(self, body: dict) -> dict:
        """``POST /api/crypto/margin-schedules/refresh {symbol, mode}``: fetch OKX's published risk tiers for ``<symbol>-USDT`` (a read-only public request) and store them as a new version."""
        from ..assets.free_data.http import FetchError, Http
        from ..assets.margin_schedules import ScheduleStore, fetch_okx

        sym = str(body.get("symbol") or "BTC").upper()
        mode = _choice(body, "mode", "cross", ("cross", "isolated"))
        try:
            s = fetch_okx(Http(), f"{sym}-USDT", None, mode)
        except (FetchError, ValueError) as exc:
            raise ApiError(str(exc)) from None
        ScheduleStore(self.api.root).save(s)
        return self._wrap("crypto_margin_schedule", {"schedule": {k: v for k, v in s.items() if k != "tiers"}, "n_tiers": len(s["tiers"]), "first_tiers": s["tiers"][:3]}, {"schedule": "downloaded"}, [], s["retrieved_at"][:10], s["notes"])

    # ------------------------------------------------------------------------------------------------------------ helpers
    @staticmethod
    def _wrap(kind: str, data: dict, parts: dict, warnings=(), as_of=None, notes=()):
        from .asset_api import _wrap

        return _wrap(kind, data, parts, warnings, as_of, notes)

    def _observed_series(self, symbol: str):
        s = self.api._series(symbol)
        if self.api.observed() and not cr.is_observed(s.source):
            raise ApiError(f"observed-only mode: {symbol} is {s.source}. Download its history (Free data) or put {symbol}.csv in data/user/crypto/")
        return s

    def _open_interest(self, symbol: str):
        import pandas as pd

        path = self.api.downloads.files("crypto", "openinterest").get(f"openinterest_{symbol}")
        if path is None:
            return None
        try:
            return cr.open_interest_view(pd.read_csv(path), self.api.downloads.meta("crypto", f"openinterest_{symbol}"))
        except (OSError, ValueError):
            return None

    def _term(self, symbol: str):
        payload = self.api.downloads.read_json("crypto", f"futures_{symbol}")
        return cr.futures_term_structure(payload) if payload else None

    def _liquidations(self, symbol: str):
        views = [cr.liquidation_view(self.api.downloads.read_json("crypto", n)) for n in (f"liquidations_{symbol}", f"liquidations_{symbol}_DERIBIT") if self.api.downloads.read_json("crypto", n)]
        views = [v for v in views if v.get("state") == "ok"]
        if not views:
            return None
        return {**views[0], "venues": views} if len(views) > 1 else views[0]

    # ------------------------------------------------------------------------------------------------------------ routes
    def calendar(self, q: dict) -> dict:
        """``GET /api/crypto/calendar?symbol=BTC``: whether the coin's marks are a 24/7 daily calendar, the annualisation that fits, and what the weekday pipeline would drop."""
        s = self._observed_series(str(q.get("symbol") or "BTC"))
        return self._wrap("crypto_calendar", {"symbol": s.symbol, "audit": cr.calendar_audit(s.spot.index)}, {"prices": s.source}, [], s.as_of)

    def market(self, q: dict) -> dict:
        """``GET /api/crypto/market?symbol=BTC&days=365``: the overview tiles with percentiles over the chosen window."""
        s = self._observed_series(str(q.get("symbol") or "BTC"))
        out = cr.market_summary(s, int(_number(q, "days", 365, 30, 3000, integer=True)))
        return self._wrap("crypto_market", out, {"prices": s.source, "funding": s.source if s.funding is not None else "unavailable", "volume": "unavailable", "open_interest": "unavailable"}, [], s.as_of, s.notes)

    def derivatives(self, q: dict) -> dict:
        """``GET /api/crypto/derivatives?symbols=BTC,ETH``: funding per interval and annualised by venue with percentiles, regimes, dispersion, the basis, and what is unavailable."""
        from .asset_api import _list

        syms = _list(q.get("symbols") or "BTC,ETH", 6)
        series = {x.upper(): self._observed_series(x) for x in syms}
        try:
            first = next(iter(series))
            out = cr.funding_intelligence(series, observed_only=self.api.observed(), days=int(_number(q, "days", 365, 60, 2000, integer=True)), open_interest=self._open_interest(first), liquidations=self._liquidations(first), term=self._term(first))
        except ValueError as exc:
            raise ApiError(str(exc)) from None
        parts = {f"{r['symbol']}@{r['venue']}": r["source"] for r in out["rows"]}
        parts.update({"open_interest": "downloaded" if out["open_interest"].get("state") == "ok" else "unavailable", "liquidations": "downloaded" if out["liquidations"].get("state") == "ok" else "unavailable", "dated_futures": "downloaded" if out["dated_futures"].get("state") == "ok" else "unavailable"})
        return self._wrap("crypto_derivatives", out, parts, [], max(x.as_of for x in series.values()))

    def carry(self, q: dict) -> dict:
        """``GET /api/crypto/carry?symbol=BTC&notional=1000000&leverage=3``: the cash-and-carry split into funding, basis, delta, costs, financing and residual, with margin and funding-reversal risk."""
        s = self._observed_series(str(q.get("symbol") or "BTC"))
        out = cr.carry_pnl(s, _number(q, "notional", 1e6, 1e3, 1e10), _number(q, "leverage", 3.0, 1.0, 125.0), _number(q, "fee_bps", 5.0, 0.0, 100.0), _number(q, "slippage_bps", 2.0, 0.0, 100.0),
                           _number(q, "financing_apr", 0.0, 0.0, 1.0), None if q.get("maintenance") in (None, "") else _number(q, "maintenance", 0.005, 0.0005, 0.5), int(_number(q, "days", 365, 30, 3000, integer=True)))
        return self._wrap("crypto_carry", out, {"prices": s.source, "funding": s.source}, [], s.as_of, s.notes)

    def executable(self, q: dict) -> dict:
        """``GET /api/crypto/executable?symbol=BTC&notional=100000``: cross-venue pairs priced on the real order-book levels of the stored snapshot, with the verdict and what is not modelled."""
        sym = str(q.get("symbol") or "BTC").upper()
        snap = self.api._snapshot(sym)
        override = None
        if q.get("taker_bps") not in (None, "") or q.get("maker_bps") not in (None, ""):
            one = {}
            if q.get("taker_bps") not in (None, ""):
                one["taker"] = _number(q, "taker_bps", 0.0, 0.0, 200.0)
            if q.get("maker_bps") not in (None, ""):
                one["maker"] = _number(q, "maker_bps", 0.0, 0.0, 200.0)
            override = {v["venue"]: dict(one) for v in (snap or {}).get("venues", [])}
        try:
            out = cr.executable_cross(snap, _number(q, "notional", 1e5, 100.0, 1e9), _choice(q, "fee_mode", "taker_taker", ("taker_taker", "maker_taker")), override,
                                      _number(q, "max_age_s", 5.0, 0.5, 86_400.0))
        except ValueError as exc:
            raise ApiError(str(exc)) from None
        return self._wrap("crypto_executable", {"symbol": sym, **out}, {"quotes": "downloaded" if snap else "unavailable"}, [], ((snap or {}).get("fetched_at") or "")[:10] or None, [])

    def execution(self, q: dict) -> dict:
        """``GET /api/crypto/execution?symbol=BTC&venue=KRAKEN&side=buy&notional=100000&order_type=twap&slices=6``: the simulated execution of one order on a stored order book."""
        sym = str(q.get("symbol") or "BTC").upper()
        snap = self.api._snapshot(sym)
        if snap is None:
            return self._wrap("crypto_execution", {"symbol": sym, **cr.unavailable("no order-book snapshot is stored", ["download the venue snapshot (Free data, crypto venues)"])}, {"book": "unavailable"})
        names = [v["venue"] for v in snap["venues"]]
        venue = str(q.get("venue") or names[0]).upper()
        book = next((v for v in snap["venues"] if v["venue"] == venue), None)
        if book is None:
            raise ApiError(f"venue must be one of {names}")
        try:
            out = cr.simulate_execution(book, _choice(q, "side", "buy", ("buy", "sell")), _number(q, "notional", 1e5, 100.0, 1e9), _choice(q, "order_type", "market", ("market", "limit", "twap", "vwap")),
                                        _number(q, "limit_bps", 0.0, -500.0, 500.0), int(_number(q, "slices", 6, 1, 200, integer=True)), _number(q, "replenish", 0.5, 0.0, 1.0),
                                        _choice(q, "profile", "flat", ("flat", "u_shape")), None, None if q.get("vol_annual") in (None, "") else _number(q, "vol_annual", 0.6, 0.0, 5.0),
                                        _number(q, "horizon_hours", 0.0, 0.0, 720.0))
        except ValueError as exc:
            raise ApiError(str(exc)) from None
        return self._wrap("crypto_execution", {"symbol": sym, "venues": names, **out}, {"book": "downloaded"}, [], (snap.get("fetched_at") or "")[:10] or None,
                          [f"Order book of {snap.get('fetched_at')}; a live book changes within seconds."])

    def validate(self, body: dict) -> dict:
        """``POST /api/crypto/validate``: a strategy of ``assets.crypto_validation`` on observed 24/7 history, as a job."""
        from .asset_api import _list

        name = str(body.get("strategy") or "tsmom")
        if name not in cv.FAMILIES:
            raise ApiError(f"strategy must be one of {sorted(cv.FAMILIES)}")
        coins = [c.upper() for c in _list(body.get("assets") or "BTC,ETH", 8)]
        if not coins:
            raise ApiError("choose at least one coin")
        series = {c: self._observed_series(c) for c in coins}
        params = body.get("params") or {}
        if not isinstance(params, dict):
            raise ApiError("params must be an object")
        spec = cv.FAMILIES[name]
        clean_params = {}
        for k, v in params.items():
            if k not in spec["params"]:
                raise ApiError(f"{name} has no setting '{k}'; its settings are {sorted(spec['params'])}")
            try:
                clean_params[k] = type(spec["params"][k])(v)
            except (TypeError, ValueError):
                raise ApiError(f"setting {k} must be a number") from None
        fee, slip = _number(body, "fees_bps", 5.0, 0.0, 200.0), _number(body, "slippage_bps", 2.0, 0.0, 200.0)
        target = None if body.get("target_vol") in (None, "") else _number(body, "target_vol", 0.2, 0.02, 2.0)
        leverage = _number(body, "max_leverage", 1.0, 0.1, 10.0)
        short = str(body.get("allow_short", "1")).lower() not in ("0", "false", "no")
        walk = str(body.get("walk_forward", "1")).lower() not in ("0", "false", "no")
        observed = self.api.observed()
        venue_funding = None
        if name == "funding_dispersion":
            s = series[coins[0]]
            venue_funding = {}
            for v in ca.VENUE_ORDER:
                f, src = ca.venue_funding(s, v)
                if not observed or cr.is_observed(src):
                    venue_funding[v] = f

        def work(progress):
            progress("running the strategy on the 24/7 calendar")
            try:
                out = cv.run(series, name, clean_params, fee, slip, target, leverage, short, observed, venue_funding, walk)
            except ValueError as exc:
                raise ApiError(str(exc)) from None
            if out.get("state") != "ok":
                raise ApiError(out.get("reason") or "the strategy cannot be run on this data")
            kinds = {s.source for s in series.values()}
            return self._wrap("crypto_validation", out, {"prices": next(iter(kinds)) if len(kinds) == 1 else "mixed"}, out["verdict"]["notes"], out["end"], out.pop("assumptions"))

        return self.api.app.submit_task(f"crypto validation: {name}", work)

    def liquidation(self, body: dict) -> dict:
        """``POST /api/crypto/liquidation``: cross-margin liquidation risk of perpetual positions (``positions``, ``collateral``); betas against BTC are measured on observed data when the coins have it."""
        positions = body.get("positions")
        if not isinstance(positions, list) or not 1 <= len(positions) <= 20 or not all(isinstance(p, dict) for p in positions):
            raise ApiError("positions must be a list of 1 to 20 positions")
        try:
            collateral = float(body.get("collateral"))
        except (TypeError, ValueError):
            raise ApiError("collateral must be a number") from None
        betas, notes = {}, []
        try:
            btc = self.api._series("BTC").spot.pct_change().dropna()
        except ApiError:
            btc = None
        for sym in sorted({str(p.get("symbol", "")).upper() for p in positions}):
            if btc is None or sym in ("BTC", ""):
                continue
            try:
                s = self.api._series(sym)
            except ApiError:
                continue
            if not cr.is_observed(s.source):
                notes.append(f"{sym}: no observed history, beta to BTC assumed 1")
                continue
            both = pd.concat([s.spot.pct_change().dropna(), btc], axis=1, join="inner").dropna().tail(365)
            if len(both) > 60 and both.iloc[:, 1].var() > 0:
                betas[sym] = float(both.iloc[:, 0].cov(both.iloc[:, 1]) / both.iloc[:, 1].var())
        for k, v in (body.get("betas") or {}).items():
            try:
                betas[str(k).upper()] = float(v)
            except (TypeError, ValueError):
                raise ApiError("betas must be numbers") from None
        try:
            tiers = None
            if body.get("tiers") is not None:
                raw = body["tiers"]
                if not isinstance(raw, dict):
                    raise ApiError("tiers must be {venue: [[notional_up_to, maintenance_rate, max_leverage], ...]}")
                tiers = {str(k).upper(): [(float(a) if a is not None else float("inf"), float(b), float(c)) for a, b, c in v] for k, v in raw.items()}
                missing = {str(p.get("venue", "")).upper() for p in positions} - set(tiers)
                if missing:
                    raise ApiError(f"tiers are missing for {sorted(missing)}")
            from ..assets.margin_schedules import ScheduleStore

            store = ScheduleStore(self.api.root)
            as_of = body.get("as_of") or None
            schedules, prices, sched_notes = {}, {}, []
            for p in positions:
                sym, venue = str(p.get("symbol", "")).upper(), str(p.get("venue", "")).upper()
                s = store.at(venue, f"{sym}-USDT-SWAP", "cross", as_of) if venue == "OKX" else None
                if s is not None:
                    schedules[(venue, sym)] = s
                    try:
                        prices[sym] = float(self.api._series(sym).spot.dropna().iloc[-1])
                    except (ApiError, IndexError):
                        schedules.pop((venue, sym), None)
                elif venue == "OKX":
                    sched_notes.append(f"{sym} on OKX: no stored schedule version in force{' on ' + str(as_of) if as_of else ''}: hypothetical margin model used (refresh the schedule to use OKX's published tiers)")
            out = cr.portfolio_liquidation(positions, collateral, betas, _number(body, "liquidation_fee_bps", 50.0, 0.0, 1000.0), use_tiers=str(body.get("use_tiers", "")).lower() in ("1", "true", "yes"), tiers=tiers,
                                           collateral_haircut=_number(body, "collateral_haircut", 0.0, 0.0, 0.9), funding_per_day=_number(body, "funding_per_day", 0.0, 0.0, 1e9), days=_number(body, "days", 0.0, 0.0, 365.0), schedules=schedules, prices=prices)
        except (ValueError, KeyError, TypeError) as exc:
            raise ApiError(str(exc)) from None
        out["betas"] = betas
        return self._wrap("crypto_liquidation", out, {"positions": "user_file", **({"betas": "platform"} if betas else {})}, [], None, notes + sched_notes + ["A shock factor moves every coin by its beta; correlations rise in stress, so real moves can be worse."])
