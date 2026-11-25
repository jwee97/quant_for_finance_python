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
                "/api/crypto/execution": self.execution, "/api/crypto/calendar": self.calendar}

    def post_routes(self) -> dict:
        return {"/api/crypto/validate": self.validate, "/api/crypto/liquidation": self.liquidation}

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
            out = cr.funding_intelligence(series, observed_only=self.api.observed(), days=int(_number(q, "days", 365, 60, 2000, integer=True)))
        except ValueError as exc:
            raise ApiError(str(exc)) from None
        parts = {f"{r['symbol']}@{r['venue']}": r["source"] for r in out["rows"]}
        parts.update({"open_interest": "unavailable", "liquidations": "unavailable", "dated_futures": "unavailable"})
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
            out = cr.portfolio_liquidation(positions, collateral, betas, _number(body, "liquidation_fee_bps", 50.0, 0.0, 1000.0))
        except (ValueError, KeyError, TypeError) as exc:
            raise ApiError(str(exc)) from None
        out["betas"] = betas
        return self._wrap("crypto_liquidation", out, {"positions": "user_file", **({"betas": "platform"} if betas else {})}, [], None, notes + ["A shock factor moves every coin by its beta; correlations rise in stress, so real moves can be worse."])
