"""The Options, Futures and Crypto workspaces' endpoints.

Each method takes a plain dictionary (the query string of a GET, or the JSON body of a POST) and returns a JSON-able dictionary. Every payload carries its provenance (``source``, ``sources``,
``warnings``, ``notes``): option chains, futures curves, the other crypto venues and the order book are MODEL OUTPUTS unless a file of yours replaces them, and the page prints which. Long
computations (strategy backtests) are submitted as jobs and polled through ``/api/job/<id>`` like the Backtest tab's.

User files, all optional, read from ``data/user/``:

``options/<NAME>.csv``   a vendor option chain (read by ``derivatives.synthetic.load_option_csv``); choose it as the underlying ``file:<NAME>``
``futures/<ROOT>.csv``   ``date, contract, expiry, price`` and optionally ``open_interest``, ``volume``, ``spot``
``crypto/<SYMBOL>.csv``  ``time, spot`` and optionally ``perp``, ``funding_daily`` or ``funding_8h``
"""

from __future__ import annotations

import re
import threading
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from ..assets import common
from ..assets import crypto_analytics as ca
from ..assets import futures_analytics as fa
from ..assets import options_analytics as oa
from ..assets.free_data.store import SAFE_NAME as DOWNLOAD_NAME
from ..assets.free_data.store import DownloadStore
from .api import ApiError, _choice, _number
from .payload import clean

SAFE_NAME = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
SYMBOL = re.compile(r"^[A-Za-z0-9._-]{1,12}$")
MAX_POSITIONS = 60


def _list(value, limit: int = 12) -> list:
    """A list from a comma-separated string or a JSON list, trimmed and bounded."""
    if value in (None, ""):
        return []
    items = value if isinstance(value, (list, tuple)) else str(value).split(",")
    out = [str(x).strip() for x in items if str(x).strip()]
    if len(out) > limit:
        raise ApiError(f"at most {limit} items")
    return out


def _floats(body: dict, key: str, default: list, lo: float, hi: float, limit: int = 9) -> list[float]:
    raw = body.get(key)
    if raw in (None, ""):
        return list(default)
    if not isinstance(raw, (list, tuple)) or not 1 <= len(raw) <= limit:
        raise ApiError(f"{key.replace('_', ' ')} must be a list of 1 to {limit} numbers")
    try:
        vals = [float(x) for x in raw]
    except (TypeError, ValueError):
        raise ApiError(f"{key.replace('_', ' ')} must be numbers") from None
    if any(not np.isfinite(v) or not lo <= v <= hi for v in vals):
        raise ApiError(f"{key.replace('_', ' ')} must be between {lo:g} and {hi:g}")
    return vals


def _wrap(kind: str, data: dict, parts: dict, warnings=(), as_of=None, notes=()) -> dict:
    """A payload with its provenance. The analytics keep their own ``notes`` and ``warnings`` inside the data they return; they are folded into the envelope's lists, and their ``source`` (a single word the
    parts already say better) is dropped, so nothing clashes with the provenance keys."""
    data = dict(data)
    notes, warnings = list(notes) + list(data.pop("notes", None) or []), list(warnings) + list(data.pop("warnings", None) or [])
    notes = list(dict.fromkeys(notes))
    data.pop("source", None)
    return clean(common.envelope(kind, data, parts, list(dict.fromkeys(warnings)), as_of, notes))


class AssetApi:
    def __init__(self, app):
        self.app = app
        self.root = Path(app.root)
        self._lock = threading.Lock()
        self._frames: dict | None = None
        self.downloads = DownloadStore(self.root)

    # ------------------------------------------------------------------------------------------------------------ routes
    def get_routes(self) -> dict:
        return {"/api/assets/catalog": self.catalog, "/api/options/chain": self.options_chain, "/api/options/surface": self.options_surface, "/api/futures/curve": self.futures_curve,
                "/api/futures/roll": self.futures_roll, "/api/crypto/overview": self.crypto_overview, "/api/crypto/funding": self.crypto_funding, "/api/crypto/cross-exchange": self.crypto_cross_exchange}

    def post_routes(self) -> dict:
        return {"/api/options/portfolio-risk": self.options_portfolio_risk, "/api/options/strategy/payoff": self.options_strategy_payoff, "/api/options/strategy/backtest": self.options_strategy_backtest,
                "/api/futures/margin": self.futures_margin, "/api/futures/strategy/backtest": self.futures_strategy_backtest, "/api/crypto/strategy/backtest": self.crypto_strategy_backtest}

    # ------------------------------------------------------------------------------------------------------------ user files
    def _user_files(self, folder: str) -> dict[str, Path]:
        d = self.root / "data" / "user" / folder
        return {p.stem: p for p in sorted(d.glob("*.csv")) if SAFE_NAME.match(p.stem)} if d.is_dir() else {}

    def _downloaded(self, kind: str, item: str | None = None) -> dict[str, dict]:
        """``{name: record}`` of the free data downloaded for a kind (see ``assets.free_data``), newest first; ``item`` keeps one sort of file (``chain``, ``curve``, ``history``)."""
        return {r["name"]: r for r in self.downloads.list(kind) if item is None or r.get("item") == item}

    def _bundle(self):
        return self.app.builder.default_bundle()

    # ------------------------------------------------------------------------------------------------------------ catalogue
    def catalog(self, q: dict | None = None) -> dict:
        """What the three workspaces can show: option underlyings (the platform's ETFs and your chain files), futures roots, crypto symbols and venues, and the strategies each workspace can run."""
        b = self._bundle()
        opt_files, fut_files, cry_files = self._user_files("options"), self._user_files("futures"), self._user_files("crypto")
        dl_chains, dl_curves, dl_coins = self._downloaded("options", "chain"), self._downloaded("futures", "curve"), self._downloaded("crypto", "history")
        underlyings = [{"id": a, "label": a, "source": "simulated chain on real prices", "has_index": a in oa.REFERENCE_INDEX} for a in b.assets if a in b.prices.columns]
        underlyings += [{"id": f"dl:{n}", "label": f"{n} (real chain, {r.get('source', 'downloaded')}, {r.get('as_of', '?')})", "source": "downloaded", "has_index": str(r.get("underlying", "")).upper() in oa.REFERENCE_INDEX,
                         "as_of": r.get("as_of"), "fetched_at": r.get("fetched_at")} for n, r in dl_chains.items()]
        underlyings += [{"id": f"file:{n}", "label": f"{n} (your file)", "source": "user_file", "has_index": False} for n in opt_files]
        roots = [{"id": r, "name": s.name, "asset_class": s.asset_class, "exchange": s.exchange, "multiplier": s.multiplier, "tick_size": s.tick_size,
                  "source": "user_file" if r in fut_files else "downloaded" if r in dl_curves else "simulated"} for r, s in fa.ROOTS.items()]
        roots += [{"id": r, "name": rec.get("root_name", r), "asset_class": rec.get("asset_class", "unknown"), "exchange": rec.get("exchange", "?"), "multiplier": float(rec.get("multiplier", 1.0)),
                   "tick_size": float(rec.get("tick_size", 0.01)), "source": "user_file" if r in fut_files else "downloaded"} for r, rec in dl_curves.items() if r not in fa.ROOTS]
        roots += [{"id": r, "name": r, "asset_class": "unknown", "exchange": "?", "multiplier": 1.0, "tick_size": 0.01, "source": "user_file"} for r in fut_files if r not in fa.ROOTS and r not in dl_curves]
        symbols = [{"id": s, "source": "user_file" if s in cry_files else "platform"} for s in ca.REAL_SYMBOLS] + [{"id": s, "source": "user_file" if s in cry_files else "downloaded" if s in dl_coins else "simulated"} for s in ca.SIMULATED_ALTS]
        symbols += [{"id": s, "source": "user_file" if s in cry_files else "downloaded"} for s in dl_coins if s not in ca.REAL_SYMBOLS and s not in ca.SIMULATED_ALTS]
        symbols += [{"id": s, "source": "user_file"} for s in cry_files if s not in ca.REAL_SYMBOLS and s not in ca.SIMULATED_ALTS and s not in dl_coins]
        return clean({"underlyings": underlyings, "option_strategies": [{"id": k, "text": v} for k, v in oa.STRATEGY_INFO.items()], "option_backtests": [{"id": k, "text": oa.BACKTEST_TEXT[k], "params": oa.backtest_params(k)} for k in oa.BACKTEST_STRATEGIES],
                      "defaults": {"rate": oa.DEFAULT_RATE, "premium": 0.15, "dividend_yields": oa.DIVIDEND_YIELDS}, "futures_roots": roots,
                      "futures_strategies": [{"id": k, "label": v["label"], "text": v["text"]} for k, v in fa.FUTURES_STRATEGIES.items()],
                      "crypto_symbols": symbols, "crypto_strategies": [{"id": k, "label": v["label"], "text": v["text"], "available": v["model"] is not None} for k, v in ca.CRYPTO_STRATEGIES.items()],
                      "venues": list(ca.VENUE_ORDER), "data_dir": "data/user/", "as_of": str(b.index[-1].date())})

    # ------------------------------------------------------------------------------------------------------------ options
    def _downloaded_chain(self, name: str, rate: float | None, dividend: float | None):
        """A real chain fetched from a free source (``assets.free_data``). Its rate and dividend yield are columns of the file (the dividend of each expiry is implied by the market's own forward), so a rate or
        dividend typed in the form does not apply to it. The realised-volatility context comes from the platform's prices when it has the underlying."""
        if not DOWNLOAD_NAME.match(name):
            raise ApiError("a downloaded chain is named like SPY or BTC-Deribit")
        known = self._downloaded("options", "chain")
        if name not in known:
            raise ApiError(f"no downloaded chain '{name}'; use the Free data card on this tab to download it (downloaded so far: {sorted(known) or 'none'})")
        rec = known[name]
        from ..derivatives.synthetic import load_option_csv

        symbol = str(rec.get("underlying") or name)
        try:
            chain = oa.prepare_chain(load_option_csv(self.downloads.folder("options") / rec["file"], underlying=symbol))
        except (ValueError, OSError) as exc:
            raise ApiError(f"{name}: {exc}") from None
        spot, quote_date = float(chain["underlying"].iloc[0]), pd.Timestamp(chain["date"].iloc[0])
        b, ctx = self._bundle(), None
        if symbol in b.prices.columns:
            try:
                ctx = replace(oa.underlying_context(symbol, b.prices[symbol], b.volume[symbol] if b.volume is not None and symbol in b.volume.columns else None), spot=spot, as_of=quote_date)
            except ValueError:
                ctx = None
        if ctx is None:
            ctx = oa.UnderlyingContext(symbol, spot, quote_date, float("nan"), float("nan"), float("nan"), pd.Series(dtype=float), float("nan"))
        notes = [f"Downloaded {str(rec.get('fetched_at', '?')).replace('T', ' ').rstrip('Z')} UTC from {rec.get('source', 'a free source')}; the quotes are for {rec.get('as_of')}."] + [str(n) for n in rec.get("notes", [])]
        if ctx.rv_21 == ctx.rv_21:
            notes.append(f"Realised volatility is from the platform's prices of {symbol} through {b.index[-1].date()}.")
        if rate is not None or dividend is not None:
            notes.append("The rate and dividend yield of a downloaded chain come with it and are not editable here.")
        return chain, ctx, "downloaded", notes

    def _option_chain(self, underlying: str, rate: float | None, dividend: float | None, premium: float):
        """``(chain, context, source, notes)`` for an underlying. ``source`` is ``downloaded`` (a real chain from a free source, ``dl:NAME``), ``user_file`` (your chain file, ``file:NAME``) or ``simulated`` (a
        model chain on an ETF's real last close and realised volatility)."""
        if underlying.startswith("dl:"):
            return self._downloaded_chain(underlying[3:], rate, dividend)
        if underlying.startswith("file:"):
            name = underlying[5:]
            files = self._user_files("options")
            if name not in files:
                raise ApiError(f"no option file '{name}.csv' in data/user/options/; available: {sorted(files)}")
            from ..derivatives.synthetic import load_option_csv

            try:
                raw = load_option_csv(files[name], underlying=name, default_rate=oa.DEFAULT_RATE if rate is None else rate, default_dividend=0.0 if dividend is None else dividend)
            except ValueError as exc:
                raise ApiError(f"{name}.csv: {exc}") from None
            chain = oa.prepare_chain(raw)
            spot = float(chain["underlying"].iloc[0])
            ctx = oa.UnderlyingContext(name, spot, pd.Timestamp(chain["date"].iloc[0]), float("nan"), float("nan"), float("nan"), pd.Series(dtype=float), float("nan"))
            notes = [f"Quotes are from your file {name}.csv (newest quote date {ctx.as_of.date()}); volatilities not in the file are inverted from the mid prices."]
            return chain, ctx, "user_file", notes
        if not SYMBOL.match(underlying) or underlying not in self._bundle().prices.columns:
            raise ApiError(f"unknown underlying '{underlying}'; the platform's ETFs are {list(self._bundle().prices.columns)}")
        b = self._bundle()
        vol_frame = b.volume[underlying] if b.volume is not None and underlying in b.volume.columns else None
        try:
            ctx = oa.underlying_context(underlying, b.prices[underlying], vol_frame)
        except ValueError as exc:
            raise ApiError(str(exc)) from None
        div = oa.DIVIDEND_YIELDS.get(underlying, 0.0) if dividend is None else dividend
        r = oa.DEFAULT_RATE if rate is None else rate
        scale = float(np.clip((ctx.dollar_volume if np.isfinite(ctx.dollar_volume) else 1e8) / 1e9, 0.05, 50.0)) * 600.0
        chain = oa.prepare_chain(oa.build_model_chain(ctx.spot, ctx.as_of, ctx.rv_21, ctx.long_run_vol, r, div, premium, underlying, scale))
        notes = [f"The chain is a MODEL (an arbitrage-free SSVI surface) quoted at {underlying}'s real last close of {ctx.spot:,.2f} on {ctx.as_of.date()} and its real 21-day realised volatility {ctx.rv_21:.1%}.",
                 f"Rate {r:.2%} and dividend yield {div:.2%} are assumptions; the implied-over-realised variance premium is {premium:.0%}. Volume, open interest and quote ages are simulated."]
        return chain, ctx, "simulated", notes

    def _option_params(self, q: dict) -> tuple[str, float | None, float | None, float]:
        underlying = str(q.get("underlying") or "SPY")
        rate = None if q.get("rate") in (None, "") else _number(q, "rate", 0.04, -0.02, 0.2)
        dividend = None if q.get("dividend") in (None, "") else _number(q, "dividend", 0.0, 0.0, 0.2)
        premium = _number(q, "premium", 0.15, -0.3, 1.0)
        return underlying, rate, dividend, premium

    def options_chain(self, q: dict) -> dict:
        """``GET /api/options/chain?underlying=SPY&expiry=2026-10-23``: the overview cards, the chain for one expiry (calls and puts side by side), its smile and skew metrics, and the term structure."""
        underlying, rate, dividend, premium = self._option_params(q)
        window = _number(q, "window", 0.15, 0.02, 0.6)
        chain, ctx, source, notes = self._option_chain(underlying, rate, dividend, premium)
        simulated = source == "simulated"
        try:
            view = oa.chain_view(chain, q.get("expiry") or None, window)
        except ValueError as exc:
            raise ApiError(str(exc)) from None
        macro = self._bundle().macro if source != "user_file" else None
        ov = oa.overview(chain, ctx, premium, macro, simulated, source)
        try:
            skew = oa.skew_metrics(chain, view["expiry"])
        except ValueError:
            skew = {}
        data = {"underlying": underlying, "overview": ov, "chain": view, "smile": oa.smile(chain, view["expiry"]), "skew": skew, "term": oa.term_structure(chain), "term_metrics": oa.term_metrics(oa.term_structure(chain)),
                "settings": {"rate": float(chain["rate"].median()), "dividend": float(chain["dividend"].median()), "premium": premium, "window": window}}       # a real chain has a yield per expiry: the middle one
        warnings = [f"{view['stale_quotes']} of {view['quotes']} quotes in this expiry are stale (older than {oa.STALE_AFTER_MIN:.0f} minutes)."] if view["stale_quotes"] else []
        warnings += [f"{view['wide_quotes']} quotes have a spread wider than 10% of the mid."] if view["wide_quotes"] else []
        parts = {"underlying_price": "platform" if simulated else source, "chain": source, "activity": source}
        if simulated or (source == "downloaded" and ov.get("reference_index")):
            parts["reference_index"] = "platform"
        return _wrap("options_chain", data, parts, warnings + ov["risk_flags"], ctx.as_of, notes)

    def options_surface(self, q: dict) -> dict:
        """``GET /api/options/surface?underlying=SPY``: the implied-volatility surface (strike by expiry), the term structure, the skew metrics of every expiry and the static no-arbitrage checks."""
        underlying, rate, dividend, premium = self._option_params(q)
        chain, ctx, source, notes = self._option_chain(underlying, rate, dividend, premium)
        simulated = source == "simulated"
        skews = []
        for e in oa.expiry_list(chain):
            if e["dte"] < 4:
                continue
            try:
                sk = oa.skew_metrics(chain, e["expiry"])
            except ValueError:
                continue
            if np.isfinite(sk.get("atm_iv", np.nan)):
                skews.append(sk)
        downloaded = source == "downloaded"
        american = bool(self._downloaded("options", "chain").get(underlying[3:], {}).get("american_style", True)) if downloaded else True
        band = 0.05 if american else 0.15
        try:
            arb = oa.arbitrage_view(chain, liquid_only=downloaded, band=band)                  # a downloaded chain is real quotes, delayed: only the live ones near the forward say anything
        except Exception:                                                                      # a diagnostic must not take the page down
            arb = {"rows": [], "tick": None, "tolerance": None}
        if downloaded:
            notes = notes + [f"The no-arbitrage checks cover real quotes that have a live bid and lie within {band:.0%} of the forward"
                             + (" (stock and ETF options can be exercised early, so in-the-money puts are worth more than put-call parity says: farther out the checks would flag the right of early exercise, not an arbitrage)." if american else ".")
                             + " A few violations are normal in a real market (stale or wide quotes); a large share would mean the data, or the rate and dividend assumptions, are wrong."]
        data = {"underlying": underlying, "surface": oa.surface_grid(chain), "term": oa.term_structure(chain), "term_metrics": oa.term_metrics(oa.term_structure(chain)), "skews": skews, "arbitrage": arb["rows"],
                "arbitrage_tick": arb["tick"], "arbitrage_tolerance": arb["tolerance"]}
        parts = {"chain": source}
        return _wrap("options_surface", data, parts, [], ctx.as_of, notes)

    def _positions(self, body: dict, chain: pd.DataFrame, as_of: pd.Timestamp) -> list[dict]:
        raw = body.get("positions")
        if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_POSITIONS:
            raise ApiError(f"positions must be a list of 1 to {MAX_POSITIONS} entries")
        out = []
        for i, p in enumerate(raw):
            if not isinstance(p, dict):
                raise ApiError(f"position {i + 1} must be an object")
            kind = str(p.get("type") or "option")
            if kind not in ("option", "stock"):
                raise ApiError(f"position {i + 1}: type must be option or stock")
            qty = _number(p, "quantity", 0.0, -1e6, 1e6)
            if kind == "stock":
                out.append({"type": "stock", "quantity": qty})
                continue
            right = _choice(p, "right", "C", ("C", "P", "c", "p")).upper()
            strike = _number(p, "strike", 0.0, 0.01, 1e7)
            if p.get("expiry"):
                try:
                    exp = pd.Timestamp(p["expiry"])
                except (ValueError, TypeError):
                    raise ApiError(f"position {i + 1}: expiry must be a date") from None
            else:
                exp = oa.nearest_expiry_to(chain, _number(p, "dte", 30, 1, 800))
            listed = oa.nearest_expiry_to(chain, (exp - as_of).days)
            iv = _number(p, "iv", 0.0, 0.0, 5.0) if p.get("iv") not in (None, "") else 0.0
            if iv <= 0:
                iv = oa.iv_lookup(chain, listed, strike)
            out.append({"type": "option", "right": right, "strike": strike, "quantity": qty, "expiry": str(exp.date()), "iv": float(iv)})
        return out

    def options_portfolio_risk(self, body: dict) -> dict:
        """``POST /api/options/portfolio-risk``: Greeks, by-expiry exposure and a stress table (spot and implied-volatility shocks) for a book of option and share positions."""
        underlying, rate, dividend, premium = self._option_params(body)
        chain, ctx, source, notes = self._option_chain(underlying, rate, dividend, premium)
        simulated = source == "simulated"
        positions = self._positions(body, chain, ctx.as_of)
        spot_shocks = _floats(body, "spot_shocks", [-0.05, -0.02, 0.0, 0.02, 0.05], -0.9, 2.0)
        vol_shocks = _floats(body, "vol_shocks", [-0.05, 0.0, 0.05], -0.5, 1.0)
        days = _number(body, "days", 0.0, 0, 365)
        try:
            risk = oa.portfolio_risk(positions, ctx.spot, float(chain["rate"].iloc[0]), float(chain["dividend"].iloc[0]), ctx.as_of, spot_shocks, vol_shocks, days)
        except ValueError as exc:
            raise ApiError(str(exc)) from None
        return _wrap("options_portfolio_risk", {"underlying": underlying, **risk}, {"prices": "platform" if simulated else source, "volatilities": source},
                                     risk["risk_flags"], ctx.as_of, notes + ["Stress P&L revalues every option with Black-Scholes at the shocked spot and volatility; time passes only if you set days."])

    def options_strategy_payoff(self, body: dict) -> dict:
        """``POST /api/options/strategy/payoff``: the legs of a named structure (or your own), their payoff at the horizon, today and half way, maximum profit and loss, break-evens, Greeks, an indicative margin, the probability of profit and expected value."""
        underlying, rate, dividend, premium = self._option_params(body)
        chain, ctx, source, notes = self._option_chain(underlying, rate, dividend, premium)
        simulated = source == "simulated"
        name = str(body.get("strategy") or "long_straddle")
        dte = _number(body, "dte", 30, 1, 700, integer=True)
        far = _number(body, "far_dte", max(dte * 2, dte + 14), dte + 1, 800, integer=True)
        e1, e2 = oa.nearest_expiry_to(chain, dte), oa.nearest_expiry_to(chain, far)
        g1 = chain[chain["expiry"] == e1]
        forward = float(g1["F"].iloc[0])
        dte1, dte2 = int(round(float(g1["T"].iloc[0]) * 365)), int(round(float(chain[chain["expiry"] == e2]["T"].iloc[0]) * 365))
        step = oa.strike_step(chain, e1)
        qty = _number(body, "quantity", 1, 1, 1000, integer=True)
        try:
            if body.get("legs"):
                legs = self._custom_legs(body["legs"], chain, ctx.as_of)
            else:
                legs = oa.strategy_legs(name, ctx.spot, forward, step, dte1, _number(body, "offset", 0.05, 0.0, 0.5), _number(body, "width", 0.05, 0.005, 0.5), max(dte2, dte1 + 1), qty)
            for leg in legs:
                if leg["type"] == "option":
                    leg["iv"] = oa.iv_lookup(chain, oa.nearest_expiry_to(chain, leg["dte"]), leg["strike"])
                    leg["dte"] = int(round(float(chain[chain["expiry"] == oa.nearest_expiry_to(chain, leg["dte"])]["T"].iloc[0]) * 365))
            atm = oa.atm_iv(g1)
            view_vol = None if body.get("view_vol") in (None, "") else _number(body, "view_vol", atm, 0.02, 3.0)
            view_drift = None if body.get("view_drift") in (None, "") else _number(body, "view_drift", 0.0, -2.0, 2.0)
            horizon = None if body.get("horizon_days") in (None, "") else _number(body, "horizon_days", dte1, 1, 800, integer=True)
            result = oa.strategy_analysis(legs, ctx.spot, forward, atm, float(chain["rate"].iloc[0]), float(chain["dividend"].iloc[0]), 0.0, view_vol, view_drift, horizon,
                                          _choice(body, "fill", "mid", ("mid", "worst")), _number(body, "spread_pct", 0.05, 0.0, 1.0), _number(body, "commission", 0.65, 0.0, 50.0))
        except ValueError as exc:
            raise ApiError(str(exc)) from None
        data = {"underlying": underlying, "strategy": name, "text": oa.STRATEGY_INFO.get(name, "Your own legs"), "expiry": str(pd.Timestamp(e1).date()), "far_expiry": str(pd.Timestamp(e2).date()), "atm_iv": atm,
                "analysis": result}
        return _wrap("options_strategy", data, {"prices": "platform" if simulated else source, "premiums": source}, result["risk_flags"], ctx.as_of,
                                     notes + ["Entry prices are the model's mids (or worse, if you chose to cross the spread). Margin is indicative. Probability of profit and expected value assume a lognormal spot with one volatility."])

    def _custom_legs(self, raw, chain: pd.DataFrame, as_of: pd.Timestamp) -> list[dict]:
        if not isinstance(raw, list) or not 1 <= len(raw) <= 8:
            raise ApiError("legs must be a list of 1 to 8 entries")
        legs = []
        for i, leg in enumerate(raw):
            if not isinstance(leg, dict):
                raise ApiError(f"leg {i + 1} must be an object")
            if str(leg.get("type") or "option") == "stock":
                legs.append({"type": "stock", "quantity": _number(leg, "quantity", 100, -1e6, 1e6)})
                continue
            legs.append({"type": "option", "right": _choice(leg, "right", "C", ("C", "P", "c", "p")).upper(), "strike": _number(leg, "strike", 0.0, 0.01, 1e7), "quantity": _number(leg, "quantity", 1, -1e4, 1e4),
                         "dte": _number(leg, "dte", 30, 1, 800, integer=True)})
        return legs

    def options_strategy_backtest(self, body: dict) -> dict:
        """``POST /api/options/strategy/backtest``: one of the rolling option strategies on the simulated option market, run as a job."""
        name = str(body.get("strategy") or "short_put")
        if name not in oa.BACKTEST_STRATEGIES:
            raise ApiError(f"strategy must be one of {list(oa.BACKTEST_STRATEGIES)}")
        try:
            params = oa.parse_backtest_params(name, body.get("params"))
        except ValueError as exc:
            raise ApiError(str(exc)) from None
        days = _number(body, "days", 252, 60, 756, integer=True)
        seed, capital = _number(body, "seed", 0, 0, 1_000_000, integer=True), _number(body, "capital", 100_000, 1e3, 1e9)
        premium, commission = _number(body, "premium", 0.25, -0.3, 1.0), _number(body, "commission", 0.65, 0.0, 50.0)
        fill = _number(body, "fill_fraction", 1.0, 0.0, 1.0)

        def work(progress):
            progress("building the simulated option market (the first run of a given length takes several seconds)")
            out = oa.run_strategy_backtest(name, params, days, seed, capital, premium, commission, fill)
            return _wrap("options_backtest", out, {"market": "simulated"}, [], out["dates"][-1] if out["dates"] else None, out.pop("notes"))

        return self.app.submit_task(f"options: {name}", work)

    # ------------------------------------------------------------------------------------------------------------ futures
    def _downloaded_curve(self, root: str, rec: dict) -> fa.CurveData:
        """A curve fetched from a free source. A root the platform simulates keeps its specification; any other takes the terms the download recorded (margins unknown)."""
        spec = fa.ROOTS.get(root)
        if spec is None:
            spec = fa.RootSpec(root, str(rec.get("root_name", root)), str(rec.get("exchange", "?")), str(rec.get("asset_class", "unknown")), float(rec.get("multiplier", 1.0)), float(rec.get("tick_size", 0.01)),
                               tuple(range(1, 13)), "last_business_day", fa.SchwartzSmithParams(), 0.0, 0.0)
        notes = [f"Downloaded {str(rec.get('fetched_at', '?')).replace('T', ' ').rstrip('Z')} UTC from {rec.get('source', 'a free source')}; the newest prices are for {rec.get('as_of')}."] + [str(n) for n in rec.get("notes", [])]
        try:
            return fa.load_curve_file(self.downloads.folder("futures") / rec["file"], root, spec=spec, source="downloaded", extra_notes=notes)
        except (ValueError, OSError) as exc:
            raise ApiError(f"{root}: {exc}") from None

    def _curve(self, root: str) -> fa.CurveData:
        root = root.upper()
        if not SYMBOL.match(root):
            raise ApiError("root must be letters and digits")
        files = self._user_files("futures")
        if root in files:
            try:
                return fa.load_curve_file(files[root], root)
            except ValueError as exc:
                raise ApiError(f"{root}.csv: {exc}") from None
        downloaded = self._downloaded("futures", "curve")
        if root in downloaded:
            return self._downloaded_curve(root, downloaded[root])
        if root not in fa.ROOTS:
            raise ApiError(f"unknown root '{root}'; the simulated roots are {sorted(fa.ROOTS)} (download it with the Free data card, or put {root}.csv in data/user/futures/ to use your own)")
        return fa.simulated_curve(root, self._bundle().index[-1])

    @staticmethod
    def _curve_parts(data: fa.CurveData, margins: bool = False) -> dict:
        """Where a curve's numbers came from, part by part. Open interest is a part only when the table has it (a free Yahoo download does not); margins are always the platform's illustrative figures."""
        parts = {"curve": data.source}
        if data.long["open_interest"].notna().any():
            parts["open_interest"] = data.source
        if margins:
            parts["margins"] = "simulated"
        return parts

    def futures_curve(self, q: dict) -> dict:
        """``GET /api/futures/curve?root=CL``: the curve, how it moved, carry, basis and the one-line snapshot."""
        data = self._curve(str(q.get("root") or "CL"))
        try:
            snap = fa.curve_snapshot(data)
            out = {"root": data.root, "curve": snap, "history": fa.curve_history(data), "carry": fa.carry_dashboard(data), "basis": fa.basis_dashboard(data), "snapshot": fa.futures_snapshot(data).to_dict(),
                   "spec": {"name": data.spec.name, "exchange": data.spec.exchange, "multiplier": data.spec.multiplier, "tick_size": data.spec.tick_size, "tick_value": data.spec.tick_value,
                            "initial_margin": data.spec.initial_margin, "maintenance_margin": data.spec.maintenance_margin}}
        except ValueError as exc:
            raise ApiError(str(exc)) from None
        flags = list(out["snapshot"]["risk_flags"])
        return _wrap("futures_curve", out, self._curve_parts(data, margins=True), flags, data.as_of, data.notes)

    def futures_roll(self, q: dict) -> dict:
        """``GET /api/futures/roll?root=CL&roll_days=5``: the contract held and the next, the suggested roll date, the open-interest crossover, the cost of the roll and its histories."""
        data = self._curve(str(q.get("root") or "CL"))
        try:
            roll = fa.roll_analysis(data, _number(q, "roll_days", 5, 0, 60, integer=True))
        except ValueError as exc:
            raise ApiError(str(exc)) from None
        return _wrap("futures_roll", {"root": data.root, "roll": roll, "spec": {"multiplier": data.spec.multiplier, "tick_value": data.spec.tick_value}}, self._curve_parts(data),
                                     roll["risk_flags"], data.as_of, data.notes + ["Roll cost is for a LONG position (negative: rolling earns the backwardation); a short position's is the opposite."])

    def futures_margin(self, body: dict) -> dict:
        """``POST /api/futures/margin``: notional, initial and maintenance margin, leverage and warnings for a book of futures positions."""
        raw = body.get("positions")
        if not isinstance(raw, list) or not 1 <= len(raw) <= 20:
            raise ApiError("positions must be a list of 1 to 20 entries")
        roots = sorted({str(p.get("root") or "").upper() for p in raw if isinstance(p, dict)})
        curves = {r: self._curve(r) for r in roots}
        positions = [{"root": str(p.get("root") or "").upper(), "contract": str(p["contract"]) if p.get("contract") else None, "quantity": _number(p, "quantity", 0, -10000, 10000)} for p in raw]
        try:
            out = fa.margin_dashboard(positions, curves, _number(body, "equity", 100_000, 1e3, 1e10), _number(body, "margin_scale", 1.0, 0.1, 10.0), _number(body, "near_expiry_days", 7, 0, 60, integer=True))
        except ValueError as exc:
            raise ApiError(str(exc)) from None
        return _wrap("futures_margin", out, {"prices": common.overall_source({r: c.source for r, c in curves.items()}), "margins": "simulated"}, out["risk_flags"],
                                     self._bundle().index[-1])

    def futures_strategy_backtest(self, body: dict) -> dict:
        """``POST /api/futures/strategy/backtest``: a strategy on the simulated futures roots through the platform pipeline, as a job."""
        name = str(body.get("strategy") or "carry_ts")
        if name not in fa.FUTURES_STRATEGIES:
            raise ApiError(f"strategy must be one of {sorted(fa.FUTURES_STRATEGIES)}")
        roots = _list(body.get("roots") or list(fa.ROOTS))
        lookback = None if body.get("lookback") in (None, "") else _number(body, "lookback", 63, 5, 756, integer=True)
        target = None if body.get("target_vol") in (None, "") else _number(body, "target_vol", 0.1, 0.01, 1.0)
        freq = _choice(body, "frequency", "weekly", ("daily", "weekly", "monthly"))
        cost = _number(body, "cost_bps", 2.0, 0.0, 200.0)
        as_of = self._bundle().index[-1]

        def work(progress):
            progress("building the simulated futures (the first run takes up to half a minute)")
            out = fa.run_futures_strategy(self.app.config, name, roots, freq, lookback, target, cost, as_of)
            return _wrap("futures_backtest", out, {"prices": "simulated"}, [], as_of, out.pop("notes"))

        return self.app.submit_task(f"futures: {name}", work)

    # ------------------------------------------------------------------------------------------------------------ crypto
    def _crypto_frames(self) -> dict:
        with self._lock:
            if self._frames is None:
                from ..data.crypto import build_crypto_frames

                d = self.root / "data" / "raw" / "crypto"
                if not (d / "BTC_funding.csv.gz").exists():
                    raise ApiError("the Deribit crypto data is not downloaded (data/raw/crypto); run `python -m src.cli backtest --model funding_carry` once, or put your own files in data/user/crypto/")
                self._frames = build_crypto_frames(d)
            return self._frames

    def _downloaded_series(self, symbol: str, rec: dict) -> ca.CryptoSeries:
        """A coin's history fetched from free sources. Its funding belongs to the venue the download says (OKX for a coin with an OKX perpetual)."""
        sources = rec.get("sources") or {}
        notes = [f"Downloaded {str(rec.get('fetched_at', '?')).replace('T', ' ').rstrip('Z')} UTC from {rec.get('source', 'a free source')}; the newest day is {rec.get('as_of')}."] + [str(n) for n in rec.get("notes", [])]
        try:
            return ca.user_series(self.downloads.folder("crypto") / rec["file"], symbol, source="downloaded", notes=notes, funding_venue="OKX" if sources.get("funding") == "OKX" else "DERIBIT")
        except (ValueError, OSError) as exc:
            raise ApiError(f"{symbol}: {exc}") from None

    def _attach_venue_funding(self, s: ca.CryptoSeries) -> ca.CryptoSeries:
        """Add the real funding series downloaded for other venues (``funding_<COIN>_<VENUE>``) to a coin's series."""
        for name, rec in self._downloaded("crypto", "funding").items():
            if rec.get("coin") != s.symbol or not rec.get("venue") or str(rec["venue"]).upper() == s.funding_venue:
                continue
            try:
                raw = pd.read_csv(self.downloads.folder("crypto") / rec["file"], parse_dates=["time"]).dropna().drop_duplicates("time").set_index("time").sort_index()["funding_daily"].astype(float)
            except (ValueError, OSError, KeyError):
                continue
            s.venue_funding[str(rec["venue"]).upper()] = raw
        return s

    def _series(self, symbol: str) -> ca.CryptoSeries:
        symbol = symbol.upper()
        if not SYMBOL.match(symbol):
            raise ApiError("symbols are letters and digits")
        files = self._user_files("crypto")
        if symbol in files:
            try:
                return ca.user_series(files[symbol], symbol)
            except ValueError as exc:
                raise ApiError(f"{symbol}.csv: {exc}") from None
        downloaded = self._downloaded("crypto", "history")
        if symbol in ca.REAL_SYMBOLS:                                                       # the platform's Deribit history is longer than any download; the downloads add venues and funding to it
            return self._attach_venue_funding(ca.platform_series(self._crypto_frames(), symbol))
        if symbol in downloaded:
            return self._attach_venue_funding(self._downloaded_series(symbol, downloaded[symbol]))
        if symbol in ca.SIMULATED_ALTS:
            return self._attach_venue_funding(ca.simulated_series(symbol, ca.platform_series(self._crypto_frames(), "BTC")))
        raise ApiError(f"unknown symbol '{symbol}'; the platform has {list(ca.REAL_SYMBOLS)}, can simulate {sorted(ca.SIMULATED_ALTS)}, and downloads others with the Free data card "
                       f"(or put {symbol}.csv in data/user/crypto/)")

    def _snapshot(self, symbol: str) -> dict | None:
        """The downloaded snapshot of a coin's quotes and order books on several venues, if there is one."""
        snap = self.downloads.read_json("crypto", f"venues_{symbol.upper()}")
        return snap if isinstance(snap, dict) and snap.get("venues") else None

    def _stable(self) -> pd.Series | None:
        try:
            s = self._crypto_frames()["stable"].copy()
        except ApiError:
            return None
        s.index = pd.DatetimeIndex(s.index).tz_localize(None) if pd.DatetimeIndex(s.index).tz is not None else pd.DatetimeIndex(s.index)
        return s

    def crypto_overview(self, q: dict) -> dict:
        """``GET /api/crypto/overview?symbol=BTC``: the cards (trend, funding, momentum, volatility, liquidity, on-chain), the spot/perpetual view, the basis dashboard and the order-book summary (the real books
        of a downloaded snapshot when there is one, else a parametric model)."""
        s = self._series(str(q.get("symbol") or "BTC"))
        snap = self._snapshot(s.symbol)
        try:
            if snap is not None:
                books = ca.snapshot_books(snap, s.symbol)
                spot_books = [b for b, v in zip(books.values(), snap["venues"]) if v["kind"] == "spot"] or list(books.values())
                book = max(spot_books, key=lambda b: b["depth"]["25bp"]["bid"] + b["depth"]["25bp"]["ask"])
            else:
                book = ca.order_book_summary(s.symbol, "BINANCE", float(s.spot.iloc[-1]), seed=int(s.as_of.toordinal()))
                books = {v: ca.order_book_summary(s.symbol, v, float(s.spot.iloc[-1]), seed=int(s.as_of.toordinal())) for v in ca.VENUE_ORDER}
            ov = ca.overview(s, self._stable(), book)
            data = {"symbol": s.symbol, "overview": ov, "spot_perp": ca.spot_perp_view(s, _number(q, "days", 365, 60, 2000, integer=True)) if s.perp is not None else None,
                    "basis": ca.basis_dashboard(s, _number(q, "fee_bps", 5.0, 0.0, 100.0)) if s.perp is not None and s.funding is not None else None, "books": books,
                    "snapshot": None if snap is None else {"fetched_at": snap.get("fetched_at"), "venues": [v["venue"] for v in snap["venues"]], "failed": snap.get("failed", {})}}
        except ValueError as exc:
            raise ApiError(str(exc)) from None
        parts = {"prices": s.source, "funding": s.source, "stablecoin_supply": "platform", "order_book": "downloaded" if snap is not None else "simulated"}
        notes = list(s.notes)
        if snap is not None:
            notes.append(f"The order books are the real ones of {snap.get('fetched_at', 'the download')}; the liquidity card uses the deepest spot venue ({book['venue']}).")
        return _wrap("crypto_overview", data, parts, ov["risk_flags"], s.as_of, notes)

    def crypto_funding(self, q: dict) -> dict:
        """``GET /api/crypto/funding?symbols=BTC,ETH,SOL``: funding by coin and venue with z-scores and charts (Deribit's BTC and ETH are data; every other venue and coin is simulated)."""
        syms = _list(q.get("symbols") or "BTC,ETH,SOL", 6)
        series = {sym.upper(): self._series(sym) for sym in syms}
        try:
            out = ca.funding_dashboard(series, days=_number(q, "days", 120, 30, 720, integer=True))
        except ValueError as exc:
            raise ApiError(str(exc)) from None
        parts = {f"{r['symbol']}@{r['venue']}": r["source"] for r in out["rows"]}
        return _wrap("crypto_funding", out, parts, out["risk_flags"], max(s.as_of for s in series.values()))

    def crypto_cross_exchange(self, q: dict) -> dict:
        """``GET /api/crypto/cross-exchange?symbol=BTC``: every pair of venues of one kind, buy at one's ask and sell at the other's bid, net of fees and slippage, with the warnings. Real quotes when a venue
        snapshot has been downloaded, else modelled ones."""
        s = self._series(str(q.get("symbol") or "BTC"))
        snap = self._snapshot(s.symbol)
        try:
            out = ca.cross_exchange(s, _number(q, "slippage_bps", 1.0, 0.0, 50.0), _number(q, "notional", 1e6, 1e3, 1e9), snapshot=snap)
        except ValueError as exc:
            raise ApiError(str(exc)) from None
        if snap is not None:
            return _wrap("crypto_cross_exchange", out, {"quotes": "downloaded"}, out.pop("risk_flags"), (snap.get("fetched_at") or "")[:10] or s.as_of, out.pop("notes"))
        return _wrap("crypto_cross_exchange", out, {"deribit": s.source, "other_venues": "simulated"}, out.pop("risk_flags"), s.as_of, out.pop("notes") + ["Deribit's price is a daily mark, so the spreads below are indicative, not executable."])

    def crypto_strategy_backtest(self, body: dict) -> dict:
        """``POST /api/crypto/strategy/backtest``: a crypto strategy on the Deribit data through the platform pipeline, as a job."""
        name = str(body.get("strategy") or "spot_momentum")
        if name not in ca.CRYPTO_STRATEGIES:
            raise ApiError(f"strategy must be one of {sorted(ca.CRYPTO_STRATEGIES)}")
        if ca.CRYPTO_STRATEGIES[name]["model"] is None:
            raise ApiError("the exchange-flow strategy needs exchange net-flow data, which the platform does not hold; use stablecoin_flow (the on-chain signal that is available) or load your own data")
        assets = [a.upper() for a in _list(body.get("assets") or "BTC,ETH", 2)]
        if not assets or any(a not in ca.REAL_SYMBOLS for a in assets):
            raise ApiError(f"the strategy backtests use the platform's data: choose from {list(ca.REAL_SYMBOLS)}")
        lookback = None if body.get("lookback") in (None, "") else _number(body, "lookback", 90, 5, 756, integer=True)
        target = None if body.get("target_vol") in (None, "") else _number(body, "target_vol", 0.2, 0.01, 2.0)
        freq = _choice(body, "frequency", "weekly", ("daily", "weekly", "monthly"))
        fees, slip = _number(body, "fees_bps", 5.0, 0.0, 200.0), _number(body, "slippage_bps", 2.0, 0.0, 200.0)
        regime = None if body.get("regime") in (None, "", "none") else _choice(body, "regime", "vol_state", ("vol_state", "hmm", "gmm"))
        self._crypto_frames()

        def work(progress):
            progress("running the strategy through the pipeline")
            out = ca.run_crypto_strategy(self.app.config, name, assets, freq, lookback, target, fees, slip, regime)
            return _wrap("crypto_backtest", out, {"prices": "platform"}, [], out["end"], out.pop("notes"))

        return self.app.submit_task(f"crypto: {name}", work)
