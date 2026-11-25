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

import functools
import os
import re
import threading
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from ..assets import common
from ..assets import crypto_analytics as ca
from ..assets import crypto_validation as cv
from ..assets import futures_analytics as fa
from ..assets import options_analytics as oa
from ..assets.free_data.store import SAFE_NAME as DOWNLOAD_NAME
from ..assets.free_data.store import DownloadStore
from ..framework import MODELS, load_library
from ..references import references
from .api import ApiError, _choice, _number, parameters
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


def _wrap(kind: str, data: dict, parts: dict, warnings=(), as_of=None, notes=(), provenance: dict | None = None) -> dict:
    """A payload with its provenance. The analytics keep their own ``notes`` and ``warnings`` inside the data they return; they are folded into the envelope's lists, and their ``source`` (a single word the
    parts already say better) is dropped, so nothing clashes with the provenance keys."""
    data = dict(data)
    notes, warnings = list(notes) + list(data.pop("notes", None) or []), list(warnings) + list(data.pop("warnings", None) or [])
    notes = list(dict.fromkeys(notes))
    data.pop("source", None)
    return clean(common.envelope(kind, data, parts, list(dict.fromkeys(warnings)), as_of, notes, provenance))


class AssetApi:
    def __init__(self, app):
        self.app = app
        self.root = Path(app.root)
        self._lock = threading.Lock()
        self._frames: dict | None = None
        self.downloads = DownloadStore(self.root)
        self._tl = threading.local()
        self.observed_default = os.environ.get("QUANT_OBSERVED_ONLY", "1").lower() not in ("0", "false", "no", "off")
        from .crypto_research_api import CryptoResearchApi

        self.research = CryptoResearchApi(self)
        from .options_studio_api import OptionsStudioApi

        self.studio = OptionsStudioApi(self)

    # ------------------------------------------------------------------------------------------------------------ data mode
    def observed(self) -> bool:
        """Whether the request being served is in observed-only mode (``data_mode=observed``, the default; ``data_mode=model`` allows the platform's simulated data, labelled as such)."""
        value = getattr(self._tl, "observed", None)
        return self.observed_default if value is None else bool(value)

    def require_model(self, what: str) -> None:
        """Refuse ``what`` in observed-only mode: it is simulated, and the page never substitutes it silently."""
        if self.observed():
            raise ApiError(f"observed-only mode: {what} is simulated. Use observed data (a Massive or downloaded chain, your own file), or switch the data mode to 'model data allowed' to study the simulation knowingly")

    def _gated(self, fn):
        def inner(arg):
            mode = str((arg or {}).get("data_mode") or "")
            if mode not in ("", "observed", "model"):
                raise ApiError("data_mode must be observed or model")
            previous = getattr(self._tl, "observed", None)
            self._tl.observed = (mode == "observed") if mode else self.observed_default
            try:
                return fn(arg)
            finally:
                self._tl.observed = previous

        functools.update_wrapper(inner, fn)
        inner.__self__ = getattr(fn, "__self__", None)                                 # callers (tests) reach the API object through the registered handler
        return inner

    # ------------------------------------------------------------------------------------------------------------ routes
    def get_routes(self) -> dict:
        return {k: self._gated(v) for k, v in {**self._get_routes(), **self.research.get_routes(), **self.studio.get_routes()}.items()}

    def post_routes(self) -> dict:
        return {k: self._gated(v) for k, v in {**self._post_routes(), **self.research.post_routes(), **self.studio.post_routes()}.items()}

    def _get_routes(self) -> dict:
        return {"/api/assets/catalog": self.catalog, "/api/options/chain": self.options_chain, "/api/options/surface": self.options_surface, "/api/options/distribution": self.options_distribution,
                "/api/options/positioning": self.options_positioning, "/api/options/surface/research": self.options_surface_research, "/api/options/vol-surface": self.options_vol_surface, "/api/futures/curve": self.futures_curve,
                "/api/futures/roll": self.futures_roll, "/api/futures/research": self.futures_research, "/api/futures/spread": self.futures_spread, "/api/experiments": self.experiments_list, "/api/options/history": self.options_history, "/api/options/backtest-modes": self.options_backtest_modes, "/api/experiments/get": self.experiments_get, "/api/futures/series": self.futures_series, "/api/crypto/overview": self.crypto_overview, "/api/crypto/funding": self.crypto_funding, "/api/crypto/cross-exchange": self.crypto_cross_exchange}

    def _post_routes(self) -> dict:
        return {"/api/options/portfolio-risk": self.options_portfolio_risk, "/api/options/strategy/payoff": self.options_strategy_payoff, "/api/options/strategy/backtest": self.options_strategy_backtest,
                "/api/options/strategy/sweep": self.options_strategy_sweep, "/api/options/strategy/walkforward": self.options_strategy_walkforward, "/api/options/strategy/compare": self.options_strategy_compare, "/api/futures/margin": self.futures_margin, "/api/experiments/save": self.experiments_save, "/api/options/snapshot": self.options_snapshot, "/api/options/historical-backtest": self.options_historical_backtest, "/api/experiments/delete": self.experiments_delete, "/api/experiments/check": self.experiments_check, "/api/experiments/compare": self.experiments_compare, "/api/futures/strategy/backtest": self.futures_strategy_backtest, "/api/futures/signals": self.futures_signals, "/api/crypto/signals": self.crypto_signals, "/api/crypto/strategy/backtest": self.crypto_strategy_backtest}

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
        dl_series = [{"id": n, "code": r.get("code", n), "region": r.get("region", ""), "name": r.get("root_name", n), "exchange": r.get("exchange", ""), "group": r.get("group", ""), "source": "downloaded",
                      "as_of": r.get("as_of"), "first_date": r.get("first_date"), "bars": r.get("bars"), "fetched_at": r.get("fetched_at"), "provider": r.get("source", "")} for n, r in self._downloaded("futures", "series").items()]
        underlyings = [{"id": a, "label": a, "source": "simulated chain on real prices", "has_index": a in oa.REFERENCE_INDEX} for a in b.assets if a in b.prices.columns]
        underlyings += [{"id": f"dl:{n}", "label": f"{n} (real chain, {r.get('source', 'downloaded')}, {r.get('as_of', '?')})", "source": "downloaded", "has_index": str(r.get("underlying", "")).upper() in oa.REFERENCE_INDEX,
                         "as_of": r.get("as_of"), "fetched_at": r.get("fetched_at")} for n, r in dl_chains.items()]
        underlyings += [{"id": f"file:{n}", "label": f"{n} (your file)", "source": "user_file", "has_index": False} for n in opt_files]

        def span(r: str) -> dict:
            rec = None if r in fut_files else dl_curves.get(r)
            return {} if rec is None else {"first_date": rec.get("first_date"), "as_of": rec.get("as_of"), "contracts": rec.get("contracts")}

        roots = [{"id": r, "name": s.name, "asset_class": s.asset_class, "exchange": s.exchange, "multiplier": s.multiplier, "tick_size": s.tick_size,
                  "source": "user_file" if r in fut_files else "downloaded" if r in dl_curves else "simulated", **span(r)} for r, s in fa.ROOTS.items()]
        roots += [{"id": r, "name": rec.get("root_name", r), "asset_class": rec.get("asset_class", "unknown"), "exchange": rec.get("exchange", "?"), "multiplier": float(rec.get("multiplier", 1.0)),
                   "tick_size": float(rec.get("tick_size", 0.01)), "source": "user_file" if r in fut_files else "downloaded", **span(r)} for r, rec in dl_curves.items() if r not in fa.ROOTS]
        roots += [{"id": r, "name": r, "asset_class": "unknown", "exchange": "?", "multiplier": 1.0, "tick_size": 0.01, "source": "user_file"} for r in fut_files if r not in fa.ROOTS and r not in dl_curves]
        symbols = [{"id": s, "source": "user_file" if s in cry_files else "platform"} for s in ca.REAL_SYMBOLS] + [{"id": s, "source": "user_file" if s in cry_files else "downloaded" if s in dl_coins else "simulated"} for s in ca.SIMULATED_ALTS]
        symbols += [{"id": s, "source": "user_file" if s in cry_files else "downloaded"} for s in dl_coins if s not in ca.REAL_SYMBOLS and s not in ca.SIMULATED_ALTS]
        symbols += [{"id": s, "source": "user_file"} for s in cry_files if s not in ca.REAL_SYMBOLS and s not in ca.SIMULATED_ALTS and s not in dl_coins]
        return clean({"underlyings": underlyings, "option_strategies": [{"id": k, "text": oa.STRATEGY_INFO[k], "group": g, "references": references("options", k)} for g, names in oa.STRATEGY_GROUPS.items() for k in names], "option_groups": list(oa.STRATEGY_GROUPS),
                      "option_backtests": [{"id": k, "text": oa.BACKTEST_TEXT[k], "params": oa.backtest_params(k), "references": references("options", k)} for k in oa.BACKTEST_STRATEGIES],
                      "option_timing": {"defaults": oa.TIMING_DEFAULTS, "references": references("options", "timing")},
                      "defaults": {"rate": oa.DEFAULT_RATE, "premium": 0.15, "dividend_yields": oa.DIVIDEND_YIELDS}, "futures_roots": roots, "futures_series": dl_series,
                      "futures_strategies": [self._futures_strategy_info(k, v) for k, v in fa.FUTURES_STRATEGIES.items()],
                      "crypto_symbols": symbols, "crypto_validation": [{"id": k, "label": v["label"], "family": v["family"], "needs": v["needs"], "params": [{"name": n, "default": d} for n, d in v["params"].items()]} for k, v in cv.FAMILIES.items()],
                      "observed_default": self.observed_default, "crypto_strategies": [self._crypto_strategy_info(k, v) for k, v in ca.CRYPTO_STRATEGIES.items()],
                      "venues": list(ca.VENUE_ORDER), "data_dir": "data/user/", "as_of": str(b.index[-1].date())})

    @staticmethod
    def _futures_strategy_info(key: str, spec: dict) -> dict:
        """One futures strategy as the page lists it: its words, its group, what it needs of the roots, the settings of the strategy behind it and where its rule comes from."""
        load_library()
        entry = next(e for e in MODELS.entries() if e.name == spec["model"])
        return {"id": key, "label": spec["label"], "text": spec["text"], "group": spec.get("group", "Other"), "needs": spec.get("needs"), "model": spec["model"], "params": parameters("model", spec["model"], entry.factory),
                "references": references("model", spec["model"])}

    @staticmethod
    def _crypto_strategy_info(key: str, spec: dict) -> dict:
        """One crypto strategy as the page lists it: its words and group, whether the platform has the data for it, the settings of the strategy behind it (with this strategy's own defaults) and its sources."""
        info = {"id": key, "label": spec["label"], "text": spec["text"], "group": spec.get("group", "Other"), "available": spec["model"] is not None, "model": spec["model"], "params": [], "references": []}
        if spec["model"] is not None:
            load_library()
            entry = next(e for e in MODELS.entries() if e.name == spec["model"])
            info["params"] = [{**p, "default": spec["params"].get(p["name"], p["default"])} for p in parameters("model", spec["model"], entry.factory)]
            info["references"] = references("model", spec["model"])
        return info

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
        price_note = ""
        if symbol in b.prices.columns:
            try:
                ctx = replace(oa.underlying_context(symbol, b.prices[symbol], b.volume[symbol] if b.volume is not None and symbol in b.volume.columns else None), spot=spot, as_of=quote_date)
                price_note = f"Realised volatility is from the platform's prices of {symbol} through {b.index[-1].date()}."
            except ValueError:
                ctx = None
        else:
            ctx, price_note = self._context_from_download(symbol, spot, quote_date)
        if ctx is None:
            ctx = oa.UnderlyingContext(symbol, spot, quote_date, float("nan"), float("nan"), float("nan"), pd.Series(dtype=float), float("nan"))
        notes = [f"Downloaded {str(rec.get('fetched_at', '?')).replace('T', ' ').rstrip('Z')} UTC from {rec.get('source', 'a free source')}; the quotes are for {rec.get('as_of')}."] + [str(n) for n in rec.get("notes", [])]
        if ctx.rv_21 == ctx.rv_21 and price_note:
            notes.append(price_note)
        elif ctx.rv_21 != ctx.rv_21:
            notes.append(f"{symbol} is not in the platform's price data, so there is no realised volatility to set beside the implied one. Tick \"Also download the underlying's daily prices\" in the Free data card to get one.")
        if rate is not None or dividend is not None:
            notes.append("The rate and dividend yield of a downloaded chain come with it and are not editable here.")
        return chain, ctx, "downloaded", notes

    def _context_from_download(self, symbol: str, spot: float, quote_date: pd.Timestamp):
        """``(context, note)`` from the daily prices downloaded for an underlying the platform does not hold (``<SYMBOL>_prices``), or ``(None, "")``."""
        rec = self._downloaded("options", "prices").get(f"{symbol}_prices")
        if rec is None:
            return None, ""
        try:
            frame = pd.read_csv(self.downloads.folder("options") / rec["file"], parse_dates=["date"]).set_index("date").sort_index()
            frame = frame[frame.index <= quote_date + pd.Timedelta(days=1)]                            # prices after the chain's date would be a look ahead
            volume = frame["volume"] if "volume" in frame and frame["volume"].notna().any() else None
            ctx = replace(oa.underlying_context(symbol, frame["close"], volume), spot=spot, as_of=quote_date)
        except (ValueError, OSError, KeyError):
            return None, ""
        return ctx, f"Realised volatility is from the daily prices of {symbol} downloaded from {rec.get('source', 'a free source')} through {frame.index[-1].date()}."

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
        self.require_model(f"the option chain of {underlying} (a model quoted on the ETF's real close)")
        b = self._bundle()
        vol_frame = b.volume[underlying] if b.volume is not None and underlying in b.volume.columns else None
        try:
            chain, ctx, r, div = oa.simulated_chain(underlying, b.prices[underlying], vol_frame, rate, dividend, premium)
        except ValueError as exc:
            raise ApiError(str(exc)) from None
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

    def options_distribution(self, q: dict) -> dict:
        """``GET /api/options/distribution?underlying=SPY&expiry=2026-11-27``: what the quotes imply for the price at one expiry (the density, quantiles and the chance of a 5 to 20% fall or rise against the
        lognormal's), the move the at-the-money straddle prices in at every expiry, the model-free implied volatility of every expiry and at 30 days (with the published index beside it where the platform has
        one), and the volatility cone of the underlying with the chain's at-the-money volatilities on it."""
        underlying, rate, dividend, premium = self._option_params(q)
        chain, ctx, source, notes = self._option_chain(underlying, rate, dividend, premium)
        try:
            expiry = pd.Timestamp(q["expiry"]) if q.get("expiry") else oa.nearest_expiry_to(chain, 30)
            dist = oa.implied_distribution(chain, expiry)
        except (ValueError, KeyError) as exc:
            raise ApiError(str(exc)) from None
        term = oa.model_free_term(chain)
        rv = ctx.rv_21 if np.isfinite(ctx.rv_21) else None
        macro = self._bundle().macro if source != "user_file" else None
        ref = oa.reference_index_view(ctx.symbol, macro, term["atm_30d"], ctx.as_of)
        if ref is not None:
            ref["model_free"] = term["iv_30d"]
        cone = oa.volatility_cone(ctx.prices) if ctx.prices is not None and len(ctx.prices) > 120 else None
        if cone and cone["rows"]:                                                       # the at-the-money volatility of an expiry as long as each window, where the chain has one that long
            atm = oa.term_structure(chain)
            for r in cone["rows"]:
                r["implied"] = oa.interpolate_iv(atm, r["calendar_days"]) if atm and atm[0]["dte"] <= r["calendar_days"] <= atm[-1]["dte"] else None
        data = {"underlying": underlying, "expiry": dist["expiry"], "expiries": [e for e in oa.expiry_list(chain) if e["dte"] >= 2], "distribution": dist, "expected_moves": oa.expected_moves(chain),
                "model_free": {**term, "realized_21d": rv, "reference": ref}, "cone": cone if cone and cone["rows"] else None,
                "references": {k: references("options", k) for k in ("distribution", "model_free", "cone")}}
        warnings = [f"The fit does not reproduce the quotes closely (largest miss {dist['fit']['max_error_vol'] * 100:.1f} volatility points): read the shape with care."] if dist["fit"]["max_error_vol"] > 0.01 else []
        warnings += ["The fitted smile admits a butterfly arbitrage, so part of the density was negative and was set to zero."] if dist["fit"]["negative_share"] > 0.001 else []
        simulated = source == "simulated"
        return _wrap("options_distribution", data, {"underlying_price": "platform" if simulated else source, "chain": source}, warnings, ctx.as_of,
                     notes + ["The distribution is the one the options are PRICED on (risk-neutral): it has the market's price of risk inside it, so a fall looks likelier than it has been in practice.",
                              "It is read from an SVI curve fitted to the out-of-the-money implied volatilities of one expiry; the model-free volatility needs no curve (the Cboe VIX rules on the quotes themselves)."])

    def options_positioning(self, q: dict) -> dict:
        """``GET /api/options/positioning?underlying=SPY&expiry=2026-11-27``: open interest and volume by strike (one expiry, or every expiry when none is given), the walls, the max-pain strike of one expiry and the
        gamma exposure with the price at which it flips sign."""
        underlying, rate, dividend, premium = self._option_params(q)
        chain, ctx, source, notes = self._option_chain(underlying, rate, dividend, premium)
        dealer = str(q.get("dealer") or oa.DEFAULT_DEALER)
        if dealer.startswith("custom:"):                                                          # custom:<call sign>,<put sign>, each between -1 and 1
            try:
                dealer = tuple(float(x) for x in dealer[7:].split(","))
                if len(dealer) != 2:
                    raise ValueError
            except ValueError:
                raise ApiError("dealer custom:<call sign>,<put sign> takes two numbers between -1 and 1") from None
        share = _number(q, "dealer_share", 1.0, 0.0, 1.0)
        try:
            expiry = pd.Timestamp(q["expiry"]) if q.get("expiry") and q.get("expiry") != "all" else None
            view = oa.positioning(chain, expiry, dealer=dealer, dealer_share=share)
        except ValueError as exc:
            raise ApiError(str(exc)) from None
        data = {"underlying": underlying, "expiry": None if expiry is None else str(expiry.date()), "expiries": [e for e in oa.expiry_list(chain) if e["dte"] >= 1], "positioning": view,
                "references": {"positioning": references("options", "positioning")}}
        warnings = ["The chain has no open interest (the file did not carry it), so there is nothing to show."] if view["call_oi"] + view["put_oi"] == 0 else []
        simulated = source == "simulated"
        if not view["uncertainty"]["sign_agrees_across_scenarios"]:
            warnings.append("The sign of the net gamma exposure depends on who is assumed to hold the other side: the four standard assumptions disagree, so the net figure is a property of the assumption, not of the market.")
        return _wrap("options_positioning", data, {"underlying_price": "platform" if simulated else source, "open_interest": source}, warnings, ctx.as_of,
                     notes + ["Open interest has no sign: who is long and who is short is not in the data. Gross gamma (open interest times gamma) is observed; the net gamma exposure is conditional on an ASSUMED "
                              "dealer position, chosen above and compared with the other standard assumptions. Whether dealers hedge dynamically, and with what price impact, is a hypothesis.",
                              "Max pain is a folk rule for the settlement of one expiry; the evidence behind it is that prices tend to cluster at strikes with large open interest near expiry."],
                     provenance={"instrument": f"{underlying} options" + ("" if expiry is None else f" {expiry.date()}"), "assumptions": view["assumptions"]})

    def options_surface_research(self, q: dict) -> dict:
        """``GET /api/options/surface/research?underlying=SPY&expiries=8``: the surface in log-forward-moneyness, delta and total-variance coordinates with a fitted SVI slice per expiry (residuals, bid-ask bands,
        butterfly condition), the calendar condition between slices, a joint SSVI fit with its arbitrage conditions, and what is not guaranteed."""
        underlying, rate, dividend, premium = self._option_params(q)
        chain, ctx, source, notes = self._option_chain(underlying, rate, dividend, premium)
        n = int(_number(q, "expiries", 8, 2, 14, integer=True))
        try:
            out = oa.surface_research(chain, max_expiries=n)
        except ValueError as exc:
            raise ApiError(str(exc)) from None
        warnings = []
        if out["arbitrage_summary"]["butterfly_violating_slices"]:
            warnings.append(f"{out['arbitrage_summary']['butterfly_violating_slices']} fitted slice(s) fail the butterfly condition on the grid.")
        if out["arbitrage_summary"]["calendar_violating_pairs"]:
            warnings.append(f"{out['arbitrage_summary']['calendar_violating_pairs']} pair(s) of neighbouring fitted slices violate the calendar condition.")
        return _wrap("options_surface_research", {"underlying": underlying, "surface": out}, {"quotes": "platform" if source == "simulated" else source, "fitted_surface": "simulated" if source == "simulated" else "model"},
                     warnings, ctx.as_of, notes + out["limits"], provenance={"instrument": f"{underlying} options", "assumptions": out["limits"],
                     "calibration": {"status": "calibrated", "model": "SVI per expiry, SSVI jointly", "rmse": float(np.mean([s["rmse_vol_points"] for s in out["slices"]]) / 100.0), "fit_date": str(ctx.as_of.date())}})

    def options_vol_surface(self, q: dict) -> dict:
        """``GET /api/options/vol-surface?underlying=SPY&date=2026-01-02&selection=otm&coordinate=k&min_dte=0&max_dte=400&nx=41&fit=1&compare=2025-12-01``: the volatility surface as built in
        ``assets/vol_surface.py``: cleaned quotes with every rejection counted, one grid for all expiries, observed-supported nodes only (masked elsewhere, never extrapolated), raw and fitted surfaces apart,
        diagnostics, the discrete smiles and at-the-money term structure (optionally with the term structure of an earlier kept date). ``date`` picks a kept dated chain (``/api/options/history``); without it
        the current chain is used. Cached by underlying, date, filters and settings."""
        from ..assets import vol_surface as vsf
        from ..assets.option_history import History
        underlying, rate, dividend, premium = self._option_params(q)
        selection, coordinate = _choice(q, "selection", "otm", vsf.SELECTIONS), _choice(q, "coordinate", "k", vsf.COORDINATES)
        min_dte, max_dte = _number(q, "min_dte", 0, 0, 5000), _number(q, "max_dte", 10_000, 1, 10_000)
        nx = _number(q, "nx", 41, 11, 161, integer=True)
        fit = str(q.get("fit", "1")).lower() not in ("0", "false", "no")
        hist = History(self.root)
        try:
            kept = hist.dates(underlying)
        except ValueError:
            kept = []
        date = str(q.get("date") or "")
        compare = str(q.get("compare") or "")
        for name, d in (("date", date), ("compare", compare)):
            if d and d != "current" and d not in kept:
                raise ApiError(f"{name} must be a kept date ({kept if kept else 'none kept yet: use Keep todays chain in the Model validation view'}) or empty for the current chain")
        chain, ctx, source, notes = self._option_chain(underlying, rate, dividend, premium)
        as_of = str(ctx.as_of.date())
        if date and date != "current":
            chain, source, as_of = hist.load(underlying, date), "snapshot", date
            chain = oa.prepare_chain(chain)
        key = (underlying, as_of, source, selection, coordinate, float(min_dte), float(max_dte), nx, fit, compare, round(float(rate or 0), 6), round(float(dividend or 0), 6))
        cache = self.__dict__.setdefault("_vol_surface_cache", OrderedDict())
        with self._lock:
            hit = cache.get(key)
            if hit is not None:
                cache.move_to_end(key)
        if hit is None:
            try:
                hit = vsf.build(chain, selection, coordinate, min_dte, max_dte, nx, fit)
            except ValueError as exc:
                raise ApiError(str(exc)) from None
            if compare:
                hit["history"] = {"date": compare, "term": vsf.term_from_chain(oa.prepare_chain(hist.load(underlying, compare)), selection)}
            with self._lock:
                cache[key] = hit
                while len(cache) > 24:
                    cache.popitem(last=False)
        simulated = source == "simulated"
        data = {"underlying": underlying, "source_kind": source, "dates": ["current"] + kept, "surface": hit}
        warnings = []
        m = hit["meta"]
        if m["status"] != "surface":
            warnings.append(m["reason"])
        elif hit["fit"] and (hit["fit"]["calendar"]["violations"] or hit["fit"]["butterfly_failing_slices"]):
            warnings.append(hit["fit"]["reading"])
        cal = {"status": "calibrated", "model": "SVI per expiry", "rmse": float(hit["fit"]["mean_rmse_vol_points"] / 100.0), "fit_date": as_of} if hit.get("fit") else {"status": "not_applicable"}
        return _wrap("options_vol_surface", data, {"quotes": "platform" if simulated else ("downloaded" if source == "snapshot" else source), "fitted_surface": "simulated" if simulated else "model"}, warnings, ctx.as_of if as_of == str(ctx.as_of.date()) else pd.Timestamp(as_of),
                     notes + hit["limits"], provenance={"instrument": f"{underlying} options", "assumptions": hit["limits"], "calibration": cal})

    def options_surface(self, q: dict) -> dict:
        """``GET /api/options/surface?underlying=SPY``: the implied-volatility surface (strike by expiry), the term structure, the skew metrics of every expiry and the static no-arbitrage checks."""
        underlying, rate, dividend, premium = self._option_params(q)
        chain, ctx, source, notes = self._option_chain(underlying, rate, dividend, premium)
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
            arb = oa.arbitrage_view(chain, liquid_only=downloaded, band=band, exercise="american" if american and downloaded else "european")                  # a downloaded chain is real quotes, delayed: only the live ones near the forward say anything
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
            tags = {k: str(p[k])[:24] for k in ("underlying", "strategy") if p.get(k)}
            if p.get("beta") not in (None, ""):
                tags["beta"] = _number(p, "beta", 1.0, -5.0, 5.0)
            if p.get("style"):
                tags["style"] = _choice(p, "style", "european", ("european", "american"))
            if p.get("spot") not in (None, ""):
                tags["spot"] = _number(p, "spot", 0.0, 0.0001, 1e9)
            if kind == "stock":
                out.append({"type": "stock", "quantity": qty, **tags})
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
                if "spot" in tags:
                    raise ApiError(f"position {i + 1}: a position with its own spot needs its own iv (the chain loaded is for {chain.attrs.get('underlying', 'another underlying')})")
                iv = oa.iv_lookup(chain, listed, strike)
            out.append({"type": "option", "right": right, "strike": strike, "quantity": qty, "expiry": str(exp.date()), "iv": float(iv), **tags})
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
            risk = oa.portfolio_risk(positions, ctx.spot, float(chain["rate"].iloc[0]), float(chain["dividend"].iloc[0]), ctx.as_of, spot_shocks, vol_shocks, days,
                                     _choice(body, "vol_rule", "sticky_strike", tuple(oa.VOL_RULES)), _number(body, "skew_slope", 0.0, -3.0, 3.0), _choice(body, "exercise", "european", ("european", "american")))
        except ValueError as exc:
            raise ApiError(str(exc)) from None
        return _wrap("options_portfolio_risk", {"underlying": underlying, **risk}, {"prices": "platform" if simulated else source, "volatilities": source},
                                     risk["risk_flags"], ctx.as_of, notes + ["Stress P&L revalues every option with Black-Scholes at the shocked spot and volatility; time passes only if you set days."])

    def options_strategy_payoff(self, body: dict) -> dict:
        """``POST /api/options/strategy/payoff``: the legs of a named structure (or your own), their payoff at the horizon, today and half way, maximum profit and loss, break-evens, Greeks, an indicative margin, the probability of profit and expected value, the odds by price, the profit by price and day, the Greeks by price, and (``compare``) up to three other structures on the same settings."""
        underlying, rate, dividend, premium = self._option_params(body)
        chain, ctx, source, notes = self._option_chain(underlying, rate, dividend, premium)
        simulated = source == "simulated"
        custom = bool(body.get("legs"))
        name = None if custom else str(body.get("strategy") or "long_straddle")
        dte = _number(body, "dte", 30, 1, 700, integer=True)
        far = _number(body, "far_dte", max(dte * 2, dte + 14), dte + 1, 800, integer=True)
        qty = _number(body, "quantity", 1, 1, 1000, integer=True)
        offset, width = _number(body, "offset", 0.05, 0.0, 0.5), _number(body, "width", 0.05, 0.005, 0.5)
        others = body.get("compare") or []
        if not isinstance(others, list) or len(others) > 3 or not all(isinstance(x, str) for x in others):
            raise ApiError("compare must be a list of up to three structure names")
        try:
            if custom:
                legs, built = oa.chain_legs(chain, self._custom_legs(body["legs"], chain, ctx.as_of)), oa.structure_context(chain, dte, far)
            else:
                legs, built = oa.named_structure(chain, ctx.spot, name, dte, offset, width, far, qty)
            atm = built["atm_iv"]
            view_vol = None if body.get("view_vol") in (None, "") else _number(body, "view_vol", atm, 0.02, 3.0)
            view_drift = None if body.get("view_drift") in (None, "") else _number(body, "view_drift", 0.0, -2.0, 2.0)
            horizon = None if body.get("horizon_days") in (None, "") else _number(body, "horizon_days", built["dte"], 1, 800, integer=True)
            common = {"rate": float(chain["rate"].iloc[0]), "dividend": float(chain["dividend"].iloc[0]), "view_vol": view_vol, "view_drift": view_drift, "horizon_days": horizon,
                      "skew_slope": _number(body, "skew_slope", oa.chain_skew_slope(chain, dte), -5.0, 5.0), "fill": _choice(body, "fill", "mid", ("mid", "worst")), "spread_pct": _number(body, "spread_pct", 0.05, 0.0, 1.0), "commission": _number(body, "commission", 0.65, 0.0, 50.0)}
            result = oa.strategy_analysis(legs, ctx.spot, built["forward"], atm, cash_secured=(name == "cash_secured_put"), **common)
            unknown = [x for x in others if x not in oa.STRATEGY_INFO]
            if unknown:
                raise ValueError(f"unknown structure {unknown[0]!r} to compare; choose from {sorted(oa.STRATEGY_INFO)}")
            compare = oa.compare_structures(chain, ctx.spot, [x for x in dict.fromkeys(others) if x != name], result["curve"]["spot"], dte=dte, offset=offset, width=width, far_dte=far, quantity=qty, analysis=common)
        except ValueError as exc:
            raise ApiError(str(exc)) from None
        data = {"underlying": underlying, "strategy": name, "group": None if custom else oa.group_of(name), "text": "Your own legs" if custom else oa.STRATEGY_INFO[name], "expiry": str(pd.Timestamp(built["expiry"]).date()),
                "far_expiry": str(pd.Timestamp(built["far_expiry"]).date()), "atm_iv": atm, "analysis": result, "compare": compare, "references": [] if custom else references("options", name)}
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

    @staticmethod
    def _backtest_arguments(body: dict) -> dict:
        """The market, costs and filter of an options backtest, checked: the arguments of ``run_strategy_backtest`` after the strategy and its settings."""
        try:
            timing = oa.parse_timing(body.get("timing"))
        except ValueError as exc:
            raise ApiError(str(exc)) from None
        return {"n_days": _number(body, "days", 252, 60, 756, integer=True), "seed": _number(body, "seed", 0, 0, 1_000_000, integer=True), "capital": _number(body, "capital", 100_000, 1e3, 1e9),
                "premium": _number(body, "premium", 0.25, -0.3, 1.0), "commission": _number(body, "commission", 0.65, 0.0, 50.0), "fill_fraction": _number(body, "fill_fraction", 1.0, 0.0, 1.0), "timing": timing}

    @staticmethod
    def _backtest_strategy(body: dict) -> tuple[str, dict]:
        name = str(body.get("strategy") or "short_put")
        if name not in oa.BACKTEST_STRATEGIES:
            raise ApiError(f"strategy must be one of {list(oa.BACKTEST_STRATEGIES)}")
        try:
            return name, oa.parse_backtest_params(name, body.get("params"))
        except ValueError as exc:
            raise ApiError(str(exc)) from None

    def options_strategy_backtest(self, body: dict) -> dict:
        """``POST /api/options/strategy/backtest``: one of the rolling option strategies on the simulated option market, run as a job."""
        self.require_model("the option market of this validation (a model with a built-in variance premium)")
        name, params = self._backtest_strategy(body)
        args = self._backtest_arguments(body)

        def work(progress):
            progress("building the simulated option market (the first run of a given length takes several seconds)")
            try:
                out = oa.run_strategy_backtest(name, params, **args)
            except ValueError as exc:
                raise ApiError(str(exc)) from None
            out["references"] = references("options", name) + (references("options", "timing") if args["timing"] else [])
            return _wrap("options_backtest", out, {"market": "simulated"}, [], out["dates"][-1] if out["dates"] else None, out.pop("notes"))

        return self.app.submit_task(f"options: {name}", work)

    def options_strategy_sweep(self, body: dict) -> dict:
        """``POST /api/options/strategy/sweep``: the same backtest at 2 to 9 values of one numeric setting (``sweep_param``, ``sweep_values``), run as a job."""
        self.require_model("the option market of this validation (a model with a built-in variance premium)")
        name, params = self._backtest_strategy(body)
        args = self._backtest_arguments(body)
        param, raw = str(body.get("sweep_param") or ""), body.get("sweep_values")
        if not isinstance(raw, list):
            raise ApiError("sweep_values must be a list of numbers")
        try:
            values = [float(x) for x in raw]
            oa.check_sweep(name, param, values, params)
        except (TypeError, ValueError) as exc:
            raise ApiError(str(exc) or "sweep_values must be numbers") from None

        def work(progress):
            progress("building the simulated option market (the first run of a given length takes several seconds)")
            out = oa.run_strategy_sweep(name, param, values, params, progress, **args)
            return _wrap("options_sweep", out, {"market": "simulated"}, [], None, ["Every value is run on the same simulated market and seed; only the setting changes."])

        return self.app.submit_task(f"options: sweep {name} {param}", work)

    def options_strategy_walkforward(self, body: dict) -> dict:
        """``POST /api/options/strategy/walkforward``: choose one setting of a rolling option strategy by walk-forward (``sweep_param``, ``sweep_values``, ``train`` and ``test`` days) and see the out-of-sample
        result against the best setting of the whole period, with a bootstrap interval; run as a job on the simulated option market."""
        self.require_model("the option market of this validation (a model with a built-in variance premium)")
        name = str(body.get("strategy") or "short_put")
        param = str(body.get("sweep_param") or "")
        values = body.get("sweep_values")
        if not isinstance(values, list) or not 2 <= len(values) <= 9:
            raise ApiError("sweep_values must be a list of 2 to 9 numbers")
        train, test = _number(body, "train", 126, 40, 400, integer=True), _number(body, "test", 42, 10, 200, integer=True)
        days = _number(body, "days", 504, 252, 756, integer=True)
        try:
            oa.check_sweep(name, param, values, body.get("params"))
        except ValueError as exc:
            raise ApiError(str(exc)) from None
        kwargs = {"n_days": days, "seed": _number(body, "seed", 0, 0, 10_000, integer=True), "premium": _number(body, "premium", 0.25, -0.3, 1.0), "commission": _number(body, "commission", 0.65, 0.0, 20.0),
                  "fill_fraction": _number(body, "fill", 1.0, 0.0, 1.0)}

        def work(progress):
            try:
                out = oa.run_strategy_walkforward(name, param, values, body.get("params"), train, test, progress, **kwargs)
            except ValueError as exc:
                raise ApiError(str(exc)) from None
            return _wrap("options_walkforward", out, {"option_market": "simulated"}, [], None, [out.pop("note")])

        return self.app.submit_task(f"options: walk-forward {name} over {param}", work)

    def options_strategy_compare(self, body: dict) -> dict:
        """``POST /api/options/strategy/compare``: every rolling strategy (or the ``strategies`` named) at its defaults on one simulated market, run as a job."""
        self.require_model("the option market of this validation (a model with a built-in variance premium)")
        args = self._backtest_arguments(body)
        names = body.get("strategies")
        if names is not None and (not isinstance(names, list) or not all(isinstance(x, str) and x in oa.BACKTEST_STRATEGIES for x in names) or len(names) < 2):
            raise ApiError(f"strategies must be a list of at least two of {list(oa.BACKTEST_STRATEGIES)}")

        def work(progress):
            progress("building the simulated option market (the first run of a given length takes several seconds)")
            out = oa.run_strategy_comparison(names, progress, **args)
            return _wrap("options_compare", out, {"market": "simulated"}, [], out["dates"][-1] if out["dates"] else None,
                         ["Every strategy is run at its default settings on the same simulated market and seed.", "The market pays the variance premium it was built with and has jumps, so premium sellers do well here by construction."])

        return self.app.submit_task("options: compare strategies", work)

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
        self.require_model(f"the {root} futures curve")
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
        """``GET /api/futures/curve?root=CL&spacing=21&dates=3&contracts=12``: the curve now and ``spacing`` trading days ago (``dates`` of them), the shape of the curve through time, its principal components,
        carry, basis and the one-line snapshot. ``contracts`` is how many of the nearest contracts the curve lists (2 to 400, or ``all``/``0`` for every contract the table has); above twelve, the shape of
        the curve through time follows it up to 24 contract months (it is eight otherwise), and the principal components stay on the nearest eight, where the daily changes are well observed."""
        data = self._curve(str(q.get("root") or "CL"))
        spacing, dates = _number(q, "spacing", 21, 1, 504, integer=True), _number(q, "dates", 3, 2, 6, integer=True)
        every = str(q.get("contracts") or "12").strip().lower() in ("0", "all")
        depth = None if every else _number(q, "contracts", 12, 2, 400, integer=True)
        try:
            snap = fa.curve_snapshot(data, depth)
            out = {"root": data.root, "curve": snap, "history": fa.curve_history(data, dates, spacing, depth), "shape": fa.curve_shape_history(data, 8 if depth is not None and depth <= 12 else min(depth or 24, 24)),
                   "pca": self._optional(fa.term_structure_pca, data), "carry": fa.carry_dashboard(data),
                   "basis": fa.basis_dashboard(data), "snapshot": fa.futures_snapshot(data).to_dict(),
                   "spec": {"name": data.spec.name, "exchange": data.spec.exchange, "multiplier": data.spec.multiplier, "tick_size": data.spec.tick_size, "tick_value": data.spec.tick_value,
                            "initial_margin": data.spec.initial_margin, "maintenance_margin": data.spec.maintenance_margin}}
        except ValueError as exc:
            raise ApiError(str(exc)) from None
        out["references"] = {name: references("futures", name) for name in ("curve", "convenience", "pca", "carry", "seasonality")}
        flags = list(out["snapshot"]["risk_flags"])
        return _wrap("futures_curve", out, self._curve_parts(data, margins=True), flags, data.as_of, data.notes)

    @staticmethod
    def _optional(fn, *args) -> dict | None:
        """The result of an analysis that needs more history than a short curve may have: ``None`` instead of an error, so that the rest of the page still draws."""
        try:
            return fn(*args)
        except ValueError:
            return None

    def futures_research(self, q: dict) -> dict:
        """``GET /api/futures/research?root=CL&date=&compare=NG,ES&from=&near=1&far=2``: the term-structure research views: the matrix of calendar spreads on a date, the history and z-score of one spread, the
        contango and backwardation regimes, how the curve changed between two dates, liquidity by contract, the expiry and roll calendar, and the shape of other markets' curves beside this one."""
        from . import asset_api as _self  # noqa: F401
        from ..assets import futures_research as fr
        data = self._curve(str(q.get("root") or "CL"))
        date = q.get("date") or None
        near, far = _number(q, "near", 1, 1, 12, integer=True), _number(q, "far", 2, 2, 24, integer=True)
        if far <= near:
            raise ApiError("far must be a later contract than near")
        out: dict = {"root": data.root}
        try:
            out["spread_matrix"] = fr.spread_matrix(data, date)
            out["spread_history"] = fr.spread_history(data, near, far)
            out["regimes"] = self._optional(fr.regime_analysis, data)
            out["liquidity"] = fr.liquidity(data, date)
            out["calendar"] = fr.expiry_calendar(data, 12)
            if q.get("from"):
                out["change"] = fr.change_decomposition(data, q["from"], date or data.as_of)
            roots = [r for r in str(q.get("compare") or "").replace(" ", "").split(",") if r and r != data.root]
            if roots:
                out["compare"] = fr.compare_curves({data.root: data, **{r: self._curve(r) for r in roots[:5]}}, date)
        except ValueError as exc:
            raise ApiError(str(exc)) from None
        out["spec"] = {"name": data.spec.name, "exchange": data.spec.exchange, "multiplier": data.spec.multiplier, "tick_size": data.spec.tick_size, "tick_value": data.spec.tick_value, "expiry_rule": data.spec.rule}
        return _wrap("futures_research", out, self._curve_parts(data, margins=False), [], data.as_of, data.notes + ["A calendar spread is a price difference between two contracts, not a profit or a cost of anything until a position is opened and closed."],
                     provenance={"instrument": f"{data.root} futures", "assumptions": ["contract identifiers and expiries follow the exchange rule for the root (holidays not modelled)", "ranks (F1, F2 ...) are rebuilt on every date from the contracts then alive"]})

    def futures_spread(self, q: dict) -> dict:
        """``GET /api/futures/spread?a=CL&b=NG&lookback=126&entry=2&exit=0.5&stop=4&cost_bps=3&simulated=0``: the relative-value bench for the spread ``ln I_A - beta ln I_B`` of two roots' excess-return indices:
        hedge ratio (OLS, total least squares, rolling), stationarity and cointegration tests, half-life, rolling correlation, seasonality, regimes and a walk-forward mean-reversion backtest. A simulated curve is
        refused unless ``simulated=1`` is sent, and then the provenance says so."""
        from ..assets import spread_lab as sl
        a, b = str(q.get("a") or "CL").upper(), str(q.get("b") or "NG").upper()
        if a == b:
            raise ApiError("choose two different roots")
        ca, cb = self._curve(a), self._curve(b)
        allow = str(q.get("simulated") or "0") in ("1", "true")
        sims = [c.root for c in (ca, cb) if c.source == "simulated"]
        if sims and not allow:
            raise ApiError(f"{', '.join(sims)} has no real curve (simulated data would be used): download or upload it, or send simulated=1 to study the simulated curve knowingly")
        lookback = _number(q, "lookback", 126, 40, 750, integer=True)
        try:
            ia, ib = sl.leg_index(ca), sl.leg_index(cb)
            frame = pd.concat([ia, ib], axis=1, keys=["a", "b"]).dropna()
            if len(frame) < 120:
                raise ValueError(f"only {len(frame)} dates are common to {a} and {b}")
            hr = sl.hedge_ratio(frame["a"], frame["b"], min(252, max(60, len(frame) // 3)))
            spread = frame["a"] - hr["ols"] * frame["b"]
            diag = sl.diagnostics(frame["a"], frame["b"], spread)
            bt = sl.backtest(frame["a"], frame["b"], lookback, _number(q, "entry", 2.0, 0.5, 6.0), _number(q, "exit", 0.5, 0.0, 3.0), _number(q, "stop", 4.0, 1.0, 10.0), _number(q, "cost_bps", 3.0, 0.0, 100.0), _number(q, "max_leverage", 1.0, 0.1, 5.0))
        except ValueError as exc:
            raise ApiError(str(exc)) from None
        step = max(1, len(spread) // 600)
        out = {"a": a, "b": b, "hedge_ratio": hr, "diagnostics": diag, "backtest": bt, "spread": {"dates": [str(d.date()) for d in spread.index[::step]], "values": [round(float(v), 5) for v in spread.iloc[::step]]},
               "hypothesis": "A statistical relationship between two prices is a hypothesis: stationarity and cointegration tests on a sample can fail to reject by chance, and relationships break."}
        warn = [] if diag["stationary_at_5pct"] else ["The spread does not pass the stationarity test at 5%: the backtest below is a test of a hypothesis the data do not support."]
        return _wrap("futures_spread", out, {"prices": common.overall_source({r: c.source for r, c in ((a, ca), (b, cb))})}, warn, frame.index[-1], ca.notes + cb.notes,
                     provenance={"instrument": f"{a}/{b} excess-return spread", "assumptions": ["legs are the log excess-return indices of the front contract (roll days 5)", "costs are one flat figure in basis points of traded notional"]})

    # ------------------------------------------------------------------------------------------------------------ chain history and calibration stability
    def options_snapshot(self, body: dict) -> dict:
        """``POST /api/options/snapshot``: keep today's chain of an underlying (one file per date) so that calibrations can be followed through time. A simulated chain is refused unless ``simulated`` is true."""
        from ..assets.option_history import History
        underlying, rate, dividend, premium = self._option_params(body)
        chain, ctx, source, notes = self._option_chain(underlying, rate, dividend, premium)
        if source == "simulated" and not body.get("simulated"):
            raise ApiError("this chain is simulated: its parameters are set by the platform, so a history of it says nothing about a market. Download a real chain (the Free data card) or send simulated=true to keep it knowingly")
        try:
            out = History(self.root).save(underlying, chain, ctx.as_of, source)
        except ValueError as exc:
            raise ApiError(str(exc)) from None
        return _wrap("options_snapshot", out, {"chain": source}, [], ctx.as_of, notes)

    def options_backtest_modes(self, q: dict) -> dict:
        """``GET /api/options/backtest-modes?underlying=SPY``: the two kinds of options backtest and which one the platform can run. Synthetic validation (a model market) runs in model-data mode and says
        so; a historical backtest needs dated chains with bid and ask, which the platform stores only as the snapshots you took. They are never ranked together."""
        from ..assets.option_history import History

        underlying = str(q.get("underlying") or "SPY")
        try:
            dates = History(self.root).dates(underlying)
        except ValueError as exc:
            raise ApiError(str(exc)) from None
        synthetic = {"id": "synthetic_validation", "label": "Synthetic validation (model market)", "state": "available" if not self.observed() else "needs_model_data",
                     "reason": "" if not self.observed() else "the market is simulated: switch the data mode to 'model data allowed' to run it", "limitations": oa.BACKTEST_LIMITATIONS}
        need = 60
        from ..assets import options_historical as oh

        need = oh.MIN_DATES
        historical = {"id": "historical", "label": "Empirical backtest (stored observed chains)", "state": "available" if len(dates) >= need else "unavailable", "snapshots": len(dates), "snapshots_needed": need,
                      "reason": (f"{len(dates)} dated chains of {underlying} are stored: enough to replay them at their own bid and ask." if len(dates) >= need else f"{len(dates)} dated chains of {underlying} are stored; an empirical backtest replays a dated bid and ask chain on each date (at least {need}). The Massive daily aggregates give "
                                "closes without bid and ask, so rebuilt chains cannot cross a spread."),
                      "needs": [] if len(dates) >= need else [f"a daily snapshot of the chain (the Snapshot button) on at least {need} dates, or a vendor history with historical bid and ask quotes"]}
        return _wrap("options_backtest_modes", {"underlying": underlying, "modes": [synthetic, historical], "ranking": "never combined: the two kinds of result answer different questions and are shown apart"},
                     {"modes": "platform"}, [], dates[-1] if dates else None)

    def options_historical_backtest(self, body: dict) -> dict:
        """``POST /api/options/historical-backtest``: an empirical backtest of a named structure on the stored observed chains, at their own bid and ask (``assets.options_historical``)."""
        from ..assets import options_historical as oh
        from ..assets.option_history import History

        underlying = str(body.get("underlying") or "SPY")
        structure = str(body.get("structure") or "cash_secured_put")
        try:
            out = oh.run(History(self.root), underlying, structure, int(_number(body, "dte", 30, 3, 400, integer=True)), int(_number(body, "every", 7, 1, 90, integer=True)), int(_number(body, "quantity", 1, 1, 100, integer=True)),
                         _number(body, "commission", 0.65, 0.0, 20.0))
        except ValueError as exc:
            raise ApiError(str(exc)) from None
        return _wrap("options_historical", out, {"chains": "downloaded"} if out.get("state") == "ok" else {"chains": "unavailable"}, [], out.get("last"), [out.get("fills", "")] if out.get("state") == "ok" else [])

    def options_history(self, q: dict) -> dict:
        """``GET /api/options/history?underlying=SPY&model=heston``: the dates kept and, with at least three, how the parameters of a model move from one date to the next (each calibration starts from the previous
        one) and how stable they are. With fewer dates the answer says how many more are needed instead of a number."""
        from ..assets.option_history import History
        from ..derivatives import calibration as cal, model_validation as mv
        underlying = str(q.get("underlying") or "SPY")
        model = _choice(q, "model", "heston", ("heston", "merton", "kou", "vg", "nig", "bs"))
        try:
            hist = History(self.root)
            dates = hist.dates(underlying)
        except ValueError as exc:
            raise ApiError(str(exc)) from None
        out: dict = {"underlying": underlying, "dates": dates, "snapshots": hist.meta(underlying), "model": model, "needed": max(3 - len(dates), 0)}
        warnings: list[str] = []
        if len(dates) < 3:
            out["stability"] = None
            warnings.append(f"{len(dates)} dated chains are kept; stability needs at least 3 (take a snapshot on each trading day: one chain per date is all a free source gives).")
        else:
            quotes, used = [], []
            for d in dates[-30:]:
                try:
                    quotes.append(cal.quotes_from_chain(hist.load(underlying, d)))
                    used.append(d)
                except ValueError as exc:
                    warnings.append(f"{d}: skipped ({exc})")
            if len(quotes) < 3:
                out["stability"] = None
                warnings.append("fewer than three of the kept chains have enough quotes to calibrate")
            else:
                try:
                    out["stability"] = mv.stability(model, quotes, used)
                except ValueError as exc:
                    raise ApiError(str(exc)) from None
        kinds = {m["source"] for m in out["snapshots"]}
        return _wrap("options_history", out, {"chain": "simulated" if kinds == {"simulated"} else "downloaded" if kinds <= {"downloaded"} else "mixed"}, warnings, pd.Timestamp(dates[-1]) if dates else None,
                     ["Each calibration starts from the previous one, which makes the path smoother than independent fits would be; the instability measure is the mean day-to-day change of each parameter over its range."])

    # ------------------------------------------------------------------------------------------------------------ saved experiments
    def _experiments(self):
        from ..assets import experiments as ex
        return ex, ex.Store(self.root)

    def experiments_list(self, q: dict) -> dict:
        """``GET /api/experiments?kind=``: the saved research runs, newest first."""
        ex, st = self._experiments()
        kind = q.get("kind") or None
        if kind and kind not in ex.KINDS:
            raise ApiError(f"kind must be one of {list(ex.KINDS)}")
        return {"experiments": st.list(kind), "kinds": list(ex.KINDS)}

    def experiments_get(self, q: dict) -> dict:
        ex, st = self._experiments()
        try:
            return st.get(str(q.get("id") or ""))
        except (KeyError, ValueError):
            raise ApiError("no such experiment") from None

    def experiments_save(self, body: dict) -> dict:
        """``POST /api/experiments/save``: keep a run with the request that produced it, its metrics and provenance, the software version and two hashes (see ``assets/experiments.py``)."""
        ex, st = self._experiments()
        try:
            return st.save(str(body.get("kind") or "other"), str(body.get("name") or ""), body.get("request") or {}, body.get("metrics") or {}, body.get("provenance") if isinstance(body.get("provenance"), dict) else None,
                           _list(body.get("assumptions") or []), _list(body.get("universe") or []), str(body.get("notes") or ""))
        except ValueError as exc:
            raise ApiError(str(exc)) from None

    def experiments_delete(self, body: dict) -> dict:
        ex, st = self._experiments()
        try:
            st.delete(str(body.get("id") or ""))
        except (KeyError, ValueError):
            raise ApiError("no such experiment") from None
        return {"deleted": body.get("id")}

    def experiments_check(self, body: dict) -> dict:
        """``POST /api/experiments/check``: ``{id, metrics}`` where ``metrics`` come from sending the stored request again; says whether the repeat reproduced the saved numbers."""
        ex, st = self._experiments()
        try:
            return st.check_repeat(str(body.get("id") or ""), body.get("metrics") or {})
        except (KeyError, ValueError):
            raise ApiError("no such experiment") from None

    def experiments_compare(self, body: dict) -> dict:
        """``POST /api/experiments/compare``: 2 to 8 saved runs side by side: metrics, the settings that differ, and what the hashes say about reproducibility."""
        ex, st = self._experiments()
        try:
            return st.compare([str(i) for i in _list(body.get("ids") or [])])
        except (KeyError, ValueError) as exc:
            raise ApiError(str(exc) if isinstance(exc, ValueError) else "no such experiment") from None

    def futures_roll(self, q: dict) -> dict:
        """``GET /api/futures/roll?root=CL&roll_days=5``: the contract held and the next, the suggested roll date, the open-interest crossover, the cost of the roll and its histories."""
        data = self._curve(str(q.get("root") or "CL"))
        try:
            roll = fa.roll_analysis(data, _number(q, "roll_days", 5, 0, 60, integer=True))
        except ValueError as exc:
            raise ApiError(str(exc)) from None
        return _wrap("futures_roll", {"root": data.root, "roll": roll, "spec": {"multiplier": data.spec.multiplier, "tick_value": data.spec.tick_value}}, self._curve_parts(data),
                                     roll["risk_flags"], data.as_of, data.notes + ["Roll cost is for a LONG position (negative: rolling earns the backwardation); a short position's is the opposite."])

    def futures_series(self, q: dict) -> dict:
        """``GET /api/futures/series?name=US_NQ``: the price history of a futures product downloaded earlier from iTick (``US_NQ``, kept, read only) or from Yahoo Finance (``YF_ES``): returns, volatility, drawdown, the calendar months and the
        largest one-day moves."""
        name = str(q.get("name") or "")
        known = self._downloaded("futures", "series")
        if not DOWNLOAD_NAME.match(name) or name not in known:
            raise ApiError(f"no downloaded price history '{name}'; use the Free data card on this tab, choose Yahoo (search all futures) and download a product (downloaded so far: {sorted(known) or 'none'})")
        rec = known[name]
        try:
            view = fa.series_view(pd.read_csv(self.downloads.folder("futures") / rec["file"]))
        except (ValueError, OSError, KeyError) as exc:
            raise ApiError(f"{name}: {exc}") from None
        notes = [f"Downloaded {str(rec.get('fetched_at', '?')).replace('T', ' ').rstrip('Z')} UTC from {rec.get('source', 'a free source')}; the newest bar is for {rec.get('as_of')}."] + [str(n) for n in rec.get("notes", [])]
        warnings = view.pop("warnings")
        out = {"name": name, "code": rec.get("code", name), "region": rec.get("region", ""), "label": rec.get("root_name", name), "exchange": rec.get("exchange", ""), "group": rec.get("group", ""), **view}
        return _wrap("futures_series", out, {"prices": "downloaded"}, warnings, rec.get("as_of"), notes)

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

    def _futures_inputs(self, body: dict) -> tuple[bool, list, dict | None, pd.Timestamp]:
        """What the strategy backtest and the signals board share: whether they run on the real curves (``data`` "real") or the simulated roots, the roots chosen, the curves of the real ones and the last day."""
        real = _choice(body, "data", "simulated", ("simulated", "real")) == "real"
        roots = _list(body.get("roots") or ([] if real else list(fa.ROOTS)))
        as_of = self._bundle().index[-1]
        curves = None
        if real:
            if not roots:
                raise ApiError("choose at least one root with a downloaded or uploaded curve")
            curves = {}
            for r in roots:
                curve = self._curve(r)
                if curve.source == "simulated":
                    raise ApiError(f"{curve.root} has no real curve to backtest: download it with the Free data card on this tab (Cboe's VX and Deribit's history keep expired contracts), or put {curve.root}.csv in data/user/futures/")
                curves[curve.root] = curve
            as_of = max(c.as_of for c in curves.values())
        return real, roots, curves, as_of

    @staticmethod
    def _settings(body: dict) -> dict:
        """The strategy's own settings from a request (``params``): a dictionary of constructor arguments, checked when the strategy is built."""
        params = body.get("params") or {}
        if not isinstance(params, dict) or len(params) > 20 or any(not isinstance(k, str) or len(k) > 40 for k in params):
            raise ApiError("params must be an object of the strategy's settings")
        return dict(params)

    def futures_strategy_backtest(self, body: dict) -> dict:
        """``POST /api/futures/strategy/backtest``: a strategy through the platform pipeline, as a job: on the simulated roots (``data`` "simulated", the default) or on the curves downloaded or uploaded
        for the roots chosen (``data`` "real"). ``params`` are the strategy's own settings (what the page lists under the strategy); ``lookback`` is the older single setting and ``params`` win over it."""
        name = str(body.get("strategy") or "carry_ts")
        if name not in fa.FUTURES_STRATEGIES:
            raise ApiError(f"strategy must be one of {sorted(fa.FUTURES_STRATEGIES)}")
        real, roots, curves, as_of = self._futures_inputs(body)
        lookback = None if body.get("lookback") in (None, "") else _number(body, "lookback", 63, 5, 756, integer=True)
        target = None if body.get("target_vol") in (None, "") else _number(body, "target_vol", 0.1, 0.01, 1.0)
        freq = _choice(body, "frequency", "weekly", ("daily", "weekly", "monthly"))
        cost = _number(body, "cost_bps", 2.0, 0.0, 200.0)
        settings = self._settings(body)

        def work(progress):
            progress("building the front-month series from the contract tables (several seconds per root the first time)" if real else "building the simulated futures (the first run takes up to half a minute)")
            out = fa.run_futures_strategy(self.app.config, name, roots, freq, lookback, target, cost, as_of, curves, settings)
            parts = {"prices": common.overall_source({r: c.source for r, c in curves.items()}) if real else "simulated"}
            return _wrap("futures_backtest", out, parts, [], as_of, out.pop("notes"))

        return self.app.submit_task(f"futures: {name}", work)

    def futures_signals(self, body: dict) -> dict:
        """``POST /api/futures/signals``: what each rule says about each root on the last day, as a job (it builds the same bundle as a backtest): on the simulated roots or the real curves chosen."""
        real, roots, curves, as_of = self._futures_inputs(body)

        def work(progress):
            progress("building the front-month series from the contract tables (several seconds per root the first time)" if real else "building the simulated futures (the first run takes up to half a minute)")
            out = fa.run_signals(roots, as_of, curves)
            out["references"] = references("futures", "signals")
            parts = {"prices": common.overall_source({r: c.source for r, c in curves.items()}) if real else "simulated"}
            return _wrap("futures_signals", out, parts, [], as_of, ["Each rule's score is divided by the average size of its own absolute score over the last two years, so a strength of 1 is a typical position and the sign is the side.",
                                                                    "The rules are the platform's strategies, run on the excess return of holding the nearest contract of each root; a rule that cannot be computed for this selection says why."])

        return self.app.submit_task("futures: signals", work)

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
            self.require_model(f"the history of {symbol}")
            return self._attach_venue_funding(ca.simulated_series(symbol, ca.platform_series(self._crypto_frames(), "BTC")))
        raise ApiError(f"unknown symbol '{symbol}'; the platform has {list(ca.REAL_SYMBOLS)}, can simulate {sorted(ca.SIMULATED_ALTS)}, and downloads others with the Free data card "
                       f"(or put {symbol}.csv in data/user/crypto/)")

    def _other_coin(self, symbol: str) -> ca.CryptoSeries | None:
        """The coin a coin is compared with on the overview (ETH for BTC, BTC for every other): ``None`` when the platform's Deribit data, which holds both, is not there to compare with."""
        try:
            return self._series("ETH" if symbol == "BTC" else "BTC")
        except ApiError:
            return None

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
            elif self.observed():                                                         # no stored book: the liquidity card says so instead of showing a parametric one
                book, books = None, {}
            else:
                book = ca.order_book_summary(s.symbol, "BINANCE", float(s.spot.iloc[-1]), seed=int(s.as_of.toordinal()))
                books = {v: ca.order_book_summary(s.symbol, v, float(s.spot.iloc[-1]), seed=int(s.as_of.toordinal())) for v in ca.VENUE_ORDER}
            ov = ca.overview(s, self._stable(), book)
            data = {"symbol": s.symbol, "overview": ov, "spot_perp": ca.spot_perp_view(s, _number(q, "days", 365, 60, 2000, integer=True)) if s.perp is not None else None,
                    "basis": ca.basis_dashboard(s, _number(q, "fee_bps", 5.0, 0.0, 100.0)) if s.perp is not None and s.funding is not None else None, "books": books,
                    "technical": ca.technical_view(s, self._other_coin(s.symbol)),
                    "references": {name: references("crypto", name) for name in ("trend", "volatility", "relative", "carry")},
                    "snapshot": None if snap is None else {"fetched_at": snap.get("fetched_at"), "venues": [v["venue"] for v in snap["venues"]], "failed": snap.get("failed", {})}}
        except ValueError as exc:
            raise ApiError(str(exc)) from None
        parts = {"prices": s.source, "funding": s.source, "stablecoin_supply": "platform", "order_book": "downloaded" if snap is not None else "unavailable" if self.observed() else "simulated"}
        notes = list(s.notes)
        if snap is not None:
            notes.append(f"The order books are the real ones of {snap.get('fetched_at', 'the download')}; the liquidity card uses the deepest spot venue ({book['venue']}).")
        return _wrap("crypto_overview", data, parts, ov["risk_flags"], s.as_of, notes)

    def crypto_funding(self, q: dict) -> dict:
        """``GET /api/crypto/funding?symbols=BTC,ETH,SOL``: funding by coin and venue with z-scores and charts (Deribit's BTC and ETH are data; every other venue and coin is simulated)."""
        syms = _list(q.get("symbols") or "BTC,ETH,SOL", 6)
        series = {sym.upper(): self._series(sym) for sym in syms}
        try:
            out = ca.funding_dashboard(series, days=_number(q, "days", 120, 30, 720, integer=True), observed_only=self.observed())
        except ValueError as exc:
            raise ApiError(str(exc)) from None
        parts = {f"{r['symbol']}@{r['venue']}": r["source"] for r in out["rows"]}
        return _wrap("crypto_funding", out, parts, out["risk_flags"], max(s.as_of for s in series.values()))

    def crypto_cross_exchange(self, q: dict) -> dict:
        """``GET /api/crypto/cross-exchange?symbol=BTC``: every pair of venues of one kind, buy at one's ask and sell at the other's bid, net of fees and slippage, with the warnings. Real quotes when a venue
        snapshot has been downloaded, else modelled ones."""
        s = self._series(str(q.get("symbol") or "BTC"))
        snap = self._snapshot(s.symbol)
        if snap is None and self.observed():
            return _wrap("crypto_cross_exchange", {"state": "unavailable", "symbol": s.symbol, "quotes": [], "pairs": [], "reason": "no order-book snapshot is stored, and the other venues' quotes are not observed by the platform",
                                                   "needs": ["download the venue snapshot (Free data, crypto venues) just before looking"]}, {"quotes": "unavailable"}, [], s.as_of, [])
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
        assets = [a.upper() for a in _list(body.get("assets") or "BTC,ETH", 6)]
        if not assets:
            raise ApiError("choose at least one coin")
        series = {}
        for a in assets:
            if a not in ca.REAL_SYMBOLS:                                                         # any coin the tab can show: a download, your file, or the simulated ones
                series[a] = self._series(a)
        lookback = None if body.get("lookback") in (None, "") else _number(body, "lookback", 90, 5, 756, integer=True)
        target = None if body.get("target_vol") in (None, "") else _number(body, "target_vol", 0.2, 0.01, 2.0)
        freq = _choice(body, "frequency", "weekly", ("daily", "weekly", "monthly"))
        fees, slip = _number(body, "fees_bps", 5.0, 0.0, 200.0), _number(body, "slippage_bps", 2.0, 0.0, 200.0)
        regime = None if body.get("regime") in (None, "", "none") else _choice(body, "regime", "vol_state", ("vol_state", "hmm", "gmm"))
        settings = self._settings(body)
        self._crypto_frames()

        def work(progress):
            progress("running the strategy through the pipeline")
            try:
                out = ca.run_crypto_strategy(self.app.config, name, assets, freq, lookback, target, fees, slip, regime, settings, series=series)
            except ValueError as exc:
                raise ApiError(str(exc)) from None
            return _wrap("crypto_backtest", out, {"prices": "platform"}, [], out["end"], out.pop("notes"))

        return self.app.submit_task(f"crypto: {name}", work)

    def crypto_signals(self, body: dict) -> dict:
        """``POST /api/crypto/signals``: what each rule says about BTC and ETH on the last day, from the platform's Deribit data, as a job."""
        self._crypto_frames()

        def work(progress):
            progress("reading every rule on the Deribit data")
            out = ca.run_crypto_signals(self.app.config)
            out["references"] = references("crypto", "signals")
            return _wrap("crypto_signals", out, {"prices": "platform"}, [], out["last_day"], ["Each rule's score is divided by the average size of its own absolute score over the last two years, so a strength of 1 is a typical position and the sign is the side.",
                                                                                           "The funding and basis rules are read on the coin's long-spot, short-perpetual index and shown under the coin; the rest on its spot price."])

        return self.app.submit_task("crypto: signals", work)
