"""The Signals tab's endpoints: follow rules on stocks and ETFs, futures, crypto and options, read them all in one run, see what changed, and keep a record of every signal issued.

``GET  /api/signals/plan``      the markets, the rules of each, the schedule and what the form offers
``GET  /api/signals/book``      what is followed, the result of the last run, the schedule and the last alerts
``POST /api/signals/follow``    follow a market's rules on some assets; ``/api/signals/unfollow`` stops (the snapshots already issued stay in the ledger)
``POST /api/signals/run``       read every follow (or the ones named) now, as a job of kind ``signals``
``GET  /api/signals/history``   the snapshots issued for one follow
``POST /api/signals/settings``  the daily schedule and the alert address; ``/api/signals/alert/test`` sends a test alert

The engine is ``src/signalbook``. This module gets each market's data the way its own tab does (the Backtest tab's universe builder, the Futures and Crypto tabs' bundles, the Options tab's chain), runs
the rule boards, and owns the daily schedule: a small thread that, while the app is running, starts one run a day at the time chosen and sends what changed to the alert address, if one is set.
Reading signals is not a backtest, so nothing here touches the trial counter.
"""

from __future__ import annotations

import datetime as dt
import json
import threading
import time
import urllib.request
from urllib.parse import urlparse

import pandas as pd

from .. import signalbook as sb
from ..signalbook.store import Book, BookError
from .api import ApiError
from .universe import UniverseError, normalise

MAX_TICKERS = 30
MAX_UNDERLYINGS = 8
CARRY_RULES = {"funding_carry", "basis_reversion"}


def _post(url: str, payload: dict) -> None:
    request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json", "User-Agent": "quant-lab-signals"}, method="POST")
    with urllib.request.urlopen(request, timeout=10) as response:                                   # noqa: S310 - the address is checked to be https (or this computer) when it is saved
        response.read(200)


class SignalsApi:
    def __init__(self, app, assets):
        self.app, self.assets = app, assets
        self.book = Book(app.root / "data" / "user" / "signals")
        self.today = lambda: pd.Timestamp.now().normalize()
        self.clock = lambda: dt.datetime.now(dt.timezone.utc)                                        # when a snapshot is issued
        self.post = _post
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        if self.book.settings()["schedule"]["enabled"]:
            self.ensure_scheduler()

    def get_routes(self) -> dict:
        return {"/api/signals/plan": self.plan, "/api/signals/book": self.view, "/api/signals/history": self.history}

    def post_routes(self) -> dict:
        return {"/api/signals/follow": self.follow, "/api/signals/unfollow": self.unfollow, "/api/signals/run": self.run, "/api/signals/settings": self.settings, "/api/signals/alert/test": self.test_alert}

    # ---------------------------------------------------------------------------------------------------------- what there is
    def boards(self) -> dict:
        from ..assets import crypto_analytics as ca
        from ..assets import futures_analytics as fa

        return {"equity": [{"id": m, "label": label} for label, m, _ in sb.EQUITY_BOARD], "futures": [{"id": m, "label": label} for label, m, _ in fa.SIGNAL_BOARD],
                "crypto": [{"id": m, "label": label} for label, m, _ in ca.CRYPTO_SIGNAL_BOARD], "options": [{"id": r, "label": label, "text": text} for r, label, text in sb.OPTION_RULES]}

    def _settings_view(self) -> dict:
        s = self.book.settings()
        return {"schedule": s["schedule"], "webhook_set": bool(s["webhook"]), "webhook_host": urlparse(s["webhook"]).hostname if s["webhook"] else None, "last_scheduled": s["last_scheduled"],
                "running": bool(self._thread and self._thread.is_alive())}

    def plan(self, q: dict | None = None) -> dict:
        from ..assets import crypto_analytics as ca
        from ..assets import futures_analytics as fa

        return {"markets": [{"id": k, "label": v} for k, v in sb.MARKETS.items()], "boards": self.boards(), "horizons": list(sb.HORIZONS), "strong": sb.engine.STRONG, "max_tickers": MAX_TICKERS,
                "max_underlyings": MAX_UNDERLYINGS, "futures_roots": sorted(fa.ROOTS), "crypto_assets": list(ca.REAL_SYMBOLS), "default_tickers": self.app.builder.default_tickers()[:MAX_TICKERS],
                "stale_after": sb.engine.STALE_AFTER, "settings": self._settings_view()}

    def view(self, q: dict | None = None) -> dict:
        counts = {}
        for s in self.book.snapshots():
            counts[s["follow"]] = counts.get(s["follow"], 0) + 1
        return {"follows": [{**f, "snapshots": counts.get(f["id"], 0)} for f in self.book.follows()], "latest": self.book.latest(), "settings": self._settings_view(), "alerts": self.book.alerts(10)}

    def history(self, q: dict) -> dict:
        try:
            follow = self.book.follow(str(q.get("id") or ""))
        except BookError as error:
            raise ApiError(str(error)) from None
        rows = []
        for s in self.book.snapshots(follow["id"])[-250:]:
            sides = [r["side"] for r in s["rows"]]
            rows.append({"issued": s["issued"], "as_of": s["as_of"], "long": sides.count("long"), "short": sides.count("short"), "flat": sides.count("flat"), "stale": s["data"]["stale"], "simulated": s["data"]["simulated"]})
        return {"follow": follow, "snapshots": rows[::-1]}

    # ---------------------------------------------------------------------------------------------------------- following
    def _config(self, market: str, raw) -> dict:
        from ..assets import futures_analytics as fa

        raw = raw or {}
        if not isinstance(raw, dict):
            raise ApiError("config must be an object")
        if market == "equity":
            tickers = raw.get("tickers") or self.app.builder.default_tickers()[:MAX_TICKERS]
            if not isinstance(tickers, list):
                raise ApiError("tickers must be a list")
            try:
                tickers = normalise(tickers)
            except UniverseError as error:
                raise ApiError(str(error)) from None
            if not 1 <= len(tickers) <= MAX_TICKERS:
                raise ApiError(f"follow between 1 and {MAX_TICKERS} tickers")
            out = {"tickers": tickers, "source": raw.get("source"), "start": raw.get("start"), "classes": raw.get("classes") if isinstance(raw.get("classes"), dict) else None}
            try:
                self.app.builder.preflight(tickers, out["start"], out["source"])
            except UniverseError as error:
                raise ApiError(str(error)) from None
            return out
        if market == "futures":
            data = raw.get("data") or "simulated"
            if data not in ("simulated", "real"):
                raise ApiError("data must be simulated or real")
            roots = raw.get("roots") or ([] if data == "real" else sorted(fa.ROOTS))
            if not isinstance(roots, list) or not roots or len(roots) > 30 or not all(isinstance(r, str) and r.strip() for r in roots):
                raise ApiError("choose between 1 and 30 futures roots")
            roots = [r.strip().upper() for r in roots]
            if data == "simulated" and [r for r in roots if r not in fa.ROOTS]:
                raise ApiError(f"unknown futures roots {[r for r in roots if r not in fa.ROOTS]}; the simulated roots are {sorted(fa.ROOTS)}")
            if data == "real":
                self.assets._futures_inputs({"data": "real", "roots": roots})                           # says which root has no real curve, before anything is followed
            return {"data": data, "roots": roots}
        if market == "crypto":
            return {}
        names = raw.get("underlyings") or ["SPY"]
        if not isinstance(names, list) or not 1 <= len(names) <= MAX_UNDERLYINGS or not all(isinstance(u, str) and 0 < len(u.strip()) <= 60 for u in names):
            raise ApiError(f"choose between 1 and {MAX_UNDERLYINGS} option underlyings")
        names = [u.strip() for u in names]
        for name in names:
            self.assets._option_chain(name, None, None, 0.15)                                        # an underlying with no chain is refused now, with the list of those that have one
        return {"underlyings": names}

    def follow(self, body: dict) -> dict:
        market = str(body.get("market") or "")
        if market not in sb.MARKETS:
            raise ApiError(f"market must be one of {', '.join(sb.MARKETS)}")
        known = [r["id"] for r in self.boards()[market]]
        rules = body.get("rules") or known
        if not isinstance(rules, list) or not rules or [r for r in rules if r not in known]:
            raise ApiError(f"rules must be a list of this market's rules: {', '.join(known)}")
        config = self._config(market, body.get("config"))
        try:
            return {"follow": self.book.add(body.get("name") or "", market, config, [r for r in known if r in rules])}
        except BookError as error:
            raise ApiError(str(error)) from None

    def unfollow(self, body: dict) -> dict:
        try:
            return self.book.remove(str(body.get("id") or ""))
        except BookError as error:
            raise ApiError(str(error)) from None

    # ---------------------------------------------------------------------------------------------------------- reading one follow
    def _equity_bundle(self, cfg: dict, refresh: bool, progress):
        """``(bundle, source label, notes)`` for a follow of tickers. The platform's own ETF files are a fixed dataset that ends when it was built, so a signal read on them is that day's signal.
        When new prices are asked for (or the data source's files for these tickers are already fresh on disk), the platform's tickers are read from the data source like any other ticker, so the
        run reaches the last close. If that download fails the run still reads the platform's files, and says so."""
        b = self.app.builder
        bundle, universe = b.resolve(cfg["tickers"], cfg.get("classes"), cfg.get("start"), refresh, cfg.get("source"), progress)
        label = str(universe.get("source") or "platform dataset")
        default = b.default_bundle()
        fixed = [t for t in cfg["tickers"] if t in default.prices.columns]
        if not fixed:
            return bundle, label, []
        src = b.source(cfg.get("source"))
        start = b.start_for(src, cfg.get("start"), True)
        if not refresh and src.store.peek(cfg["tickers"], start)[1]:
            return bundle, label, []                                                                 # not asked to download, and not all on disk: the platform's files, with their date
        frames, errors = src.store.get(cfg["tickers"], start, False, progress)
        if errors:
            first = next(iter(errors.items()))
            return bundle, label, [f"New prices could not be downloaded from {src.label} ({first[0]}: {first[1]}), so this was read on the platform's own files."]
        try:
            live = b._combine(default, cfg["tickers"], frames, {k.upper(): v for k, v in (cfg.get("classes") or {}).items()}, start)
        except UniverseError as error:
            return bundle, label, [f"The downloaded prices could not be used ({error}), so this was read on the platform's own files."]
        return live, src.label, []

    def _read(self, follow: dict, refresh: bool, progress):
        """``(rows, skipped, last_day, simulated, source, prices, notes)`` of one follow on the data of its own tab."""
        from ..assets import crypto_analytics as ca
        from ..assets import futures_analytics as fa
        from ..assets import options_analytics as oa
        from ..assets.strategy_runs import signals_board
        from ..framework.crypto_bundle import load_crypto_bundle

        market, cfg, wanted = follow["market"], follow["config"], set(follow["rules"])
        if market == "equity":
            bundle, source, notes = self._equity_bundle(cfg, refresh, progress)
            rules = tuple(r for r in sb.EQUITY_BOARD if r[1] in wanted)
            rows, skipped = sb.board_rows(signals_board(bundle, rules))
            return rows, skipped, bundle.index[-1], False, source, bundle.prices, notes
        if market == "futures":
            real, roots, curves, as_of = self.assets._futures_inputs({"data": cfg["data"], "roots": cfg["roots"]})
            bundle = fa._real_bundle_for(dict(sorted(curves.items())))[0] if real else fa.futures_bundle_for(sorted(roots), as_of)
            rows, skipped = sb.board_rows(signals_board(bundle, tuple(r for r in fa.SIGNAL_BOARD if r[1] in wanted)))
            return rows, skipped, bundle.index[-1], not real, "real curves" if real else "simulated", bundle.prices, []
        if market == "crypto":
            bundle = load_crypto_bundle(self.app.config)
            board = signals_board(bundle, tuple(r for r in ca.CRYPTO_SIGNAL_BOARD if r[1] in wanted))
            for row in board["signals"]:                                                              # a carry rule trades the coin's carry index; the board shows it under the coin
                if "strength" in row:
                    where = (lambda a: f"{a}_CARRY") if row["id"] in CARRY_RULES else (lambda a: a)   # noqa: E731
                    row["score"], row["strength"] = ({a: row[k].get(where(a)) for a in ca.REAL_SYMBOLS} for k in ("score", "strength"))
            board["assets"] = list(ca.REAL_SYMBOLS)
            rows, skipped = sb.board_rows(board, lambda rule, asset: f"{asset}_CARRY" if rule in CARRY_RULES else asset)
            return rows, skipped, bundle.index[-1], False, "platform (Deribit)", bundle.prices, []
        rows, skipped, days, simulated, sources = [], [], [], False, set()
        for name in cfg["underlyings"]:
            chain, ctx, source, _ = self.assets._option_chain(name, None, None, 0.15)
            macro = self.assets._bundle().macro if source != "user_file" else None
            r, s = sb.option_rows(oa.overview(chain, ctx, 0.15, macro, source == "simulated", source))
            rows += [x for x in r if x["rule"] in wanted]
            skipped += [{**x, "label": f"{x['label']} ({ctx.symbol})"} for x in s if x["rule"] in wanted]
            days.append(pd.Timestamp(ctx.as_of))
            simulated = simulated or source == "simulated"
            sources.add(source)
        return rows, skipped, min(days), simulated, ", ".join(sorted(sources)), None, []

    def run_now(self, ids: list[str] | None = None, refresh: bool = False, progress=None, scheduled: bool = False) -> dict:
        """Read the follows named (all of them by default), append a snapshot of each to the ledger, and return what they say, what changed and the record so far."""
        progress = progress or (lambda text: None)
        follows = [f for f in self.book.follows() if ids is None or f["id"] in ids]
        if not follows:
            raise ApiError("nothing is followed yet: follow a market first" if ids is None else "none of those follows exist")
        issued, today, out = self.clock().strftime("%Y-%m-%dT%H:%M:%SZ"), self.today(), []
        for i, follow in enumerate(follows):
            progress(f"{follow['name']} ({i + 1} of {len(follows)}): reading the rules")
            entry = {"follow": follow, "error": None}
            try:
                rows, skipped, last_day, simulated, source, prices, notes = self._read(follow, refresh, progress)
                fresh = sb.freshness(follow["market"], last_day, today, simulated)
                fresh["problems"] += notes
                snap = sb.snapshot(follow, rows, skipped, fresh, issued, source)
                earlier = self.book.snapshots(follow["id"])
                previous = next((s for s in reversed(earlier) if s["as_of"] != snap["as_of"]), None) or (earlier[-1] if earlier else None)
                self.book.record(snap)
                record = sb.track_record(earlier + [snap], prices) if prices is not None else None
                entry.update(snapshot=snap, consensus=sb.consensus(rows), changes=sb.changes(previous, snap), compared_with=previous and {"issued": previous["issued"], "as_of": previous["as_of"]},
                             record=record, snapshots=len(earlier) + 1)
            except (ApiError, UniverseError, BookError, ValueError, KeyError, OSError) as error:
                entry["error"] = str(error.args[0] if isinstance(error, KeyError) and error.args else error)
            out.append(entry)
        result = {"ran_at": issued, "scheduled": bool(scheduled), "follows": out, "changed": sum(len(e.get("changes") or []) for e in out),
                  "problems": [f"{e['follow']['name']}: {p}" for e in out if not e["error"] for p in e["snapshot"]["data"]["problems"]] + [f"{e['follow']['name']} could not be read: {e['error']}" for e in out if e["error"]]}
        self.book.save_latest(result)
        if scheduled:
            self._alert(result)
        return result

    def run(self, body: dict) -> dict:
        ids = body.get("ids")
        if ids is not None and (not isinstance(ids, list) or not all(isinstance(x, str) for x in ids)):
            raise ApiError("ids must be a list of follow ids")
        if not self.book.follows():
            raise ApiError("nothing is followed yet: follow a market first")
        refresh = bool(body.get("refresh"))
        return self.app.submit_task(f"signals: {len(ids) if ids else len(self.book.follows())} follow(s)", lambda progress: self.run_now(ids, refresh, progress), "signals")

    # ---------------------------------------------------------------------------------------------------------- the schedule and alerts
    def settings(self, body: dict) -> dict:
        try:
            self.book.save_settings(body.get("schedule"), body.get("webhook"))
        except BookError as error:
            raise ApiError(str(error)) from None
        if self.book.settings()["schedule"]["enabled"]:
            self.ensure_scheduler()
        return self._settings_view()

    @staticmethod
    def alert_text(result: dict) -> str:
        lines = [f"Signals read {result['ran_at']}: {result['changed']} change{'s' if result['changed'] != 1 else ''}."]
        for e in result["follows"]:
            for c in (e.get("changes") or [])[:12]:
                lines.append(f"{e['follow']['name']}: {c['asset']} {c['label']} {c['from']} -> {c['to']}")
        lines += result["problems"][:6]
        return "\n".join(lines[:40])

    def _send(self, text: str, extra: dict) -> dict:
        url = self.book.settings()["webhook"]
        entry = {"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "text": text, "delivered": None, "error": None, **extra}
        if url:
            try:
                self.post(url, {"text": text, "content": text})
                entry["delivered"] = True
            except Exception as error:                                                               # an alert that cannot be sent is written down, never raised into the run
                entry["delivered"], entry["error"] = False, f"{type(error).__name__}: {error}"[:200]
        self.book.alert(entry)
        return entry

    def _alert(self, result: dict) -> None:
        if result["changed"] or result["problems"]:
            self._send(self.alert_text(result), {"changed": result["changed"], "problems": len(result["problems"])})

    def test_alert(self, body: dict) -> dict:
        if not self.book.settings()["webhook"]:
            raise ApiError("set an alert address first")
        entry = self._send("A test alert from the Signals tab: alerts reach this address.", {"test": True})
        if not entry["delivered"]:
            raise ApiError(f"the alert could not be sent: {entry['error']}")
        return {"ok": True}

    def tick(self, now: dt.datetime | None = None) -> bool:
        """Start today's scheduled run if it is due and has not been started. Returns whether it did."""
        now = now or dt.datetime.now()
        s = self.book.settings()
        plan, today = s["schedule"], now.strftime("%Y-%m-%d")
        if not plan["enabled"] or s["last_scheduled"] == today or (plan["weekdays_only"] and now.weekday() >= 5) or now.strftime("%H:%M") < plan["time"] or not self.book.follows():
            return False
        self.book.save_settings(last_scheduled=today)
        self.app.submit_task("signals: the daily run", lambda progress: self.run_now(None, plan["refresh"], progress, scheduled=True), "signals")
        return True

    def ensure_scheduler(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        def loop():
            while not self._stop.wait(30):
                try:
                    if not self.book.settings()["schedule"]["enabled"]:
                        return                                                                       # switched off: the thread ends, and is started again when it is switched on
                    self.tick()
                except Exception:                                                                    # noqa: S110 - a schedule that fails once tries again in half a minute
                    pass

        self._thread = threading.Thread(target=loop, name="signals-schedule", daemon=True)
        self._thread.start()
