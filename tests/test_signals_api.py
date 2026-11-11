"""The Signals tab's endpoints and ``quant signals``: following each of the four markets, one run that reads them all and appends to the ledger, what changed on a later data day, the record,
old and simulated data said aloud, every refusal as a message, the daily schedule (due once a day, never twice), alerts to an address that is never sent anything unless it is set, and the terminal."""

from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src import cli
from src.utils.config import load_config
from src.webapp import api
from src.webapp.api import ApiError, App
from src.webapp.universe import TickerStore, UniverseBuilder
from tests.ml_helpers import make_bundle
from tests.test_webapp import _fetch

ROOT = Path(__file__).resolve().parents[1]


def make_app(tmp_path, n=900, days=None, fetch=_fetch):
    """An app whose default universe can be cut short (``days[0]`` rows), so that a later run sees later prices."""
    config = load_config()
    full = make_bundle(n_assets=4, n=n, seed=3, alpha=0.0)
    days = days if days is not None else [n]

    def loader(c):
        from dataclasses import replace

        cut = full.index[: days[0]]
        fields = {}
        for name in ("prices", "returns", "volume", "open", "high", "low", "investable", "volume_dollar"):
            frame = getattr(full, name, None)
            if isinstance(frame, pd.DataFrame):
                fields[name] = frame.loc[frame.index.intersection(cut)]
        return replace(full, **fields) if days[0] < n else full

    app = App(config, builder=UniverseBuilder(config, TickerStore(tmp_path / "prices", fetch=fetch), default_loader=loader), root=tmp_path)
    return app


@pytest.fixture()
def app(tmp_path):
    return make_app(tmp_path)


def plain(value):
    json.dumps(value, allow_nan=False)
    return value


def _wait(app, job, timeout=240):
    t0 = time.time()
    while time.time() - t0 < timeout:
        view = app.job(job)
        if view["status"] in ("done", "error", "interrupted"):
            return view
        time.sleep(0.05)
    raise AssertionError("the job did not end")


def follow(app, **body):
    return app.extra_post["/api/signals/follow"](body)["follow"]


def test_the_routes_the_version_and_the_plan(app):
    for path in ("/api/signals/plan", "/api/signals/book", "/api/signals/history"):
        assert path in app.extra_get
    for path in ("/api/signals/follow", "/api/signals/unfollow", "/api/signals/run", "/api/signals/settings", "/api/signals/alert/test"):
        assert path in app.extra_post
    assert api.API_VERSION >= 17 and api.DOC_ROOT_FILES["signals"] == "signals.md" and (ROOT / "docs" / "signals.md").exists()
    plan = plain(app.extra_get["/api/signals/plan"]({}))
    assert [m["id"] for m in plan["markets"]] == ["equity", "futures", "crypto", "options"] and set(plan["boards"]) == {"equity", "futures", "crypto", "options"}
    assert all(len(rules) >= 3 and all(r["id"] and r["label"] for r in rules) for rules in plan["boards"].values()) and plan["horizons"] == [1, 5, 21]
    assert plan["settings"]["schedule"]["enabled"] is False and plan["settings"]["webhook_set"] is False and plan["settings"]["running"] is False, "no schedule thread until one is asked for"
    book = plain(app.extra_get["/api/signals/book"]({}))
    assert book == {"follows": [], "latest": None, "settings": plan["settings"], "alerts": []}
    assert not (app.root / "data" / "user" / "signals").exists(), "looking writes nothing"


def test_every_rule_of_the_equity_board_is_a_registered_strategy():
    from src import signalbook as sb
    from src.framework import MODELS, load_library

    load_library()
    names = {e.name for e in MODELS.entries()}
    assert not [m for _, m, _ in sb.EQUITY_BOARD if m not in names]


def test_one_run_reads_stocks_futures_and_options_and_writes_the_ledger(app):
    before = {k: set(v) for k, v in app.trials.items()}
    eq = follow(app, name="ETFs", market="equity", config={"tickers": ["A00", "A01", "A02"]}, rules=["tsmom", "short_term_reversal"])
    follow(app, name="Futures", market="futures", config={"roots": ["ES", "CL"]}, rules=["tsmom", "carry_ts"])
    opt = follow(app, name="Vol", market="options", config={"underlyings": ["A00"]})
    assert eq["rules"] == ["tsmom", "short_term_reversal"] and eq["config"]["tickers"] == ["A00", "A01", "A02"] and len(opt["rules"]) == 3
    view = _wait(app, app.extra_post["/api/signals/run"]({})["job"])
    assert view["status"] == "done", view
    d = plain(view["result"])
    by = {e["follow"]["market"]: e for e in d["follows"]}
    assert [e["error"] for e in d["follows"]] == [None, None, None]
    e = by["equity"]
    assert len(e["snapshot"]["rows"]) == 6 and {r["asset"] for r in e["snapshot"]["rows"]} == {"A00", "A01", "A02"} and {r["side"] for r in e["snapshot"]["rows"]} <= {"long", "short", "flat"}
    assert e["snapshot"]["data"]["stale"] and "not today's" in e["snapshot"]["data"]["problems"][0] and not e["snapshot"]["data"]["simulated"], "the test prices end years ago, and the run says so"
    assert e["changes"] == [] and e["compared_with"] is None and e["snapshots"] == 1 and e["record"]["overall"] == {} and e["record"]["snapshots"] == 1
    assert [c["asset"] for c in e["consensus"]] == ["A00", "A01", "A02"]
    assert by["futures"]["snapshot"]["data"]["simulated"] and any("simulated data" in p for p in d["problems"]) and {r["asset"] for r in by["futures"]["snapshot"]["rows"]} == {"CL", "ES"}
    assert by["options"]["record"] is None and {r["price_key"] for r in by["options"]["snapshot"]["rows"]} == {None} and by["options"]["snapshot"]["data"]["simulated"]
    assert d["changed"] == 0 and d["scheduled"] is False
    book = app.extra_get["/api/signals/book"]({})
    assert book["latest"]["ran_at"] == d["ran_at"] and [f["snapshots"] for f in book["follows"]] == [1, 1, 1]
    assert len(app.signals.book.snapshots()) == 3 and {k: set(v) for k, v in app.trials.items()} == before, "reading signals is not a trial"
    hist = plain(app.extra_get["/api/signals/history"]({"id": eq["id"]}))
    assert len(hist["snapshots"]) == 1 and hist["snapshots"][0]["long"] + hist["snapshots"][0]["short"] + hist["snapshots"][0]["flat"] == 6
    assert any(j["kind"] == "signals" for j in app.jobs_list()["jobs"])


def test_crypto_is_read_on_the_platforms_data_and_carry_rules_are_scored_on_the_carry_index(app):
    f = follow(app, name="Coins", market="crypto", rules=["tsmom", "funding_carry"])
    d = app.signals.run_now([f["id"]])
    e = d["follows"][0]
    assert e["error"] is None and {r["asset"] for r in e["snapshot"]["rows"]} == {"BTC", "ETH"}
    assert {(r["rule"], r["price_key"]) for r in e["snapshot"]["rows"]} == {("tsmom", "BTC"), ("tsmom", "ETH"), ("funding_carry", "BTC_CARRY"), ("funding_carry", "ETH_CARRY")}
    assert not e["snapshot"]["data"]["simulated"]


def test_a_later_data_day_shows_what_changed_and_scores_the_earlier_signals(tmp_path):
    days = [860]
    app = make_app(tmp_path, n=900, days=days)
    f = follow(app, name="ETFs", market="equity", config={"tickers": ["A00", "A01", "A02", "A03"]})
    full = make_bundle(n_assets=4, n=900, seed=3, alpha=0.0).index
    app.signals.clock = lambda: full[859].to_pydatetime().replace(hour=22)                            # issued on the evening of the data's last day
    first = app.signals.run_now()["follows"][0]
    again = app.signals.run_now()["follows"][0]
    assert again["changes"] == [] and again["snapshots"] == 2 and again["record"]["snapshots"] == 1, "a second run on the same data day adds no day to the record"
    days[0] = 900
    app.builder._default = None                                                                   # a new day of data: the builder keeps the default universe it loaded
    app.builder._cache.clear()
    app.signals.clock = lambda: full[899].to_pydatetime().replace(hour=22)
    later = app.signals.run_now()["follows"][0]
    assert later["snapshot"]["as_of"] > first["snapshot"]["as_of"] and later["compared_with"]["as_of"] == first["snapshot"]["as_of"]
    flipped = {(c["rule"], c["asset"]) for c in later["changes"]}
    was = {(r["rule"], r["asset"]): r["side"] for r in first["snapshot"]["rows"]}
    now = {(r["rule"], r["asset"]): r["side"] for r in later["snapshot"]["rows"]}
    assert flipped == {k for k in now if k in was and was[k] != now[k]} and flipped, "forty trading days later some rule has changed side"
    rec = later["record"]
    assert rec["snapshots"] == 2 and rec["overall"]["21"]["n"] > 0 and 0.0 <= rec["overall"]["21"]["hit_rate"] <= 1.0 and rec["first"] == first["snapshot"]["as_of"]
    views = [r for r in first["snapshot"]["rows"] if r["side"] != "flat"]
    assert rec["overall"]["1"]["n"] == len(views), "every long or short signal of the first day was scored once at one day, and the new day's are too young"
    assert f["id"] == later["follow"]["id"] and rec["late"] == 0
    app.signals.clock = lambda: pd.Timestamp("2026-11-09 22:00").to_pydatetime()                        # the same data read years later: in the ledger, never in the record
    stale = app.signals.run_now()["follows"][0]
    assert stale["record"]["snapshots"] == 2 and stale["record"]["overall"] == rec["overall"]


def test_asking_for_new_prices_brings_the_platforms_own_tickers_up_to_date_and_a_failed_download_is_said(tmp_path):
    """The platform's ETF files end when they were built. A run that downloads reads those tickers from the data source instead, to the last close; one that cannot says it used the old files."""
    days = pd.bdate_range("2019-01-01", "2026-11-06")
    calls = []

    def fetch(tickers, start):
        calls.append(list(tickers))
        return {t: pd.DataFrame({"close": 100.0 + 0.01 * pd.Series(range(len(days)), index=days) * (1 + i), "high": 101.0, "low": 99.0, "volume": 1e6}, index=days) for i, t in enumerate(tickers)}

    app = make_app(tmp_path / "live", fetch=fetch)
    app.signals.today = lambda: pd.Timestamp("2026-11-09")
    follow(app, name="ETFs", market="equity", config={"tickers": ["A00", "A01"]}, rules=["tsmom"])
    stale = app.signals.run_now()["follows"][0]
    assert stale["snapshot"]["data"]["stale"] and stale["snapshot"]["source"] == "platform dataset" and calls == [], "without asking, nothing is downloaded"
    live = app.signals.run_now(refresh=True)["follows"][0]
    assert live["snapshot"]["as_of"] == "2026-11-06" and not live["snapshot"]["data"]["stale"] and live["snapshot"]["data"]["problems"] == [] and live["snapshot"]["source"] != "platform dataset"
    assert calls == [["A00", "A01"]]
    again = app.signals.run_now()["follows"][0]
    assert again["snapshot"]["as_of"] == "2026-11-06" and calls == [["A00", "A01"]], "the files downloaded today are read again without another download"
    offline = make_app(tmp_path / "offline")
    follow(offline, name="ETFs", market="equity", config={"tickers": ["A00", "A01"]}, rules=["tsmom"])
    e = offline.signals.run_now(refresh=True)["follows"][0]
    assert e["error"] is None and e["snapshot"]["source"] == "platform dataset" and any("could not be downloaded" in p and "platform's own files" in p for p in e["snapshot"]["data"]["problems"])


def test_a_follow_that_cannot_be_read_does_not_stop_the_others(app, monkeypatch):
    good = follow(app, name="ETFs", market="equity", config={"tickers": ["A00", "A01"]}, rules=["tsmom"])
    bad = follow(app, name="Vol", market="options", config={"underlyings": ["A00"]})
    monkeypatch.setattr(app.signals.assets, "_option_chain", lambda *a, **k: (_ for _ in ()).throw(ApiError("the chain is gone")))
    d = plain(app.signals.run_now())
    by = {e["follow"]["id"]: e for e in d["follows"]}
    assert by[good["id"]]["error"] is None and by[bad["id"]]["error"] == "the chain is gone" and "snapshot" not in by[bad["id"]]
    assert any("Vol could not be read: the chain is gone" in p for p in d["problems"]) and len(app.signals.book.snapshots()) == 1


def test_every_refusal_is_a_message_and_nothing_is_followed(app):
    for body, why in [({"market": "bonds"}, "market must be one of"), ({"market": "equity", "name": ""}, "1 to 60 characters"), ({"market": "equity", "name": "x", "rules": ["nope"]}, "rules must be a list of this market's rules"),
                      ({"market": "equity", "name": "x", "config": {"tickers": "A00"}}, "tickers must be a list"), ({"market": "equity", "name": "x", "config": {"tickers": [f"T{i}" for i in range(31)]}}, "between 1 and 30"),
                      ({"market": "futures", "name": "x", "config": {"roots": ["NOPE"]}}, "unknown futures roots"), ({"market": "futures", "name": "x", "config": {"data": "live"}}, "simulated or real"),
                      ({"market": "futures", "name": "x", "config": {"data": "real"}}, "between 1 and 30 futures roots"), ({"market": "options", "name": "x", "config": {"underlyings": ["NOPE"]}}, "unknown underlying"),
                      ({"market": "options", "name": "x", "config": {"underlyings": []}}, None), ({"market": "equity", "name": "x", "config": "all"}, "config must be an object")]:
        if why is None:
            continue
        with pytest.raises(ApiError, match=why):
            app.extra_post["/api/signals/follow"](body)
    assert app.signals.book.follows() == []
    with pytest.raises(ApiError, match="nothing is followed yet"):
        app.extra_post["/api/signals/run"]({})
    with pytest.raises(ApiError, match="no follow with id"):
        app.extra_post["/api/signals/unfollow"]({"id": "nope"})
    with pytest.raises(ApiError, match="no follow with id"):
        app.extra_get["/api/signals/history"]({})
    follow(app, name="Coins", market="crypto")
    with pytest.raises(ApiError, match="ids must be a list"):
        app.extra_post["/api/signals/run"]({"ids": "all"})
    with pytest.raises(ApiError, match="none of those follows exist"):
        app.signals.run_now(["nope"])
    with pytest.raises(ApiError, match="already exists"):
        follow(app, name="coins", market="crypto")


def test_the_daily_run_is_due_once_a_day_after_its_time_and_not_at_the_weekend(app):
    sig = app.signals
    monday = dt.datetime(2026, 11, 9, 19, 0)
    assert sig.tick(monday) is False, "off by default"
    view = app.extra_post["/api/signals/settings"]({"schedule": {"enabled": True, "time": "18:30"}})
    assert view["schedule"]["enabled"] and view["running"], "switching it on starts the schedule"
    assert sig.tick(monday) is False, "nothing is followed, so there is nothing to run"
    follow(app, name="ETFs", market="equity", config={"tickers": ["A00", "A01"]}, rules=["tsmom"])
    assert sig.tick(dt.datetime(2026, 11, 9, 18, 29)) is False and sig.tick(dt.datetime(2026, 11, 7, 19, 0)) is False, "before the time, and on a Saturday"
    assert sig.tick(monday) is True and sig.tick(monday) is False and sig.tick(dt.datetime(2026, 11, 9, 23, 59)) is False, "once a day"
    assert sig.book.settings()["last_scheduled"] == "2026-11-09"
    job = [j for j in app.jobs_list()["jobs"] if j["kind"] == "signals"][0]
    assert _wait(app, job["id"])["status"] == "done" and sig.book.latest()["scheduled"] is True
    assert sig.tick(dt.datetime(2026, 11, 10, 18, 30)) is True, "and again the next day"
    app.extra_post["/api/signals/settings"]({"schedule": {"enabled": True, "time": "18:30", "weekdays_only": False}})
    assert sig.tick(dt.datetime(2026, 11, 14, 19, 0)) is True, "a Saturday, when weekends are asked for"
    with pytest.raises(ApiError, match="HH:MM"):
        app.extra_post["/api/signals/settings"]({"schedule": {"time": "evening"}})
    app.extra_post["/api/signals/settings"]({"schedule": {"enabled": False}})
    sig._stop.set()


def test_alerts_go_to_the_address_only_when_one_is_set_and_a_failure_is_written_down(app):
    sig, sent = app.signals, []
    sig.post = lambda url, payload: sent.append((url, payload))
    follow(app, name="ETFs", market="equity", config={"tickers": ["A00", "A01"]}, rules=["tsmom"])
    sig.run_now(scheduled=True)
    assert sent == [] and sig.book.alerts()[0]["delivered"] is None and "not today's" in sig.book.alerts()[0]["text"], "no address: the alert is kept in the book and sent nowhere"
    sig.run_now()
    assert len(sig.book.alerts()) == 1, "a run you started yourself raises no alert"
    with pytest.raises(ApiError, match="set an alert address first"):
        app.extra_post["/api/signals/alert/test"]({})
    for bad in ("http://example.com/hook", "example.com"):
        with pytest.raises(ApiError, match="https://"):
            app.extra_post["/api/signals/settings"]({"webhook": bad})
    view = app.extra_post["/api/signals/settings"]({"webhook": "https://hooks.example.com/services/SECRET"})
    assert view["webhook_set"] and view["webhook_host"] == "hooks.example.com" and "SECRET" not in json.dumps(view) and "SECRET" not in json.dumps(app.extra_get["/api/signals/book"]({})), "the address is never sent back to the page"
    assert app.extra_post["/api/signals/alert/test"]({}) == {"ok": True} and sent[-1][0].endswith("/SECRET") and sent[-1][1]["text"] == sent[-1][1]["content"]
    sig.run_now(scheduled=True)
    assert len(sent) == 2 and sent[-1][1]["text"].startswith("Signals read ") and sig.book.alerts()[0]["delivered"] is True
    sig.post = lambda url, payload: (_ for _ in ()).throw(OSError("no route"))
    assert sig.run_now(scheduled=True)["follows"][0]["error"] is None, "an alert that cannot be sent does not fail the run"
    assert sig.book.alerts()[0]["delivered"] is False and "no route" in sig.book.alerts()[0]["error"]
    with pytest.raises(ApiError, match="could not be sent"):
        app.extra_post["/api/signals/alert/test"]({})
    assert app.extra_post["/api/signals/settings"]({"webhook": ""})["webhook_set"] is False


def test_the_alert_text_names_each_change():
    from src.webapp.signals_api import SignalsApi

    text = SignalsApi.alert_text({"ran_at": "2026-11-09T18:30:00Z", "changed": 1, "problems": ["ETFs: old data"],
                                  "follows": [{"follow": {"name": "ETFs"}, "changes": [{"asset": "SPY", "label": "Trend, one year", "from": "long", "to": "flat"}]}, {"follow": {"name": "Broken"}}]})
    assert text.splitlines() == ["Signals read 2026-11-09T18:30:00Z: 1 change.", "ETFs: SPY Trend, one year long -> flat", "ETFs: old data"]


def test_the_new_files_run_in_the_lanes_that_read_their_dependencies():
    if not (ROOT / ".github/workflows/ci.yml").exists():
        pytest.skip("the workflow files are not in this tree")
    ci = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())

    def text(job):
        return " ".join(str(step.get("run", "")) for step in ci["jobs"][job]["steps"])

    assert "tests/test_signals_api.py" in text("replit-python") and "tests/test_signalbook.py" in text("compat")


# ------------------------------------------------------------------------------------------------------------------------------ the terminal
def test_the_terminal_follows_runs_lists_and_shows_the_history(app, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr("src.webapp.api.App", lambda config: app)
    assert cli.main(["signals", "list"]) == 0 and "nothing is followed yet" in capsys.readouterr().out
    assert cli.main(["signals", "run"]) == 2 and "nothing is followed yet" in capsys.readouterr().err
    assert cli.main(["signals", "follow", "--market", "equity", "--name", "ETFs", "--assets", "A00", "A01", "--rules", "tsmom", "xs_momentum"]) == 0
    assert "following ETFs (equity, 2 rules)" in capsys.readouterr().out
    assert cli.main(["signals", "follow", "--market", "futures", "--name", "Fut", "--assets", "NOPE"]) == 2 and "unknown futures roots" in capsys.readouterr().err
    saved = tmp_path / "signals.json"
    assert cli.main(["signals", "run", "--json", str(saved)]) == 0
    out = capsys.readouterr().out
    assert "ETFs (equity)" in out and "A00" in out and "long" in out and "! The data ends on" in out and "record: no signal is old enough to score yet" in out
    assert [e["follow"]["name"] for e in json.loads(saved.read_text())["follows"]] == ["ETFs"]
    assert cli.main(["signals", "list"]) == 0
    listing = capsys.readouterr().out
    fid = listing.split()[0]
    assert "1 snapshots" in listing and cli.main(["signals", "history", "--id", fid]) == 0 and "issued" in capsys.readouterr().out
    assert cli.main(["signals", "unfollow", "--id", fid]) == 0 and cli.main(["signals", "unfollow", "--id", fid]) == 2
