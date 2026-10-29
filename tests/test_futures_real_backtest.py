"""A futures strategy backtested on real contract tables (downloaded or your own file), and the engine behaviour that real data exposed: an account that loses more than it has.

The contract tables here are built by the tests (monthly contracts that expire on the 15th, priced as a random walk with a gentle slope), so they are not real prices; what they check is the machinery that
real prices go through: which dates count as the front of the market, how weekends and closures are handled, what is refused and why, and how the loss of the whole account is booked.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.assets import bundles, futures_analytics as fa
from src.assets.futures import front_month_dates
from src.assets.free_data.store import Download, DownloadStore
from src.assets.strategy_runs import compact_result
from src.utils.config import load_config
from src.webapp.api import ApiError
from tests.test_webapp_assets_ml import _app, finished, wait
from tests.test_webapp_labs import _rich


def monthly_table(start: str, end: str, seed: int = 0, calendar: str = "B", life_days: int = 120, drift: float = 0.0) -> pd.DataFrame:
    """Monthly contracts that expire on the 15th of each month, each priced from ``life_days`` before its expiry to the expiry: ``date, contract, expiry, price``."""
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, end, freq=calendar)
    base = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(drift, 0.01, len(days)))), index=days)
    rows = []
    for expiry in pd.date_range(pd.Timestamp(start).replace(day=1), pd.Timestamp(end) + pd.DateOffset(months=6), freq="MS") + pd.Timedelta(days=14):
        alive = days[(days >= expiry - pd.Timedelta(days=life_days)) & (days <= expiry)]
        for d in alive:
            rows.append({"date": d, "contract": f"C{expiry:%y%m}", "expiry": expiry, "price": float(base[d]) * (1.0 + 0.002 * (expiry - d).days / 30.0)})
    return pd.DataFrame(rows)


def curve_of(root: str, table: pd.DataFrame, asset_class: str = "commodity", source: str = "downloaded") -> fa.CurveData:
    spec = fa.RootSpec(root, root, "TEST", asset_class, 1.0, 0.01, tuple(range(1, 13)), "x", fa.SchwartzSmithParams(), 0.0, 0.0)
    t = table.assign(ttm=(table["expiry"] - table["date"]).dt.days / 365.0, open_interest=np.nan, volume=np.nan)
    return fa.CurveData(root, spec, t[["date", "contract", "expiry", "price", "ttm", "open_interest", "volume"]].reset_index(drop=True), None, source, [])


# ------------------------------------------------------------------------------------------------------------------ which dates are the front of the market
def test_a_table_that_keeps_expired_contracts_has_a_front_month_and_one_that_does_not_has_none_before_the_fetch():
    full = monthly_table("2021-09-01", "2023-06-01")
    assert front_month_dates(full).all(), "with the expired contracts in the table the nearest contract is never more than a month away"
    fetched = pd.Timestamp("2023-06-01")
    yahoo_like = full[full["expiry"] >= fetched]                                           # what a free Yahoo download holds on the day it is fetched: only what has not expired
    ok = front_month_dates(yahoo_like)
    assert not ok[:"2023-03-31"].any(), "months before the fetch have a nearest contract that is the one expiring in June: it is not their front"
    assert ok["2023-05-10":].all()
    assert not front_month_dates(full[full["contract"] == "C2201"]).any(), "one contract says nothing about the spacing of expiries, so nothing is called the front"


def test_a_history_that_starts_with_only_far_contracts_is_not_called_the_front_until_the_front_appears():
    full = monthly_table("2021-09-01", "2023-06-01")
    late = full[full["expiry"] >= pd.Timestamp("2022-03-01")]                              # the nearer contracts of the first months are missing from the file, as the first Deribit months are
    ok = front_month_dates(late)
    assert not ok[:"2021-12-31"].any() and ok["2022-02-10":"2022-03-01"].all()


# ------------------------------------------------------------------------------------------------------------------ closures and weekends
def test_a_root_whose_exchange_was_closed_is_held_flat_across_the_closure_and_not_beyond_its_last_day():
    a, b = monthly_table("2021-06-01", "2022-12-30", seed=1), monthly_table("2021-06-01", "2022-12-30", seed=2)
    closed = pd.Timestamp("2022-05-30")                                                       # a holiday on which B did not trade
    b = b[b["date"] != closed]
    b = b[b["date"] <= pd.Timestamp("2022-12-20")]
    held = bundles.futures_bundle({"A": a, "B": b}, hold_days=5)
    loose = bundles.futures_bundle({"A": a, "B": b})
    assert np.isnan(loose.prices.at[closed, "B"]) and held.prices.at[closed, "B"] == held.prices.at[closed - pd.Timedelta(days=3), "B"] and held.returns.at[closed, "B"] == 0.0
    assert held.prices["B"].last_valid_index() == pd.Timestamp("2022-12-20"), "a root is not carried past its own last day"
    assert held.macro["CARRY_B"].last_valid_index() <= pd.Timestamp("2022-12-20")
    assert held.prices["A"].equals(loose.prices["A"]), "a root with a price every day is untouched"


def test_the_weekend_of_a_market_that_never_closes_is_folded_into_monday_and_a_year_is_252_returns():
    seven = monthly_table("2020-01-01", "2023-06-30", seed=3, calendar="D")
    bundle, report = fa.real_futures_bundle({"X": curve_of("X", seven)})
    weekdays = seven[seven["date"].dt.day_of_week < 5]
    assert (bundle.prices.index.day_of_week < 5).all() and report["X"]["file_days"] == weekdays["date"].nunique(), "the weekend rows are left out before anything is measured"
    expected = bundles.excess_return_series(weekdays)
    monday = next(d for d in expected.index if d.day_of_week == 0 and d > expected.index[300])
    assert bundle.returns.at[monday, "X"] == pytest.approx(expected[monday]), "Monday's return is the move from Friday's close"


def test_a_closed_exchange_is_held_for_five_days_and_not_for_a_sixth():
    a, b = monthly_table("2019-01-02", "2023-12-29", seed=11), monthly_table("2019-01-02", "2023-12-29", seed=12)
    week = pd.bdate_range("2022-05-23", "2022-05-27")                                          # five weekdays on which B did not trade
    six = pd.bdate_range("2022-08-15", "2022-08-22")                                           # and six
    b = b[~b["date"].isin(week.union(six))]
    bundle, report = fa.real_futures_bundle({"A": curve_of("A", a), "B": curve_of("B", b)})
    before_week, before_six = bundle.prices.at[pd.Timestamp("2022-05-20"), "B"], bundle.prices.at[pd.Timestamp("2022-08-12"), "B"]
    assert all(bundle.prices.at[d, "B"] == before_week for d in week) and all(bundle.returns.at[d, "B"] == 0.0 for d in week), "five closed days are held flat"
    assert all(bundle.prices.at[d, "B"] == before_six for d in six[:5]) and np.isnan(bundle.prices.at[six[5], "B"]), "and the sixth is a day without a price"
    assert report["A"]["front_days"] > report["B"]["front_days"] > 1000


def test_a_root_needs_a_year_of_front_months_to_the_day():
    table = monthly_table("2021-01-04", "2022-12-30", seed=13)
    days = front_month_dates(table[table["date"].dt.day_of_week < 5])
    enough, short = days[days].index[251], days[days].index[250]                              # the 252nd and the 251st day on which the nearest contract is the front
    ok, report = fa.real_futures_bundle({"X": curve_of("X", table[table["date"] <= enough])})
    assert report["X"]["front_days"] == 252 and report["X"]["first"] == str(days[days].index[0].date()) and report["X"]["last"] == str(enough.date()), "252 days are enough"
    assert report["X"]["file_days"] == report["X"]["front_days"] and report["X"]["contracts"] == int(table[table["date"] <= enough]["contract"].nunique()) and report["X"]["source"] == "downloaded"
    with pytest.raises(ValueError, match=r"X has 251 days on which its nearest contract is the front of the market \(251 days in the file\); a strategy needs at least 252"):
        fa.real_futures_bundle({"X": curve_of("X", table[table["date"] <= short])})
    assert fa.MIN_FRONT_DAYS == 252 and fa.HOLD_DAYS == 5
    with pytest.raises(ValueError, match="at least 100"):
        fa.real_futures_bundle({"X": curve_of("X", table[table["date"] <= days[days].index[98]])}, min_days=100)
    assert fa.real_futures_bundle({"X": curve_of("X", table[table["date"] <= days[days].index[99]])}, min_days=100)[1]["X"]["front_days"] == 100


# ------------------------------------------------------------------------------------------------------------------ real tables through the pipeline
def test_a_real_table_runs_through_the_pipeline_with_its_own_coverage_and_notes():
    curves = {"AAA": curve_of("AAA", monthly_table("2019-01-02", "2023-12-29", seed=4)), "BBB": curve_of("BBB", monthly_table("2019-01-02", "2023-12-29", seed=5))}
    out = fa.run_futures_strategy(load_config(), "tsmom", frequency="weekly", cost_bps=2.0, curves=curves)
    assert out["source"] == "real" and out["roots"] == ["AAA", "BBB"] and out["n_days"] > 252 and set(out["coverage"]) == {"AAA", "BBB"}
    cov = out["coverage"]["AAA"]
    assert cov["front_days"] >= 252 and cov["front_days"] <= cov["file_days"] and cov["contracts"] > 40 and cov["source"] == "downloaded"
    assert any("front of the market" in n for n in out["notes"]) and any("not a forecast" in n for n in out["notes"]) and not any("simulation" in n for n in out["notes"])
    assert out["warnings"] == [], "a strategy that traded and survived has nothing to warn about"
    again = fa.run_futures_strategy(load_config(), "tsmom", frequency="weekly", cost_bps=2.0, curves=curves)
    assert again["metrics"]["sharpe"] == out["metrics"]["sharpe"], "the cached bundle gives the same answer"


def test_a_real_table_with_less_than_a_year_of_front_month_is_refused_with_the_reason():
    full = monthly_table("2021-09-01", "2023-06-01")
    yahoo_like = full[full["expiry"] >= pd.Timestamp("2023-06-01")]
    with pytest.raises(ValueError, match=r"CL has \d+ days on which its nearest contract is the front.*Yahoo download holds only the contracts that have not expired.*VX"):
        fa.real_futures_bundle({"CL": curve_of("CL", yahoo_like)})
    with pytest.raises(ValueError, match="at least 252"):
        fa.run_futures_strategy(load_config(), "carry_ts", curves={"CL": curve_of("CL", monthly_table("2022-01-03", "2022-09-30"))})


def test_the_strategies_that_cannot_trade_the_chosen_roots_say_so_instead_of_returning_a_flat_line():
    one = {"AAA": curve_of("AAA", monthly_table("2019-01-02", "2022-12-30", seed=6), "volatility"), "BBB": curve_of("BBB", monthly_table("2019-01-02", "2022-12-30", seed=7), "crypto")}
    with pytest.raises(ValueError, match=r"within an asset class.*AAA is volatility.*BBB is crypto"):
        fa.run_futures_strategy(load_config(), "carry_xs", curves=one)
    with pytest.raises(ValueError, match="commodity roots"):
        fa.run_futures_strategy(load_config(), "basis_momentum", curves=one)
    with pytest.raises(ValueError, match="at least one root"):
        fa.run_futures_strategy(load_config(), "carry_ts", curves={})
    with pytest.raises(ValueError, match="within an asset class"):
        fa.run_futures_strategy(load_config(), "carry_xs", ["CL", "GC"])                       # simulated roots: a commodity and a metal are not ranked against each other


# ------------------------------------------------------------------------------------------------------------------ what a futures strategy is told to run, and what it is refused
class Calls:
    """Stand-ins for the pipeline run and the bundles, so that what ``run_futures_strategy`` decides can be read from what it asks for."""

    def __init__(self, monkeypatch):
        from src.assets import strategy_runs

        self.runs = []
        monkeypatch.setattr(strategy_runs, "run_strategy", lambda bundle, config, model, params, **kwargs: self.runs.append((bundle, model, params, kwargs)) or {"metrics": {"sharpe": 1.0}})
        monkeypatch.setattr(strategy_runs, "subset_bundle", lambda bundle, columns: (bundle, tuple(columns)))
        monkeypatch.setattr(fa, "_real_bundle_for", lambda curves: ("real bundle", {r: {"front_days": 300, "first": "2020-01-02", "last": "2021-03-31", "file_days": 310, "contracts": 14, "source": "downloaded"} for r in curves}))
        monkeypatch.setattr(fa, "futures_bundle_for", lambda roots, as_of=None: "simulated bundle")


def stand_in(classes: dict) -> dict:
    return {r: SimpleNamespace(spec=SimpleNamespace(asset_class=c)) for r, c in classes.items()}


def test_a_strategy_is_refused_the_roots_it_cannot_rank_and_run_with_the_parameters_its_description_gives(monkeypatch):
    calls = Calls(monkeypatch)
    config = load_config()
    with pytest.raises(ValueError, match=r"unknown futures strategy 'vote'; choose from \['basis_momentum', 'carry_ts', 'carry_xs', 'donchian', 'tsmom', 'xs_momentum'\]"):
        fa.run_futures_strategy(config, "vote", curves=stand_in({"A": "commodity"}))
    for strategy in ("carry_xs", "xs_momentum"):
        with pytest.raises(ValueError, match="a cross-sectional strategy needs at least two roots"):
            fa.run_futures_strategy(config, strategy, curves=stand_in({"A": "commodity"}))
    with pytest.raises(ValueError, match=r"choose at least two roots of the same class \(chosen: A is commodity, B is fx\)"):
        fa.run_futures_strategy(config, "carry_xs", curves=stand_in({"A": "commodity", "B": "fx"}))
    assert calls.runs == [], "nothing was run for any of them"
    fa.run_futures_strategy(config, "carry_xs", curves=stand_in({"A": "commodity", "B": "commodity", "C": "fx"}))                 # two of one class are enough
    with pytest.raises(ValueError, match=r"basis momentum ranks commodities: choose at least two commodity roots$"):
        fa.run_futures_strategy(config, "basis_momentum", curves=stand_in({"A": "commodity", "B": "fx"}))
    with pytest.raises(ValueError, match=r"two commodity roots \(CL, NG, HG\)$"):
        fa.run_futures_strategy(config, "basis_momentum", ["CL", "ES"])
    fa.run_futures_strategy(config, "basis_momentum", curves=stand_in({"A": "commodity", "B": "commodity"}))
    assert [call[1] for call in calls.runs] == ["carry_xs", "basis_momentum"]
    calls.runs.clear()
    fa.run_futures_strategy(config, "tsmom", curves=stand_in({"A": "commodity"}), lookback=60, frequency="weekly", target_vol=0.12, cost_bps=3)
    (bundle, model, params, kwargs), = calls.runs
    assert bundle == ("real bundle", ("A",)) and model == "tsmom" and params == {"lookback": 60} and kwargs == {"rebalance": "weekly", "target_vol": 0.12, "cost_bps": 3.0, "label": "Trend", "allocator": "score_stack"}
    calls.runs.clear()
    for strategy, expected in (("donchian", {"entry": 30}), ("carry_ts", {"smooth": 30}), ("carry_xs", {})):
        fa.run_futures_strategy(config, strategy, curves=stand_in({"A": "commodity", "B": "commodity"}), lookback=30)
    assert [call[2] for call in calls.runs] == [{"entry": 30}, {"smooth": 30}, {}], "the lookback is the parameter the strategy names, and a strategy with none ignores it"
    calls.runs.clear()
    fa.run_futures_strategy(config, "carry_ts", curves=stand_in({"A": "commodity"}))
    assert calls.runs[0][2] == {} and calls.runs[0][3] == {"rebalance": None, "target_vol": None, "cost_bps": 2.0, "label": "Carry (time-series)", "allocator": "sleeves"}, "no lookback, no parameters; two basis points by default"


def test_the_result_says_what_ran_and_on_what_and_a_simulation_is_not_called_a_backtest(monkeypatch):
    Calls(monkeypatch)
    config = load_config()
    real = fa.run_futures_strategy(config, "tsmom", curves=stand_in({"b": "commodity", "a": "equity"}))
    assert real["source"] == "real" and real["roots"] == ["A", "B"] and real["strategy"] == "tsmom" and real["label"] == "Trend" and real["cost_bps"] == 2.0 and real["text"].startswith("Long what has risen")
    assert real["coverage"]["A"]["front_days"] == 300 and set(real["coverage"]) == {"A", "B"}
    assert real["notes"][0] == "A: 300 days from 2020-01-02 to 2021-03-31 on which the nearest contract is the front of the market (of 310 in the file, 14 contracts)." and len(real["notes"]) == 5
    assert real["notes"][2].startswith("The result is the excess return of holding the nearest contract of each root and rolling it 5 days before it expires") and real["notes"][3].startswith("Costs are the one-way basis points")
    assert real["notes"][4].startswith("Weekends are folded into Monday's return and an exchange holiday is held flat for at most 5 days")
    simulated = fa.run_futures_strategy(config, "tsmom", ["cl", "es"], cost_bps=1.5)
    assert simulated["source"] == "simulated" and simulated["roots"] == ["CL", "ES"] and "coverage" not in simulated and simulated["cost_bps"] == 1.5
    assert len(simulated["notes"]) == 2 and simulated["notes"][0].startswith("The futures are a Schwartz-Smith model simulation")
    with pytest.raises(ValueError, match=r"unknown futures roots \['ZZ'\]; the simulated roots are \["):
        fa.run_futures_strategy(config, "tsmom", ["CL", "zz"])
    with pytest.raises(ValueError, match="choose at least one root"):
        fa.run_futures_strategy(config, "tsmom", curves={})


def test_the_bundle_of_a_real_table_is_built_once_for_the_same_table_and_the_three_latest_are_kept(monkeypatch):
    built = []
    monkeypatch.setattr(fa, "real_futures_bundle", lambda curves: built.append(sorted(curves)) or ("bundle", {}))
    monkeypatch.setattr(fa, "_BUNDLES", fa.OrderedDict())

    def curves_of(root: str, price: float = 1.0) -> dict:
        return {root: SimpleNamespace(long=pd.DataFrame({"date": pd.to_datetime(["2021-01-04", "2021-01-05"]), "price": [price, price]}))}

    for root in "ABCD":
        fa._real_bundle_for(curves_of(root))
    assert built == [["A"], ["B"], ["C"], ["D"]] and len(fa._BUNDLES) == 3, "the oldest of four is dropped"
    fa._real_bundle_for(curves_of("B"))
    fa._real_bundle_for(curves_of("C"))
    assert len(built) == 4, "B and C are still there"
    fa._real_bundle_for(curves_of("A"))
    assert built[-1] == ["A"] and len(built) == 5, "A is not"
    fa._real_bundle_for(curves_of("C", 2.0))
    assert len(built) == 6, "the same root with other prices is another table"


# ------------------------------------------------------------------------------------------------------------------ an account that loses more than it has
def test_the_compact_result_warns_of_a_wiped_out_account_and_of_a_strategy_that_never_traded():
    idx = pd.bdate_range("2024-01-02", periods=40)
    net = pd.Series(0.001, index=idx)
    wiped = net.copy()
    wiped.iloc[20], wiped.iloc[21:] = -1.0, 0.0

    def result(returns, weight):
        weights = pd.DataFrame({"A": weight}, index=idx)
        return SimpleNamespace(window=returns, benchmarks={}, weights=weights, metrics={"sharpe": 1.0}, validation={})

    out = compact_result(result(wiped, 0.5))
    assert len(out["warnings"]) == 1 and f"wiped out on {idx[20].date()}" in out["warnings"][0]
    flat = compact_result(result(net, 0.0))
    assert len(flat["warnings"]) == 1 and "held no position" in flat["warnings"][0]
    fine = compact_result(result(net, 0.5))
    assert fine["warnings"] == []


# ------------------------------------------------------------------------------------------------------------------ the route
@pytest.fixture()
def app_with_curves(tmp_path):
    store = DownloadStore(tmp_path)
    for root, seed, cls in (("AAA", 8, "commodity"), ("BBB", 9, "commodity")):
        frame = monthly_table("2019-01-02", "2023-12-29", seed=seed)
        store.save(Download("futures", root, "curve", frame=frame, meta={"source_id": "test", "source": "a test", "root": root, "root_name": f"{root} test", "exchange": "TEST", "asset_class": cls, "multiplier": 1.0,
                                                                             "tick_size": 0.01, "as_of": "2023-12-29", "first_date": "2019-01-02", "contracts": int(frame["contract"].nunique())}))
    return _app(tmp_path, _rich, root=tmp_path)


def test_the_backtest_route_runs_on_the_downloaded_curves_and_says_so(app_with_curves):
    app = app_with_curves
    cat = app.extra_get["/api/assets/catalog"]({})
    aaa = next(r for r in cat["futures_roots"] if r["id"] == "AAA")
    assert aaa["source"] == "downloaded" and aaa["first_date"] == "2019-01-02" and aaa["as_of"] == "2023-12-29" and aaa["contracts"] > 40
    out = finished(app, app.extra_post["/api/futures/strategy/backtest"]({"strategy": "tsmom", "data": "real", "roots": ["AAA", "BBB"], "frequency": "weekly"}))
    assert out["sources"]["prices"] == "downloaded" and out["roots"] == ["AAA", "BBB"] and out["coverage"]["AAA"]["front_days"] >= 252 and out["as_of"] == "2023-12-29"
    assert any("front of the market" in n for n in out["notes"]) and np.isfinite(out["metrics"]["sharpe"])
    sim = finished(app, app.extra_post["/api/futures/strategy/backtest"]({"strategy": "carry_ts", "roots": ["CL", "GC"]}))
    assert sim["sources"]["prices"] == "simulated" and "coverage" not in sim


def test_the_backtest_route_refuses_what_it_cannot_run_before_it_queues_a_job(app_with_curves):
    post = app_with_curves.extra_post["/api/futures/strategy/backtest"]
    with pytest.raises(ApiError, match="at least one root"):
        post({"strategy": "tsmom", "data": "real"})
    with pytest.raises(ApiError, match="CL has no real curve"):
        post({"strategy": "tsmom", "data": "real", "roots": ["CL"]})
    with pytest.raises(ApiError, match="data must be one of"):
        post({"strategy": "tsmom", "data": "imagined", "roots": ["AAA"]})
    with pytest.raises(ApiError, match="unknown root"):
        post({"strategy": "tsmom", "data": "real", "roots": ["ZZZ"]})


def test_a_real_backtest_that_the_data_cannot_support_is_a_failed_job_with_a_message(app_with_curves):
    app = app_with_curves
    short = monthly_table("2023-01-02", "2023-09-29", seed=10)
    DownloadStore(app.root).save(Download("futures", "SHORT", "curve", frame=short, meta={"source_id": "test", "source": "a test", "root": "SHORT", "asset_class": "commodity", "multiplier": 1.0, "tick_size": 0.01}))
    response = app.extra_post["/api/futures/strategy/backtest"]({"strategy": "carry_ts", "data": "real", "roots": ["SHORT"]})
    job = wait(app, response)
    assert job["status"] == "error" and "at least 252" in job["error"]
