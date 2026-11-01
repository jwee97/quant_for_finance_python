"""P0-15 and P0-16: a value is usable from the instant it was public, and a bar from the instant it was final, on the real clock.

Sessions: the UTC instant at which each market's daily bar is final is checked against hand-computed literals across the daylight-saving changes of the United States, Europe and Australia; what a decision at one
close may see is checked for Asian, European, US and continuous crypto sessions, for a holiday and for a weekend; a label convention whose entry price precedes the decision is refused.
Point in time: a series joined on availability is compared with a brute-force scan of the records on hand-built and random vintages (a late publication, a later restatement, a release on the decision day, a date-only stamp), against
the join on the reference date that leaks, and the quality of every macro input is classified.
"""

from datetime import time

import numpy as np
import pandas as pd
import pytest

from src.ml import audit as A, config as mc, datasets as dsm, pit, sessions as S
from tests.ml_helpers import make_bundle, probe_inputs


def utc(text):
    return pd.Timestamp(text, tz="UTC")


# ------------------------------------------------------------------------------------------------------------------ when a bar is final
CLOSES = [
    ("us_equity", "2024-03-08", "2024-03-08 21:00"), ("us_equity", "2024-03-11", "2024-03-11 20:00"),                               # New York changes on 10 March 2024
    ("us_equity", "2024-11-01", "2024-11-01 20:00"), ("us_equity", "2024-11-04", "2024-11-04 21:00"),                               # and back on 3 November
    ("uk_equity", "2024-03-28", "2024-03-28 16:35"), ("uk_equity", "2024-04-02", "2024-04-02 15:35"),                               # London changes on 31 March
    ("uk_equity", "2024-10-25", "2024-10-25 15:35"), ("uk_equity", "2024-10-29", "2024-10-29 16:35"),                               # and on 27 October
    ("eu_equity", "2024-03-28", "2024-03-28 16:35"), ("eu_equity", "2024-04-02", "2024-04-02 15:35"),
    ("tokyo_equity", "2024-03-05", "2024-03-05 06:30"), ("tokyo_equity", "2024-07-05", "2024-07-05 06:30"), ("hk_equity", "2024-03-05", "2024-03-05 08:10"),
    ("au_equity", "2024-01-15", "2024-01-15 05:10"), ("au_equity", "2024-07-15", "2024-07-15 06:10"),                               # Sydney is UTC+11 in January, UTC+10 in July
    ("fx", "2024-03-05", "2024-03-05 22:00"), ("fx", "2024-07-05", "2024-07-05 21:00"),
    ("crypto", "2024-03-05", "2024-03-06 00:00"), ("crypto", "2024-12-31", "2025-01-01 00:00"),
]


@pytest.mark.parametrize("name,date,expected", CLOSES)
def test_the_instant_a_bar_is_final_matches_the_hand_computed_utc_across_daylight_saving(name, date, expected):
    assert S.get(name).close_utc([date])[0] == utc(expected)


def test_the_gap_between_two_closes_changes_with_daylight_saving_but_their_order_does_not():
    """From 11 to 29 March 2024 New York is on summer time and London and Paris are not: the gap is one hour shorter than in the weeks either side."""
    days = pd.DatetimeIndex(["2024-03-08", "2024-03-15", "2024-04-02"])
    gap = (S.get("us_equity").close_utc(days) - S.get("uk_equity").close_utc(days)) / pd.Timedelta(hours=1)
    assert list(gap) == [pytest.approx(4.4167, abs=1e-3), pytest.approx(3.4167, abs=1e-3), pytest.approx(4.4167, abs=1e-3)]
    assert (S.get("tokyo_equity").close_utc(days) < S.get("uk_equity").close_utc(days)).all() and (S.get("uk_equity").close_utc(days) < S.get("us_equity").close_utc(days)).all()


def test_unknown_and_unrepresentable_sessions_are_refused():
    with pytest.raises(S.SessionError, match="unknown session 'mars'"):
        S.get("mars")
    with pytest.raises(S.SessionError, match="does not exist exactly once"):
        S.Session("night", "America/New_York", time(2, 30)).close_utc(["2024-03-10"])               # the clock jumps from 02:00 to 03:00 that day


def test_a_ticker_is_given_the_session_its_name_or_class_implies_and_the_basis_is_said():
    cases = {"SPY": ("us_equity", "default"), "7203.T": ("tokyo_equity", "suffix .T"), "VOD.L": ("uk_equity", "suffix .L"), "SAP.DE": ("eu_equity", "suffix .DE"), "BTC-USD": ("crypto", "crypto asset"),
             "EURUSD=X": ("fx", "suffix =X"), "^N225": ("tokyo_equity", "index symbol"), "^GSPC": ("us_equity", "index symbol")}
    for ticker, want in cases.items():
        assert S.session_for(ticker) == want
    assert S.session_for("FOO", "crypto") == ("crypto", "crypto asset")


# ------------------------------------------------------------------------------------------------------------------ what a decision may see
@pytest.fixture(scope="module")
def panel():
    idx = pd.bdate_range("2024-03-04", "2024-03-15")
    n = np.arange(1.0, len(idx) + 1)
    return pd.DataFrame({"NY": n, "TK": n, "LN": n, "BTC": n}, index=idx)


SESS = {"NY": "us_equity", "TK": "tokyo_equity", "LN": "uk_equity", "BTC": "crypto"}


def test_a_us_close_decision_sees_asia_and_europe_of_the_same_date_but_not_a_crypto_bar_with_that_label(panel):
    values, ages = S.asof_panel(panel, SESS, "us_equity")
    for col, shift in (("NY", 0), ("TK", 0), ("LN", 0), ("BTC", 1)):
        want = panel[col].shift(shift)
        assert np.array_equal(values[col].to_numpy(), want.to_numpy(), equal_nan=True), col
    est, edt = pd.Timestamp("2024-03-08"), pd.Timestamp("2024-03-12")                            # winter and summer time in New York
    assert ages.loc[est].to_dict() == {"NY": 0.0, "TK": pytest.approx(14.5), "LN": pytest.approx(4.4167, abs=1e-3), "BTC": pytest.approx(21.0)}
    assert ages.loc[edt].to_dict() == {"NY": 0.0, "TK": pytest.approx(13.5), "LN": pytest.approx(3.4167, abs=1e-3), "BTC": pytest.approx(20.0)}


def test_a_tokyo_close_decision_sees_none_of_the_other_markets_bars_of_its_own_date(panel):
    values, ages = S.asof_panel(panel, SESS, "tokyo_equity")
    assert np.array_equal(values["TK"].to_numpy(), panel["TK"].to_numpy())
    for col in ("NY", "LN", "BTC"):
        assert np.array_equal(values[col].to_numpy(), panel[col].shift(1).to_numpy(), equal_nan=True), f"{col}: the bar with the date of the decision closes after the Tokyo decision"
    tue = pd.Timestamp("2024-03-05")
    assert ages.loc[tue, "NY"] == pytest.approx(9.5) and ages.loc[tue, "BTC"] == pytest.approx(6.5) and ages.loc[tue, "LN"] == pytest.approx(13.9167, abs=1e-3)
    mon = pd.Timestamp("2024-03-11")                                                             # the previous bar of a Monday decision is Friday's: the weekend is in the age
    assert ages.loc[mon, "NY"] == pytest.approx(57.5) and ages.loc[mon, "BTC"] == pytest.approx(54.5)


def test_a_crypto_midnight_decision_sees_every_bar_of_the_day_before_and_nothing_of_the_new_one(panel):
    """The decision session is crypto: the bar of date d is final at midnight and the decision for d is taken then, so the US bar of d (21:00 UTC) is in hand and nothing that closes after midnight is."""
    values, _ = S.asof_panel(panel, SESS, "crypto")
    for col in panel.columns:
        assert np.array_equal(values[col].to_numpy(), panel[col].to_numpy()), col


def test_a_closed_market_has_no_bar_so_an_older_one_is_the_latest_and_its_age_says_so(panel):
    shut = panel.copy()
    shut.loc["2024-03-06", "NY"] = np.nan                                                        # a holiday in New York
    values, ages = S.asof_panel(shut, SESS, "us_equity")
    day = pd.Timestamp("2024-03-06")
    assert values.loc[day, "NY"] == panel.loc["2024-03-05", "NY"] and ages.loc[day, "NY"] == pytest.approx(24.0)
    limited, _ = S.asof_panel(shut, SESS, "us_equity", max_age_hours=20)
    assert np.isnan(limited.loc[day, "NY"]) and limited.loc[pd.Timestamp("2024-03-07"), "NY"] == panel.loc["2024-03-07", "NY"]


def test_the_information_lag_is_zero_for_markets_that_close_first_and_one_for_those_that_close_after():
    cases = {("tokyo_equity", "us_equity"): 0, ("uk_equity", "us_equity"): 0, ("eu_equity", "us_equity"): 0, ("us_equity", "us_equity"): 0, ("us_equity", "crypto"): 0,
             ("crypto", "us_equity"): 1, ("fx", "us_equity"): 1, ("us_equity", "tokyo_equity"): 1, ("us_equity", "uk_equity"): 1, ("uk_equity", "tokyo_equity"): 1, ("crypto", "tokyo_equity"): 1}
    for (market, decision), want in cases.items():
        assert S.information_lag(market, decision) == want, (market, decision)
    assert S.information_lags({"A": "crypto", "B": "tokyo_equity"}, "us_equity") == {"A": 1, "B": 0}


def test_two_markets_whose_order_of_closing_changes_with_daylight_saving_are_refused():
    late_utc = S.Session("evening", "UTC", time(20, 30))                                         # before the New York close in winter (21:00 UTC), after it in summer (20:00 UTC)
    with pytest.raises(S.SessionError, match="daylight saving"):
        S.information_lag(late_utc, "us_equity")


def test_a_label_convention_whose_entry_price_is_in_the_past_of_the_decision_is_refused():
    S.entry_check({"A": "us_equity", "B": "crypto", "C": "tokyo_equity"}, "us_equity", lag=1)
    S.entry_check({"A": "us_equity", "B": "crypto"}, "us_equity", lag=0)
    S.entry_check({"A": "us_equity"}, "tokyo_equity", lag=0)
    with pytest.raises(S.SessionError, match="with lag 0 the entry price of C \\(tokyo_equity\\)"):
        S.entry_check({"A": "us_equity", "C": "tokyo_equity"}, "us_equity", lag=0)
    with pytest.raises(S.SessionError, match="uk_equity"):
        S.entry_check({"L": "uk_equity"}, "us_equity", lag=0)


def test_a_scheduled_release_is_public_to_a_new_york_close_the_same_day_and_to_a_tokyo_close_the_next():
    days = pd.DatetimeIndex(["2024-03-05", "2024-03-12"])
    release = S.release_instant(days, time(8, 30), "America/New_York")
    assert list(release) == [utc("2024-03-05 13:30"), utc("2024-03-12 12:30")], "08:30 New York is 13:30 UTC in winter and 12:30 UTC in summer"
    assert S.release_known(release, "us_equity", days).all() and not S.release_known(release, "tokyo_equity", days).any()
    assert S.release_known(release, "tokyo_equity", days + pd.Timedelta(days=1)).all()
    after = S.release_instant(days, time(16, 30), "America/New_York")                            # after the New York close
    assert not S.release_known(after, "us_equity", days).any() and S.release_known(after, "us_equity", days + pd.Timedelta(days=1)).all()


# ------------------------------------------------------------------------------------------------------------------ point in time
def cpi():
    return pd.DataFrame({
        "reference_date": ["2024-04-01", "2024-05-01", "2024-06-01", "2024-06-01", "2024-07-01"],
        "value": [300.0, 301.0, 302.0, 302.5, 303.0],
        "available_at": ["2024-05-15 12:30", "2024-06-12 12:30", "2024-07-11 12:30", "2024-08-14 12:30", "2024-08-14 12:30"],   # 08:30 New York in summer
        "vintage": [0, 0, 0, 1, 0]})


def oracle(records, decision, dates):
    """The brute-force definition: among the records public at the decision instant, the latest reference period, and of that the latest vintage."""
    inst = S.decision_instants(decision, dates).asi8
    avail = pit._instants(records["available_at"])
    ref = pd.to_datetime(records["reference_date"]).to_numpy("datetime64[ns]").astype("int64")
    vint = records["vintage"].to_numpy(dtype=int)
    out = []
    for t in inst:
        ok = np.flatnonzero(avail <= t)
        if not len(ok):
            out.append(np.nan)
            continue
        top = ok[ref[ok] == ref[ok].max()]
        pick = top[np.lexsort((vint[top], avail[top]))[-1]]
        out.append(float(records["value"].iloc[pick]))
    return np.array(out)


DECISIONS = ["us_equity", "tokyo_equity", "uk_equity", "crypto"]


@pytest.mark.parametrize("decision", DECISIONS)
def test_the_availability_join_equals_the_brute_force_scan(decision):
    days = pd.bdate_range("2024-05-01", "2024-08-30")
    got = pit.vintage_asof(cpi(), decision, days)
    assert np.array_equal(got.to_numpy(), oracle(cpi(), decision, days), equal_nan=True)


def test_a_late_publication_is_seen_on_the_first_decision_after_its_release_not_on_its_reference_date():
    days = pd.bdate_range("2024-06-03", "2024-08-16")
    us, tokyo = pit.vintage_asof(cpi(), "us_equity", days), pit.vintage_asof(cpi(), "tokyo_equity", days)
    d = lambda s: pd.Timestamp(s)                                                              # noqa: E731
    assert us[d("2024-06-03")] == 300.0 and us[d("2024-06-12")] == 301.0 and us[d("2024-07-10")] == 301.0, "June's number is not there in early June"
    assert us[d("2024-07-11")] == 302.0, "08:30 New York is before the 16:00 close of the same day"
    assert tokyo[d("2024-07-11")] == 301.0 and tokyo[d("2024-07-12")] == 302.0, "a Tokyo close at 06:30 UTC precedes a 12:30 UTC release: one day later"


def test_a_later_restatement_is_public_only_from_its_own_release():
    days = pd.bdate_range("2024-08-12", "2024-08-16")
    us = pit.vintage_asof(cpi(), "us_equity", days)
    assert us[pd.Timestamp("2024-08-13")] == 302.0, "the restated June figure (302.5) is not public yet"
    assert us[pd.Timestamp("2024-08-14")] == 303.0, "July's first print is the latest period once it is public"
    assert pit.known_value(cpi(), "2024-06-01", "2024-08-13 20:00") == 302.0 and pit.known_value(cpi(), "2024-06-01", "2024-08-14 12:30") == 302.5
    assert np.isnan(pit.known_value(cpi(), "2024-06-01", "2024-07-11 12:29")) and np.isnan(pit.known_value(cpi(), "2023-01-01", "2024-08-14"))


def test_the_join_on_the_reference_date_hands_a_model_numbers_that_did_not_exist_yet():
    days = pd.bdate_range("2024-06-03", "2024-07-10")
    leaked, honest = pit.join_on_reference(cpi(), days), pit.vintage_asof(cpi(), "us_equity", days)
    assert (leaked != honest).sum() >= 20
    assert leaked[pd.Timestamp("2024-06-03")] == 302.5 and honest[pd.Timestamp("2024-06-03")] == 300.0, "on 3 June, the leak knows the June print (and its restatement); the world knew April's"


def test_a_stamp_with_only_a_date_is_public_from_the_end_of_that_day():
    rec = pd.DataFrame({"reference_date": ["2024-06-01"], "value": [5.0], "available_at": ["2024-07-11"]})
    days = pd.bdate_range("2024-07-10", "2024-07-15")
    got = pit.vintage_asof(rec, "us_equity", days)
    assert np.isnan(got[pd.Timestamp("2024-07-11")]) and got[pd.Timestamp("2024-07-12")] == 5.0


def test_records_without_an_availability_time_are_refused():
    with pytest.raises(pit.PITError, match="available_at"):
        pit.vintage_asof(cpi().drop(columns="available_at"), "us_equity", pd.bdate_range("2024-07-01", periods=3))


@pytest.mark.parametrize("seed", range(5))
def test_the_join_equals_the_scan_on_random_vintages_with_missing_values_and_every_session(seed):
    rng = np.random.default_rng(seed)
    n = 40
    ref = pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 400, n), unit="D")
    lag = pd.to_timedelta(rng.integers(10, 60, n), unit="D") + pd.to_timedelta(rng.integers(0, 24 * 60, n), unit="min")
    records = pd.DataFrame({"reference_date": ref, "value": np.where(rng.random(n) < 0.1, np.nan, rng.normal(size=n)), "available_at": (ref + lag).tz_localize("UTC"), "vintage": rng.integers(0, 3, n)})
    days = pd.bdate_range("2023-02-01", "2024-03-29")
    for decision in DECISIONS:
        assert np.array_equal(pit.vintage_asof(records, decision, days).to_numpy(), oracle(records, decision, days), equal_nan=True), (seed, decision)


# ------------------------------------------------------------------------------------------------------------------ the quality of the evidence
def test_every_macro_input_is_classified_by_how_its_availability_is_known():
    assert pit.series_quality("VIX")["quality"] == "observed" and pit.series_quality("CARRY_CL")["quality"] == "observed"
    m1 = pit.series_quality("M1SL", {"release_lag_days": 45})
    assert m1["quality"] == "estimated" and m1["release_lag_days"] == 45 and "latest vintage" in m1["basis"]
    assert pit.series_quality("MY_SERIES")["quality"] == "unavailable"
    assert pit.series_quality("MY_SERIES", {"quality": "observed", "basis": "ALFRED vintages"})["basis"] == "ALFRED vintages"
    assert [pit.fundamentals_quality(q) for q in ("filed", "restated", "estimated", "nonsense")] == ["observed", "observed", "estimated", "unavailable"]


def test_a_report_restricts_validation_claims_unless_every_input_is_observed():
    ok = pit.pit_report(["VIX", "DGS10"])
    assert not ok["claims_restricted"] and ok["counts"] == {"observed": 2, "estimated": 0, "unavailable": 0}
    mixed = pit.pit_report(["VIX", "M1SL", "MY_SERIES"], {"M1SL": {"release_lag_days": 45}})
    assert mixed["claims_restricted"] and mixed["counts"] == {"observed": 1, "estimated": 1, "unavailable": 1} and "restricted" in mixed["statement"]
    none = pit.pit_report([])
    assert not none["claims_restricted"] and "no macro" in none["statement"]


def test_the_macro_columns_a_feature_set_reads_are_found_from_the_feature_names():
    x = probe_inputs()
    names = ["macro_VIX_z", "macro_DGS10_chg21", "carry", "iv_level", "funding_ann", "stable_growth_30", "ret_5"]
    used = pit.macro_columns_used(names, x.macro.columns, x.assets)
    assert set(used) == {"VIX", "DGS10", "CARRY_SPY", "SLOPE_SPY", "BASISMOM_SPY", "CARRY_GLD", "SLOPE_GLD", "BASISMOM_GLD", "CARRY_A03", "SLOPE_A03", "BASISMOM_A03", "GVZ", "FUNDING_BTC", "BASIS_BTC", "STABLE_SUPPLY"}
    assert used == sorted(used) and pit.macro_columns_used(["ret_5", "vol_21"], x.macro.columns, x.assets) == []


def test_a_dataset_reports_the_point_in_time_quality_of_the_macro_series_its_features_read():
    cfg = mc.experiment_config({"dataset": {"feature_groups": ["price", "macro"], "min_history": 100}}, defaults=mc.DEFAULTS)
    bundle = make_bundle(5, 500, seed=2)
    ds = dsm.build_dataset(bundle, cfg["dataset"], "pit")
    report = ds.meta["audit"]["pit"]
    assert set(report["series"]) == {"DGS10", "VIX", "T10Y2Y"} and report["series"]["VIX"]["quality"] == "observed" and report["series"]["DGS10"]["quality"] == "observed"
    assert not report["claims_restricted"]
    bundle.macro_meta = {"DGS10": {"release_lag_days": 1, "quality": "estimated", "basis": "fixed lag on the latest vintage"}}
    ds2 = dsm.build_dataset(bundle, cfg["dataset"], "pit2")
    restricted = ds2.meta["audit"]["pit"]
    assert restricted["series"]["DGS10"]["quality"] == "estimated" and restricted["claims_restricted"] and "restricted" in restricted["statement"]
    price_only = dsm.build_dataset(bundle, mc.experiment_config({"dataset": {"feature_groups": ["price"], "min_history": 100}}, defaults=mc.DEFAULTS)["dataset"], "none")
    assert price_only.meta["audit"]["pit"]["series"] == {} and not price_only.meta["audit"]["pit"]["claims_restricted"]


def test_the_audit_module_still_measures_every_feature_after_the_additions():
    assert A.violations(A.feature_dependencies(probe_inputs(), names=["macro_VIX_z", "carry", "ret_21"])) == []
