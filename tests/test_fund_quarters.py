"""Year-to-date figures kept as reported; standalone quarters derived from them, with their derivation, availability and warnings; TTM built on them."""

from __future__ import annotations

import pytest

from src.fundamentals import quarters
from src.fundamentals.pit import PeriodTables
from src.fundamentals.providers.base import CompanyProfile
from src.fundamentals.store import FundamentalsStore

CUM = {"Q1": {"value": 100, "period_end": "2023-03-31", "filing_date": "2023-05-05"}, "H1": {"value": 250, "period_end": "2023-06-30", "filing_date": "2023-08-04"},
       "Q3YTD": {"value": 390, "period_end": "2023-09-30", "filing_date": "2023-11-03"}, "FY": {"value": 600, "period_end": "2023-12-31", "filing_date": "2024-02-20"}}


@pytest.fixture()
def store(tmp_path):
    s = FundamentalsStore(tmp_path / "f.db")
    s.upsert_security(CompanyProfile("manual", "us:Q", "QQQ", "Quarterly Inc", exchange="US", market="US", country="US", currency="USD"))
    return s


def test_cumulative_cash_flow_becomes_standalone_quarters_with_the_derivation_and_the_day_each_was_public():
    q = quarters.decumulate(CUM)
    assert [q[k]["value"] for k in ("Q1", "Q2", "Q3", "Q4")] == [100, 150, 140, 210] and sum(q[k]["value"] for k in q) == 600
    assert q["Q3"]["derivation"].startswith("nine months 390") and "250" in q["Q3"]["derivation"]
    assert q["Q2"]["available_from"] == "2023-08-04" and q["Q4"]["available_from"] == "2024-02-20", "a quarter is public when its latest input was"
    assert [i["period"] for i in q["Q4"]["inputs"]] == ["FY", "Q3YTD"] and q["Q4"]["days"] == 92 and not q["Q4"]["flags"]


def test_a_missing_input_means_no_derivation_never_an_estimate():
    no_half = {k: v for k, v in CUM.items() if k != "H1"}
    q = quarters.decumulate(no_half)
    assert set(q) == {"Q1", "Q4"} and q["Q4"]["value"] == 210, "Q2 and Q3 cannot be separated without the half year, so they are not derived"
    h2 = quarters.decumulate({"H1": CUM["H1"], "FY": CUM["FY"]})
    assert set(h2) == {"H2"} and h2["H2"]["value"] == 350
    assert quarters.decumulate({"FY": CUM["FY"]}) == {} and quarters.decumulate({"H1": CUM["H1"]}) == {}


def test_a_restated_input_is_flagged_and_a_fifty_three_week_quarter_is_not_forced_onto_91_days():
    cum = {k: dict(v) for k, v in CUM.items()}
    cum["H1"].update(value=240, vintage=1)
    assert any("vintages" in f for f in quarters.decumulate(cum)["Q2"]["flags"])
    wk = {"Q1": {"value": 10, "period_end": "2020-12-26", "filing_date": "2021-01-28"}, "H1": {"value": 25, "period_end": "2021-03-27", "filing_date": "2021-04-29"},
          "Q3YTD": {"value": 40, "period_end": "2021-06-26", "filing_date": "2021-07-29"}, "FY": {"value": 70, "period_end": "2021-09-25", "filing_date": "2021-10-29"}}
    assert quarters.decumulate(wk)["Q2"]["days"] == 91 and not quarters.decumulate(wk)["Q4"]["flags"]
    wk["FY"]["period_end"] = "2021-10-02"                                                               # a 14-week fourth quarter
    assert quarters.decumulate(wk)["Q4"]["days"] == 98 and not quarters.decumulate(wk)["Q4"]["flags"]
    wk["FY"]["period_end"] = "2022-03-01"
    assert quarters.decumulate(wk)["Q4"]["flags"], "a 'quarter' of 248 days is queried"


def test_cumulative_observations_are_kept_and_derived_quarters_feed_ttm_from_their_availability_date(store):
    sid = int(store.securities().iloc[0]["security_id"])
    for k, v in CUM.items():
        quarters.save_cumulative(store, sid, "operating_cash_flow", 2023, k, v["period_end"], v["value"], "USD", v["filing_date"])
    derived = quarters.derive(store, sid, write=True)
    assert [d["value"] for d in derived] == [100, 150, 140, 210]
    with store.db.transaction() as c:
        assert c.execute("SELECT COUNT(*) FROM ytd_observations").fetchone()[0] == 4, "the cumulative figures as reported are still there"
        src = c.execute("SELECT source_metric, quality_flag FROM fundamental_observations WHERE provider = 'derived' ORDER BY id").fetchall()
    assert all(s[1] == "derived" and s[0].startswith("derived: ") for s in src) and len(src) == 4
    t = PeriodTables(store, [sid])
    ttm = t.ttm[t.ttm["period_end"] == "2023-12-31"]
    assert float(ttm["operating_cash_flow"].iloc[0]) == 600 and "TTM" in set(t.ttm["fiscal_period"])
    assert str(ttm["available_date"].iloc[0])[:10] >= "2024-02-20"


def test_cumulative_figures_in_two_currencies_are_not_differenced(store):
    sid = int(store.securities().iloc[0]["security_id"])
    quarters.save_cumulative(store, sid, "revenue", 2023, "Q1", "2023-03-31", 10, "USD", "2023-05-01")
    quarters.save_cumulative(store, sid, "revenue", 2023, "H1", "2023-06-30", 25, "KRW", "2023-08-01")
    out = quarters.derive(store, sid)
    assert out[0]["value"] is None and "different currencies" in out[0]["flags"][0]
    with pytest.raises(ValueError):
        quarters.save_cumulative(store, sid, "total_assets", 2023, "Q1", "2023-03-31", 1, "USD")
