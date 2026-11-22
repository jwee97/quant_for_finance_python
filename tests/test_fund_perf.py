"""Generous ceilings on the research-side hot spots, so a change that makes them quadratic is caught (``benchmarks/fundamentals_perf.py`` prints the timings at any size)."""

from __future__ import annotations

import time

import pytest

from src.fundamentals import datahealth, snapshot, sources
from src.fundamentals.service import FundamentalsService


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    svc = FundamentalsService(tmp_path_factory.mktemp("perf"))
    svc.load_simulated(n_companies=40, start_year=2010, end_year=2016, seed=2)
    return svc.store


def test_data_health_of_forty_companies_loads_the_tables_once_not_once_per_company(store):
    t = time.perf_counter()
    d = datahealth.overview(store)
    took = time.perf_counter() - t
    assert d["n"] == 40 and took < 30, f"{took:.1f}s"


def test_a_bulk_comparison_equals_the_one_company_comparison(store):
    sid = int(store.securities().iloc[3]["security_id"])
    one = sources.compare(store, sid)
    ids = [int(i) for i in store.securities()["security_id"]]
    periods, obs = store.periods(ids), store.observations(ids)
    with store.db.transaction() as c:
        conf = {(r[0], r[1]) for r in c.execute("SELECT financial_period_id, metric_code FROM observation_provenance WHERE verified = 1").fetchall()}
    bulk = sources.compare(store, sid, frames={"periods": periods[periods["security_id"] == sid], "obs": obs[obs["security_id"] == sid], "prefs": store.source_preferences(ids), "confirmed": conf})
    assert one == bulk


def test_the_fingerprint_is_fast_and_stable(store):
    t = time.perf_counter()
    a, b = snapshot.fingerprint(store), snapshot.fingerprint(store)
    assert a["hash"] == b["hash"] and time.perf_counter() - t < 20
