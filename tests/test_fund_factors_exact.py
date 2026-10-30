"""The factors built on the scores of the books (Altman, Magic Formula, owner earnings, PEG, Graham, Merton), each against a hand calculation on three small companies whose statements and prices are
written out, and the three published composites against their parts."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.fundamentals import factors as F
from src.fundamentals import transforms as T
from src.fundamentals.research import ResearchContext, UniverseSpec
from src.fundamentals.service import FundamentalsService
from src.fundamentals.store import CompanyProfile, FundamentalsStore, PeriodStatement

DAY = pd.Timestamp("2021-06-30")                      # the date everything is read on: FY2020 (filed 1 March 2021) is the latest statement

# three companies, annual statements for 2017-2020 (the same each year but for earnings per share and net income), constant prices with a little noise
A = dict(revenue=1800.0, operating_income=250.0, total_assets=2000.0, current_assets=900.0, current_liabilities=500.0, cash_and_equivalents=200.0, short_term_debt=100.0, long_term_debt=400.0, total_debt=500.0,
         ppe=700.0, shareholders_equity=1000.0, total_liabilities=1000.0, retained_earnings=600.0, shares_outstanding=100.0, depreciation_amortization=60.0, capital_expenditure=100.0,
         operating_cash_flow=210.0, dividends_paid=45.0)
B = dict(revenue=900.0, operating_income=100.0, total_assets=1400.0, current_assets=600.0, current_liabilities=300.0, cash_and_equivalents=100.0, short_term_debt=50.0, long_term_debt=150.0, total_debt=200.0,
         ppe=750.0, shareholders_equity=900.0, total_liabilities=500.0, retained_earnings=300.0, shares_outstanding=100.0, depreciation_amortization=40.0, capital_expenditure=70.0,
         operating_cash_flow=120.0, dividends_paid=20.0)
C = dict(revenue=3000.0, operating_income=300.0, total_assets=3600.0, current_assets=1500.0, current_liabilities=700.0, cash_and_equivalents=200.0, short_term_debt=100.0, long_term_debt=500.0, total_debt=600.0,
         ppe=1300.0, shareholders_equity=2400.0, total_liabilities=1200.0, retained_earnings=900.0, shares_outstanding=100.0, depreciation_amortization=90.0, capital_expenditure=150.0,
         operating_cash_flow=330.0, dividends_paid=60.0)
EPS = {"A": {2017: 1.0, 2018: 1.2, 2019: 1.5, 2020: 2.0}, "B": {y: 1.0 for y in range(2017, 2021)}, "C": {y: 1.5 for y in range(2017, 2021)}}
PRICE = {"A": 40.0, "B": 10.0, "C": 60.0}


def _world(tmp_path, companies: dict | None = None):
    store = FundamentalsStore(tmp_path / "w.db")
    days = pd.bdate_range("2019-01-02", "2021-12-31")
    ids = {}
    for k, (name, base) in enumerate((companies or {"A": A, "B": B, "C": C}).items()):
        sid = ids[name] = store.upsert_security(CompanyProfile("simulated", name, name, name, exchange="NYSE", market="NYSE", country="US", currency="USD", sector="X"))
        store.store_financials(sid, "simulated", [PeriodStatement("FY", f"{y}-12-31", dict(base, eps_diluted=EPS[name][y], net_income=EPS[name][y] * 100.0), fiscal_year=y, filing_date=f"{y + 1}-03-01",
                                                                  reporting_currency="USD") for y in range(2017, 2021)])
        price = PRICE[name] * (1.0 + 0.004 * np.sin(np.arange(len(days)) / (3.0 + k)))             # a smooth wiggle: a volatility that is not zero
        store.store_prices(pd.DataFrame({"security_id": sid, "date": days, "adj_close": price, "close": price, "volume": 1e6}))
    return ResearchContext(store, UniverseSpec()), ids


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return _world(tmp_path_factory.mktemp("exact"))


def _at(ctx, ids, name, who):
    return F.compute(ctx, name).loc[DAY, ids[who]]


def _price(ctx, ids, who):
    return float(ctx.price().loc[DAY, ids[who]])


def test_the_owner_earnings_yield_is_profit_plus_depreciation_less_capital_spending_over_market_value(world):
    ctx, ids = world
    mcap = _price(ctx, ids, "A") * 100.0
    assert _at(ctx, ids, "owner_earnings_yield", "A") == pytest.approx((200.0 + 60.0 - 100.0) / mcap, rel=1e-12)
    assert _at(ctx, ids, "owner_earnings_yield", "B") == pytest.approx((100.0 + 40.0 - 70.0) / (_price(ctx, ids, "B") * 100.0), rel=1e-12)


def test_the_inverse_peg_is_growth_in_per_cent_times_the_earnings_yield_and_needs_both_positive(world):
    ctx, ids = world
    growth = (2.0 / 1.0) ** (1 / 3) - 1
    pe = _price(ctx, ids, "A") / 2.0
    peg = pe / (100 * growth)
    assert _at(ctx, ids, "peg_inverse", "A") == pytest.approx(1.0 / peg, rel=1e-12) and _at(ctx, ids, "peg_inverse", "A") == pytest.approx(100 * growth * 200.0 / (_price(ctx, ids, "A") * 100.0), rel=1e-12)
    assert math.isnan(_at(ctx, ids, "peg_inverse", "B"))                                       # no growth in earnings: the ratio has no meaning


def test_the_graham_number_over_the_price_and_the_net_current_asset_value_over_the_price(world):
    ctx, ids = world
    eps3, bvps = (2.0 + 1.5 + 1.2) / 3, 1000.0 / 100.0
    assert _at(ctx, ids, "graham_number_to_price", "A") == pytest.approx(math.sqrt(22.5 * eps3 * bvps) / _price(ctx, ids, "A"), rel=1e-12)
    assert _at(ctx, ids, "ncav_to_price", "A") == pytest.approx((900.0 - 1000.0) / 100.0 / _price(ctx, ids, "A"), rel=1e-12)          # owes more than the current assets cover: negative
    assert _at(ctx, ids, "ncav_to_price", "B") == pytest.approx((600.0 - 500.0) / 100.0 / _price(ctx, ids, "B"), rel=1e-12)


def test_the_original_altman_score_sets_the_market_value_of_the_equity_against_the_liabilities(world):
    ctx, ids = world
    mcap = _price(ctx, ids, "A") * 100.0
    expected = 1.2 * (400 / 2000) + 1.4 * (600 / 2000) + 3.3 * (250 / 2000) + 0.6 * (mcap / 1000.0) + 1.0 * (1800 / 2000)
    assert _at(ctx, ids, "altman_z_original", "A") == pytest.approx(expected, rel=1e-12)
    assert _at(ctx, ids, "altman_z_double_prime", "A") == pytest.approx(4.18, abs=1e-12) and _at(ctx, ids, "retained_earnings_to_assets", "A") == pytest.approx(0.3, rel=1e-12)


def test_the_magic_formula_averages_two_percentile_ranks_among_the_companies_of_the_date(world):
    ctx, ids = world
    roc = {w: _at(ctx, ids, "greenblatt_roc", w) for w in "ABC"}
    assert roc == pytest.approx({"A": 0.25, "B": 0.10, "C": 0.15}, rel=1e-12)
    ebit_ev = {w: float(F.compute(ctx, "ebit_to_ev").loc[DAY, ids[w]]) for w in "ABC"}
    assert ebit_ev["B"] > ebit_ev["A"] > ebit_ev["C"]
    magic = {w: _at(ctx, ids, "magic_formula", w) for w in "ABC"}
    assert magic == pytest.approx({"A": (3 / 3 + 2 / 3) / 2, "B": (1 / 3 + 3 / 3) / 2, "C": (2 / 3 + 1 / 3) / 2}, rel=1e-12)


def test_the_naive_distance_to_default_is_the_closed_form_of_bharath_and_shumway(world):
    ctx, ids = world
    for who, face in (("A", 100.0 + 0.5 * 400.0), ("B", 50.0 + 0.5 * 150.0)):
        sid = ids[who]
        equity, sigma_e, drift = float(ctx.market_cap().loc[DAY, sid]), float(ctx.volatility(252).loc[DAY, sid]), float(ctx.past(252, 0).loc[DAY, sid])
        total = equity + face
        sigma_d = 0.05 + 0.25 * sigma_e
        sigma_v = equity / total * sigma_e + face / total * sigma_d
        expected = (math.log(total / face) + drift - 0.5 * sigma_v ** 2) / sigma_v
        assert sigma_e > 0 and _at(ctx, ids, "naive_distance_to_default", who) == pytest.approx(expected, rel=1e-12)


def test_a_company_with_no_debt_at_all_has_no_distance_to_default_not_an_infinite_one(tmp_path):
    clean = {k: v for k, v in A.items() if k not in ("short_term_debt", "long_term_debt", "total_debt")}
    ctx, ids = _world(tmp_path, {"A": clean})
    assert math.isnan(F.compute(ctx, "naive_distance_to_default").loc[DAY, ids["A"]])


def test_the_new_factors_have_a_prior_direction_a_family_and_a_text():
    names = ("altman_z_double_prime", "altman_z_prime", "altman_z_original", "retained_earnings_to_assets", "naive_distance_to_default", "greenblatt_roc", "owner_earnings_margin", "sustainable_growth",
             "internal_growth", "eps_cagr_3y", "owner_earnings_yield", "peg_inverse", "graham_number_to_price", "ncav_to_price", "magic_formula", "noa_to_assets", "noa_change",
             "qmj_style_composite", "contrarian_value_composite", "piotroski_value_composite")
    for name in names:
        f = F.get(name)
        assert f.family in F.FAMILIES and f.direction in (1, -1) and len(f.text) > 30, name
    assert F.get("noa_to_assets").direction == -1 and F.get("noa_change").direction == -1 and F.get("altman_z_double_prime").direction == 1 and F.get("magic_formula").family == "value"


@pytest.fixture(scope="module")
def simulated(tmp_path_factory):
    svc = FundamentalsService(tmp_path_factory.mktemp("composites"))
    svc.load_simulated(n_companies=50, start_year=2009, end_year=2016, seed=8)
    return ResearchContext(svc.store, UniverseSpec())


def test_the_contrarian_composite_subtracts_the_growth_leg(simulated):
    ctx = simulated

    def z(name):
        return T.apply(F.compute(ctx, name), [{"op": "winsorize", "pct": 1.0}, {"op": "zscore"}], ctx)

    t = 70
    parts = [z("cfo_yield").iloc[t], z("earnings_yield").iloc[t], z("book_to_price").iloc[t], -z("revenue_cagr_3y").iloc[t]]
    frame = pd.concat(parts, axis=1)
    ok = frame.notna().all(axis=1)
    got = F.compute(ctx, "contrarian_value_composite").iloc[t]
    assert ok.sum() > 20 and np.allclose(got[ok], frame[ok].mean(axis=1))
    assert "minus revenue_cagr_3y" in F.get("contrarian_value_composite").text


def test_the_quality_style_composite_flips_every_part_that_is_bad_when_high(simulated):
    ctx = simulated

    def z(name):
        return T.apply(F.compute(ctx, name), [{"op": "winsorize", "pct": 1.0}, {"op": "zscore"}], ctx) * (1.0 if F.get(name).direction >= 0 else -1.0)

    names = ["gross_profitability", "roa", "cash_return_on_assets", "accruals_ratio", "debt_to_assets", "beta", "earnings_stability", "share_growth"]
    t = 90
    frame = pd.concat([z(n).iloc[t] for n in names], axis=1)
    full = frame.notna().all(axis=1)
    got = F.compute(ctx, "qmj_style_composite").iloc[t]
    assert full.sum() > 15 and np.allclose(got[full], frame[full].mean(axis=1))
    enough = frame.notna().sum(axis=1) >= 5
    assert enough.sum() >= full.sum() and got[~enough].isna().all()                            # fewer than five parts: no score


def test_cheap_and_improving_is_book_to_price_and_the_f_score(simulated):
    ctx = simulated

    def z(name):
        return T.apply(F.compute(ctx, name), [{"op": "winsorize", "pct": 1.0}, {"op": "zscore"}], ctx)

    t = 80
    frame = pd.concat([z("book_to_price").iloc[t], z("piotroski_f").iloc[t]], axis=1)
    ok = frame.notna().all(axis=1)
    assert ok.sum() > 15 and np.allclose(F.compute(ctx, "piotroski_value_composite").iloc[t][ok], frame[ok].mean(axis=1))
