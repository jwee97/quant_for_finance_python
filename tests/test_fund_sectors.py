"""Sector-aware applicability and sector figures from confirmed custom observations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.fundamentals import factors, sectors
from src.fundamentals.research import ResearchContext, UniverseSpec
from src.fundamentals.service import FundamentalsService


def test_families_are_read_from_sector_and_industry_text_and_unclassified_is_not_guessed():
    f = sectors.family_of
    assert f("Financials", "Banks")[0] == "bank" and f("Financials", "Insurance - Property & Casualty")[0] == "insurer"
    assert f("Real Estate", "REIT - Retail")[0] == "reit" and f("Technology", "Software")[0] == "general"
    assert f(None, None) == ("general", "unclassified") and f("", "")[1] == "unclassified"


@pytest.fixture()
def world(tmp_path):
    svc = FundamentalsService(tmp_path)
    svc.load_simulated(n_companies=14, start_year=2012, end_year=2016, seed=9)
    return svc


def _label(store, ids_to_fam):
    with store.db.transaction() as c:
        for sid, (sector, industry) in ids_to_fam.items():
            c.execute("UPDATE securities SET sector = ?, industry = ? WHERE security_id = ?", (sector, industry, int(sid)))


def test_a_bank_has_no_gross_margin_or_current_ratio_instead_of_a_low_score(world):
    base = ResearchContext(world.store, UniverseSpec())
    bank, reit, other = base.ids[0], base.ids[1], base.ids[2]
    before = factors.compute(base, "gross_margin")
    assert before[bank].notna().any(), "the simulated company has a gross margin until it is called a bank"
    _label(world.store, {bank: ("Financials", "Banks"), reit: ("Real Estate", "Equity REITs")})
    ctx = ResearchContext(world.store, UniverseSpec())
    gm, cr, inv = factors.compute(ctx, "gross_margin"), factors.compute(ctx, "current_ratio"), factors.compute(ctx, "inventory_turnover")
    assert gm[bank].isna().all() and cr[bank].isna().all() and inv[reit].isna().all() and gm[other].notna().any(), "only the companies it does not apply to lose it"
    ey = factors.compute(ctx, "earnings_yield")
    assert ey[bank].notna().any(), "earnings yield applies to a bank"
    a = sectors.applicability(ctx, "gross_margin")
    assert sorted(a["not_applicable_companies"]) == sorted([bank, reit]) and a["families"] == ["bank", "reit"] and "deposits" in a["reason"]["bank"]
    assert sectors.applicability(ctx, "gross_margin")["unclassified_companies"] >= 0


def test_a_persons_family_choice_overrides_the_text(world):
    ctx0 = ResearchContext(world.store, UniverseSpec())
    sid = ctx0.ids[3]
    world.store.set_family(sid, "insurer")
    ctx = ResearchContext(world.store, UniverseSpec())
    assert sectors.families(ctx)[sid] == "insurer" and factors.compute(ctx, "fcf_yield")[sid].isna().all()
    world.store.set_family(sid, None)
    with pytest.raises(ValueError):
        world.store.set_family(sid, "spaceship")


def _confirm_custom(store, sid, code, value, period_end, filing):
    with store.db.transaction() as c:
        c.execute("INSERT INTO custom_observations (security_id, metric_code, label, value, currency, unit, unit_multiplier, period_end, period_type, filing_date, pit_quality, vintage, confirmed_at, created_at) "
                  "VALUES (?, ?, ?, ?, NULL, 'percent', 1, ?, 'ANNUAL', ?, 'filed', 0, '2020-01-01T00:00:00Z', '2020-01-01T00:00:00Z')", (int(sid), code, code, value, period_end, filing))


def test_confirmed_sector_figures_are_factors_public_only_from_their_filing_date_and_only_for_their_family(world):
    base = ResearchContext(world.store, UniverseSpec())
    bank, other = base.ids[0], base.ids[1]
    _label(world.store, {bank: ("Financials", "Banks")})
    _confirm_custom(world.store, bank, "custom:cet1_ratio", 13.5, "2013-12-31", "2014-02-20")
    _confirm_custom(world.store, other, "custom:cet1_ratio", 99.0, "2013-12-31", "2014-02-20")          # not a bank: the figure does not apply
    ctx = ResearchContext(world.store, UniverseSpec())
    p = factors.compute(ctx, "custom:cet1_ratio")
    before, after = p.index[p.index < "2014-02-21"], p.index[(p.index > "2014-02-22") & (p.index < "2015-01-01")]
    assert p.loc[before, bank].isna().all() and (p.loc[after, bank] == 13.5).all(), "known from the filing date, not the period end"
    assert p[other].isna().all(), "a non-bank has no CET1 ratio"
    late = p.index[p.index > "2016-12-01"]
    assert len(late) == 0 or p.loc[late, bank].isna().all(), "stale after 540 days"
    assert "custom:cet1_ratio" in {c["name"] for c in sectors.catalogue()}
