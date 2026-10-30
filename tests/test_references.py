"""The sources shown under each method: well formed, attached to methods that exist, and carried by the catalogue the page reads."""

from __future__ import annotations

import re

import pytest

from src import references as refs
from src.framework import ALLOCATORS, MODELS, load_library
from src.utils.config import load_config
from src.webapp.api import App
from src.webapp.universe import TickerStore, UniverseBuilder
from tests.ml_helpers import make_bundle
from tests.test_webapp import _fetch

load_library()
DOI_URL = re.compile(r"^https://doi\.org/10\.\d{4,9}/\S+$")

TABLES = {"model": refs.MODELS, "allocator": refs.ALLOCATORS, **refs.EXTRA}


def every_ref():
    for kind, table in TABLES.items():
        for name, items in table.items():
            for ref in items:
                yield kind, name, ref


def test_every_reference_is_a_citation_with_a_year_and_an_https_link_or_none():
    seen = 0
    for kind, name, ref in every_ref():
        seen += 1
        assert re.search(r"\(\d{4}\)", ref.cite), (kind, name, ref.cite)                       # authors (year), title, venue
        assert ref.cite == ref.cite.strip() and "  " not in ref.cite and ref.cite.count(",") >= 2, ref.cite
        assert ref.url is None or ref.url.startswith("https://"), (name, ref.url)
        if ref.url and "doi.org" in ref.url:
            assert DOI_URL.match(ref.url), ref.url
    assert seen > 100


def test_no_method_lists_the_same_source_twice_and_every_method_exists():
    from src.forecast import BLENDS, METHODS

    registries = {"model": {e.name for e in MODELS.entries()}, "allocator": {e.name for e in ALLOCATORS.entries()}, "forecaster": set(METHODS) | set(BLENDS)}
    for kind, table in TABLES.items():
        for name, items in table.items():
            assert len({r.cite for r in items}) == len(items), (kind, name)
            assert len(items) >= 1, (kind, name)
            if kind in registries:
                assert name in registries[kind], f"{kind} '{name}' has references but is not registered"


def test_the_new_book_rules_cite_their_sources():
    for name, must in {"carver_ewmac": "Carver", "clenow_momentum": "Clenow", "tstat_momentum": "Baltas", "variance_ratio_switch": "MacKinlay", "connors_rsi": "Connors", "double_seven": "Connors",
                       "skew_premium": "Amaya", "cppi": "Black"}.items():
        assert any(must in r.cite for r in refs.MODELS[name]), name


def test_every_forecasting_method_but_the_two_plain_baselines_cites_its_source():
    from src.forecast import BLENDS, METHODS

    bare = set(METHODS) | set(BLENDS)
    assert bare - set(refs.FORECASTERS) == {"random_walk", "historical"}                         # a zero-mean random walk and the empirical distribution need no citation
    assert len(refs.FORECAST_EVALUATION) >= 6 and all(r.url for r in refs.FORECAST_EVALUATION)


def test_lookup_returns_plain_dictionaries_and_an_empty_list_for_a_method_with_none():
    got = refs.references("model", "tsmom")
    assert got and set(got[0]) == {"cite", "url"} and got[0]["url"].startswith("https://doi.org/")
    assert refs.references("model", "no_such_method") == [] and refs.references("no_such_kind", "tsmom") == []


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("refs_app")
    config = load_config()
    bundle = make_bundle(n_assets=4, n=600, seed=3, alpha=0.0)
    return App(config, builder=UniverseBuilder(config, TickerStore(tmp / "prices", fetch=_fetch), default_loader=lambda c: bundle), root=tmp)


def test_the_catalogue_the_page_reads_carries_the_sources_of_strategies_and_allocators(app):
    catalog = app.catalog()
    models = {m["name"]: m for m in catalog["models"]}
    assert models["carver_ewmac"]["references"][0]["cite"].startswith("Carver (2015)")
    assert all("references" in m for m in catalog["models"]) and all("references" in a for a in catalog["allocators"])
    assert {a["name"]: a for a in catalog["allocators"]}["black_litterman"]["references"][0]["url"] == "https://doi.org/10.2469/faj.v48.n5.28"
    assert models["carver_ewmac"]["book"] == "sleeves" and models["clenow_momentum"]["min_assets"] >= 2
    names = {p["name"] for p in models["carver_ewmac"]["params"]}
    assert {"fast_spans", "slow_multiple", "vol_span", "target_abs", "cap", "min_history", "diversification", "fdm_cap"} <= names


def test_the_factors_and_scores_of_the_books_name_their_sources_and_the_catalogue_carries_them():
    from src.fundamentals import factors

    assert set(refs.FACTOR_SOURCES) <= set(factors.FACTORS)
    new = ("altman_z_double_prime", "altman_z_prime", "altman_z_original", "retained_earnings_to_assets", "naive_distance_to_default", "greenblatt_roc", "magic_formula", "owner_earnings_yield",
           "owner_earnings_margin", "sustainable_growth", "internal_growth", "peg_inverse", "eps_cagr_3y", "graham_number_to_price", "ncav_to_price", "noa_to_assets", "noa_change", "qmj_style_composite",
           "contrarian_value_composite", "piotroski_value_composite")
    assert all(n in refs.FACTOR_SOURCES for n in new)
    by_name = {c["name"]: c for c in factors.catalogue()}
    assert by_name["magic_formula"]["sources"][0]["cite"].startswith("Greenblatt (2005)") and by_name["magic_formula"]["sources"][0]["url"] is None             # a book has no link
    assert by_name["naive_distance_to_default"]["sources"][0]["url"] == "https://doi.org/10.1093/rfs/hhn044"
    assert by_name["volatility_63"]["sources"] == []                                                                                                 # a factor with no source on record says so, not a guess
    assert any("Buffett (1986)" in r["cite"] and r["url"] == "https://www.berkshirehathaway.com/letters/1986.html" for r in by_name["owner_earnings_yield"]["sources"])


def test_the_company_cards_of_the_fundamentals_tab_each_have_sources():
    assert set(refs.FUNDAMENTALS_VIEWS) == {"distress", "graham", "valuation", "growth", "owner_earnings", "return_on_capital", "net_operating_assets", "dupont", "anomaly", "piotroski"}
    assert refs.references("fundamentals", "growth")[0]["url"] == "https://doi.org/10.2307/3665251" and refs.references("fundamentals", "nothing") == []
