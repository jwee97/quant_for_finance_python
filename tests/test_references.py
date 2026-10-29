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
    registries = {"model": {e.name for e in MODELS.entries()}, "allocator": {e.name for e in ALLOCATORS.entries()}}
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
