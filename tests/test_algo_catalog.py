"""The taxonomy catalogue is true (every location imports), the algorithm factory parses what it is given and refuses what it is not, and the command line runs."""

from __future__ import annotations

import numpy as np
import pytest

from src import cli
from src.algo import catalog
from src.algo.factory import ALL_ALGORITHMS, TACTICS, build_algo, efficient_frontier, run_order
from src.algo.tactics import AIM, TargetCost

# The items of the taxonomy as the request listed them: each must be in the catalogue.
TAXONOMY = {
    "1a. Investment: alpha generating": ["Long-term", "Short-term", "Company outlook", "Company news", "Corporate action", "Mispricing"],
    "1b. Investment: portfolio rebalance": ["Asset allocation", "Index reconstitution", "Market outlook", "Market neutral", "Flight to quality", "Model driven"],
    "1c. Investment: risk management": ["Risk reduction", "Hedging", "Liquidation costs"],
    "1d. Investment: cash flow": ["Cash deposit", "Redemption", "Cash dividend", "Liabilities", "Payments"],
    "1e. Investment: economic outlook": ["Yield curve strategy", "Credit strategy"],
    "2. Trading algorithm styles": ["Aggressive", "Working order", "Passive"],
    "3. Specific algorithm types": ["VWAP", "TWAP", "POV / Volume", "Arrival price", "Implementation shortfall", "Basket / portfolio algorithms", "Black-box: pair trading",
                                    "Black-box: auto market making", "Black-box: statistical arbitrage", "Liquidity seeking"],
    "4. High-frequency trading": ["Auto market making (AMM)", "Quantitative trading / statistical arbitrage", "Rebate / liquidity trading"],
    "5. Best execution goals": ["Minimise cost", "Minimise cost with a risk constraint", "Minimise risk with a cost constraint", "Balance cost and risk", "Price improvement"],
    "6. Adaptation tactics": ["Target cost", "Aggressive in the money (AIM)", "Passive in the money (PIM)"],
    "7. Schedule and portfolio optimisation": ["Quadratic programming", "Trade schedule exponential", "Residual schedule exponential", "Trade rate parameter", "Portfolio optimisation with TCA (third wave)"],
    "8. Advanced execution and risk tactics": ["Minimum trading risk quantity", "Maximum trading opportunity", "Program-block decomposition"],
}


def test_every_item_of_the_taxonomy_is_in_the_catalogue_with_an_honest_status():
    have = {(i.section, i.item) for i in catalog.ITEMS}
    for section, items in TAXONOMY.items():
        for item in items:
            assert (section, item) in have, (section, item)
    assert len(have) == len(catalog.ITEMS)                                                                       # no item twice
    assert {i.status for i in catalog.ITEMS} <= {"built", "existing", "proxy", "data", "simulator"}
    assert all(i.where and i.note for i in catalog.ITEMS)


def test_every_location_the_catalogue_names_really_exists():
    for item in catalog.ITEMS:
        for where in item.where:
            assert catalog.resolve(where) is not None, (item.item, where)
    with pytest.raises(ValueError):
        catalog.resolve("nowhere:thing")
    with pytest.raises(KeyError):
        catalog.resolve("model:no_such_model")
    with pytest.raises(AttributeError):
        catalog.resolve("py:src.algo.algos.NoSuchAlgorithm")


def test_the_coverage_page_is_the_catalogue_written_out_and_the_data_items_really_need_data():
    from pathlib import Path

    page = (Path(__file__).resolve().parents[1] / "docs" / "algorithmic_trading.md").read_text(encoding="utf-8")
    assert page == catalog.markdown(), "docs/algorithmic_trading.md is stale: run `python -m src.algo.catalog`"
    for item in catalog.ITEMS:
        assert f"| {item.item} |" in page
    # an item marked as needing data names a model that refuses to run without its file
    import pandas as pd

    from src.framework import MODELS, bundle_from_prices
    from src.strategies import alpha_styles

    prices = pd.DataFrame(100.0 + pd.Series(range(300), dtype=float).to_numpy()[:, None] * [0.01, 0.02], index=pd.bdate_range("2015-01-05", periods=300), columns=["A", "B"])
    bundle = bundle_from_prices(prices, name="x")
    saved = alpha_styles.USER_DATA
    alpha_styles.USER_DATA = Path("/nonexistent/for/this/test")
    try:
        for name in ("news_sentiment", "panel_signal"):
            with pytest.raises(KeyError, match="needs the file"):
                MODELS.create(name).score(bundle)
    finally:
        alpha_styles.USER_DATA = saved
    assert {i.status for i in catalog.ITEMS if i.item in ("Company news", "Company outlook")} == {"data"}


def test_the_catalogue_table_has_a_row_per_item_and_shows_where_to_look():
    t = catalog.table()
    assert len(t) == len(catalog.ITEMS) and {"section", "item", "status", "where", "note"} <= set(t.columns)
    assert t.loc[t["item"] == "VWAP", "where"].iloc[0] == "src.algo.algos.VWAP"


# --------------------------------------------------------------------------------------------------------------------------- the factory
@pytest.mark.parametrize("spec,name", [("vwap", "vwap"), ("  twap ", "twap"), ("is:risk_aversion=0.01", "is"), ("pov:rate=0.15", "pov"), ("aim+vwap", "aim(vwap)"), ("pim+is:risk_aversion=0.003", "pim(is)"),
                                       ("target_cost+is:risk_aversion=0.003", "target_cost(is)"), ("liquidity_seeking:floor=0.2", "liquidity_seeking"), ("exp_trade:kappa=2.5", "exp_trade")])
def test_a_specification_builds_the_algorithm_it_names(spec, name):
    assert build_algo(spec).name == name


def test_settings_reach_the_algorithm_and_wrappers_wrap_it():
    algo = build_algo("aim+pov:rate=0.2")
    assert isinstance(algo, AIM) and algo.base.rate == 0.2
    assert isinstance(build_algo("target_cost+vwap"), TargetCost) and set(TACTICS) == {"aim", "pim", "target_cost"}
    assert build_algo("is:risk_aversion=0.01,alpha_bps=-20").alpha_bps == -20


@pytest.mark.parametrize("spec", ["nope", "aim+nope", "xx+vwap", "vwap:rate", "vwap:colour=red", "pov:rate=2", "liquidity_seeking:floor=0"])
def test_a_specification_that_cannot_be_built_is_refused_with_a_reason(spec):
    with pytest.raises(ValueError):
        build_algo(spec)


def test_run_order_returns_a_row_a_schedule_and_the_market_profile_per_algorithm():
    out = run_order(["twap", "vwap", "aim+vwap"], paths=60, scenario="mean_reverting")
    assert [r["algo"] for r in out["rows"]] == ["twap", "vwap", "aim+vwap"] and set(out["schedules"]) == {"twap", "vwap", "aim+vwap"}
    assert sum(out["schedules"]["twap"]) == pytest.approx(1.0) and sum(out["volume_profile"]) == pytest.approx(1.0) and out["intervals"] == 26
    assert out["order"]["participation"] == pytest.approx(0.1) and out["scenario"]["name"] == "mean_reverting" and out["style"]["name"] == "aggressive"
    assert run_order("vwap", paths=20)["rows"][0]["algo"] == "vwap"
    assert "prob_beat_target" in run_order("vwap", paths=20, target_bps=20.0)["rows"][0]


def test_an_alpha_override_changes_what_the_market_does_to_the_order():
    base = run_order("twap", paths=300, seed=2)["rows"][0]["shortfall_bps"]
    up = run_order("twap", paths=300, seed=2, alpha_bps=40.0)["rows"][0]["shortfall_bps"]
    assert up > base + 5.0                                                                                       # a buyer into a rising price pays more


@pytest.mark.parametrize("kwargs", [{"side": "hold"}, {"style": "sneaky"}, {"scenario": "apocalypse"}, {"paths": 5}, {"paths": 10_000}, {"algos": []}, {"algos": ["vwap"] * 9}, {"shares": -1.0}, {"adv": 0.0}])
def test_an_order_that_cannot_be_run_is_refused(kwargs):
    kwargs = {"algos": ["vwap"], **kwargs}
    with pytest.raises(ValueError):
        run_order(**kwargs)


def test_the_frontier_helper_returns_a_monotone_trade_off_and_the_two_benchmarks():
    out = efficient_frontier(points=9)
    cost, risk = [r["cost_bps"] for r in out["frontier"]], [r["risk_bps"] for r in out["frontier"]]
    assert len(cost) == 9 and np.all(np.diff(cost) >= -1e-9) and np.all(np.diff(risk) <= 1e-9)
    assert out["vwap"]["cost_bps"] <= out["twap"]["cost_bps"] + 1e-9 and ALL_ALGORITHMS["vwap"] is not None


# --------------------------------------------------------------------------------------------------------------------------- the command line
def test_the_algo_commands_run_and_say_what_they_show(capsys):
    assert cli.main(["algo", "list"]) == 0
    out = capsys.readouterr().out
    assert "Implementation shortfall" in out and "liquidity_seeking" in out and "aim, pim, target_cost" in out
    assert cli.main(["algo", "run", "--algos", "twap", "aim+vwap", "--paths", "60", "--schedule", "--target-bps", "25"]) == 0
    out = capsys.readouterr().out
    assert "shortfall_bps" in out and "aim+vwap" in out and "prob_beat_target" in out and "share of the order traded in each interval" in out
    assert cli.main(["algo", "frontier"]) == 0 and "VWAP: cost" in capsys.readouterr().out
    assert cli.main(["algo", "basket", "--size", "6"]) == 0
    out = capsys.readouterr().out
    assert "joint" in out and "minimum trading risk quantity" in out and "program-block" in out
    for sim in ("pairs", "etf", "rebate", "amm"):
        assert cli.main(["algo", "hft", "--sim", sim]) == 0
    assert "research simulators" in capsys.readouterr().out


def test_a_command_line_order_that_cannot_run_fails_with_a_message_not_a_traceback():
    with pytest.raises(ValueError):
        cli.main(["algo", "run", "--algos", "nope"])
