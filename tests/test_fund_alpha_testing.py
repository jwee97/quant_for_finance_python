"""How well an alpha is tested, against known truth: the power of a study (a closed form that a simulation agrees with), the interval around a Sharpe ratio (it covers the true value about as often as it
says), the premium regression (it recovers a planted premium, finds nothing in noise and sees through a premium that is a control in disguise), the information-lag sweep (a signal that lives in the days
after a filing fades when it is used later), the bars of a search (three named settings, each stricter than the last), the count of ideas tried across runs on one question, and the calibration of a
search on returns shuffled among the companies (where nothing can be found)."""

from __future__ import annotations

import math
import shutil

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.fundamentals import calibration as C
from src.fundamentals import discovery as D
from src.fundamentals import evaluate as E
from src.fundamentals import expr as X
from src.fundamentals import report as R
from src.fundamentals import transforms
from src.fundamentals.errors import FundamentalsError
from src.fundamentals.evaluate import holding_returns
from src.fundamentals.research import ResearchContext, UniverseSpec
from src.fundamentals.service import FundamentalsService
from src.fundamentals.store import CompanyProfile, FundamentalsStore, PeriodStatement
from src.fundamentals.workbench import Workbench

PLANTED = {"delta_roic": 0.011, "fcf_yield": 0.0068, "asset_growth": 0.0045}
Z95 = 1.959964


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    svc = FundamentalsService(tmp_path_factory.mktemp("alpha_testing"))
    svc.load_simulated(n_companies=100, start_year=2009, end_year=2018, seed=13, planted=PLANTED)
    with svc.store.db.transaction() as conn:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    return svc, ResearchContext(svc.store, UniverseSpec())


def fresh(world, tmp_path, name="copy") -> D.Discovery:
    """A discovery engine with an empty library over a copy of the loaded database, so that one test's graveyard is not another's."""
    svc, ctx = world
    target = tmp_path / f"{name}.db"
    shutil.copy(svc.store.db.path, target)
    return D.Discovery(FundamentalsStore(target), ctx)


def spearman_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    rx, ry = stats.rankdata(x, axis=1), stats.rankdata(y, axis=1)
    rx, ry = rx - rx.mean(axis=1, keepdims=True), ry - ry.mean(axis=1, keepdims=True)
    return (rx * ry).sum(axis=1) / np.sqrt((rx ** 2).sum(axis=1) * (ry ** 2).sum(axis=1))


# ------------------------------------------------------------------------------------------------------------------------ the power of a study
def test_the_smallest_effect_a_study_can_see_is_its_closed_form_and_a_simulation_agrees():
    n, t = 101, 100
    d = E.detectable_effect(n, t, 21, 21)
    se = 0.1 / math.sqrt(t)                                                           # 1 / sqrt(N - 1) per date, over sqrt(T)
    assert d["available"] and d["null_sd_per_date"] == pytest.approx(0.1) and d["se_of_mean_ic"] == pytest.approx(se)
    assert d["mde_ic"] == pytest.approx((Z95 + stats.norm.ppf(0.8)) * se) and d["effective_dates"] == t and d["adequate"] is True
    assert E.detectable_effect(n, t, 63, 21)["se_of_mean_ic"] == pytest.approx(se * math.sqrt(3))        # a quarterly horizon sampled monthly shares each window with two neighbours
    assert E.detectable_effect(101, 100, 21, 21, ic_sd=0.2)["se_of_mean_ic"] == pytest.approx(0.02)       # a measured spread replaces the null one
    for row in d["table"]:
        assert row["expected_t"] == pytest.approx(row["ic"] / se) and row["power_t2"] >= row["power_t3"]
    at_mde = stats.norm.cdf(d["mde_ic"] / se - Z95) + stats.norm.cdf(-d["mde_ic"] / se - Z95)
    assert at_mde == pytest.approx(0.8, abs=0.001)
    assert E.detectable_effect(3, 100, 21, 21)["available"] is False and E.detectable_effect(50, 1, 21, 21)["available"] is False
    small = E.detectable_effect(20, 60, 21, 21)
    assert small["adequate"] is False and "Add companies" in small["reading"]

    rng = np.random.default_rng(11)
    reps = 400
    null = spearman_rows(rng.standard_normal((20_000, n)), rng.standard_normal((20_000, n)))
    assert null.std() == pytest.approx(1 / math.sqrt(n - 1), rel=0.03)                # the spread of a rank correlation of nothing
    for target in (0.02, d["mde_ic"], 0.04):
        rho = 2 * math.sin(math.pi * target / 6)                                      # the Pearson correlation whose Spearman is the target
        x = rng.standard_normal((reps * t, n))
        y = rho * x + math.sqrt(1 - rho ** 2) * rng.standard_normal((reps * t, n))
        ic = spearman_rows(x, y).reshape(reps, t)
        mean = ic.mean()
        seen = (np.abs(ic.mean(axis=1) / (ic.std(axis=1, ddof=1) / math.sqrt(t))) > Z95).mean()
        predicted = stats.norm.cdf(mean / se - Z95) + stats.norm.cdf(-mean / se - Z95)
        assert seen == pytest.approx(predicted, abs=0.07), target                     # four hundred studies of a true effect: the share that reach |t| > 1.96 is the share the formula gives


def test_a_sharpe_interval_covers_the_true_ratio_about_as_often_as_it_says():
    sd = 0.04
    mu = 1.0 / math.sqrt(12) * sd                                                     # a true annual Sharpe ratio of 1
    scale = sd / math.sqrt(4 / 2)

    def coverage(draw, seed, reps=300, n=120):
        rng = np.random.default_rng(seed)
        hit = 0
        for _ in range(reps):
            lo, hi = E.sharpe_interval(pd.Series(draw(rng, n)), 12)
            hit += lo <= 1.0 <= hi
        return hit / reps

    assert 0.92 <= coverage(lambda r, n: r.normal(mu, sd, n), 1) <= 0.99
    assert coverage(lambda r, n: mu + scale * r.standard_t(4, n), 2) >= 0.88          # fat tails widen the interval: it still covers
    short, long = (E.sharpe_interval(pd.Series(np.random.default_rng(0).normal(mu, sd, n)), 12) for n in (60, 240))
    assert (short[1] - short[0]) > 1.6 * (long[1] - long[0]) and short[0] < 1.0 < short[1]
    assert all(math.isnan(x) for x in E.sharpe_interval(pd.Series([0.01] * 5), 12)) and all(math.isnan(x) for x in E.sharpe_interval(pd.Series([0.01] * 30), 12))


# ------------------------------------------------------------------------------------------------------------------------ the premium
def _panel(ctx, a):
    return pd.DataFrame(a, index=ctx.dates, columns=ctx.ids)


def _z(df):
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)


def test_the_premium_regression_recovers_a_planted_premium_finds_none_in_noise_and_sees_through_a_control_in_disguise(world):
    ctx = world[1]
    rng = np.random.default_rng(3)
    shape = (len(ctx.dates), ctx.n)
    size_panel = E.control_panels(ctx, ["size"])["size"]
    size = _z(size_panel)
    signal = _panel(ctx, rng.standard_normal(shape))
    h, step = 21, 21

    planted = E.fama_macbeth(ctx, signal, 0.01 * _z(signal) + 0.08 * _panel(ctx, rng.standard_normal(shape)), {}, h, step)
    got = planted["with_controls"]
    assert 0.0075 < got["premium"] < 0.0105 and got["t_stat"] > 6 and got["ci95"][0] < got["premium"] < got["ci95"][1] and "survives" not in planted["reading"]     # clipped tails shade the slope a little low
    assert planted["controls"] == [] and planted["with_controls"]["premium"] == planted["without_controls"]["premium"]

    nothing = E.fama_macbeth(ctx, signal, 0.08 * _panel(ctx, rng.standard_normal(shape)), {"size": size_panel}, h, step)
    assert abs(nothing["with_controls"]["t_stat"]) < 2.6 and nothing["with_controls"]["ci95"][0] < 0 < nothing["with_controls"]["ci95"][1]
    false_alarms = sum(abs(E.fama_macbeth(ctx, _panel(ctx, rng.standard_normal(shape)), 0.08 * _panel(ctx, rng.standard_normal(shape)), {"size": size_panel}, h, step)["with_controls"]["t_stat"]) > Z95
                       for _ in range(40))
    assert false_alarms <= 8                                                          # about one noise signal in twenty looks real; eight of forty would be a broken error

    disguised = size + 0.5 * _panel(ctx, rng.standard_normal(shape))                  # a signal that is mostly size, in a world where size is what pays
    pays_for_size = 0.02 * size + 0.08 * _panel(ctx, rng.standard_normal(shape))
    fm = E.fama_macbeth(ctx, disguised, pays_for_size, {"size": size_panel}, h, step)
    assert fm["without_controls"]["t_stat"] > 10 and abs(fm["with_controls"]["t_stat"]) < 2.6 and fm["explained_by_controls"]["t_stat"] > 5
    assert "does not survive the controls" in fm["reading"] and fm["average_companies"] > 80

    own = 0.02 * size + 0.015 * _z(signal) + 0.08 * _panel(ctx, rng.standard_normal(shape))
    fm = E.fama_macbeth(ctx, signal, own, {"size": size_panel}, h, step)
    assert fm["with_controls"]["t_stat"] > 10 and "survives the controls" in fm["reading"] and fm["with_controls"]["premium"] == pytest.approx(fm["without_controls"]["premium"], rel=0.2)


def test_two_versions_of_a_premium_are_compared_by_their_difference_and_not_by_which_one_is_significant():
    base = {"controls": ["size", "beta"], "with_controls": {"premium": 0.005, "t_stat": 1.4}, "without_controls": {"premium": 0.008, "t_stat": 2.2}, "explained_by_controls": {"premium": 0.003, "t_stat": 1.1}}
    noisy = E.premium_reading(base)
    assert "cannot say whether the controls explain it" in noisy and "does not survive" not in noisy                    # clear on its own, not clear with controls, but the two differ by no more than noise
    assert "does not survive the controls" in E.premium_reading({**base, "explained_by_controls": {"premium": 0.003, "t_stat": 3.1}})
    assert "survives the controls" in E.premium_reading({**base, "with_controls": {"premium": 0.007, "t_stat": 2.6}})
    assert "account for about 38%" in E.premium_reading({**base, "with_controls": {"premium": 0.005, "t_stat": 2.6}, "explained_by_controls": {"premium": 0.003, "t_stat": 3.1}})
    assert E.premium_reading({**base, "controls": []}).endswith("(t = 1.4).") and "Neither version" in E.premium_reading({**base, "without_controls": {"premium": 0.002, "t_stat": 0.5}})
    assert E.premium_reading({"controls": [], "with_controls": {"n_dates": 3}, "without_controls": {"n_dates": 3}}) == "Too few dates to run the regression."


def test_a_study_carries_its_power_its_intervals_and_on_request_its_premium(world):
    ctx = world[1]
    plain = E.evaluate_factor(ctx, E.EvalSpec(factor="fcf_yield", breakdowns=(), premium=False))
    assert plain["premium"] is None and plain["power"]["available"] and plain["power"]["companies"] > 80
    assert E.evaluate_factor(ctx, E.EvalSpec(factor="fcf_yield", breakdowns=()))["premium"]["controls"] == list(E.CONTROLS)          # a single study holds size, beta, volatility and momentum fixed unless told otherwise
    ic, sharpe = plain["ic"], plain["portfolio"]["net"]
    assert ic["ci95"][0] < ic["mean_ic"] < ic["ci95"][1] and ic["se"] == pytest.approx((ic["ci95"][1] - ic["ci95"][0]) / (2 * 1.96))
    assert sharpe["sharpe_ci95"][0] < sharpe["sharpe"] < sharpe["sharpe_ci95"][1]
    with_premium = E.evaluate_factor(ctx, E.EvalSpec(factor="fcf_yield", breakdowns=(), premium=True, controls=("size", "momentum")))
    fm = with_premium["premium"]
    assert fm["controls"] == ["size", "momentum"] and fm["with_controls"]["premium"] > 0 and fm["with_controls"]["t_stat"] > 3 and fm["without_controls"]["t_stat"] > 3 and fm["series"] and "premium" in fm["reading"]
    bare = E.evaluate_factor(ctx, E.EvalSpec(factor="fcf_yield", breakdowns=(), premium=True, controls=()))["premium"]["without_controls"]
    assert bare["ci95"][0] < PLANTED["fcf_yield"] < bare["ci95"][1]                   # the world pays 0.68% a month for a standard deviation of free-cash-flow yield: the regression finds the number that was planted
    only_roic = E.evaluate_factor(ctx, E.EvalSpec(factor="fcf_yield", breakdowns=(), premium=True, controls=("roic",)))["premium"]
    assert only_roic["controls"] == ["roic"]                                          # any factor of the library can be held fixed
    own = E.evaluate_factor(ctx, E.EvalSpec(factor="fcf_yield", breakdowns=(), premium=True, controls=("fcf_yield", "size")))["premium"]
    assert own["controls"] == ["size"]                                                # a factor is never held fixed against itself
    small = ResearchContext(ctx.store, UniverseSpec(), security_ids=ctx.ids[:25])
    tiny = E.evaluate_factor(small, E.EvalSpec(factor="fcf_yield", breakdowns=(), min_assets=10))
    assert tiny["power"]["adequate"] is False and any("this study is small" in w for w in tiny["warnings"])
    with pytest.raises(FundamentalsError, match="unknown control"):
        E.control_panels(ctx, ["vibes"])


# ------------------------------------------------------------------------------------------------------------------------ the information lag
def _post_filing_world(tmp_path, n=100, seed=8):
    """A world where the market takes a month to digest each year's figure, in the month after the first month-end that follows the filing (filed on 15 March, priced during April): a signal built from
    the figure predicts that month only if it is used as soon as it is public, and not when it is used a month, two months or three months later."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2011-01-03", "2019-12-31")
    store = FundamentalsStore(tmp_path / "post_filing.db")
    ids = [store.upsert_security(CompanyProfile("simulated", f"S{i}", f"S{i}", f"S{i}", exchange="NYSE", market="NYSE", country="US", currency="USD", sector="X")) for i in range(n)]
    drift = np.zeros((len(days), n))
    figures = {}
    for year in range(2010, 2019):
        z = rng.standard_normal(n)
        figures[year] = z
        drift[(days.year.to_numpy() == year + 1) & (days.month.to_numpy() == 4)] += 0.004 * z          # usable at the end of March, priced during April
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.012, size=(len(days), n)) + drift, axis=0))
    store.store_prices(pd.DataFrame({"security_id": np.tile(ids, len(days)), "date": np.repeat(days.to_numpy(), n), "adj_close": prices.ravel(), "close": prices.ravel(), "volume": 1e6}))
    for k, sid in enumerate(ids):
        store.store_financials(sid, "simulated", [PeriodStatement("FY", f"{y}-12-31", {"net_income": float(figures[y][k]), "revenue": 100.0}, fiscal_year=y, filing_date=f"{y + 1}-03-15", reporting_currency="USD")
                                                  for y in range(2010, 2019)])
    return store


def test_a_signal_that_lives_in_the_days_after_a_filing_fades_when_it_is_used_later(tmp_path, world):
    store = _post_filing_world(tmp_path)
    audit = E.look_ahead_audit(store, E.EvalSpec(factor="net_margin", breakdowns=(), min_assets=20), UniverseSpec())
    lag = {r["lag_days"]: r for r in audit["by_lag"]}
    assert list(lag) == [1, 30, 60, 90] and lag[1]["n_dates"] > 80
    assert lag[1]["mean_ic"] > 0.05 and lag[1]["t_stat"] > 3                          # used as soon as it is public, the figure predicts the month that follows
    assert all(abs(lag[d]["mean_ic"]) < 0.5 * lag[1]["mean_ic"] for d in (30, 60, 90))
    assert audit["fades_when_waiting"] is True and "falls to less than half" in audit["lag_verdict"] and audit["flattered"] is False
    steady = E.look_ahead_audit(world[0].store, E.EvalSpec(factor="fcf_yield", breakdowns=()), UniverseSpec(), lags=(1, 30, 90))
    assert [r["lag_days"] for r in steady["by_lag"]] == [1, 30, 90] and steady["fades_when_waiting"] is False and "not about being first" in steady["lag_verdict"]
    assert all(r["mean_ic"] > 0.03 for r in steady["by_lag"])                         # a slow, persistent fundamental is still there three months after the filing


# ------------------------------------------------------------------------------------------------------------------------ the bars of a search
def test_the_three_settings_are_each_stricter_than_the_last_and_a_named_bar_always_wins():
    loose, standard, strict = (D.DiscoverySpec(strictness=s).limits() for s in D.STRICTNESS)
    assert loose == D.Thresholds() and D.DEFAULT_STRICTNESS == "standard" and D.DiscoverySpec().strictness == "exploratory"
    assert (loose.t_ic, standard.t_ic, strict.t_ic) == (1.65, 2.0, 3.0) and (loose.max_q, standard.max_q, strict.max_q) == (0.25, 0.10, 0.05)
    for bar in ("min_coverage", "min_dates", "t_ic", "min_abs_ic", "positive_share", "min_icir", "best_year_t", "monotonicity", "ls_t", "ls_sharpe", "net_sharpe", "val_t", "oos_t"):
        a, b, c = getattr(loose, bar), getattr(standard, bar), getattr(strict, bar)
        assert a <= b <= c, bar                                                       # a higher bar is a stricter one
    for bar in ("max_turnover", "max_q", "max_correlation"):
        a, b, c = getattr(loose, bar), getattr(standard, bar), getattr(strict, bar)
        assert a >= b >= c, bar                                                       # a lower limit is a stricter one
    assert D.DiscoverySpec(strictness="strict", thresholds={"t_ic": 2.5}).limits().t_ic == 2.5 and D.DiscoverySpec(strictness="strict", thresholds={"t_ic": 2.5}).limits().max_q == 0.05
    with pytest.raises(FundamentalsError, match="strictness must be one of"):
        D.DiscoverySpec(strictness="reckless").limits()
    assert D.DiscoverySpec.from_dict({"n_candidates": 5}).strictness == "exploratory"       # a record from before the bars were named ran with the first-look bars


def test_the_bars_are_part_of_what_makes_two_results_the_same_but_only_when_they_are_not_the_first_look_ones():
    first = D.DiscoverySpec()
    assert first.conditions() == first.question() and "strictness" not in first.conditions() and "thresholds" not in first.conditions()      # a database from before keeps its identities
    standard = D.DiscoverySpec(strictness="standard")
    assert standard.conditions()["strictness"] == "standard" and standard.question() == first.question()
    assert D.DiscoverySpec(thresholds={"t_ic": 2.0}).conditions()["thresholds"] == {"t_ic": 2.0}
    from src.fundamentals.library import settings_hash
    digests = {settings_hash(s.conditions()) for s in (first, standard, D.DiscoverySpec(strictness="strict"), D.DiscoverySpec(thresholds={"t_ic": 2.0}))}
    assert len(digests) == 4 and settings_hash(standard.question()) == settings_hash(first.question())          # four sets of bars, one question


def test_the_workbench_starts_from_the_standard_bars_and_refuses_what_is_not_a_bar(world):
    wb = Workbench(service=world[0])
    assert wb.discovery_spec({}).strictness == "standard" and wb.discovery_spec({"strictness": "strict"}).strictness == "strict"
    for bad, text in (({"strictness": "wild"}, "strictness must be one of"), ({"thresholds": {"t_ic": "high"}}, "must be a number"), ({"thresholds": [1]}, "dictionary"), ({"thresholds": {"magic": 1}}, "unknown threshold")):
        with pytest.raises(FundamentalsError, match=text):
            wb.discovery_spec(bad)
    cat = wb.catalogue()["discovery"]
    assert cat["strictness"] == ["exploratory", "standard", "strict"] and cat["defaults"]["strictness"] == "standard" and cat["presets"]["strict"]["t_ic"] == 3.0 and cat["presets"]["exploratory"]["t_ic"] == 1.65


def test_a_stricter_setting_lets_fewer_formulas_through_the_first_screen_and_still_finds_the_planted_value_effect(world, tmp_path):
    found, passed = {}, {}
    for name in D.STRICTNESS:
        disc = fresh(world, tmp_path, name)
        out = disc.run(disc.create(D.DiscoverySpec(n_candidates=80, seed=3, evolve=0.0, strictness=name)))
        passed[name] = out["counts"]["fast_ic"]
        found[name] = {a["text"] for a in disc.library.list("Candidate")}
        assert out["strictness"] == name and out["limits"]["t_ic"] == D.PRESETS[name].get("t_ic", 1.65)
    assert passed["exploratory"] >= passed["standard"] >= passed["strict"] and passed["exploratory"] > passed["strict"]            # the same formulas, a higher bar at the first screen
    assert "div(fcf,market_cap)" in found["exploratory"] and "div(fcf,market_cap)" in found["standard"]
    assert "Bars: strict (first screen |t| of 3" in R.format_discovery(out, 3)


# ------------------------------------------------------------------------------------------------------------------------ the ideas tried across runs
def test_the_correction_for_the_number_tried_counts_every_idea_tried_on_the_question_across_runs(world, tmp_path):
    disc = fresh(world, tmp_path)
    first = disc.run(disc.create(D.DiscoverySpec(n_candidates=60, seed=3, evolve=0.0, strictness="exploratory", seeds=[])))                        # random formulas only
    second = disc.run(disc.create(D.DiscoverySpec(n_candidates=30, seed=4, evolve=0.0, strictness="exploratory", seeds=["div(fcf,market_cap)", "div(net_income,market_cap)"])))
    c1, c2 = first["counts"], second["counts"]
    assert c1["runs_in_family"] == 1 and c1["n_tested_family"] == c1["n_tested"]                                   # the first run on a question is alone on it
    assert c2["runs_in_family"] == 2 and c2["n_tested"] <= c2["n_tested_family"] <= c1["n_tested"] + c2["n_tested"] and c2["n_tested_family"] > c1["n_tested"]
    assert c1["family"] == c2["family"]
    pool = {}
    for run in (first["run_id"], second["run_id"]):
        for c in disc.candidates(run, limit=10_000):
            p = c["metrics"].get("p_train")
            if p is not None:
                pool[c["metrics"]["signal_key"]] = p                                                             # an idea tried twice counts once
    assert len(pool) == c2["n_tested_family"]
    q = dict(zip(pool, D.bh_qvalues(np.array(list(pool.values())))))
    scored = [c for c in disc.candidates(second["run_id"], limit=10_000) if c["metrics"].get("q_value") is not None]
    assert len(scored) >= 2 and all(c["metrics"]["q_value"] == pytest.approx(q[c["metrics"]["signal_key"]], abs=1e-7) for c in scored)       # the q-value is that of the whole pool, as if the formulas had been searched at once (stored to eight places)
    assert all(c["metrics"]["q_value_run"] is not None for c in disc.candidates(second["run_id"], status="candidate", limit=100))
    top = disc.library.list("Candidate")
    assert top and all(a["metrics"]["n_tested"] >= a["metrics"]["n_tested_run"] and a["metrics"]["runs_on_question"] in (1, 2) for a in top)
    later = [a for a in top if a["metrics"]["run_id"] == second["run_id"]]
    assert all(a["metrics"]["n_tested"] == c2["n_tested_family"] and a["metrics"]["n_tested_run"] == c2["n_tested"] for a in later)


def test_another_question_or_the_same_question_with_other_bars_is_counted_apart_or_together(world, tmp_path):
    disc = fresh(world, tmp_path)
    base = D.DiscoverySpec(n_candidates=40, seed=3, evolve=0.0, strictness="exploratory")
    one = disc.run(disc.create(base))
    other_bars = disc.run(disc.create(D.DiscoverySpec(n_candidates=40, seed=3, evolve=0.0, strictness="standard")))
    assert other_bars["counts"]["family"] == one["counts"]["family"] and other_bars["counts"]["runs_in_family"] == 2
    assert other_bars["counts"]["tested"] > 0                                                                   # the bars are part of the identity of a verdict: nothing was skipped as already seen
    another_horizon = disc.run(disc.create(D.DiscoverySpec(n_candidates=40, seed=3, evolve=0.0, horizon="3m")))
    assert another_horizon["counts"]["family"] != one["counts"]["family"] and another_horizon["counts"]["runs_in_family"] == 1
    again = disc.run(disc.create(base))
    assert again["counts"]["skipped_seen"] + again["counts"]["duplicate_idea"] >= 30 and again["counts"]["n_tested_family"] == one["counts"]["n_tested_family"] + other_bars["counts"]["tested"] - overlap(disc, one, other_bars)


def overlap(disc, a, b) -> int:
    ka = {c["metrics"]["signal_key"] for c in disc.candidates(a["run_id"], limit=10_000) if c["metrics"].get("p_train") is not None}
    kb = {c["metrics"]["signal_key"] for c in disc.candidates(b["run_id"], limit=10_000) if c["metrics"].get("p_train") is not None}
    return len(ka & kb)


# ------------------------------------------------------------------------------------------------------------------------ what a search could find
def test_the_preflight_says_what_a_search_could_find_and_agrees_with_the_formula(world):
    wb = Workbench(service=world[0])
    ctx = wb.context({}, "M", 1)
    rows = {}
    for name in D.STRICTNESS:
        spec = wb.discovery_spec({"strictness": name, "n_candidates": 200})
        pf = wb.preflight(ctx, spec)
        rows[name] = {r["ic"]: r for r in pf["table"]}
        limits = D.DiscoverySpec(strictness=name).limits()
        se = {k: pf["power"][k]["se_of_mean_ic"] for k in ("train", "validation", "oos")}
        r = rows[name][0.03]
        assert r["first_screen"] == pytest.approx(stats.norm.cdf(0.03 / se["train"] - limits.t_ic) + stats.norm.cdf(-0.03 / se["train"] - limits.t_ic))
        assert r["found"] == pytest.approx(r["first_screen"] * r["validation"] * r["locked_test"])
        assert pf["luck_first_screen"] == pytest.approx(200 * 2 * stats.norm.sf(limits.t_ic)) and pf["blocks"]["dates"]["train"] > pf["blocks"]["dates"]["oos"] > 12
    for ic in E.TYPICAL_IC:
        assert rows["exploratory"][ic]["found"] >= rows["standard"][ic]["found"] >= rows["strict"][ic]["found"]               # a stricter search finds less of what is real
    found = [rows["standard"][ic]["found"] for ic in E.TYPICAL_IC]
    assert found == sorted(found) and found[-1] > 0.9 and found[0] < 0.05
    tiny = ResearchContext(world[0].store, UniverseSpec(), security_ids=world[1].ids[:30])
    weak = wb.preflight(tiny, wb.discovery_spec({"n_candidates": 50}))
    assert weak["adequate"] is False and "underpowered" in weak["reading"]
    with pytest.raises(FundamentalsError, match="too little history"):
        wb.preflight(ctx, D.DiscoverySpec(min_assets=10_000))


def test_the_size_of_a_study_is_read_from_the_database_before_anything_is_built(world, tmp_path):
    empty = Workbench(tmp_path)
    s = empty.study_size()
    assert s["available"] is False and "at least 20 companies" in s["reading"]
    wb = Workbench(service=world[0])
    size = wb.study_size()
    assert size["available"] and 90 <= size["companies"] <= 100 and 100 <= size["dates"] <= 125 and size["mde_ic"] == pytest.approx(E.detectable_effect(size["companies"], size["dates"], 21, 21)["mde_ic"])
    assert wb.study_size() is wb.study_size() and wb.overview()["study_size"]["dates"] == size["dates"]                    # kept until the database changes
    assert abs(size["companies"] - world[1].n) <= 5                                                                       # close to what the study then finds


# ------------------------------------------------------------------------------------------------------------------------ the calibration on shuffled returns
def test_the_shuffle_deals_each_dates_returns_to_other_companies_and_leaves_everything_else_alone(world):
    ctx = world[1]
    before = holding_returns(ctx).copy()
    forward = ctx.forward("1m").copy()
    placebo = C.PlaceboContext(ctx, 5)
    for horizon in ("1m", "3m"):
        real, dealt = ctx.forward(horizon), placebo.forward(horizon)
        assert (real.notna().to_numpy() == dealt.notna().to_numpy()).all()                                           # the same companies have a return on the same dates
        assert np.allclose(np.sort(real.to_numpy(float), axis=1), np.sort(dealt.to_numpy(float), axis=1), equal_nan=True)         # the same returns, on every date, held by other companies
        assert not np.allclose(real.to_numpy(float), dealt.to_numpy(float), equal_nan=True)
    held = holding_returns(placebo)
    assert np.allclose(np.sort(before.to_numpy(float), axis=1), np.sort(held.to_numpy(float), axis=1), equal_nan=True) and not np.allclose(before.to_numpy(float), held.to_numpy(float), equal_nan=True)
    assert holding_returns(ctx).equals(before) and ctx.forward("1m").equals(forward)                                 # the context being studied is untouched
    assert placebo.dates.equals(ctx.dates) and placebo.panel("revenue", "fy") is not None and placebo.market_cap().equals(ctx.market_cap())
    once = C.PlaceboContext(ctx, 5).forward("1m")
    assert once.equals(placebo.forward("1m")) and not once.equals(C.PlaceboContext(ctx, 6).forward("1m"))             # the same seed deals the same cards
    both = [C.PlaceboContext(ctx, 9).forward("1m"), holding_returns(C.PlaceboContext(ctx, 9))]
    src = [ctx.forward("1m"), before]
    seen = [pd.Series([a.iloc[i].corr(b.iloc[i], method="spearman") for i in range(5, 100)]).mean() for a, b in ((both[0], both[1]), (src[0], src[1]))]
    assert seen[0] == pytest.approx(seen[1], abs=0.05)                                                                # one relabelling for every window that starts on a date


def test_a_signal_that_works_in_the_real_data_does_nothing_in_the_shuffled_data(world):
    ctx = world[1]
    spec = E.EvalSpec(factor="fcf_yield", breakdowns=())
    real = E.evaluate_factor(ctx, spec)["ic"]
    ics = [E.evaluate_factor(C.PlaceboContext(ctx, seed), spec)["ic"] for seed in range(4)]
    assert real["t_stat"] > 4 and all(abs(i["mean_ic"]) < 0.02 and abs(i["t_stat"]) < 3 for i in ics)
    assert abs(np.mean([i["mean_ic"] for i in ics])) < 0.012


def test_the_exact_interval_for_a_count_of_rare_events():
    lo, hi = C.poisson_interval(0)
    assert lo == 0.0 and hi == pytest.approx(3.689, abs=0.001)
    lo, hi = C.poisson_interval(10)
    assert (lo, hi) == pytest.approx((4.795, 18.390), abs=0.002)
    assert C.poisson_interval(10, exposure=2.0) == pytest.approx([2.3975, 9.195], abs=0.002)


def _runs(t_values, **each):
    base = {"seed": 1, "status": "finished", "formulas": 200, "tested": len(t_values), "funnel": dict.fromkeys(D.STAGES, 0), "survivors": 0, "finalists": 0, "candidates": 0, "failed_locked_test": 0,
            "min_q": 0.4, "t": list(map(float, t_values))}
    return [{**base, **each}]


def test_the_spread_of_the_shuffled_t_statistics_says_whether_the_standard_errors_can_be_believed():
    rng = np.random.default_rng(0)
    spec = D.DiscoverySpec(strictness="standard")
    right = C.summarise(_runs(rng.standard_normal(600)), spec)
    assert right["summary"]["t_calibration"] == "close" and 0.9 < right["summary"]["t_sd"] < 1.1 and "about right" in right["reading"]
    assert right["summary"]["tail"]["1.65"]["share"] == pytest.approx(C.EXPECTED_TAIL[1.65], abs=0.03) and C.EXPECTED_TAIL[1.65] == pytest.approx(0.0989, abs=0.0005)
    generous = C.summarise(_runs(1.5 * rng.standard_normal(600)), spec)
    assert generous["summary"]["t_calibration"] == "generous" and "too generous" in generous["reading"] and generous["summary"]["t_bar_5pct"] == pytest.approx(1.5 * Z95, rel=0.15)
    careful = C.summarise(_runs(0.6 * rng.standard_normal(600)), spec)
    assert careful["summary"]["t_calibration"] == "careful" and "careful" in careful["reading"]
    few = C.summarise(_runs(1.4 * rng.standard_normal(80)), spec)
    assert few["summary"]["t_calibration"] == "unknown" and "Too few formulas" in few["reading"]                      # eighty formulas cannot say how wide a spread is
    mild = C.summarise(_runs(1.18 * rng.standard_normal(900)), spec)
    assert "a little generous" in mild["reading"] and C.summarise([], spec)["reading"] == "No placebo run finished."


def test_what_luck_lets_through_is_set_beside_what_a_real_run_let_through():
    spec = D.DiscoverySpec(strictness="standard")
    runs = _runs(np.zeros(200), survivors=2) + _runs(np.zeros(200), survivors=1)
    out = C.summarise(runs, spec, {"tested": 200, "survivors": 12, "candidates": 5})["summary"]
    assert out["survivors_mean"] == pytest.approx(1.5) and out["luck_share"] == pytest.approx(1.5 / 12) and out["luck_share_interval"][0] < out["luck_share"] < out["luck_share_interval"][1]
    nothing = C.summarise(_runs(np.zeros(200), survivors=6), spec, {"tested": 200, "survivors": 5, "candidates": 0})
    assert nothing["summary"]["luck_share"] == 1.0 and "no more than luck" in nothing["reading"]
    assert "let nothing through" in C.summarise(_runs(np.zeros(200)), spec, {"tested": 200, "survivors": 0, "candidates": 0})["reading"]


def _state(wb) -> tuple:
    """What a placebo must not change: the library and graveyard, the runs, and the count of looks at the locked dates."""
    with wb.store.db.transaction() as conn:
        looks = dict(conn.execute("SELECT key, value FROM meta WHERE key LIKE 'oos_looks:%'").fetchall())
    return wb.library.counts(), len(wb.discovery().runs(50)), looks


def test_a_search_on_shuffled_returns_finds_nothing_and_leaves_the_database_alone(world, tmp_path):
    target = tmp_path / "studied.db"
    shutil.copy(world[0].store.db.path, target)
    wb = Workbench(service=FundamentalsService(tmp_path, db_path=target))
    real = D.Discovery(wb.store, wb.context({}, "M", 1))
    spec = wb.discovery_spec({"n_candidates": 80, "seed": 5, "evolve": 0.0, "strictness": "exploratory"})
    outcome = real.run(real.create(spec))
    before = _state(wb)
    lines = []
    result = wb.placebo(spec, 2, {"tested": outcome["counts"]["tested"], "survivors": outcome["counts"]["survivors"], "candidates": outcome["counts"]["candidates"]}, lines.append)
    assert result["complete"] and result["summary"]["runs"] == 2 and result["summary"]["candidates_max"] == 0 and result["summary"]["tested"] > 80
    assert result["experiment_id"].startswith("P") and wb.experiments("placebo", 5)[0]["experiment_id"] == result["experiment_id"] and any("shuffled returns 2 of 2" in x for x in lines)
    assert _state(wb) == before                                                                                  # nothing was written to the library, the graveyard or the record of looks at the locked dates
    assert outcome["counts"]["survivors"] >= 4 and result["summary"]["luck_share"] < 0.5 and "expected to be luck" in result["reading"]
    assert result["strictness"] == "exploratory" and set(result["summary"]["tail"]) == {"1.65", "2.0", "3.0"} and sum(result["t_histogram"]["counts"]) > 0
    stopped = wb.placebo(spec, 3, None, None, lambda: True)
    assert stopped["complete"] is False and stopped["reading"] == "No placebo run finished." and "experiment_id" not in stopped
    with pytest.raises(FundamentalsError, match="between 1 and 5"):
        wb.placebo(spec, 9)


# ------------------------------------------------------------------------------------------------------------------------ the simulated world's daily prices
@pytest.fixture(scope="module")
def sim():
    from src.fundamentals.providers.simulated import SimConfig, SimulatedWorld
    world = SimulatedWorld(SimConfig(n_companies=80, start_year=2010, end_year=2017, seed=3))
    frame = world.daily_prices()
    adj = frame.pivot(index="date", columns="ticker", values="adj_close")
    return world, adj.reindex(columns=list(world.ticker))


def test_the_daily_prices_end_every_month_where_the_monthly_world_says_they_do(sim):
    world, adj = sim
    month_end = adj.groupby(adj.index.to_period("M")).last()
    logp = np.log(month_end.to_numpy())
    # the first month starts from 100, every later month from the one before: each month's return is the one the monthly world drew
    first = logp[0] - np.log(100.0)
    both = np.isfinite(first) & np.isfinite(world.logret[1])
    assert both.sum() > 60 and np.allclose(first[both], world.logret[1][both], atol=1e-9)
    step = np.diff(logp, axis=0)
    expected = world.logret[2:len(step) + 2]
    ok = np.isfinite(step) & np.isfinite(expected)
    assert ok.sum() > 3000 and np.allclose(step[ok], expected[ok], atol=1e-9)


def test_the_daily_prices_are_as_volatile_as_the_monthly_ones_say_and_not_a_straight_line(sim):
    world, adj = sim
    daily = adj.pct_change(fill_method=None)
    market = daily.mean(axis=1).dropna()
    vol, sharpe = market.std() * math.sqrt(252), market.mean() / market.std() * math.sqrt(252)
    assert 0.09 < vol < 0.25 and abs(sharpe) < 2.0                                    # a diversified portfolio of shares moves by a tenth or two a year (it was 4% with a Sharpe ratio of 3 when only the month ends were random)
    month = adj.groupby(adj.index.to_period("M")).last().pct_change(fill_method=None).mean(axis=1).dropna()
    ratio = market.var() * 21.0 / month.var()
    assert 0.5 < ratio < 1.3                                                           # a day's variance times the days in a month is about a month's (it was 0.06 before; the slow parts of a return, spread evenly over the days, add less than their share)
    one = daily.iloc[:, 0].dropna()
    assert one.std() > 0.005 and one.abs().max() < 0.6                                 # a share moves every day, and not by half its value in one


def test_a_beta_measured_on_daily_returns_is_the_beta_the_world_gave_the_company(sim):
    world, adj = sim
    daily = adj.pct_change(fill_method=None)
    market = daily.mean(axis=1)
    measured = np.array([daily[t].cov(market) / market.var() for t in adj.columns])
    assert np.corrcoef(measured, world.beta)[0, 1] > 0.7 and abs(measured.mean() - world.beta.mean()) < 0.15


# ------------------------------------------------------------------------------------------------------------------------ exact values and edges the first planted-bug audit found nothing checking
def _bucketed(ctx, levels: int) -> pd.DataFrame:
    """A signal that takes exactly ``levels`` different values on every date: the market capitalisation cut into that many equal groups."""
    cap = ctx.market_cap_usd()
    return np.ceil(cap.rank(axis=1, pct=True) * levels).where(cap.notna())


def test_the_spread_between_the_top_and_the_bottom_bucket_is_their_difference_whichever_way_the_signal_points(world):
    ctx = world[1]
    spec = E.EvalSpec(breakdowns=(), premium=False)
    knows = _bucketed(ctx, 5) * 0.0 + ctx.forward("1m").rank(axis=1, pct=True)                                   # a signal that knows what comes next
    up, down = E.evaluate(ctx, knows, spec, "foresight"), E.evaluate(ctx, -knows, spec, "hindsight")
    for result, sign in ((up, 1), (down, -1)):
        mean = result["quantiles"]["mean_forward_return"]
        assert result["quantiles"]["spread_top_minus_bottom"] == pytest.approx(mean["5"] - mean["1"]) and np.sign(result["quantiles"]["spread_top_minus_bottom"]) == sign
        assert result["quantiles"]["spread_top_minus_bottom"] * sign > 0.05


def test_a_signal_that_takes_few_values_is_warned_about_at_the_edges_the_study_states(world):
    ctx = world[1]
    spec = E.EvalSpec(breakdowns=(), premium=False)
    fifteen, fourteen = (E.evaluate(ctx, _bucketed(ctx, k), spec, "levels")["warnings"] for k in (15, 14))
    assert not any("distinct values" in w for w in fifteen) and any("only 14 distinct values" in w for w in fourteen)         # a typical date with fifteen different values is enough
    two = E.evaluate(ctx, _bucketed(ctx, 2), spec, "two levels")["warnings"]
    assert any("only 2 of 5 buckets are ever populated" in w for w in two)
    five = E.evaluate(ctx, _bucketed(ctx, 5), spec, "five levels")["warnings"]
    assert not any("buckets are ever populated" in w for w in five)                                              # five values fill five buckets exactly
    half_empty = _bucketed(ctx, 20)
    half_empty.iloc[: int(len(half_empty) * 0.6)] = np.nan                                                         # no figure at all on most dates: they do not count as dates with no values
    assert not any("distinct values" in w for w in E.evaluate(ctx, half_empty, spec, "half empty")["warnings"])


def test_a_direction_chosen_on_the_data_flips_a_factor_that_works_the_wrong_way_round_and_says_so(world):
    ctx = world[1]
    raw = {}
    for name in ("current_ratio", "debt_to_equity", "operating_margin", "revenue_growth", "asset_growth", "roic"):
        raw[name] = E.evaluate_factor(ctx, E.EvalSpec(factor=name, direction=1, breakdowns=(), premium=False))["ic"]["mean_ic"]
    wrong = min(raw, key=raw.get)
    assert raw[wrong] < -0.002                                                                                   # the world has a factor whose plain sign earned nothing but a loss
    auto = E.evaluate_factor(ctx, E.EvalSpec(factor=wrong, direction="auto", breakdowns=(), premium=False))
    assert auto["ic"]["mean_ic"] == pytest.approx(-raw[wrong]) and any("sign was flipped" in w for w in auto["warnings"])
    right = max(raw, key=raw.get)
    kept = E.evaluate_factor(ctx, E.EvalSpec(factor=right, direction="auto", breakdowns=(), premium=False))
    assert raw[right] > 0 and kept["ic"]["mean_ic"] == pytest.approx(raw[right]) and any("kept the factor's own sign" in w for w in kept["warnings"])


def test_a_figure_known_before_it_was_public_is_called_flattering_by_the_audit_and_a_slow_signal_is_not(tmp_path):
    from tests.test_fund_evaluate import _anticipation_world
    store = _anticipation_world(tmp_path)
    audit = E.look_ahead_audit(store, E.EvalSpec(factor="net_margin", breakdowns=(), min_assets=20), UniverseSpec(), lags=(1,))
    assert audit["flattered"] is True and audit["naive"]["mean_ic"] > 0.04 > audit["honest"]["mean_ic"] and "flatters this factor a great deal" in audit["verdict"]
    assert audit["naive_minus_honest_ic"] == pytest.approx(audit["naive"]["mean_ic"] - audit["honest"]["mean_ic"])


def test_the_blocks_of_dates_must_hold_twelve_each_and_a_formula_at_a_bar_has_met_it(world):
    ctx = world[1]
    spec = D.DiscoverySpec()
    n = int((ctx.forward(spec.horizon).notna().sum(axis=1).to_numpy() >= spec.min_assets).sum())
    embargo = 1                                                                                                   # a month's horizon on a monthly grid
    total = (n - embargo - 12 + 0.5) / n                                                                          # the share of the dates that leaves the locked block twelve dates
    exactly = D.splits(ctx, D.DiscoverySpec(train_fraction=0.55, validation_fraction=total - 0.55))
    assert exactly["dates"]["oos"] == 12
    with pytest.raises(FundamentalsError, match="too little history"):
        D.splits(ctx, D.DiscoverySpec(train_fraction=0.55, validation_fraction=total - 0.55 + 1.5 / n))        # one date fewer
    blocks = D.splits(ctx, spec)
    formula = X.parse("div(fcf,market_cap)")
    first = D.Screen(ctx, spec, blocks, D.BoundedCache()).run(formula)
    t, ic = abs(first["metrics"]["t_train"]), abs(first["metrics"]["ic_raw"])
    assert first["status"] == "survivor"
    at_bar = D.Screen(ctx, D.DiscoverySpec(thresholds={"t_ic": t, "min_abs_ic": ic}), blocks, D.BoundedCache()).run(formula)
    assert at_bar["status"] == "survivor"                                                                         # a t-statistic and an IC exactly at their bars are met
    for bar in ({"t_ic": t + 1e-9}, {"min_abs_ic": ic + 1e-12}):
        over = D.Screen(ctx, D.DiscoverySpec(thresholds=bar), blocks, D.BoundedCache()).run(formula)
        assert over["status"] == "rejected" and over["stage"] == 1 and "too weak on the training dates" in over["reason"]
    screen = D.Screen(ctx, spec, blocks, D.BoundedCache())
    lock = screen.locked_test(formula, 1.0)
    assert lock["passed"] is True
    exact = D.Screen(ctx, D.DiscoverySpec(thresholds={"oos_t": lock["oos_t"]}), blocks, D.BoundedCache()).locked_test(formula, 1.0)
    above = D.Screen(ctx, D.DiscoverySpec(thresholds={"oos_t": lock["oos_t"] + 1e-9}), blocks, D.BoundedCache()).locked_test(formula, 1.0)
    assert exact["passed"] is True and above["passed"] is False                                                   # the locked test is passed at exactly its bar and not just under it


def test_a_run_counts_the_formulas_it_generated_and_the_ones_it_skipped_as_known(world, tmp_path):
    disc = fresh(world, tmp_path)
    spec = D.DiscoverySpec(n_candidates=30, seed=3, evolve=0.0)
    first = disc.run(disc.create(spec))
    assert first["counts"]["generated"] == 30 and first["counts"]["next_index"] == 30 and first["counts"]["tested"] + first["counts"]["skipped_seen"] + first["counts"]["duplicate_idea"] <= 30
    again = disc.run(disc.create(spec))
    assert again["counts"]["generated"] == 30 and again["counts"]["skipped_seen"] >= 20                           # the same thirty ideas, found already: counted as generated and as known


def test_neutralising_needs_the_companies_the_regression_has_parameters_for(world):
    ctx = world[1]
    controls = ("mcap", "beta")
    date = ctx.dates[60]
    panels = transforms._controls(ctx, controls, ctx.dates[[60]])
    complete = [c for c in ctx.ids if all(np.isfinite(panels[name].loc[date, c]) for name in controls)]            # companies whose controls are known on the date
    need = transforms._dummies(ctx, ("sector",)).shape[1] + len(controls) + 5                                     # as many companies as parameters and five more
    assert len(complete) > need + 5
    values = np.random.default_rng(0).standard_normal(len(complete))

    def with_values(count: int) -> pd.DataFrame:
        frame = pd.DataFrame(np.nan, index=ctx.dates, columns=ctx.ids)
        frame.loc[date, complete[:count]] = values[:count]
        return frame

    enough = transforms.neutralize(with_values(need), ctx, ("sector",), controls, min_obs=5)
    assert enough.loc[date].notna().sum() == need                                                                 # enough: every one of them gets a residual
    short = transforms.neutralize(with_values(need - 1), ctx, ("sector",), controls, min_obs=5)
    assert short.loc[date].notna().sum() == 0                                                                     # one fewer: the date is left missing, not fitted on too little
    assert transforms.neutralize(with_values(need), ctx, ("sector",), controls).loc[date].notna().sum() == 0      # and the bar of thirty companies applies unless it is lowered
    assert transforms.neutralize(with_values(30), ctx, ("sector",), controls).loc[date].notna().sum() == 30       # thirty are enough


def test_alphas_combined_by_equal_weight_get_equal_weights_and_two_are_the_fewest_that_can_be_combined_or_clustered(world):
    from src.fundamentals import combine as CB
    from src.fundamentals import factors as F
    ctx = world[1]
    signals = {n: F.compute(ctx, n) * (1.0 if F.get(n).direction >= 0 else -1.0) for n in ("fcf_yield", "d_roic", "mom_12_1")}
    three = CB.combine(ctx, signals, "equal")
    assert np.allclose(three.weights.to_numpy(), 1.0 / 3.0) and list(three.weights.columns) == ["fcf_yield", "d_roic", "mom_12_1"]
    two = CB.combine(ctx, {k: signals[k] for k in ("fcf_yield", "d_roic")}, "equal")
    assert np.allclose(two.weights.to_numpy(), 0.5)
    with pytest.raises(FundamentalsError, match="at least two alphas"):
        CB.combine(ctx, {"fcf_yield": signals["fcf_yield"]}, "equal")
    corr = CB.correlation({k: signals[k] for k in ("fcf_yield", "d_roic")})
    two_clusters = CB.cluster(corr, cut=0.5)
    assert len(two_clusters["links"]) == 1 and sorted(two_clusters["order"]) == ["d_roic", "fcf_yield"] and set(two_clusters["labels"]) == {"fcf_yield", "d_roic"}
    one = CB.cluster(corr.iloc[:1, :1])
    assert one["links"] == [] and one["order"] == ["fcf_yield"]


def test_a_book_reports_in_plain_words_and_starts_with_a_million_dollars():
    from src.fundamentals import bridge as BR
    assert BR.PortfolioSpec().capital_usd == 1_000_000.0
    assert BR._pct(None) == "n/a" and BR._pct(float("nan")) == "n/a" and BR._pct(0.1234) == "+12.3%" and BR._pct(-0.05) == "-5.0%"
    assert BR._num(None) == "n/a" and BR._num(float("nan")) == "n/a" and BR._num(1.2345) == "1.23" and BR._num(-0.5) == "-0.50"
    days = pd.bdate_range("2020-01-01", periods=6)
    five, six = pd.Series([0.01, -0.02, 0.015, 0.0, 0.005], index=days[:5]), pd.Series([0.01, -0.02, 0.015, 0.0, 0.005, 0.002], index=days)
    assert BR._stats(five) == {"n_obs": 5} and BR._stats(six)["n_obs"] == 6                                       # a Sharpe ratio is not offered for five days
