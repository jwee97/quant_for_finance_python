"""Release 3, P0-26: null-control studies run the WHOLE adaptive pipeline on placebo data, many times, under a frozen design, and report the false-promotion rate with its uncertainty.

What is tested: the placebo generator's properties (it breaks the feature-return link and nothing else), the exact binomial arithmetic, the verdict rules (a smoke run is labelled as plumbing; an interval that
clears or breaks the tolerance is said to), the freeze protocol (a stored design cannot change; a resumed study repeats nothing), and that every replication reruns the search on its own placebo dataset rather than
permuting a selected strategy's returns."""

import json
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from src.ml import datasets as dsm, evaluation as ev, null_controls as nc, registry as rg, research_controls as rc, strategy_search as ss
from src.ml.config import build_key, experiment_config
from src.ml.targets import forward_return
from tests.ml_helpers import make_bundle

warnings.filterwarnings("ignore")
FEATURES = ["price", "momentum", "volatility", "mean_reversion", "relative"]
BASE = {"dataset": {"feature_groups": FEATURES}, "split": {"min_train_days": 400, "refit_every": 126}, "model": {"kind": "ridge", "params": {}}}


@pytest.fixture(scope="module")
def bundle():
    return make_bundle(6, 1300, seed=21, alpha=1.0)


@pytest.fixture(scope="module")
def dataset(bundle):
    cfg = experiment_config(BASE)
    return dsm.build_dataset(bundle, cfg["dataset"], "null-base")


def mean_ic(d, feature):
    fwd = forward_return(d.prices, d.horizon, d.lag).reindex(index=d.dates, columns=d.assets)
    wide = d.X[feature].unstack("asset").reindex(index=d.dates, columns=d.assets)
    return float(ev.rank_ic_wide(wide, fwd).dropna().mean())


# --------------------------------------------------------------------------------------------------------------------- the generator
def test_the_placebo_keeps_each_side_intact_and_breaks_the_link_between_them(dataset):
    k, n = 430, len(dataset.dates)
    p = nc.placebo_dataset(dataset, k)
    assert p.X.equals(dataset.X) and list(p.dates) == list(dataset.dates) and p.meta["placebo"] == {"generator": "circular_shift_returns", "offset": k}
    r0, r1 = dataset.returns.to_numpy(), p.returns.to_numpy()
    rows = np.array([t for t in range(2, n) if t != k])
    assert np.allclose(r1[rows], r0[(rows - k) % n]), "returns are the real returns, shifted"
    assert (r1[k] == 0.0).all(), "the seam's single return is zero"
    assert float(np.abs(dataset.returns.corr().to_numpy() - p.returns.corr().to_numpy()).max()) < 0.02, "the cross-asset structure moves together"
    step = p.prices.pct_change(fill_method=None).to_numpy()
    assert np.nanmax(np.abs(step[1:] - r1[1:])) < 1e-12, "prices are rebuilt from the shifted returns, so they agree with them"
    lag, h = dataset.lag, dataset.horizon                                                      # the labels follow the placebo prices: the close of t + lag to the close of t + lag + horizon
    px = p.prices.to_numpy()
    dpos = pd.Series(np.arange(n), index=dataset.dates).reindex(p.Y.index.get_level_values(0)).to_numpy()
    apos = pd.Index(dataset.assets).get_indexer(p.Y.index.get_level_values(1))
    last = dpos + lag + h < n
    want = px[dpos[last] + lag + h, apos[last]] / px[dpos[last] + lag, apos[last]] - 1.0
    assert np.allclose(p.Y[dataset.target].to_numpy()[last], want, atol=1e-10, equal_nan=True)
    assert p.Y[dataset.target].notna().sum() == dataset.Y[dataset.target].notna().sum(), "no label is lost: the placebo has as many labelled rows as the real data"
    real = mean_ic(dataset, "ret_63")
    shifted = [mean_ic(nc.placebo_dataset(dataset, s), "ret_63") for s in (430, 700, 950)]
    assert real > 0.08 and max(abs(x) for x in shifted) < 0.5 * real, f"the planted momentum signal ({real:.3f}) is gone in the placebo ({shifted})"


def test_the_shift_is_deterministic_bounded_and_refuses_data_too_short_to_shift():
    a, b = nc.offset_for(7, 1300, 200), nc.offset_for(7, 1300, 200)
    assert a == b and 200 <= a <= 1100 and nc.offset_for(8, 1300, 200) != a
    with pytest.raises(ValueError, match="needs at least"):
        nc.offset_for(1, 300, 200)
    with pytest.raises(ValueError, match="zero days"):
        nc.placebo_dataset(type("D", (), {"dates": pd.bdate_range("2020-01-01", periods=10)})(), 10)


# --------------------------------------------------------------------------------------------------------------------- the arithmetic
def test_binomial_intervals_match_closed_forms():
    zero = nc.binomial_intervals(0, 20)
    assert zero["rate"] == 0.0 and zero["exact"][0] == 0.0 and zero["exact"][1] == pytest.approx(1 - 0.025 ** (1 / 20), abs=1e-12) and zero["resolution"] == 0.05
    full = nc.binomial_intervals(20, 20)
    assert full["exact"][1] == 1.0 and full["exact"][0] == pytest.approx(0.025 ** (1 / 20), abs=1e-12)
    k, n = 5, 50
    z, p = sps.norm.ppf(0.975), k / n
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    got = nc.binomial_intervals(k, n)
    assert got["wilson"] == pytest.approx([centre - half, centre + half], abs=1e-12)
    assert got["exact"][0] == pytest.approx(sps.beta.ppf(0.025, 5, 46), abs=1e-12) and got["exact"][1] == pytest.approx(sps.beta.ppf(0.975, 6, 45), abs=1e-12)
    assert nc.binomial_intervals(0, 0)["rate"] is None


def rows(n, k):
    return [{"promoted": i < k} for i in range(n)]


def test_the_verdict_distinguishes_a_smoke_run_a_controlled_rate_a_violation_and_an_inconclusive_one():
    spec = nc.NullSpec(replications=400, min_replications=100, tolerance=0.10, precision=0.06)
    smoke = nc.summarize(spec, rows(8, 0))
    assert smoke["sufficient"] is False and smoke["controlled"] is None and "plumbing only" in smoke["verdict"] and smoke["rate"] == 0.0, "a clean smoke run says nothing about the error rate"
    good = nc.summarize(spec, rows(400, 4))
    assert good["sufficient"] and good["controlled"] is True and good["false_promotions"] == 4 and good["exact_interval"][1] <= 0.10 and "within the predeclared tolerance" in good["verdict"]
    bad = nc.summarize(spec, rows(400, 80))
    assert bad["sufficient"] and bad["controlled"] is False and bad["exact_interval"][0] > 0.10 and "ABOVE" in bad["verdict"]
    edge = nc.summarize(spec, rows(400, 40))
    assert edge["sufficient"] and edge["controlled"] is None and "inconclusive" in edge["verdict"]
    wide = nc.summarize(nc.NullSpec(replications=120, min_replications=100, tolerance=0.10, precision=0.01), rows(120, 2))
    assert wide["sufficient"] is False and "precision" in wide["verdict"], "enough replications but an interval wider than the predeclared precision is still not enough"
    assert any("not a null for volatility forecasts" in s for s in good["limitations"]) and good["resolution"] == pytest.approx(1 / 400)
    assert nc.summarize(spec, [{"promoted": False, "test_mean": -0.1}] * 3)["false_promotions"] == 0, "negative null results are fine; nothing requires every null Sharpe to be negative"


# --------------------------------------------------------------------------------------------------------------------- the protocol
def fake_replicate(seed):
    return {"seed": seed, "promoted": seed % 7919 == 0 and False or (seed // 7919) % 5 == 0, "selected": {"number": seed % 3, "params": {"model.kind": "ridge"}}, "test_p": 0.4}


def test_the_design_is_frozen_before_the_first_replication_and_cannot_change_after(tmp_path):
    reg = rg.Registry(tmp_path / "ml")
    spec = nc.NullSpec(study_id="s9", replications=6, seed=3, min_replications=4, precision=0.5)
    made = nc.register(reg, spec)
    stored = nc.load(reg, made["null_id"])
    assert stored["state"] == "frozen" and stored["spec_hash"] == made["spec_hash"] == nc.spec_hash(spec) and stored["spec"]["generator"] == "circular_shift_returns" and stored["done"] == 0
    assert nc.spec_hash(nc.NullSpec(study_id="s9", replications=6, seed=4)) != made["spec_hash"], "any change of the design is a different study"
    with reg._conn() as c:                                                                   # someone edits the stored design after seeing results
        c.execute("UPDATE null_studies SET spec=? WHERE null_id=?", (json.dumps({**stored["spec"], "tolerance": 0.9}), made["null_id"]))
    with pytest.raises(rc.ControlError, match="no longer matches its hash"):
        nc.run(reg, made["null_id"], fake_replicate)


def test_a_study_runs_resumes_and_stores_every_seed_and_selection(tmp_path):
    reg = rg.Registry(tmp_path / "ml")
    spec = nc.NullSpec(study_id="s9", replications=6, seed=3, min_replications=4, precision=0.9)
    made = nc.register(reg, spec)
    seen = []
    stop_after_two = lambda: len(seen) >= 2                                                    # noqa: E731
    first = nc.run(reg, made["null_id"], lambda s: seen.append(s) or fake_replicate(s), stop=stop_after_two)
    assert first["state"] == "stopped" and first["replications"] == 2
    done = nc.run(reg, made["null_id"], lambda s: seen.append(s) or fake_replicate(s))
    assert done["state"] == "done" and done["replications"] == 6 and len(seen) == 6 and len(set(seen)) == 6, "a resumed study repeats nothing"
    stored = nc.load(reg, made["null_id"])
    assert [r["seed"] for r in stored["replication_records"]] == [3 + i * 7919 for i in range(6)] and all(r["selected"]["params"] == {"model.kind": "ridge"} for r in stored["replication_records"])
    assert stored["result"]["replications"] == 6 and stored["result"]["false_promotions"] == sum(r["promoted"] for r in stored["replication_records"]) and stored["state"] == "done"
    assert stored["result"]["exact_interval"][0] <= stored["result"]["rate"] <= stored["result"]["exact_interval"][1] and stored["result"]["sufficient"] is True
    lin = rc.lineage(reg, "null-" + made["null_id"])
    assert lin["raw_looks"] == 0 and lin["by_state"] == {} or all(not o["counts"] for o in lin["operations"]), "null replications are never counted as looks at the real study"
    assert rc.raw_looks(reg, "s9") == 0
    with pytest.raises(rc.ControlError, match="unknown null study"):
        nc.load(reg, "nope")


# --------------------------------------------------------------------------------------------------------------------- the whole pipeline, on placebo data
def test_every_replication_reruns_the_search_on_its_own_placebo_data(bundle, tmp_path, monkeypatch):
    seen = []
    real = ss.SearchRunner.evaluate

    def spy(self, params, stage="random"):
        placebo = self._datasets[build_key(self.base)].meta.get("placebo")
        seen.append((self.study, placebo["offset"] if placebo else None, dict(params).get("model.kind")))
        return real(self, params, stage)

    monkeypatch.setattr(ss.SearchRunner, "evaluate", spy)
    reg = rg.Registry(tmp_path / "ml")
    spec = nc.NullSpec(study_id="u-etf", base=BASE, kinds=("ridge", "lasso"), n_random=2, n_bayes=0, replications=2, seed=11, min_replications=50, min_shift=200)
    made = nc.register(reg, spec)
    out = nc.run(reg, made["null_id"], nc.search_pipeline(spec, bundle=bundle))
    assert len(seen) == 2 * spec.n_random, "the search ran afresh in every replication"
    assert len({s[0] for s in seen}) == 2 and all(s[1] is not None for s in seen), "each replication has its own placebo shift: the trials never saw the real returns"
    assert {s[2] for s in seen} == {"ridge", "lasso"}
    assert out["sufficient"] is False and out["controlled"] is None and "plumbing only" in out["verdict"], "two replications are a smoke run"
    records = nc.load(reg, made["null_id"])["replication_records"]
    for r in records:
        assert r["n_candidates"] == 2 and r["selected"] is not None and r["offset"] >= 200 and set(r["selection"]) == {"dsr", "spa"} and r["reason"]
        assert isinstance(r["promoted"], bool) and (r["promoted"] is False or (r["selection_passed"] and r["holdout_passed"]))
    assert records[0]["offset"] != records[1]["offset"]
