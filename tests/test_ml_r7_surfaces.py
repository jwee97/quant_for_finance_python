"""Release 7 through the API and the command line: the delay report, the new comparison families, combinations and ablations, the mixture study and the regime-exposure settings.

The modules are tested directly elsewhere. Here the claims are about the doors: unknown fields and nonsense are refused, a sealed segment stays sealed, a card is frozen once and run once, a combination is only
built from components that were run, and the command line says the same things for the same reasons."""

import json
import warnings

import pytest

from src.cli import main
from src.ml.registry import Registry
from src.webapp.api import ApiError
from tests.ml_helpers import make_bundle
from tests.test_webapp_assets_ml import _app, _safe, finished, wait

warnings.filterwarnings("ignore")
BASE = {"dataset": {"feature_groups": ["price", "momentum", "volatility"], "target": "fwd_ret_5"}, "split": {"min_train_days": 400, "refit_every": 63}, "signal": {"method": "top_n_long_short", "n": 2, "holding_period": 1},
        "model": {"kind": "ridge"}, "explain": False}
GROUPS = ["price", "momentum", "volatility"]
REGIME = {"signal": {"regime_exposure": "vol_quantile", "regime_states": 2, "regime_min_train": 100, "regime_min_days": 30}}


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("r7api")
    bundle = make_bundle(n_assets=8, n=1300, seed=19, alpha=1.0)
    return _app(tmp, lambda config: bundle, root=tmp)


@pytest.fixture(scope="module")
def parent(app):
    return finished(app, app.extra_post["/api/ml/train"]({**BASE, "name": "parent7", "mode": "development"}))


def get(app, path, **q):
    return _safe(app.extra_get[path](q))


def data(out):
    return out.get("data", out)


# ------------------------------------------------------------------------------------------------------------------ the delay report
def test_the_delay_report_takes_declared_delays_refuses_sealed_segments_and_is_recorded(app, parent):
    assert "/api/ml/delay-report" in app.extra_get
    with pytest.raises(ApiError, match="does not take"):
        get(app, "/api/ml/delay-report", id=parent["id"], nonsense="1")
    with pytest.raises(ApiError, match="whole numbers of sessions"):
        get(app, "/api/ml/delay-report", id=parent["id"], delays="1,2")
    with pytest.raises(ApiError, match="whole numbers of sessions"):
        get(app, "/api/ml/delay-report", id=parent["id"], delays="0,abc")
    with pytest.raises(ApiError, match="protected"):
        get(app, "/api/ml/delay-report", id=parent["id"], segment="test")
    d = data(get(app, "/api/ml/delay-report", id=parent["id"], participation="0.1"))
    assert d["eligibility"] == "applied" and [r["delay"] for r in d["rows"]] == [0, 1, 2, 5] and d["record_seq"] >= 1 and d["declaration"]["hash"] and d["rule"].startswith("stale book")
    assert all(r["timing"]["execution_lag_sessions"] == 1 + r["delay"] for r in d["rows"]) and all(r["portfolio"]["eligibility"] == "applied" for r in d["rows"])
    assert all(r["capacity"]["status"] == "measured" for r in d["rows"]), "the universe's volume was found, so capacity at the given participation is measured"
    stored = data(get(app, "/api/ml/delay-report", id=parent["id"], stored="1"))
    assert stored["stored"] is True and stored["record_seq"] == d["record_seq"]
    fresh = data(get(app, "/api/ml/delay-report", id=parent["id"], delays="0,3"))
    assert [r["delay"] for r in fresh["rows"]] == [0, 3] and fresh["record_seq"] > d["record_seq"] and all(r["capacity"]["status"] == "not_established" for r in fresh["rows"])


# ------------------------------------------------------------------------------------------------------------------ the new families
def test_the_new_families_are_validated_before_anything_runs(app, parent):
    post = app.extra_post["/api/ml/improvement-experiment"]
    ok = {"family": "rebalancing", "hypothesis": "a no-trade region saves more cost than it gives up", "one_change": "the rebalance rule", "baseline": BASE, "presets": ["no_trade:2.0"], "n_boot": 300}
    for body, text in (({**ok, "family": "ensemble"}, "needs members"), ({**ok, "select_baseline": True}, "applies to the ensemble family"), ({**ok, "options": {"leverage": 3}}, "options holds"),
                       ({**ok, "family": "ensemble", "members": "ridge"}, "between two and eight"), ({**ok, "family": "ensemble", "members": "ridge;;lasso"}, "cannot read"),
                       ({"family": "combination", "hypothesis": "h", "one_change": "c"}, "combination_of is a list"), ({**ok, "family": "combination", "combination_of": [{"card_id": "a", "candidate": "b"}] * 2}, "does not apply to a combination"),
                       ({"family": "combination", "hypothesis": "h", "one_change": "c", "combination_of": [{"card_id": "a"}]}, "combination_of is a list"),
                       ({"family": "combination", "hypothesis": "h", "one_change": "c", "combination_of": [{"card_id": "a", "candidate": "b"}] * 2, "parsimony": {"rule": "one_se"}}, "parsimony does not apply")):
        with pytest.raises(ApiError, match=text):
            post(body)
    with pytest.raises(ApiError, match="must be a number"):
        post({**ok, "presets": ["no_trade:abc"]})
    res = wait(app, post({"family": "combination", "hypothesis": "h", "one_change": "c", "combination_of": [{"card_id": "cmp-nope", "candidate": "x"}, {"card_id": "cmp-nope2", "candidate": "y"}]}))
    assert res["status"] == "error" and "unknown comparison card" in res["error"]


def test_a_rebalancing_card_and_a_risk_control_card_run_once_and_a_combination_is_built_from_them_and_ablated(app, parent):
    post = app.extra_post["/api/ml/improvement-experiment"]
    a = finished(app, post({"family": "rebalancing", "hypothesis": "a no-trade region saves more cost than it gives up", "one_change": "the rebalance rule", "baseline": BASE, "presets": ["no_trade:2.0"],
                            "n_boot": 300, "study": "r7-api"}))
    b = finished(app, post({"family": "risk_control", "hypothesis": "a causal volatility regime lowers risk", "one_change": "regime exposure", "baseline": BASE, "candidates": [REGIME], "n_boot": 300,
                            "study": "r7-api"}))
    for out in (a, b):
        card = out["card"]
        arm = card["results"]["candidates"][0]
        assert card["state"] == "run" and card["gate_decision"]["status"] == "not_evaluated" and arm["status"] == "compared" and arm["arm"] == "re-run of the baseline's saved predictions"
    assert b["card"]["results"]["candidates"][0]["metrics"]["risk_matched"]["eligibility"] == "applied"
    parents = [{"card_id": o["card_id"], "candidate": o["card"]["results"]["candidates"][0]["name"]} for o in (a, b)]
    combo = finished(app, post({"family": "combination", "hypothesis": "the two together beat the baseline", "one_change": "no-trade region plus regime exposure", "combination_of": parents, "n_boot": 300}))
    cc = combo["card"]
    assert cc["family"] == "combination" and cc["lineage"]["kind"] == "combination" and cc["multiple_testing_scope"]["family_size"] == 3 and cc["results"]["candidates"][0]["status"] == "compared"
    abl = finished(app, post({"family": "combination", "ablation": True, "combination_of": parents, "n_boot": 300}))
    assert abl["card"]["lineage"]["kind"] == "ablation" and len(abl["card"]["results"]["ablation"]) == 2 and abl["card"]["hypothesis"].startswith("each component")
    again = finished(app, post({"family": "combination", "hypothesis": "the two together beat the baseline", "one_change": "no-trade region plus regime exposure", "combination_of": parents, "n_boot": 300}))
    assert again["card"]["reused"] is True and again["card_id"] == combo["card_id"], "the same declaration is the same card, run once"
    listing = data(get(app, "/api/ml/improvement-experiment", study="r7-api"))["cards"]
    assert {c["family"] for c in listing} >= {"rebalancing", "risk_control", "combination"}


def test_an_ensemble_card_is_compared_with_the_single_model_development_would_choose(app, parent):
    post = app.extra_post["/api/ml/improvement-experiment"]
    out = finished(app, post({"family": "ensemble", "hypothesis": "a stacked ensemble beats the best single member", "one_change": "the model", "baseline": BASE, "members": "ridge,lasso[alpha=0.05],elasticnet[alpha=0.05]",
                              "select_baseline": True, "presets": ["weighting:stacked"], "n_boot": 300, "study": "r7-ens"}))
    card = out["card"]
    sel = card["baseline_selection"]
    assert sel["n_candidates"] == 3 and sel["selected"]["kind"] in ("ridge", "lasso", "elasticnet") and "optimistic" in sel["note"] and card["results"]["baseline"]["kind"] == sel["selected"]["kind"]
    arm = card["results"]["candidates"][0]
    assert arm["status"] == "compared" and arm["arm"] == "retrained" and arm["kind"].startswith("model_ensemble"[:5]) and card["primary_metric"] == "net_sharpe"


# ------------------------------------------------------------------------------------------------------------------ the mixture study
def test_combining_experiments_with_a_development_end_adds_the_study_of_the_best_single_member(app, parent):
    other = finished(app, app.extra_post["/api/ml/train"]({**BASE, "name": "second7", "mode": "development", "model": {"kind": "lasso", "params": {"alpha": 0.05}}}))
    body = {"ids": [parent["id"], other["id"]], "methods": ["equal", "inverse_vol"], "min_history": 40, "rebalance": 5}
    plain = finished_or_direct(app, body)
    assert "mixture_study" not in plain
    with pytest.raises(ApiError, match="development_end is a date"):
        app.extra_post["/api/ml/combine"]({**body, "development_end": "someday"})
    start = plain["start"]
    cut = str((__import__("pandas").Timestamp(start) + __import__("pandas").Timedelta(days=125)).date())
    out = finished_or_direct(app, {**body, "development_end": cut})
    ms = out["mixture_study"]
    assert ms["best_single"]["chosen_on"] == "the development period only" and [r["method"] for r in ms["rows"]] == ["equal", "inverse_vol"] and ms["evaluation"][0] > ms["development"][1]
    assert all(r["vs_best_single"]["eligibility"] in ("applied", "insufficient") for r in ms["rows"])


def finished_or_direct(app, body):
    res = app.extra_post["/api/ml/combine"](body)
    res = _safe(res) if not isinstance(res, dict) or "job" not in res else finished(app, res)
    return data(res)


# ------------------------------------------------------------------------------------------------------------------ the command line
@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("r7cli")
    b = make_bundle(8, 1300, seed=35, alpha=1.0, macro=False)
    csv = tmp / "toy.csv"
    b.prices.rename_axis("date").to_csv(csv)
    reg = str(tmp / "ml")
    assert main(["ml-train", "--prices", str(csv), "--groups", *GROUPS, "--model", "ridge", "--refit-every", "126", "--min-train-days", "400", "--registry", reg, "--no-explain", "--name", "base", "--development"]) == 0
    exp = Registry(reg).list_experiments(5)[0]["id"]
    spec = tmp / "spec.json"
    spec.write_text(json.dumps({"family": "rebalancing", "hypothesis": "a no-trade region saves more cost than it gives up", "one_change": "the rebalance rule", "n_boot": 300,
                                "baseline": {"dataset": {"feature_groups": GROUPS, "target": "fwd_ret_5", "universe": "toy"}, "split": {"min_train_days": 400, "refit_every": 126}, "model": {"kind": "ridge", "params": {}},
                                             "signal": {"method": "top_n_long_short", "n": 2, "holding_period": 1}},
                                "candidates": [{"signal": {"no_trade_margin": 2.0}}]}))
    return str(csv), reg, exp, str(spec)


def test_the_command_line_prints_the_delay_report_and_runs_a_rebalancing_card(world, capsys):
    csv, reg, exp, spec = world
    capsys.readouterr()
    assert main(["ml-delay", exp, "--registry", reg]) == 0
    out = capsys.readouterr().out
    assert "stale book" in out and "break-even bps" in out and "calendar days" in out and "declared one-way cost" in out
    assert main(["ml-delay", exp, "--registry", reg, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["eligibility"] == "applied"
    assert main(["ml-delay", exp, "--registry", reg, "--delays", "1,2"]) == 2
    assert "whole numbers of sessions" in capsys.readouterr().err
    assert main(["ml-delay", exp, "--registry", reg, "--segment", "test"]) == 2
    assert "protected" in capsys.readouterr().err
    assert main(["ml-improve", "--registry", reg, "--spec", spec, "--prices", csv, "--json"]) == 0
    card = json.loads(capsys.readouterr().out)
    assert card["family"] == "rebalancing" and card["state"] == "run" and card["results"]["candidates"][0]["status"] == "compared" and card["gate_decision"]["status"] == "not_evaluated"
