"""Release 6 through the API and the command line: alpha diagnostics, portfolio diagnostics and the controlled-comparison cards.

The service functions are tested directly elsewhere. Here the claims are about the doors: unknown fields and bad identifiers are refused, a sealed segment stays sealed, the payloads are strict JSON, a card is
frozen once and run once, and the command line says the same things for the same reasons."""

import json
import warnings

import pytest

from src.cli import main
from src.ml.registry import Registry
from src.webapp.api import ApiError
from tests.ml_helpers import make_bundle
from tests.test_webapp_assets_ml import _app, _safe, finished, wait

warnings.filterwarnings("ignore")
BASE = {"dataset": {"feature_groups": ["price", "momentum", "volatility"], "target": "fwd_ret_5"}, "split": {"min_train_days": 504}, "signal": {"n": 2}, "model": {"kind": "ridge"}, "explain": False}
GROUPS = ["price", "momentum", "volatility"]


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("r6api")
    bundle = make_bundle(n_assets=8, n=1300, seed=17, alpha=1.0)
    return _app(tmp, lambda config: bundle, root=tmp)


@pytest.fixture(scope="module")
def parent(app):
    return finished(app, app.extra_post["/api/ml/train"]({**BASE, "name": "parent", "mode": "development"}))


def get(app, path, **q):
    out = _safe(app.extra_get[path](q))
    return out


def test_the_routes_exist_and_take_only_what_they_declare(app, parent):
    for path in ("/api/ml/alpha-diagnostics", "/api/ml/portfolio-diagnostics", "/api/ml/improvement-experiment"):
        assert path in app.extra_get
    assert "/api/ml/improvement-experiment" in app.extra_post
    with pytest.raises(ApiError, match="does not take"):
        get(app, "/api/ml/alpha-diagnostics", id=parent["id"], nonsense="1")
    with pytest.raises(ApiError, match="not a valid identifier"):
        get(app, "/api/ml/alpha-diagnostics", id="../x")
    with pytest.raises(ApiError, match="not a valid identifier"):
        get(app, "/api/ml/alpha-diagnostics", id=parent["id"], baseline="../x")
    with pytest.raises(ApiError, match="horizon"):
        get(app, "/api/ml/portfolio-diagnostics", id=parent["id"], horizon="2")


def test_alpha_diagnostics_are_computed_on_development_data_recorded_as_a_look_and_strict_json(app, parent):
    out = get(app, "/api/ml/alpha-diagnostics", id=parent["id"])
    d = out.get("data", out)
    assert d["eligibility"] == "applied" and d["segment"] == "validation" and d["record_seq"] >= 1 and d["ic"]["mean"]["settings"]["lags"] >= 4
    assert len(d["decay"]["cells"]) == 16 and d["redundancy"]["rows_scope"].startswith("train and validation rows only")
    seq2 = get(app, "/api/ml/alpha-diagnostics", id=parent["id"]).get("data", {}).get("record_seq")
    assert seq2 is None or seq2 > d["record_seq"], "each request is another record in the append-only table"
    for seg in ("test", "forward"):
        with pytest.raises(ApiError, match="protected"):
            get(app, "/api/ml/alpha-diagnostics", id=parent["id"], segment=seg)
        with pytest.raises(ApiError, match="protected"):
            get(app, "/api/ml/portfolio-diagnostics", id=parent["id"], segment=seg)


def test_portfolio_diagnostics_report_the_book_the_estimators_and_the_allocation_sensitivity(app, parent):
    out = get(app, "/api/ml/portfolio-diagnostics", id=parent["id"], dates="8")
    d = out.get("data", out)
    assert d["eligibility"] == "applied" and d["book"]["eligibility"] == "applied" and len(d["risk_models"]["estimators"]) == 4 and d["record_seq"] >= 1
    assert d["allocation"]["eligibility"] == "applied" and d["allocation"]["base_status"] == "optimal" and d["risk_contributions"]["estimator"] in d["risk_models"]["estimators"]


def test_an_improvement_experiment_is_validated_frozen_run_once_and_read_back(app, parent):
    post = app.extra_post["/api/ml/improvement-experiment"]
    ok = {"family": "target", "hypothesis": "a market-neutral target ranks assets better", "one_change": "the target only", "baseline": BASE, "candidates": [{"dataset": {"target": "excess_ret_5"}}],
          "n_boot": 300}
    for body, text in (({**ok, "family": "vibes"}, "family must be one of"), ({**ok, "hypothesis": " "}, "hypothesis is required"), ({**ok, "candidates": []}, "1 to 6"), ({**ok, "candidates": ["x"]}, "1 to 6"),
                       ({**ok, "baseline": "x"}, "baseline is a configuration"), ({**ok, "primary_metric": "sortino"}, "primary_metric"), ({**ok, "min_effect": -1}, "min_effect"), ({**ok, "now": 1}, "does not take"),
                       ({**ok, "parsimony": "x"}, "parsimony is a rule")):
        with pytest.raises(ApiError, match=text):
            post(body)
    job = post(ok)
    out = finished(app, job)
    card = out["card"]
    assert out["card_id"].startswith("cmp-") and card["state"] == "run" and card["reused"] is False and card["gate_decision"]["status"] == "not_evaluated"
    cand = card["results"]["candidates"][0]
    assert cand["status"] == "compared" and cand["verdict"] in ("supported_on_development", "worse", "inconclusive") and cand["metrics"]["ic"]["difference"]["eligibility"] == "applied"
    again = finished(app, post(ok))["card"]
    assert again["reused"] is True and again["frozen_at"] == card["frozen_at"], "the same declaration is the same card, run once"
    read = get(app, "/api/ml/improvement-experiment", id=out["card_id"])
    assert read.get("data", read)["results"]["candidates"][0]["verdict"] == cand["verdict"]
    listing = get(app, "/api/ml/improvement-experiment")
    cards = listing.get("data", listing)["cards"]
    assert any(c["card_id"] == out["card_id"] and c["state"] == "run" for c in cards)
    with pytest.raises(ApiError, match="unknown comparison card"):
        get(app, "/api/ml/improvement-experiment", id="cmp-nope")
    sneaky = {**ok, "candidates": [{"dataset": {"target": "excess_ret_5"}, "backtest": {"cost_bps": 0.0}}]}
    res = wait(app, post(sneaky))
    assert res["status"] == "error" and "does not allow" in res["error"], "a candidate that also changes the costs is not a controlled comparison"


def test_a_cost_aware_allocation_can_be_compared_with_the_existing_sizing_as_a_portfolio_card(app):
    post = app.extra_post["/api/ml/improvement-experiment"]
    body = {"family": "portfolio", "hypothesis": "weighing the forecast against risk and cost beats the rank rule net of costs", "one_change": "the position rule",
            "baseline": {**BASE, "signal": {"method": "top_n_long_short", "n": 2, "holding_period": 5}},
            "candidates": [{"signal": {"method": "cost_aware_mean_variance", "holding_period": 1, "risk_aversion": 10.0, "opt_every": 5, "max_weight": 0.3}}], "n_boot": 300}
    out = finished(app, post(body))["card"]
    cand = out["results"]["candidates"][0]
    assert out["primary_metric"] == "net_sharpe" and cand["status"] == "compared" and cand["metrics"]["net_sharpe"]["difference"]["eligibility"] == "applied"
    assert any(p.startswith("signal.") for p in out["candidates"][0]["changed"]) and set(p.split(".")[0] for p in out["candidates"][0]["changed"]) == {"signal"}
    assert out["results"]["baseline"]["kind"] == "ridge"


# ------------------------------------------------------------------------------------------------------------------ the command line
@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("r6cli")
    b = make_bundle(8, 1300, seed=33, alpha=1.0, macro=False)
    csv = tmp / "toy.csv"
    b.prices.rename_axis("date").to_csv(csv)
    reg = str(tmp / "ml")
    assert main(["ml-train", "--prices", str(csv), "--groups", *GROUPS, "--model", "ridge", "--refit-every", "126", "--min-train-days", "400", "--registry", reg, "--no-explain", "--name", "base", "--development"]) == 0
    exp = Registry(reg).list_experiments(5)[0]["id"]
    spec = tmp / "spec.json"
    spec.write_text(json.dumps({"family": "target", "hypothesis": "a market-neutral target ranks assets better", "one_change": "the target only", "n_boot": 300,
                                "baseline": {"dataset": {"feature_groups": GROUPS, "target": "fwd_ret_5", "universe": "toy"}, "split": {"min_train_days": 400, "refit_every": 126}, "model": {"kind": "ridge", "params": {}}},
                                "candidates": [{"dataset": {"target": "excess_ret_5"}}]}))
    return str(csv), reg, exp, str(spec)


def test_the_command_line_prints_alpha_and_portfolio_diagnostics_and_refuses_what_the_api_refuses(world, capsys):
    csv, reg, exp, _ = world
    capsys.readouterr()
    assert main(["ml-alpha", exp, "--registry", reg]) == 0
    out = capsys.readouterr().out
    assert "mean rank IC" in out and "Newey-West lag" in out and "decay (mean IC by extra delay" in out and "coverage" in out
    assert main(["ml-alpha", exp, "--registry", reg, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["eligibility"] == "applied"
    assert main(["ml-alpha", exp, "--registry", reg, "--segment", "test"]) == 2
    assert "protected" in capsys.readouterr().err
    assert main(["ml-alpha", "no-such-experiment", "--registry", reg]) == 2
    assert main(["ml-portfolio", exp, "--registry", reg, "--dates", "8"]) == 0
    out = capsys.readouterr().out
    assert "covariance estimators on the risk the book realised" in out and "allocation sensitivity on" in out and "lw_constant_correlation" in out
    assert main(["ml-portfolio", exp, "--registry", reg, "--segment", "forward"]) == 2


def test_the_command_line_freezes_and_runs_a_card_and_shows_it_again(world, capsys):
    csv, reg, _, spec = world
    capsys.readouterr()
    assert main(["ml-improve", "--spec", spec, "--registry", reg, "--prices", csv]) == 0
    out = capsys.readouterr().out
    assert "Target engineering" in out and "one change: the target only" in out and "gate: not_evaluated" in out and ("supported_on_development" in out or "inconclusive" in out or "worse" in out)
    card_id = next(w for w in out.split() if w.startswith("cmp-"))
    assert main(["ml-improve", "--card", card_id, "--registry", reg]) == 0
    assert card_id in capsys.readouterr().out
    assert main(["ml-improve", "--card", "cmp-nope", "--registry", reg]) == 2
    assert main(["ml-improve", "--spec", spec, "--registry", reg, "--prices", csv, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["reused"] is True
