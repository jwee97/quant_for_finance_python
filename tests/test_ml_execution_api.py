"""Release 4 through the API and the gates: the execution settings a request may carry, the trading-diagnostics route, a realistic re-run as a child job, and Gate B reading the execution evidence.

The service functions are exercised directly in ``test_ml_execution_pipeline.py``; here the claim is that the routes validate what they take, that nothing in a request can supply a result or a status, and
that Gate B treats a vectorized result as unable to pass a requirement it does not model."""

import json
import warnings

import pytest

from src.ml import gates, verification as vf
from src.webapp.api import ApiError
from tests.ml_helpers import make_bundle
from tests.test_webapp_assets_ml import _app, _safe, finished

warnings.filterwarnings("ignore")
BASE = {"dataset": {"feature_groups": ["price", "momentum", "volatility"]}, "split": {"min_train_days": 504}, "signal": {"n": 2}, "model": {"kind": "ridge"}, "explain": False}


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("execution")
    bundle = make_bundle(n_assets=8, n=1300, seed=5, alpha=1.0)
    return _app(tmp, lambda config: bundle, root=tmp)


@pytest.fixture(scope="module")
def parent(app):
    return finished(app, app.extra_post["/api/ml/train"]({**BASE, "name": "parent", "mode": "development"}))


def get(app, path, **q):
    return _safe(app.extra_get[path](q))


def registry_of(app):
    return app.extra_get["/api/ml/lineage"].__self__.registry                                   # the MlApi instance behind the routes


def test_the_routes_exist_and_the_execution_section_rejects_what_it_does_not_take(app):
    assert "/api/ml/trading-diagnostics" in app.extra_get
    for body, text in (({"execution": {"delay": 2}}, "has no setting 'delay'"), ({"execution": {"borrow": 1}}, "has no setting 'borrow'"), ({"execution": {"capital": -5}}, "capital"),
                       ({"execution": {"mode": "yolo"}}, "mode must be one of"), ({"execution": {"sizing": "later"}}, "sizing must be one of"), ({"execution": {"max_participation": 5}}, "max_participation")):
        with pytest.raises(ApiError, match=text):
            app.extra_post["/api/ml/train"]({**BASE, **body})
    with pytest.raises(ApiError, match="not a valid identifier"):
        app.extra_get["/api/ml/trading-diagnostics"]({"id": "../x"})
    with pytest.raises(ApiError, match="can change only the signal, backtest, execution"):
        app.extra_post["/api/ml/rebacktest"]({"id": "x", "model": {"kind": "lasso"}})


def test_a_vectorized_result_is_read_as_such_and_a_realistic_rerun_is_a_child_with_its_diagnostics(app, parent):
    legacy = get(app, "/api/ml/trading-diagnostics", id=parent["id"])
    assert legacy["mode"] == "vectorized_legacy" and legacy["recorded"] is True and "capacity" in legacy["cannot_support"] and legacy["parent"] is None
    job = app.extra_post["/api/ml/rebacktest"]({"id": parent["id"], "execution": {"mode": "realistic_cash_equity", "capital": 500_000, "impact_coefficient": 0.3, "max_participation": 0.05}})
    child = finished(app, job)
    assert child["parent"] == parent["id"] and child["id"] != parent["id"] and child["report"]["execution"]["mode"] == "realistic_cash_equity"
    diag = get(app, "/api/ml/trading-diagnostics", id=child["id"])
    assert diag["parent"] == parent["id"] and diag["mode"] == "realistic_cash_equity" and diag["assumptions"]["capital"] == 500_000 and diag["assumptions"]["max_participation"] == 0.05
    assert diag["assumptions"]["impact"]["coefficient"] == 0.3 and diag["reconciliation"]["reconciled"] and diag["timeline"]["delay_sessions"] == 1
    assert set(diag["costs"]) == {"fee", "spread", "slippage", "impact", "borrow", "financing"} and diag["capacity"]["levels"] and diag["manifest"]["files"]["fills"]["sha256"]
    assert any("assumption" in x for x in diag["limitations"]), "an impact coefficient that was not calibrated is said to be an assumption"
    json.dumps(diag, allow_nan=False)
    # no participation limit is a stated limitation, not a silent default
    free = finished(app, app.extra_post["/api/ml/rebacktest"]({"id": parent["id"], "execution": {"max_participation": None}}))
    assert free["report"]["execution"]["assumptions"]["max_participation"] is None and any("no participation limit was applied" in x for x in free["report"]["execution"]["limitations"])


def test_an_experiment_saved_before_modes_existed_is_read_as_vectorized_legacy(app, parent):
    path = registry_of(app).root / "experiments" / parent["id"] / "report.json"
    report = json.loads(path.read_text())
    report.pop("execution")
    path.write_text(json.dumps(report))
    old = get(app, "/api/ml/trading-diagnostics", id=parent["id"])
    assert old["mode"] == "vectorized_legacy" and old["recorded"] is False and "saved before execution modes were recorded" in old["note"]


def test_gate_b_blocks_a_vectorized_result_and_reads_the_machinery_and_the_data_of_a_realistic_one(app):
    reg = registry_of(app)
    full = finished(app, app.extra_post["/api/ml/train"]({**BASE, "name": "full"}))
    vec = gates.evaluate(reg, full["id"])
    item = next(i for i in vec["gates"]["B"]["items"] if i["id"] == "realistic_net_oos")
    assert item["status"] == "blocked" and "vectorized_legacy run" in item["reason"] and "quant ml-realistic" in item["reason"]
    rerun = finished(app, app.extra_post["/api/ml/rebacktest"]({"id": full["id"], "execution": {"mode": "realistic_cash_equity", "capital": 1_000_000}}))
    res = gates.evaluate(reg, rerun["id"])
    item = next(i for i in res["gates"]["B"]["items"] if i["id"] == "realistic_net_oos")
    assert item["status"] == "missing" and "requirement P1-" in item["reason"], "the execution machinery has no verification rows yet: not a pass"
    for req in gates.EXECUTION_REQUIREMENTS:
        reg.log_check(req, "fixture", "pass", subject_hash=vf.source_hash(vf.REQUIREMENTS[req][1]), reason="fixture row")
    item = next(i for i in gates.evaluate(reg, rerun["id"])["gates"]["B"]["items"] if i["id"] == "realistic_net_oos")
    assert item["status"] == "missing" and "P1-08" in item["reason"], "this long-short book held shorts, so the borrow machinery's verification is required too"
    reg.log_check("P1-08", "fixture", "pass", subject_hash=vf.source_hash(vf.REQUIREMENTS["P1-08"][1]), reason="fixture row")
    item = next(i for i in gates.evaluate(reg, rerun["id"])["gates"]["B"]["items"] if i["id"] == "realistic_net_oos")
    assert item["status"] == "blocked" and "no borrow records" in item["reason"], "shorts were held and no borrow records exist: the claim is blocked, not assumed"


def test_a_manifest_that_no_longer_matches_its_files_fails_the_item(app):
    reg = registry_of(app)
    base = finished(app, app.extra_post["/api/ml/train"]({**BASE, "name": "tamper-base", "mode": "development"}))
    child = finished(app, app.extra_post["/api/ml/rebacktest"]({"id": base["id"], "execution": {"capital": 800_000}}))
    d = reg.root / "experiments" / child["id"]
    target = d / "execution_fills.parquet"
    target.write_bytes(target.read_bytes() + b"x")
    item = next(i for i in gates.evaluate(reg, child["id"])["gates"]["B"]["items"] if i["id"] == "realistic_net_oos")
    assert item["status"] == "failed" and "differs from the hash in the manifest" in item["reason"]


def test_the_provider_passes_only_with_verified_machinery_real_data_and_no_blockers(tmp_path):
    """The provider's logic on a hand-made report: every condition is separately necessary."""
    from src.ml import registry as rg

    reg = rg.Registry(tmp_path / "ml")
    folder = tmp_path / "ml" / "experiments" / "exp-1"
    folder.mkdir(parents=True)
    (folder / "execution_fills.parquet").write_bytes(b"fills")
    sha = gates.digest_file(folder / "execution_fills.parquet")
    good = {"mode": "realistic_cash_equity", "manifest": {"sha256": "m1", "files": {"fills": {"file": "execution_fills.parquet", "sha256": sha, "rows": 1}}},
            "reconciliation": {"reconciled": True, "ledger_ok": True, "gap": 0.0}, "short_positions": False,
            "capability": {"blockers": [], "limitations": ["only closing prices"], "data_status": "vendor"}}
    for req in gates.EXECUTION_REQUIREMENTS:
        reg.log_check(req, "fixture", "pass", subject_hash=vf.source_hash(vf.REQUIREMENTS[req][1]), reason="fixture row")
    ok = gates._execution_item(reg, "exp-1", {"execution": good})
    assert ok.status == "ok" and "only closing prices" in ok.reason
    for mutate, status, text in ((lambda b: b["capability"].update(data_status="simulated"), "blocked", "simulated"), (lambda b: b["capability"].update(blockers=["no volume: x"]), "blocked", "no volume"),
                                 (lambda b: b["reconciliation"].update(reconciled=False, gap=5.0), "failed", "does not reconcile"), (lambda b: b.update(short_positions=True), "missing", "P1-08"),
                                 (lambda b: b.pop("manifest"), "missing", "no artifact manifest")):
        bad = json.loads(json.dumps(good))
        mutate(bad)
        got = gates._execution_item(reg, "exp-1", {"execution": bad})
        assert got.status == status and text in got.reason, (got.status, got.reason)
    reg.log_check("P1-05", "fixture", "fail", subject_hash=vf.source_hash(vf.REQUIREMENTS["P1-05"][1]), reason="a failing row")
    assert gates._execution_item(reg, "exp-1", {"execution": good}).status == "failed"
