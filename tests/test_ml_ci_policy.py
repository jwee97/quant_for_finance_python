"""Release 3, P0-29: the model-correctness checks run in a blocking lane, a missing or skipped family cannot be read as success, and the heavy checks are scheduled separately with their evidence kept.

These tests read the workflow files and the verification runner; they do not claim anything about a CI run that has not happened. What they pin is the configuration that makes a green lane mean something."""

from pathlib import Path

import pytest
import yaml

from src.ml import cli, verification as vf
from src.ml.registry import Registry

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / ".github/workflows/ci.yml").exists():                                       # the Docker image is built without `.github` (see .dockerignore): the workflows are checked where they exist
    pytest.skip("the workflow files are not in this tree", allow_module_level=True)
CI = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
HEAVY = yaml.safe_load((ROOT / ".github/workflows/ml-heavy.yml").read_text())
RELEASE_3 = ["tests/test_ml_provenance.py", "tests/test_ml_holdout_controls.py", "tests/test_ml_statistical_eligibility.py", "tests/test_ml_null_pipeline.py", "tests/test_ml_promotion_gates.py",
             "tests/test_ml_registry_migrations.py", "tests/test_ml_research_api.py", "tests/test_ml_research_cli.py", "tests/test_ml_ci_policy.py"]


RELEASE_4 = ["tests/test_ml_execution_ledger.py", "tests/test_ml_execution_fixtures.py", "tests/test_ml_execution_pipeline.py", "tests/test_ml_execution_api.py", "tests/test_ml_execution_cli.py"]


RELEASE_5 = ["tests/test_ml_nested_selection.py", "tests/test_ml_statistical_inference.py", "tests/test_ml_matched_benchmarks.py", "tests/test_ml_regime_evaluation.py", "tests/test_ml_sensitivity.py",
             "tests/test_ml_calibration_scores.py", "tests/test_ml_multiseed.py", "tests/test_ml_prospective_commitments.py", "tests/test_ml_forward_equity.py", "tests/test_ml_validation_policy_governance.py",
             "tests/test_ml_statistical_api.py", "tests/test_ml_statistical_cli.py"]


def steps(job: dict) -> list[str]:
    return [str(s.get("run", "")) for s in job.get("steps", [])]


def test_the_correctness_lane_requires_every_requirement_to_pass_and_be_current():
    job = CI["jobs"]["ml-correctness"]
    assert any("ml-verify --run --require-pass" in s for s in steps(job)), "the lane must fail unless every registered requirement passed on the current source"
    assert not job.get("continue-on-error") and all(not s.get("continue-on-error") for s in job["steps"])


def test_no_job_of_the_pipeline_can_fail_without_failing_the_run():
    for name, job in CI["jobs"].items():
        assert not job.get("continue-on-error"), f"{name} must not be allowed to fail"
        for s in job["steps"]:
            run = str(s.get("run", ""))
            assert not s.get("continue-on-error"), f"a step of {name} is allowed to fail"
            checks = "pytest" in run or "ml-verify --run" in run or "--require-pass" in run
            assert not (checks and ("|| true" in run or "exit 0" in run)), f"a step of {name} that runs checks swallows its own failure"       # an evidence step that only writes JSON may follow


def test_every_registered_requirement_names_tests_and_sources_that_exist():
    for req, (tests, group) in vf.REQUIREMENTS.items():
        assert tests and group in vf.SOURCES, req
        for t in tests:
            assert (ROOT / t).exists(), f"{req} names {t}, which does not exist"
        for entry in vf.SOURCES[group]:
            assert (ROOT / entry).exists(), f"the source group {group} names {entry}, which does not exist"
    assert {f"P0-{n}" for n in range(23, 30)} <= set(vf.REQUIREMENTS) and "P1-26" in vf.REQUIREMENTS
    assert {f"P1-{n:02d}" for n in range(1, 15)} <= set(vf.REQUIREMENTS), "every requirement of Release 4 is registered"
    assert {f"P1-{n}" for n in range(15, 27)} <= set(vf.REQUIREMENTS), "every requirement of Release 5 is registered"


def test_the_release_3_tests_run_in_the_base_install_and_the_dashboard_lanes():
    for lane in ("ml-base", "replit-python"):
        text = " ".join(steps(CI["jobs"][lane]))
        for t in RELEASE_3:
            assert t in text, f"{t} is not run by the {lane} lane"
        assert "pytest" in text


def test_the_release_4_tests_run_in_the_base_install_and_the_dashboard_lanes():
    for lane in ("ml-base", "replit-python"):
        text = " ".join(steps(CI["jobs"][lane]))
        for t in RELEASE_4:
            assert t in text, f"{t} is not run by the {lane} lane"
    named = {t for tests, _ in vf.REQUIREMENTS.values() for t in tests}
    assert set(RELEASE_4) <= named, "every Release 4 test file is the evidence of some requirement"


def test_a_skipped_family_is_blocked_and_a_registry_with_no_evidence_fails_the_gate(tmp_path, capsys):
    skipped = [{"file": "tests/x.py", "passed": 5, "failed": 0, "skipped": 2, "errors": 0, "failures": [], "skips": ["test_a: torch is not installed", "test_b: torch is not installed"]}]
    status, reason = vf.status_of(skipped)
    assert status == "blocked" and "not verified" in reason, "a family whose tests were skipped has not been verified"
    justified = [{"file": "tests/x.py", "passed": 5, "failed": 0, "skipped": 1, "errors": 0, "failures": [], "skips": ["test_a: NOT_APPLICABLE ngboost cannot coexist with pandas 3"]}]
    assert vf.status_of(justified)[0] == "pass"
    from argparse import Namespace

    empty = Registry(tmp_path / "ml")
    rows = vf.current(empty)
    assert all(r["status"] == "not_run" for r in rows) and len(rows) == len(vf.REQUIREMENTS), "nothing that never ran is shown as passing"
    args = Namespace(registry=str(tmp_path / "ml"), run=False, requirement=[], json=False, require_pass=True)
    assert cli.cmd_verify(args, None) == 1, "the gate CI uses exits non-zero when no evidence exists"


def test_the_heavy_null_study_is_scheduled_separately_and_keeps_its_evidence():
    assert "schedule" in HEAVY[True] and "workflow_dispatch" in HEAVY[True] and "push" not in HEAVY[True] and "pull_request" not in HEAVY[True]
    job = HEAVY["jobs"]["null-controls"]
    run = " ".join(steps(job))
    assert "ml-null-controls" in run and "--min-replications 100" in run and "--replications" in run and not job.get("continue-on-error")
    assert int(HEAVY[True]["workflow_dispatch"]["inputs"]["replications"]["default"]) >= 100
    uploads = [s for s in job["steps"] if "upload-artifact" in str(s.get("uses", ""))]
    assert uploads and uploads[0].get("if") == "always()" and "artifacts/ml-heavy" in str(uploads[0]["with"]["path"]), "the evidence is kept even when the study fails"
    assert "SIMULATED" in (ROOT / ".github/workflows/ml-heavy.yml").read_text(), "the workflow says what its data is"


def test_the_test_suite_registers_its_pytest_markers():
    cfg = (ROOT / "pyproject.toml").read_text()
    assert "[tool.pytest" in cfg
    import re

    used = set()
    for path in (ROOT / "tests").glob("test_ml_*.py"):
        used |= set(re.findall(r"^\s*@pytest\.mark\.(\w+)", path.read_text(), flags=re.M))
    used -= {"skipif", "skip", "parametrize", "usefixtures", "xfail", "filterwarnings"}               # the built-in marks need no registration
    unknown = {m for m in used if m not in cfg}
    assert not unknown, f"markers used but not registered: {sorted(unknown)}"


@pytest.mark.parametrize("name", ["ml-correctness", "ml-base", "replit-python"])
def test_the_lanes_are_present(name):
    assert name in CI["jobs"]


def test_the_release_5_tests_run_in_the_base_install_and_the_dashboard_lanes_and_the_browser_lane():
    for lane in ("ml-base", "replit-python"):
        text = " ".join(steps(CI["jobs"][lane]))
        for t in RELEASE_5:
            assert t in text, f"{t} is not run by the {lane} lane"
    assert "tests/test_webapp_ml_validation_browser.py" in " ".join(steps(CI["jobs"]["browser"]))
    named = {t for tests, _ in vf.REQUIREMENTS.values() for t in tests}
    assert set(RELEASE_5) <= named, "every Release 5 test file is the evidence of some requirement"
    for lane in ("ml-base", "ml-correctness"):
        assert CI["jobs"][lane]["timeout-minutes"] >= 45, "the lanes that run the growing verification set have headroom"
