"""Release 2 evidence where a user meets it: the saved experiment carries an inference bundle, the report and the command line show fold lineage, the dataset audit and the treatment of overlapping labels, and the
forward tracker refuses drifted data and says which of its signals are replays of history the saved model had already seen."""

import json
import warnings

import pytest

from src.ml import bundle as B, cli, experiments as xe, monitoring as mon, registry as rg
from src.ml.datasets import _slice_bundle
from tests.ml_helpers import make_bundle

warnings.filterwarnings("ignore")
BASE = {"dataset": {"feature_groups": ["price", "volatility"], "min_history": 130}, "split": {"min_train_days": 300, "refit_every": 150, "sample_weighting": "uniqueness"}, "signal": {"n": 2, "holding_period": 3}}


@pytest.fixture(scope="module")
def market():
    return make_bundle(6, 1000, seed=14, alpha=1.0)


@pytest.fixture(scope="module")
def saved(market, tmp_path_factory):
    reg = rg.Registry(tmp_path_factory.mktemp("ml"))
    cut = market.index[800]
    exp = xe.run_experiment(BASE, bundle=_slice_bundle(market, slice(None, cut)), registry=reg, explain=False)
    return reg, exp, cut


def test_a_saved_experiment_carries_an_inference_bundle_beside_its_model(saved):
    reg, exp, cut = saved
    folder = reg.root / "experiments" / exp.id
    assert (folder / "model.pkl").exists() and (folder / "model.qfb").exists() and not (folder / "bundle_error.txt").exists()
    bundle = B.InferenceBundle.load(folder / "model.qfb")
    assert bundle.header["kind"] == "ridge" and bundle.header["schema"]["features"] == exp.final_model.features_ and bundle.header["fit_info"]["fit_cutoff"] == str(cut.date())
    assert bundle.header["label"]["horizon"] == exp.cfg["dataset"]["horizon"] and bundle.header["state"]["information_cutoff"] == str(cut.date())


def test_the_stored_report_shows_fold_lineage_the_dataset_audit_and_the_treatment_of_overlapping_labels(saved):
    reg, exp, _ = saved
    report = json.loads(json.dumps(reg.report(exp.id)))
    first = report["folds"]["list"][0]
    assert first["fit_cutoff"] == first["test_start"] and first["label_end_max"] <= first["fit_cutoff"] and len(first["train_rows_hash"]) == 16 and first["dependence"]["treatment"] == "uniqueness_weights"
    assert report["dependence"]["treatment"] == "uniqueness_weights" and 0.1 < report["dependence"]["mean_uniqueness"] < 0.5
    audit = report["audit"]
    assert audit["label"]["target"] == exp.cfg["dataset"]["target"] and audit["cells"] == audit["rows_kept"] + sum(audit["excluded"].values()) and audit["backward_fill"] == "none"
    assert audit["pit"]["claims_restricted"] is False and audit["sessions"]["policy"] == "common_close"


def test_the_command_line_report_prints_the_lineage_the_audit_and_the_dependence(saved):
    reg, exp, _ = saved
    text = cli.text_report(reg.report(exp.id))
    for needle in ("fold lineage (fitted on rows whose labels had ended by the fit cutoff)", "fold 0: fitted", "overlapping labels: mean uniqueness", "dataset audit:", "labels: entry close of day t + lag",
                   "stale prices: none", "point-in-time:", "sessions: decision at the us_equity close; policy common_close"):
        assert needle in text, needle
    assert cli.audit_lines(None) == ["audit: not recorded (the panel was built before dataset audits existed)"]
    old = {**reg.report(exp.id), "folds": {"list": [{"fold": 0, "skipped": False, "test_start": "2020-01-01", "test_end": "2020-02-01"}]}, "audit": None, "dependence": None}
    assert "fold lineage: not recorded" in cli.text_report(old)


def test_forward_signals_for_dates_the_model_had_already_seen_are_flagged_as_replays(saved, market):
    reg, exp, cut = saved
    tracker = mon.ForwardTracker(reg, exp.id)
    assert tracker.bundle is not None and tracker.seen_through == str(cut.date())
    live = tracker.step(market, as_of=market.index[801])
    assert live["replay"] is False and live["note"] is None
    same_close = tracker.step(market, as_of=cut)
    assert same_close["replay"] is False, "a model fitted at the close of a day may decide that close"
    replay = tracker.step(market, as_of=market.index[780])
    assert replay["replay"] is True and "replay of history it has seen" in replay["note"] and replay["model_seen_through"] == str(cut.date())
    summary = tracker.summary(market)
    assert summary["n_replay"] == 1 and summary["model_seen_through"] == str(cut.date()) and any("replays, not forecasts" in a for a in summary["alerts"])


def test_the_forward_tracker_refuses_data_whose_schema_has_drifted(saved, market, monkeypatch):
    reg, exp, _ = saved
    tracker = mon.ForwardTracker(reg, exp.id)
    real = tracker._dataset

    def drifted(bundle, as_of):
        ds, b = real(bundle, as_of)
        ds.X = ds.X.rename(columns={ds.X.columns[0]: "renamed_" + ds.X.columns[0]})
        return ds, b

    monkeypatch.setattr(tracker, "_dataset", drifted)
    with pytest.raises(B.SchemaDriftError, match="missing"):
        tracker.step(market, as_of=market.index[850])


def test_an_experiment_saved_without_a_bundle_still_tracks_forward_through_its_model(saved, market, tmp_path):
    reg, exp, cut = saved
    import shutil

    copy = tmp_path / "ml"
    shutil.copytree(reg.root, copy)
    (copy / "experiments" / exp.id / "model.qfb").unlink()
    tracker = mon.ForwardTracker(rg.Registry(copy), exp.id)
    assert tracker.bundle is None and tracker.seen_through == str(cut.date())
    assert tracker.step(market, as_of=market.index[802])["replay"] is False


def test_the_html_report_shows_the_audit_the_dependence_treatment_and_the_fold_lineage(saved):
    from src.ml.visualization import render_html

    reg, exp, _ = saved
    page = render_html(reg.report(exp.id))
    for needle in ("Data and training integrity", "labels: entry close of day t + lag", "point-in-time:", "Overlapping labels: mean uniqueness", "<b>uniqueness_weights</b>", "Fold lineage", "rows hash"):
        assert needle in page, needle
    first = reg.report(exp.id)["folds"]["list"][0]
    assert first["fit_cutoff"] in page and first["train_rows_hash"] in page
    legacy = {**reg.report(exp.id), "audit": None, "dependence": None, "folds": {"list": [{"fold": 0, "skipped": False, "test_start": "2020-01-01", "test_end": "2020-02-01"}]}}
    old = render_html(legacy)
    assert "audit: not recorded" in old and "Fold lineage: not recorded" in old


def test_forward_testing_refuses_an_experiment_whose_saved_rule_misreads_its_models_output(saved, tmp_path):
    reg, exp, _ = saved
    import shutil

    import yaml

    copy = tmp_path / "ml_legacy"
    shutil.copytree(reg.root, copy)
    path = copy / "experiments" / exp.id / "config.yaml"
    cfg = yaml.safe_load(path.read_text())
    mon.ForwardTracker(rg.Registry(copy), exp.id)                                                         # as saved it is fine
    cfg["signal"]["method"] = "target_weights"                                                            # a rule that reads the output as portfolio weights, on a model that outputs a return
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="cannot be forward-tested.*target_weights reads the model's output as portfolio weights"):
        mon.ForwardTracker(rg.Registry(copy), exp.id)
    cfg["signal"]["reinterpret_outputs"] = True                                                           # the one explicit override, which the report records, still works
    path.write_text(yaml.safe_dump(cfg))
    mon.ForwardTracker(rg.Registry(copy), exp.id)
