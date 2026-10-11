"""A package that is installed but cannot load (a missing system library such as libgomp.so.1) is unavailable, not fatal. The Machine learning tab used to fail to open on such a machine because the
model catalogue imported LightGBM to see whether it was there and only expected ``ImportError``."""

from __future__ import annotations

import sys

import pytest

from src.ml import models as md
from src.strategies.ml import make_tree_learner
from src.utils.optional import OptionalPackageError, describe_failure, import_optional, unavailable_reason


def test_a_package_that_is_not_installed_says_how_to_install_it():
    reason = unavailable_reason("surely_not_a_real_package_name")
    assert reason == "the 'surely_not_a_real_package_name' package is not installed (pip install surely_not_a_real_package_name)"
    assert unavailable_reason("json") is None
    with pytest.raises(OptionalPackageError, match="not installed") as caught:
        import_optional("surely_not_a_real_package_name")
    assert isinstance(caught.value, ImportError), "callers that already catch ImportError keep working"


def test_a_package_that_needs_a_missing_package_names_the_missing_one(tmp_path, monkeypatch):
    (tmp_path / "half_installed_pkg.py").write_text("import surely_not_a_real_dependency\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    assert unavailable_reason("half_installed_pkg") == "the 'half_installed_pkg' package is installed but needs the 'surely_not_a_real_dependency' package, which is not (pip install surely_not_a_real_dependency)"


def test_a_package_that_cannot_load_is_unavailable_and_the_reason_says_what_to_install(boosters_without_openmp, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    reason = unavailable_reason("lightgbm")
    assert "installed but cannot load" in reason and "libgomp.so.1: cannot open shared object file" in reason and "apt-get install -y libgomp1" in reason
    xgb = unavailable_reason("xgboost")                                                       # a ValueError subclass of the package's own, not an OSError: still just unavailable
    assert "installed but cannot load" in xgb and "XGBoost Library (libxgboost.so) could not be loaded" in xgb and "apt-get install -y libgomp1" in xgb and "Likely causes" not in xgb
    with pytest.raises(OptionalPackageError, match="libgomp"):
        import_optional("lightgbm")
    monkeypatch.setattr(sys, "platform", "darwin")
    assert "brew install libomp" in describe_failure("lightgbm", OSError("dlopen: libomp.dylib not found"))
    monkeypatch.setattr(sys, "platform", "win32")
    assert "Visual C++" in describe_failure("lightgbm", OSError("DLL load failed: vcomp140.dll"))
    assert describe_failure("somepackage", OSError("some other loader failure")) == "the 'somepackage' package is installed but cannot load (some other loader failure)"


def test_the_model_catalogue_lists_a_booster_that_cannot_load_as_unavailable_instead_of_failing(boosters_without_openmp):
    catalogue = {m["kind"]: m for m in md.model_catalog()}                                    # this raised OSError before
    for kind in ("lightgbm", "xgboost"):
        assert catalogue[kind]["available"] is False and "installed but cannot load" in catalogue[kind]["reason"], kind
    assert catalogue["catboost"]["available"] is False                                        # not installed in the test environment, or broken the same way: either way unavailable
    for kind in ("ridge", "random_forest", "hist_gb", "kalman"):
        assert catalogue[kind]["available"] is True and catalogue[kind]["reason"] is None, kind
    with pytest.raises(md.ModelUnavailable, match=r"libgomp.*choose another model"):
        md.make_model("lightgbm")
    with pytest.raises(md.ModelUnavailable, match="choose another model"):
        md.make_model("xgboost", task="classification")
    assert md.make_model("hist_gb") is not None                                               # the scikit-learn histogram boosting needs none of it


def test_the_strategy_learners_say_the_same_thing(boosters_without_openmp):
    for learner in ("lightgbm", "xgboost"):
        with pytest.raises(ImportError, match=r"cannot be used here.*installed but cannot load.*hgb, random_forest and extra_trees need nothing extra"):
            make_tree_learner(learner, 10, 3, 0.1, 10, 0)
    assert make_tree_learner("hgb", 10, 3, 0.1, 10, 0) is not None
