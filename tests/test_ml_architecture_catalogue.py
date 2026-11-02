"""P0-08, the part that needs no PyTorch: the architecture manifest reaches the places a person reads it (the catalogue the model form is built from), every network is described, the reduced ones say what they
leave out, and reading the manifest does not load a network library."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.ml.deep.manifest import ARCHITECTURES
from tests.ml_helpers import make_bundle
from tests.test_webapp_assets_ml import _app, _safe

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("variant", "reference", "input_axes", "output_axes", "masking", "normalization", "residuals", "attention", "horizon_mapping")


@pytest.fixture(scope="module")
def catalog(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("architectures_api")
    app = _app(tmp, lambda config: make_bundle(n_assets=6, n=400, seed=3, alpha=1.0), root=tmp)
    return _safe(app.extra_get["/api/ml/catalog"]({}))


def test_the_catalog_carries_the_architecture_manifest_with_the_reduced_networks_labelled(catalog):
    arch, kinds = catalog["architectures"], {m["kind"] for m in catalog["models"]}
    assert {"lstm", "gru", "tcn", "patchtst", "nbeats", "nhits", "tft"} <= set(arch) <= kinds
    assert set(arch) == set(ARCHITECTURES) and {k for k, a in arch.items() if a["reduced"]} == {"patchtst", "nbeats", "nhits", "tft"}
    for kind, a in arch.items():
        assert a["kind"] == kind and a["manifest_version"] == 1
        assert all(isinstance(a[key], str) and a[key] for key in FIELDS), kind
        assert bool(a.get("differences")) is a["reduced"], f"{kind}: a reduced network lists what it leaves out, and a faithful one lists nothing"
    assert arch["patchtst"]["variant"].startswith("REDUCED") and arch["nbeats"]["variant"].startswith("REDUCED") and arch["nhits"]["variant"].startswith("REDUCED") and arch["tft"]["variant"].startswith("COMPACT")
    assert "no static covariates" in arch["tft"]["differences"]


def test_a_network_the_manifest_does_not_describe_is_not_invented_for_the_form(catalog):
    models = {m["kind"]: m for m in catalog["models"]}
    assert "ridge" in models and "ridge" not in catalog["architectures"], "only networks have an architecture entry; the form shows none for a linear model"
    assert all(k in models for k in catalog["architectures"]), "an entry names a model the catalogue offers"


def test_reading_the_manifest_loads_no_network_library():
    code = ("import sys, json\nfrom src.ml.deep.manifest import ARCHITECTURES, architecture\nfrom src.webapp import ml_api\n"
            "architecture('lstm')\nprint(json.dumps({'torch': 'torch' in sys.modules, 'n': len(ARCHITECTURES)}))")
    run = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stderr[-1500:]
    out = json.loads(run.stdout.strip().splitlines()[-1])
    assert out == {"torch": False, "n": len(ARCHITECTURES)}
