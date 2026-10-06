"""The tutorial notebooks are executed from scratch: a notebook that no longer runs is documentation that lies."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("nbclient")
pytest.importorskip("ipykernel")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docs" / "notebooks"))


def _execute(name: str):
    from nbclient import NotebookClient
    from nbformat.v4 import new_notebook

    from build_notebooks import NOTEBOOKS

    nb = new_notebook(cells=NOTEBOOKS[name], metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
    try:
        NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT / "docs" / "notebooks")}}).execute()
    except Exception as exc:
        if "No such kernel" in str(exc):
            pytest.skip("python3 kernel not installed")
        raise
    return "\n".join(o.get("text", "") for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.get("output_type") == "stream")


def test_first_backtest_notebook_runs():
    out = _execute("01_first_backtest.ipynb")
    assert "net Sharpe" in out and "15 assets" in out


def test_regime_notebook_runs():
    out = _execute("02_regimes_and_adaptation.ipynb")
    assert "share of placebos at least as good" in out


def test_multiple_testing_notebook_runs_and_shows_the_deflation():
    out = _execute("03_how_many_ideas_did_you_try.ipynb")
    assert "counting all 200 tries" in out and "after Benjamini-Hochberg" in out
    import re

    alone, deflated = (float(x) for x in re.search(r"idea: ([\d.]+); counting all 200 tries: ([\d.]+)", out).groups())
    assert deflated < 0.5 < alone, "the best of 200 noise strategies must look significant alone and not after deflation"


def test_committed_notebooks_have_outputs_and_no_errors():
    import nbformat

    for path in sorted((ROOT / "docs" / "notebooks").glob("*.ipynb")):
        nb = nbformat.read(path, 4)
        code = [c for c in nb.cells if c.cell_type == "code"]
        assert code and all(c.get("outputs") is not None for c in code)
        assert any(c.outputs for c in code), path.name
        assert not any(o.get("output_type") == "error" for c in code for o in c.outputs), path.name
