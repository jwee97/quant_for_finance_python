"""Every stage script and every module imports, and every registered stage has a script with a ``main``.

The unit tests exercise the library; this catches the failure they cannot: a stage script that no longer
imports after a refactor or an automated clean-up.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

import pytest

from experiments.run_all import STAGES

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("number,module", [(n, m) for n, m, *_ in STAGES])
def test_every_registered_stage_imports_and_has_a_main(number, module):
    imported = importlib.import_module(f"experiments.{module}")
    assert callable(getattr(imported, "main", None)), f"stage {number} has no main()"
    assert str(number).zfill(2) == module[5:7]


def test_every_stage_script_on_disk_is_registered():
    on_disk = {p.stem for p in (ROOT / "experiments").glob("stage[0-9][0-9]_*.py")}
    assert on_disk == {m for _, m, *_ in STAGES}


def test_every_library_module_imports():
    failures = []
    for module in pkgutil.walk_packages([str(ROOT / "src")], "src."):
        if module.name.endswith("__main__"):
            continue
        try:
            importlib.import_module(module.name)
        except Exception as error:                                  # report all failures, not the first
            failures.append((module.name, repr(error)))
    assert not failures, failures
