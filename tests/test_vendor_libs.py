import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "vendor_libs.py"


def _load():
    spec = importlib.util.spec_from_file_location("vendor_libs", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_script_copies_the_bundles_out_of_installed_packages(tmp_path, monkeypatch):
    (tmp_path / "pkgs" / "plotly" / "package_data").mkdir(parents=True)
    (tmp_path / "pkgs" / "plotly" / "__init__.py").write_text("")
    (tmp_path / "pkgs" / "plotly" / "package_data" / "plotly.min.js").write_text("/* plotly */")
    (tmp_path / "pkgs" / "dash_ag_grid").mkdir()
    (tmp_path / "pkgs" / "dash_ag_grid" / "__init__.py").write_text("")
    (tmp_path / "pkgs" / "dash_ag_grid" / "ag-grid-community.min.js").write_text("/* ag */")
    (tmp_path / "pkgs" / "dash_ag_grid" / "ag-theme-quartz.css").write_text("/* css */")
    monkeypatch.syspath_prepend(str(tmp_path / "pkgs"))
    for m in ("plotly", "dash_ag_grid"):
        monkeypatch.delitem(sys.modules, m, raising=False)
    mod = _load()
    monkeypatch.setattr(mod, "VENDOR", tmp_path / "vendor")
    assert mod.main([]) == 0
    assert (tmp_path / "vendor" / "plotly.min.js").read_text() == "/* plotly */" and (tmp_path / "vendor" / "ag-grid-community.min.js").is_file() and (tmp_path / "vendor" / "ag-theme-quartz.css").is_file()


def test_check_mode_copies_nothing(tmp_path, monkeypatch):
    mod = _load()
    monkeypatch.setattr(mod, "VENDOR", tmp_path / "vendor")
    assert mod.main(["--check"]) == 0 and not (tmp_path / "vendor").exists()


def test_an_installed_plotly_package_is_served_without_copying_anything(tmp_path, monkeypatch):
    pkg = tmp_path / "pkgs" / "plotly" / "package_data"
    pkg.mkdir(parents=True)
    (tmp_path / "pkgs" / "plotly" / "__init__.py").write_text("")
    (pkg / "plotly.min.js").write_text("/* plotly */")
    monkeypatch.syspath_prepend(str(tmp_path / "pkgs"))
    monkeypatch.delitem(sys.modules, "plotly", raising=False)
    from src.webapp import server
    hit = server.static_file("vendor/plotly.min.js")
    assert hit is not None and hit.read_text() == "/* plotly */" and server.static_file("vendor/nothing.js") is None
