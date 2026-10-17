"""The Replit configuration: the Run button starts the dashboard from the workflow, and Publish needs a run command of its own (without one Replit says "Could not find run command")."""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:                                  # Python 3.10: pytest itself depends on tomli there
    import tomli as tomllib

CONFIG = Path(__file__).resolve().parents[1] / ".replit"
pytestmark = pytest.mark.skipif(not CONFIG.exists(), reason="the Replit configuration is not part of this copy of the repository (the Docker image, for one)")


def _config() -> dict:
    return tomllib.loads(CONFIG.read_text(encoding="utf-8"))


def test_publishing_has_a_run_command_that_serves_the_dashboard_on_the_port_replit_exposes():
    cfg = _config()
    run = cfg["deployment"]["run"]
    assert run[:2] == ["sh", "-c"] and "src serve" in run[2] and "--host 0.0.0.0" in run[2]
    port = next(p["localPort"] for p in cfg["ports"] if p["externalPort"] == 80)
    assert f"--port {port}" in run[2]


def test_the_run_button_and_publishing_start_the_dashboard_the_same_way():
    cfg = _config()
    workflow = next(w for w in cfg["workflows"]["workflow"] if w["name"] == "Quant Lab")
    assert cfg["deployment"]["run"][2] in workflow["tasks"][0]["args"]
