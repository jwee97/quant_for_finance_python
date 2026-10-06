import json
import re

import pytest

from src.framework.dashboard import build_dashboard
from src.utils.config import load_config


def test_dashboard_embeds_the_ledger_and_is_valid_json(tmp_path):
    config = load_config()
    path = build_dashboard(config, tmp_path / "dashboard.html")
    html = path.read_text(encoding="utf-8")
    payload = re.search(r"const D=(\{.*?\});\nconst \$=", html, re.S).group(1)
    data = json.loads(payload.replace("<\\/", "</"))
    registry_lines = [l for l in (config.root / "experiments" / "registry.jsonl").read_text().splitlines() if l.strip()]
    assert len(data["ledger"]) == len(registry_lines)
    assert {r["decision"] for r in data["ledger"]} >= {"retain", "reject"}
    for tab in ("Decision ledger", "Strategy library", "Your runs", "What can we detect?"):
        assert tab in html


def test_dashboard_renders_without_script_errors(tmp_path):
    pw = pytest.importorskip("playwright.sync_api")
    import os
    exe = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"
    if not os.path.exists(exe):
        pytest.skip("no chromium")
    path = build_dashboard(load_config(), tmp_path / "dashboard.html")
    errors = []
    with pw.sync_playwright() as p:
        browser = p.chromium.launch(executable_path=exe, args=["--no-sandbox"])
        page = browser.new_page()
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(f"file://{path}")
        for tab in ("library", "runs", "ic", "walk", "risk", "figs", "power"):
            page.click(f'nav button[data-t="{tab}"]')
        assert page.locator("#ledgerTable tr").count() > 10
        browser.close()
    assert errors == []
