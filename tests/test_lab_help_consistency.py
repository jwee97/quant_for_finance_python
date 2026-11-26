"""What the Alpha Lab's own guidance says about saving must be what the research API does."""

from __future__ import annotations

import pytest

from src.utils.config import load_config
from src.webapp import lab_api
from src.webapp.api import App
from src.webapp.universe import TickerStore, UniverseBuilder
from tests.app_roots import isolated_root
from tests.test_webapp import _default, _fetch


@pytest.fixture()
def app(tmp_path):
    config = load_config()
    return App(config, builder=UniverseBuilder(config, TickerStore(tmp_path / "prices", fetch=_fetch), default_loader=_default), root=isolated_root(tmp_path / "root", config))


def test_the_lab_no_longer_says_nothing_can_be_saved_and_names_what_can(app):
    plan = app.extra_get["/api/lab/plan"]({})
    text = " ".join(plan["notes"])
    assert "Nothing here is saved" not in text and "not saved to a research session yet" not in (lab_api.__doc__ or "")
    for study in ("factor study", "portfolio comparison", "trading study"):
        assert study in text, study
    assert "single-signal alpha study and the risk study cannot yet" in text


def test_every_study_the_help_says_can_be_saved_is_accepted_by_the_save_route():
    import inspect

    from src.webapp import research_api

    src = inspect.getsource(research_api.ResearchApi.artifact_save)
    for needle in ("portfolio_lab_artifact", "factor_study_artifact", "trading_study_artifact"):
        assert needle in src
