"""Your own strategies: every ``*.py`` file in the ``user_strategies/`` folder of the project is imported when the library loads.

``quant new-strategy my_idea`` writes a ready-to-edit template there. Importing a file registers whatever it registers, so the model appears in
``quant list models``, in ``quant backtest --model``, and in the dashboard's strategy list. Files whose name starts with ``_`` are skipped.
These files run as ordinary Python with your permissions: only put code you wrote or trust in that folder.
"""

from __future__ import annotations

import re
import sys
import types
from pathlib import Path

FOLDER = "user_strategies"
NAME = re.compile(r"^[a-z][a-z0-9_]{2,40}$")

TEMPLATE = '''"""{title}: replace this sentence with what the strategy bets on and why it might work."""

from src.framework.forecasting import ForecastModel
from src.framework.registry import register_model


@register_model("{name}", "custom", "Describe the bet in one sentence of more than thirty characters; it becomes the strategy's explanation")
class {cls}(ForecastModel):
    """Why might this work? Write the economic reason here: the framework asks for it."""

    name, family = "{name}", "custom"
    position_mode = "cross_sectional"          # "cross_sectional": rank assets against each other (needs two or more tickers); "time_series": each asset on its own (runs on one ticker)
    # min_assets = 3                            # uncomment if the idea needs more tickers than its mode implies (a pair needs 2, a basket 3); the dashboard then refuses a smaller universe

    def __init__(self, window: int = 63):
        self.window = window                    # parameters with defaults show up as form fields in the dashboard

    def score(self, data):
        """Return a table (dates x assets). Higher = expect a higher return. Use only information up to each date."""
        momentum = data.prices / data.prices.shift(self.window) - 1.0          # <- your idea goes here
        return momentum.where(data.investable)                                   # always mask with data.investable
'''


def folder(root: Path) -> Path:
    return Path(root) / FOLDER


def new_strategy(root: Path, name: str) -> Path:
    """Write the template for ``name`` (lower-case letters, digits, underscores) and return its path; never overwrites."""
    if not NAME.match(name):
        raise ValueError("a strategy name is 3-41 characters: lower-case letters, digits and underscores, starting with a letter")
    target = folder(root) / f"{name}.py"
    if target.exists():
        raise FileExistsError(f"{target} already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(TEMPLATE.format(title=name.replace("_", " ").capitalize(), name=name, cls="".join(p.capitalize() for p in name.split("_"))), encoding="utf-8")
    return target


def load_user_strategies(root: Path, reload: bool = False) -> dict[str, str]:
    """Import every file in the folder. Returns ``{file name: "" or the error}`` so a broken file is reported, not fatal."""
    out: dict[str, str] = {}
    directory = folder(root)
    if not directory.is_dir():
        return out
    for path in sorted(directory.glob("*.py")):
        if path.name.startswith("_"):
            continue
        module_name = f"user_strategies.{path.stem}"
        try:
            if module_name in sys.modules and not reload:
                out[path.name] = ""
                continue
            if module_name in sys.modules:
                _forget(path.stem)
            module = types.ModuleType(module_name)
            module.__file__ = str(path)
            sys.modules[module_name] = module
            exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)      # from source, never a stale bytecode cache
            out[path.name] = ""
        except Exception as error:                               # a typo in your file must not stop the library from loading
            sys.modules.pop(module_name, None)
            out[path.name] = f"{type(error).__name__}: {error}"
    return out


def _forget(stem: str) -> None:
    """Drop what a previous import of this file registered, so reloading after an edit replaces it instead of colliding."""
    from ..framework.registry import MODELS
    module = sys.modules.pop(f"user_strategies.{stem}", None)
    for entry in list(MODELS.entries()):
        if getattr(entry.factory, "__module__", "") == getattr(module, "__name__", None):
            MODELS.unregister(entry.name)
