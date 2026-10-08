"""How many tickers a strategy needs, and the message to show when a universe is too small.

A rule that ranks assets against each other (cross-sectional momentum, value, a pair or a basket) has nothing to say about a single ticker: every rank is the same and the book is
empty. A covariance-based allocation needs several assets to diversify between. Each strategy declares what it needs (``ForecastModel.required_assets``) and each allocator likewise
(``Allocator.required_assets``); this module combines them for the allocator a spec would actually end up with, so the dashboard can refuse a run that could only produce an empty book.
"""

from __future__ import annotations

from .pipeline import allocator_choice
from .registry import ALLOCATORS, MODELS

ONE_TICKER_EXAMPLES = ("tsmom", "ma_crossover", "rsi2", "donchian", "macd_trend")
_FACTORY_MIN: dict = {}


def factory_min_assets(factory) -> int:
    """How many tickers a strategy needs, from an instance with default settings (so a rule whose need depends on its settings reports the default), else from the class."""
    if factory not in _FACTORY_MIN:
        try:
            _FACTORY_MIN[factory] = int(factory().required_assets())
        except Exception:                                           # a strategy that cannot be built without arguments
            _FACTORY_MIN[factory] = max(int(getattr(factory, "min_assets", 1) or 1), 2 if getattr(factory, "position_mode", "cross_sectional") == "cross_sectional" else 1)
    return _FACTORY_MIN[factory]


def _model_reason(model) -> str:
    if getattr(model, "position_mode", "cross_sectional") == "cross_sectional" and int(model.min_assets) <= 1:
        return f"'{model.name}' ranks the tickers against each other"
    return f"'{model.name}' trades a group of tickers together"


def _assess(models: list, allocation: dict | None) -> tuple[int, list[tuple[str, str]]]:
    """``(fewest tickers, [(reason, source)])`` where ``source`` is ``"model"`` or ``"allocation"``: what sets the bar, strictest first."""
    found: list[tuple[int, str, str]] = []
    for model in models:
        need = int(model.required_assets())
        if need > 1:
            found.append((need, _model_reason(model), "model"))
    name, params = allocator_choice(allocation, models)
    need = int(ALLOCATORS.create(name, **params).required_assets())
    if need > 1 and name != "model_weights":
        found.append((need, "this allocation scores the tickers against each other" if name in ("forecast_stack", "score_stack") else
                      f"the '{name}' allocation spreads the portfolio over several tickers", "allocation"))
    top = max([1] + [n for n, _, _ in found])
    return top, [(reason, source) for n, reason, source in found if n == top] if top > 1 else []


def minimum_assets(models: list, allocation: dict | None = None) -> tuple[int, list[str]]:
    """``(fewest tickers, reasons)`` for these strategy instances under this allocation spec. ``reasons`` says what sets the bar (empty when one ticker is enough)."""
    need, found = _assess(models, allocation)
    return need, [reason for reason, _ in found]


def universe_problem(models: list, allocation: dict | None, tickers: int) -> str | None:
    """A sentence saying why a universe of ``tickers`` is too small for these strategies, or ``None`` when it is big enough."""
    need, found = _assess(models, allocation)
    if tickers >= need:
        return None
    reason, source = found[0]
    more = need - tickers
    advice = f"Add {more} more ticker{'s' if more > 1 else ''}"
    if need == 2 and tickers == 1:
        if source == "model":
            examples = [n for n in ONE_TICKER_EXAMPLES if n in MODELS][:4]
            advice += f", or choose a strategy that works on one ticker (for example {', '.join(examples)})."
        else:
            advice += ", or pick 'Independent sleeves' under Portfolio (each ticker is traded on its own signal)."
    else:
        advice += "."
    return f"{reason}, so it needs at least {need} tickers and you have {tickers}. {advice}"
