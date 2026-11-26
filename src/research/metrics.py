"""The metric registry: what each saved metric means, which way is better, which saved results can carry it and when two values of it are comparable.

A ranking is only as honest as the direction it sorts in and the question the metrics answer. Sorting RMSE or CRPS "highest first" puts the worst forecast on top, ranking a rank IC measured over 5 days against one over 21 days compares
different questions, and a Sharpe ratio next to a log loss is not a comparison at all. Every metric the comparison ranks by is declared here once:

``direction``   ``higher`` or ``lower`` is better (``max_drawdown`` is stored as a negative number, so higher, i.e. closer to zero, is better);
``family``      ``trading`` (needs a return series), ``predictive`` (a forecast or factor score) or ``calibration`` (how honest the stated uncertainty is);
``kinds``       the artifact kinds that carry it; a ranking over results that do not all carry it is refused;
``horizon``     True if its value depends on the forecast horizon, so that two values are comparable only at the same horizon;
``scale``       what must be the same for two values to be comparable beyond the horizon: ``units`` (error metrics are in the units of the target, so only the same target and transformation compare) or ``none``.

Nothing here computes a metric; it says how a computed one may be used.
"""

from __future__ import annotations

from dataclasses import dataclass

TRADING_KINDS = ("backtest", "portfolio_lab", "ml_experiment", "options", "futures", "crypto", "regime")


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    direction: str                                  # "higher" | "lower"
    family: str                                     # "trading" | "predictive" | "calibration"
    kinds: tuple[str, ...]
    horizon: bool = False
    scale: str = "none"
    note: str = ""


def _m(key, label, direction, family, kinds, **kw) -> tuple[str, Metric]:
    return key, Metric(key, label, direction, family, tuple(kinds), **kw)


REGISTRY: dict[str, Metric] = dict([
    # trading performance, recomputed on the common grid
    _m("sharpe", "Sharpe ratio", "higher", "trading", TRADING_KINDS),
    _m("sortino", "Sortino ratio", "higher", "trading", TRADING_KINDS),
    _m("calmar", "Calmar ratio", "higher", "trading", TRADING_KINDS),
    _m("cagr", "CAGR", "higher", "trading", TRADING_KINDS),
    _m("total_return", "Total return", "higher", "trading", TRADING_KINDS),
    _m("max_drawdown", "Max drawdown", "higher", "trading", TRADING_KINDS, note="stored as a negative number, so the value closer to zero is better"),
    _m("ann_vol", "Volatility (annual)", "lower", "trading", TRADING_KINDS),
    _m("turnover_annual", "Turnover (annual)", "lower", "trading", TRADING_KINDS),
    _m("cost_drag_annual", "Costs (annual)", "lower", "trading", TRADING_KINDS),
    # predictive quality of a signal or a forecast
    _m("best_combination_test_ic", "Rank IC of the best combination (test period)", "higher", "predictive", ("factor_study",), horizon=True),
    _m("rank_ic", "Rank information coefficient", "higher", "predictive", ("factor_study", "ml_experiment"), horizon=True),
    _m("best_skill_common", "Skill against the random walk (common dates)", "higher", "predictive", ("forecast",), horizon=True),
    _m("directional_accuracy", "Directional accuracy", "higher", "predictive", ("forecast", "ml_experiment"), horizon=True),
    _m("rmse", "RMSE", "lower", "predictive", ("forecast", "ml_experiment"), horizon=True, scale="units"),
    _m("mae", "MAE", "lower", "predictive", ("forecast", "ml_experiment"), horizon=True, scale="units"),
    _m("mape", "MAPE", "lower", "predictive", ("forecast", "ml_experiment"), horizon=True),
    _m("crps", "CRPS", "lower", "calibration", ("forecast",), horizon=True, scale="units"),
    _m("pinball", "Pinball loss", "lower", "calibration", ("forecast",), horizon=True, scale="units"),
    _m("log_loss", "Log loss", "lower", "predictive", ("forecast", "ml_experiment"), horizon=True),
    _m("brier", "Brier score", "lower", "calibration", ("forecast", "ml_experiment"), horizon=True),
    _m("best_coverage90", "Coverage of the 90% interval", "target", "calibration", ("forecast",), horizon=True, note="nearest to 0.90 is best; neither higher nor lower"),
])

DEFAULT_TRADING = "sharpe"
DEFAULT_PREDICTIVE = {"factor_study": "best_combination_test_ic", "forecast": "best_skill_common"}
TARGETS = {"best_coverage90": 0.90}


class MetricError(ValueError):
    pass


def get(key: str) -> Metric:
    if key not in REGISTRY:
        raise MetricError(f"unknown metric {key!r}; the metrics a comparison can rank by are: " + ", ".join(sorted(REGISTRY)))
    return REGISTRY[key]


def sort_key(key: str, value):
    """A key under which ``sorted`` puts the best first for this metric's direction, and missing values last."""
    m = get(key)
    if value is None:
        return (1, 0.0)
    v = float(value)
    if m.direction == "higher":
        return (0, -v)
    if m.direction == "lower":
        return (0, v)
    return (0, abs(v - TARGETS[key]))


def rank(values: dict[str, float | None], key: str) -> list[str]:
    """Ids best first by the metric's own direction; a result without the value is last."""
    return [k for k, _ in sorted(values.items(), key=lambda kv: sort_key(key, kv[1]))]


def applicability(arts: list[dict], key: str) -> list[str]:
    """Why ``key`` cannot rank these results (an empty list means it can). Checks the kind each result is, that each carries the number (a trading metric is recomputed, so it needs a series), that predictive values
    are at one horizon and, for metrics in the units of the target, that the target is the same."""
    m = get(key)
    why: list[str] = []
    wrong = [a["title"] for a in arts if a["kind"] not in m.kinds]
    if wrong:
        why.append(f"{m.label} is not defined for " + ", ".join(sorted(set(wrong))) + f" (it applies to {', '.join(k.replace('_', ' ') for k in m.kinds)})")
    if m.family == "trading":
        none = [a["title"] for a in arts if not a.get("series")]
        if none:
            why.append(f"{m.label} is recomputed from a return series and " + ", ".join(none) + " saved none")
    else:
        none = [a["title"] for a in arts if (a.get("metrics") or {}).get(key) is None]
        if none and not wrong:
            why.append(f"{m.label} was not saved with " + ", ".join(none))
        if m.horizon and len({(a.get("config") or {}).get("horizon") for a in arts}) > 1:
            why.append(f"{m.label} depends on the horizon and these results look at different horizons")
        if m.scale == "units" and len({(a["context"].get("assets") and tuple(a["context"]["assets"]), (a.get("config") or {}).get("target")) for a in arts}) > 1:
            why.append(f"{m.label} is in the units of what was predicted, and these results predict different instruments or targets")
    return why


def describe() -> list[dict]:
    """The registry as the page and the report show it."""
    return [{"key": m.key, "label": m.label, "direction": m.direction, "family": m.family, "kinds": list(m.kinds), "horizon_specific": m.horizon, "note": m.note} for m in REGISTRY.values()]
