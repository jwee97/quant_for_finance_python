"""Find good settings for a strategy without guessing, and say whether to believe them.

A strategy's settings (the two noise levels of the Kalman trend filter, a momentum lookback, how many assets to hold) are normally chosen by hand, and trying them by hand is slow. Handing the job to a
search is quick, but the best of many tries looks better than it is: the more settings are tried on the same prices, the better the winner looks, whether or not it is any good. This is the beginner's
version of a proper search, and what makes it safe is what it does around the search:

* every setting of the strategy gets a range (a hand-picked one for the common strategies, one around the standard value for the rest), so the person only chooses how long to look;
* the first thing tried is the strategy's own settings, so the answer always has something to be compared with, and "keep what you have" is a possible result;
* the days are split in time: the search sees the first part only (the *tuning window*) and never the last part (the *hold-out*, 30% of the days unless asked otherwise), which is used once, at the
  end, to compare the starting settings with the winner;
* the answer says in words what happened. Settings that win on the tuning window and lose on the hold-out are overfitted, and the verdict says so; the chance that the winner's Sharpe ratio is luck after so
  many tries (the deflated Sharpe ratio) and the probability of backtest overfitting come from the tries' own return streams (``ops.hpo.Study.selection_report``).

``tune`` is independent of how a strategy is run: it takes ``run(params) -> Run`` (the net daily returns of the strategy with those settings, and optionally its weights), so the search can be tested with
synthetic return streams, and ``webapp.tune_api`` supplies the real pipeline. The search is a space-filling start (a scrambled Sobol sequence) followed by a Tree-structured Parzen Estimator or a
Gaussian process (``ops.hpo``); no configuration is run twice.

What is done with the search's result is as much of the method as the search:

* the winner can be the trial whose *neighbourhood* did best (``Study.plateau``) rather than the single best score, which on a noisy objective is mostly luck;
* the tries that were drawn before the search learned anything do not depend on the data, so they can be used to ask a second question honestly: had the settings been re-chosen on the past
  alone, again and again, would they have beaten the starting settings on what came next (``walk_forward_check``)?
* a gain that held on the hold-out is called *good* only when it is also more than the hold-out's own noise (a paired t statistic of the gain), when the winner is not what the best of that many
  lucky tries would show (the Romano-Wolf p-value over every try's return stream) and when that re-choosing did not lose. The thresholds were set by simulation on strategies with a known truth
  (``experiments/tune_selection_study.py``): the hold-out gain alone said "good" one time in five when no setting had any edge.
"""

from __future__ import annotations

import logging
import math
import re
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from ..backtest.metrics import annualised_volatility, cagr, calmar_ratio, sharpe_ratio, sortino_ratio
from ..features.returns import cumulative_returns, max_drawdown
from ..ops.hpo import Categorical, Float, Int, LogFloat, Param, Study
from ..stats.inference import sharpe_standard_error

MIN_TUNE_DAYS, MIN_HOLD_DAYS = 252, 63                                     # a year to learn from, a quarter to check on: less says nothing
MIN_TRIALS, MAX_TRIALS = 6, 300                                            # a search of fewer than six tries is not a search; more than 300 is a way to fool oneself
MAX_TICKED = 6                                                             # settings ticked at first: more than that cannot be searched in a few dozen tries
CURVE_POINTS = 500
EDGE = 0.1                                                                 # a best value in the outer tenth of its range (on the range's own scale) is flagged: the range may be too narrow
STABILITY_BLOCKS, STABILITY_PENALTY = 4, 0.5                               # the consistency score: mean of the metric over four equal parts, minus half their spread
SEARCH, SELECT = "tpe", "best"                                             # how tries are proposed and which one wins (see ``tune``)
SEARCHES = {"tpe": {"label": "Smart search (recommended)", "text": "An evenly spread start, then a model of where the good settings and the bad ones sat, which proposes values like the good ones. Works with any mix of numbers and choices."},
            "bayes": {"label": "Gaussian process", "text": "Fits a smooth surface to the scores and tries where an improvement looks likeliest. Strong with a few numeric settings; with more than about eight, or over a hundred tries, it is slower and no better."},
            "random": {"label": "Random (a baseline)", "text": "Every try is drawn blindly. Use it to see how much the smarter searches add."}}
SELECTS = {"best": {"label": "The highest score", "text": "The setting that scored best on the tuning window. The usual choice, and the one that measured best on strategies whose neighbouring settings behave alike."},
           "plateau": {"label": "The middle of the best region", "text": "The setting whose neighbours also scored well, so a lone spike is passed over. Useful when a score jumps about between nearby values."}}
WALK_FOLDS = 4                                                             # the later stretches of the tuning window that the walk-forward check re-chooses for
HOLD_T_MIN, RW_MAX = 1.0, 0.3                                              # what a "good" verdict asks of the untouched days (a paired t of the gain) and of the search (the Romano-Wolf p of the winner): see ``judge``

EFFORTS = {"quick": {"label": "Quick", "trials": 20, "seconds": 90, "text": "up to 20 tries or about a minute and a half: a first look at which settings matter"},
           "standard": {"label": "Standard", "trials": 50, "seconds": 300, "text": "up to 50 tries or five minutes: enough for most strategies with three or four settings"},
           "thorough": {"label": "Thorough", "trials": 120, "seconds": 900, "text": "up to 120 tries or fifteen minutes: for many settings, or when the answer will be relied on"}}

def cautious_sharpe(returns: pd.Series, errors: float = 1.0) -> float:
    """The annual Sharpe ratio less ``errors`` standard errors of that estimate (skewness and kurtosis included, Mertens 2002): a lower bound. Two settings with the same ratio are not equally good
    when one of them rests on less evidence (a shorter stretch, fatter tails), and this marks that one down."""
    r = _clean(returns)
    if len(r) < 60 or not r.std(ddof=1) > 1e-12:                                          # a stream that never moves has no ratio (and no standard error)
        return float("nan")
    return float((r.mean() / r.std(ddof=1) - errors * sharpe_standard_error(r.to_numpy(), "mertens")) * math.sqrt(252))


OBJECTIVES = {
    "sharpe": {"label": "Return for the risk taken (Sharpe ratio)", "short": "Sharpe ratio", "unit": "ratio", "fn": sharpe_ratio,
               "text": "Yearly return divided by yearly ups and downs. The usual yardstick: a higher number means a smoother ride for the same gain."},
    "sortino": {"label": "Return for the downside risk (Sortino ratio)", "short": "Sortino ratio", "unit": "ratio", "fn": sortino_ratio,
                "text": "Like the Sharpe ratio, but only losing days count as risk, so a strategy is not marked down for big gains."},
    "calmar": {"label": "Growth against the worst fall (Calmar ratio)", "short": "Calmar ratio", "unit": "ratio", "fn": calmar_ratio,
               "text": "Yearly growth divided by the deepest fall from a peak. Favours settings that avoid one terrible stretch."},
    "cagr": {"label": "Yearly growth", "short": "yearly growth", "unit": "pct", "fn": cagr,
             "text": "How fast the money grew, with no allowance for risk: it rewards bold settings, so read the worst fall beside it."},
    "cautious": {"label": "Cautious Sharpe ratio (a lower bound)", "short": "cautious Sharpe ratio", "unit": "ratio", "fn": cautious_sharpe,
                 "text": "The Sharpe ratio less one standard error of that estimate: a result that rests on little evidence (a short stretch, big swings in one direction) is marked down. Steadier than the plain ratio on a short history."},
}


class TuningError(ValueError):
    """A request or a search the person can do something about; the message is shown in the page."""


@dataclass
class Run:
    """What one strategy run gives the tuner: the net daily returns, and the position weights when they are known (for the table by ticker)."""

    returns: pd.Series
    weights: pd.DataFrame | None = None


# ------------------------------------------------------------------------------------------------------------------ settings and their ranges
def _r(low, high, scale="linear", tune=True):
    return (low, high, scale, tune)


# Hand-picked ranges. Where a rule needs one setting to stay below another (a fast average shorter than a slow one, an entry level below an exit level), the two ranges do not overlap, so that no
# combination is invalid. ``tune=False`` lists a setting that can be tuned but is left unticked at first (here: ``clip``, which only ever acts through its product with ``scale``).
CURATED = {
    "kalman_trend": {"level_noise": _r(0.002, 0.5, "log"), "slope_noise": _r(1e-7, 1e-3, "log"), "vol_halflife": _r(10, 250, "log"), "scale": _r(0.03, 1.0, "log"), "clip": _r(1.0, 10.0, "log", False)},
    "kalman_pairs": {"delta": _r(1e-7, 1e-3, "log"), "obs_var": _r(1e-4, 1e-1, "log"), "clip": _r(1.5, 5.0)},
    "momentum": {"lookback": _r(42, 378), "skip": _r(0, 21), "vol_lookback": _r(21, 252)},
    "xs_momentum": {"lookback": _r(63, 504), "skip": _r(0, 42)},
    "dual_momentum": {"lookback": _r(63, 504), "skip": _r(0, 42), "top_k": _r(1, 8)},
    "smooth_momentum": {"lookback": _r(126, 504), "skip": _r(0, 63)},
    "tsmom": {"lookback": _r(21, 504), "target_vol": _r(0.1, 0.8, "log"), "max_scale": _r(1.0, 8.0), "halflife": _r(20, 180, "log")},
    "ma_crossover": {"fast": _r(10, 90), "slow": _r(110, 300)},
    "trend_atr": {"fast": _r(10, 90), "slow": _r(110, 300), "k": _r(0.1, 2.0, "log"), "atr_window": _r(7, 30)},
    "macd_trend": {"fast": _r(5, 20), "slow": _r(24, 80), "signal": _r(4, 20), "vol_window": _r(10, 60)},
    "donchian": {"entry": _r(30, 150), "exit": _r(8, 28)},
    "kama_trend": {"er_window": _r(5, 30), "fast": _r(2, 8), "slow": _r(15, 60), "vol_window": _r(10, 60)},
    "adx_trend": {"window": _r(7, 40), "threshold": _r(10, 40)},
    "ichimoku_trend": {"tenkan": _r(4, 20), "kijun": _r(22, 60), "senkou": _r(64, 150)},
    "supertrend": {"atr_window": _r(5, 30), "multiplier": _r(1.5, 6.0)},
    "keltner_breakout": {"window": _r(10, 60), "k": _r(1.0, 4.0), "atr_window": _r(7, 30)},
    "bollinger_reversion": {"window": _r(10, 60), "k": _r(1.0, 3.5)},
    "rsi2": {"rsi_window": _r(2, 5), "entry": _r(3, 30), "trend_window": _r(100, 300), "exit_window": _r(2, 10)},
    "ibs_reversion": {"entry": _r(0.05, 0.4), "exit": _r(0.5, 0.95), "trend_window": _r(100, 300)},
    "stochastic_reversion": {"window": _r(5, 30), "smooth": _r(1, 6), "entry": _r(5, 35), "exit": _r(45, 80), "trend_window": _r(100, 300)},
    "ou_reversion": {"window": _r(60, 300), "max_half_life": _r(10, 90), "df_critical": _r(-3.5, -2.0)},
    "vol_managed_long": {"window": _r(10, 126), "max_scale": _r(1.0, 6.0)},
    "online_ridge": {"forgetting": _r(0.9, 0.999), "ridge": _r(1.0, 1000.0, "log"), "step": _r(0.005, 0.5, "log"), "process_to_observation": _r(1e-5, 1e-1, "log")},
    "ml_ridge": {"alpha": _r(1.0, 10000.0, "log")},
    "ml_trees": {"n_estimators": _r(50, 400), "max_depth": _r(2, 6), "learning_rate": _r(0.01, 0.3, "log"), "min_samples_leaf": _r(20, 400, "log")},
    # the ones whose constructors refuse a range centred on the default (a probability below one, a count of at most five, a month of the year)
    "consecutive_down": {"days": _r(2, 8)},
    "earnings_season_premium": {"volume_multiple": _r(1.5, 8.0)},
    "evt_risk_managed": {"p": _r(0.9, 0.995)},
    "flight_to_quality": {"threshold": _r(0.2, 0.9)},
    "fundamental_dcf": {"terminal_growth": _r(0.005, 0.04)},
    "paa": {"protection": _r(0, 2)},
    "rebalancing_flow": {"lead": _r(0, 5)},
    "sell_in_may": {"start_month": _r(1, 12), "end_month": _r(1, 12)},
}
TUNE_CHOICES = {("bollinger_reversion", "shorts"), ("rsi2", "shorts"), ("ibs_reversion", "shorts"), ("macd_trend", "use")}   # a yes/no or either/or that is a real choice, ticked at first
OPERATIONAL = re.compile(r"^(seed|min_.*|refit(_every)?|embargo|lag(_days)?|expiry|epochs|max_epochs|patience|batch|restarts|paths|max_points|optimize_every|train_months)$")   # how a model is fitted or fed, not what it believes

HELP = {
    ("kalman_trend", "level_noise"): "How freely the filter lets its estimate of today's price level move, compared with the day-to-day noise in prices. Higher: it follows prices closely and reacts fast. Lower: it smooths harder.",
    ("kalman_trend", "slope_noise"): "How quickly the filter lets the trend itself change. Higher: a trend can flip within weeks (more trading). Lower: it holds on to a trend for many months.",
    ("kalman_trend", "vol_halflife"): "Days over which recent ups and downs are averaged, so that a trend is measured in units of the asset's own risk. Half the weight sits on this many most recent days.",
    ("kalman_trend", "scale"): "The strength of a typical trend, in units of risk. The position grows with the trend up to roughly this strength and flattens beyond it. Lower: bigger positions on weak trends.",
    ("kalman_trend", "clip"): "Flattens the signal for very strong trends. It only acts together with scale (only their product matters), so tune one of the two, not both.",
    ("kalman_pairs", "delta"): "How fast the hedge ratio between the two prices is allowed to drift. Higher: it follows the pair closely. Lower: it assumes a steadier relationship.",
    ("kalman_pairs", "obs_var"): "How noisy the spread between the two prices is assumed to be. Higher: the filter trusts each day's price less.",
    ("online_ridge", "process_to_observation"): "How fast the Kalman filter lets the model's coefficients drift, compared with the noise in returns. Only used when the updater is kalman.",
}
GENERIC_HELP = [
    (r"^(lookback|.*_lookback)$", "How many days of past prices the rule looks at. A longer look reacts more slowly but is steadier."),
    (r"^(window|.*_window)$", "How many days of history this part of the rule uses. Longer is smoother and slower to react."),
    (r"^skip$", "The most recent days left out of the lookback, because the latest weeks often reverse."),
    (r"^(fast|slow|tenkan|kijun|senkou|er_window)$", "Length in days of one of the averages the rule compares: the short one reacts quickly, the long one sets the direction."),
    (r"^(k|multiplier)$", "How many standard deviations (or average daily ranges) from normal count as a signal. Higher: fewer, stronger signals."),
    (r"^(top_k|top)$", "How many of the best-ranked assets are held."),
    (r"^shrink.*$", "How far the estimate is pulled toward a neutral guess: 0 trusts the data, 1 ignores it."),
    (r"^(entry|exit)$", "The level of the indicator at which the rule opens or closes a position."),
    (r"^threshold$", "How strong the signal must be before the rule acts on it."),
    (r"^shorts$", "Whether the rule may also bet on falling prices (short), or only buy."),
    (r"^(max_scale|target_vol|scale|clip)$", "Sets how big the positions are. Bigger positions change the risk more than the skill, so compare risk-adjusted numbers."),
    (r"^(halflife|half_life|.*_halflife)$", "Days over which the weight on older data halves: smaller reacts faster."),
    (r"^(alpha|ridge)$", "How strongly the model is held back from fitting noise: higher gives simpler, steadier fits."),
    (r"^(learning_rate|step)$", "How big each learning step is: higher learns faster but is noisier."),
    (r"^forgetting$", "How fast old data is forgotten: 0.97 remembers about a month, 0.999 about four years."),
    (r"^(n_estimators|trees)$", "How many small decision trees are added."),
    (r"^max_depth$", "How deep each tree may grow: deeper trees fit more detail and overfit more easily."),
    (r"^components$", "How many hidden market factors are taken out before looking at what is left."),
]


@dataclass
class Setting:
    """One setting of a strategy: its standard value, the range to search, and whether it is searched (ticked) at first."""

    name: str
    kind: str                                    # int | float | choice | bool
    default: object
    low: float | None = None
    high: float | None = None
    scale: str = "linear"                        # linear | log
    choices: tuple = ()
    tune: bool = True
    curated: bool = False
    help: str = ""
    start: object = None                         # the value the comparison starts from (the standard value unless the person changed it)

    @property
    def label(self) -> str:
        return self.name.replace("_", " ")

    def space(self) -> Param:
        if self.kind in ("choice", "bool"):
            return Categorical(*self.choices)
        if self.kind == "int":
            return Int(int(self.low), int(self.high))
        return LogFloat(float(self.low), float(self.high)) if self.scale == "log" else Float(float(self.low), float(self.high))

    def view(self) -> dict:
        return {"name": self.name, "label": self.label, "kind": self.kind, "default": self.default, "low": self.low, "high": self.high, "scale": self.scale, "choices": list(self.choices),
                "tune": self.tune, "curated": self.curated, "help": self.help}


def _round_sig(x: float, digits: int = 3) -> float:
    return float(f"{x:.{digits}g}")


def auto_range(kind: str, default) -> tuple | None:
    """A range around the standard value for a setting nobody has curated: a third to three times an integer, a third to three times a float of one or more, and ten times either way for a small float
    (never above 1, which is the top of almost every small float setting). ``None`` when there is nothing to centre on (zero, negative, missing)."""
    if isinstance(default, bool) or default is None or not isinstance(default, (int, float)):
        return None
    if kind == "int":
        if default < 1:
            return None
        low = max(1, round(default / 3))
        return (int(low), int(max(low + 2, round(default * 3))), "linear")
    if default <= 0 or not math.isfinite(default):
        return None
    if default >= 1:
        return (_round_sig(default / 3), _round_sig(default * 3), "log")
    if default >= 0.1:
        return (_round_sig(default / 3), _round_sig(min(1.0, default * 3)), "log")
    return (_round_sig(default / 10), _round_sig(min(1.0, default * 10)), "log")


def explain(strategy: str, name: str, default, low, high) -> str:
    if (strategy, name) in HELP:
        return HELP[(strategy, name)]
    for pattern, text in GENERIC_HELP:
        if re.match(pattern, name):
            return text
    return f"A setting of this strategy. Its standard value is {default}" + (f"; the range tried is {low:g} to {high:g}." if low is not None and high is not None else ".")


def settings_for(strategy: str, params: list[dict]) -> list[Setting]:
    """The settings of a strategy that can be searched, from its constructor's parameters (``webapp.api.parameters``): numbers, yes/no flags and fixed lists of choices. Text and structured settings
    are not searched. At most ``MAX_TICKED`` are ticked at first, the curated ones before the others and never a setting about how a model is fitted (``OPERATIONAL``)."""
    curated = CURATED.get(strategy, {})
    out: list[Setting] = []
    for p in params:
        name, kind, default = p["name"], p["kind"], p["default"]
        if p.get("required") or kind in ("str", "json") or default is None:
            continue
        if kind == "choice":
            choices = tuple(p["choices"] or ())
            if len(choices) < 2:
                continue
            out.append(Setting(name, "choice", default, choices=choices, tune=(strategy, name) in TUNE_CHOICES, curated=(strategy, name) in TUNE_CHOICES, help=explain(strategy, name, default, None, None)))
        elif kind == "bool":
            out.append(Setting(name, "bool", bool(default), choices=(False, True), tune=(strategy, name) in TUNE_CHOICES, curated=(strategy, name) in TUNE_CHOICES, help=explain(strategy, name, default, None, None)))
        elif kind in ("int", "float"):
            given = curated.get(name)
            rng = given or (auto_range(kind, default) + (True,) if auto_range(kind, default) else None)
            if rng is None:
                continue
            low, high, scale, tune = rng
            if kind == "int":
                low, high = int(low), int(high)
                if given is None and OPERATIONAL.match(name):                      # a minimum history or a refit interval: the default is a floor the constructor insists on, so look above it
                    low = max(low, int(default))
                    high = max(high, low + 2)
            out.append(Setting(name, kind, default, low, high, scale, tune=bool(tune) and not OPERATIONAL.match(name), curated=given is not None, help=explain(strategy, name, default, low, high)))
    ticked = 0
    for s in out:
        if s.tune:
            ticked += 1
            if ticked > MAX_TICKED:
                s.tune = False
    return out


# ------------------------------------------------------------------------------------------------------------------ windows, scores, metrics
@dataclass
class Windows:
    start: pd.Timestamp                          # first scored day: the starting settings' first traded day
    split: pd.Timestamp                          # first day of the hold-out
    end: pd.Timestamp                            # last day of the hold-out
    n_tune: int
    n_hold: int
    tuning_end: pd.Timestamp                     # last day of the tuning window


def make_windows(index, holdout: float) -> Windows:
    """Split the days the starting settings traded into the tuning window (first) and the hold-out (the last ``holdout`` share). Refuses a split that leaves either too short to say anything."""
    idx = pd.DatetimeIndex(index).sort_values()
    n = len(idx)
    n_hold = int(round(n * holdout))
    if n - n_hold < MIN_TUNE_DAYS or n_hold < MIN_HOLD_DAYS:
        raise TuningError(f"there are {n} trading days to work with: {n - n_hold} to tune on and {n_hold} to check on. At least {MIN_TUNE_DAYS} and {MIN_HOLD_DAYS} are needed. "
                          "Use a longer history (an earlier 'Use data from' date in the Backtest tab), or a smaller hold-out share.")
    return Windows(idx[0], idx[n - n_hold], idx[-1], n - n_hold, n_hold, idx[n - n_hold - 1])


def _clean(returns: pd.Series) -> pd.Series:
    r = pd.Series(returns).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    return r[~r.index.duplicated(keep="last")].sort_index()


def metrics(returns: pd.Series) -> dict:
    """The numbers shown for a stretch of net daily returns."""
    r = _clean(returns)
    if len(r) < 2:
        return {"days": int(len(r)), "sharpe": None, "sortino": None, "calmar": None, "cagr": None, "cautious": None, "vol": None, "max_drawdown": None, "total_return": None}
    curve = cumulative_returns(r)
    return {"days": int(len(r)), "sharpe": sharpe_ratio(r), "sortino": sortino_ratio(r), "calmar": calmar_ratio(r), "cagr": cagr(r), "cautious": cautious_sharpe(r), "vol": annualised_volatility(r),
            "max_drawdown": max_drawdown(curve), "total_return": float(curve.iloc[-1] - 1.0)}


def score(returns: pd.Series, objective: str, stable: bool = True) -> float:
    """What the search maximises, from the returns of the tuning window only. With ``stable`` the window is cut into four equal parts and a setting scores the mean of the metric over the parts less half
    their spread: a result that rests on one good stretch ranks below one that works throughout."""
    fn = OBJECTIVES[objective]["fn"]
    r = _clean(returns)
    if stable and len(r) >= STABILITY_BLOCKS * 60:
        edges = np.linspace(0, len(r), STABILITY_BLOCKS + 1).astype(int)
        parts = np.array([fn(r.iloc[a:b]) for a, b in zip(edges[:-1], edges[1:])], dtype=float)
        if np.isfinite(parts).all():
            return float(parts.mean() - STABILITY_PENALTY * parts.std(ddof=1))
    value = fn(r)
    return float(value) if value is not None else float("nan")


@contextmanager
def quiet(*names: str):
    """The backtest engine writes a line to the log for every run; a search runs it dozens of times. Raise those loggers to warnings for the duration (the single worker runs one job at a time)."""
    loggers = [logging.getLogger(n) for n in names]
    levels = [lg.level for lg in loggers]
    try:
        for lg in loggers:
            lg.setLevel(logging.WARNING)
        yield
    finally:
        for lg, level in zip(loggers, levels):
            lg.setLevel(level)


# ------------------------------------------------------------------------------------------------------------------ reading the trials
def _importance(setting: Setting, values: list, scores: list) -> float | None:
    """Share of the variation in scores that goes with this setting (between-group over total sum of squares, by quartile of its range or by choice): near 0, it did not matter; near 1, it did."""
    y = np.asarray(scores, dtype=float)
    if len(y) < 8:
        return None
    total = float(((y - y.mean()) ** 2).sum())
    if total <= 0:
        return 0.0
    if setting.kind in ("choice", "bool"):
        groups = [y[[v == c for v in values]] for c in setting.choices]
    else:
        space = setting.space()
        u = np.array([space.to_unit(v) for v in values], dtype=float)
        bins = np.digitize(u, np.unique(np.quantile(u, [0.25, 0.5, 0.75])))
        groups = [y[bins == b] for b in np.unique(bins)]
    between = sum(len(g) * (float(g.mean()) - float(y.mean())) ** 2 for g in groups if len(g))
    return float(min(1.0, between / total))


def _thin(values: np.ndarray, points: int = CURVE_POINTS) -> np.ndarray:
    n = len(values)
    return np.arange(n) if n <= points else np.unique(np.linspace(0, n - 1, points).astype(int))


def _fmt(objective: str, value) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value * 100:.1f}%" if OBJECTIVES[objective]["unit"] == "pct" else f"{value:.2f}"


def _gain(objective: str, value) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value * 100:+.1f} points" if OBJECTIVES[objective]["unit"] == "pct" else f"{value:+.2f}"


def _span(a, b) -> str:
    return f"{pd.Timestamp(a).date()} to {pd.Timestamp(b).date()}"


def spans(win: Windows) -> tuple[str, str]:
    """The dates of the tuning window and of the hold-out, for the verdict's sentences."""
    return _span(win.start, win.tuning_end), _span(win.split, win.end)


def judge(objective: str, start: dict, best: dict, tuning_span: str, hold_span: str, same: bool, selection: dict | None, tried: int, prior: int = 0, edges: list[str] | None = None,
          benchmark: dict | None = None, labels: tuple[str, str, str] = ("the tuning window", "the untouched days", "these days"), walk: dict | None = None, hold_t: float | None = None) -> dict:
    """The verdict, in words: did the winner's improvement survive the days the search never saw? ``start`` and ``best`` are ``{"tuning": {measure: value}, "holdout": {...}}``; ``tuning_span`` and
    ``hold_span`` are the dates to print (or empty); ``edges`` names the settings whose best value sits at an end of the range tried, ``benchmark`` is the plain hold-everything alternative over the same
    days, ``labels`` name the two parts of the data (the Machine learning tab says "the validation segment" and "the test and forward segments"), ``walk`` is the walk-forward check
    (``walk_forward_check``) and ``hold_t`` the paired t statistic of the gain on the untouched days (``paired_t``).

    A gain that held up on the untouched days is still not called good unless the evidence is more than the gain itself: the t statistic of that gain must reach ``HOLD_T_MIN``, the Romano-Wolf p-value
    of the winner (the chance that the best of everything tried is this good by luck, allowing for how many were tried and how alike they are) must be under ``RW_MAX``, and re-choosing the settings on the
    past alone must not have lost to the starting settings. The thresholds come from a simulation on strategies with a known truth (``experiments/tune_selection_study.py``, docs/tuning.md): the
    hold-out gain rule alone said "good" for a strategy with no edge one time in five."""
    name = OBJECTIVES[objective]["short"]
    tuning_name, hold_name, hold_these = labels
    where = lambda label, span: f"{label} ({span})" if span else label                       # noqa: E731
    d_t, b_t, d_h, b_h = start["tuning"].get(objective), best["tuning"].get(objective), start["holdout"].get(objective), best["holdout"].get(objective)
    reasons: list[str] = []
    if same:
        level, headline = "keep", f"Keep what you have: none of the {tried - 1} other settings beat it"
        reasons.append(f"Nothing tried did better than the starting settings on {where(tuning_name, tuning_span)}, so there is nothing to carry over.")
        if d_h is not None and np.isfinite(d_h):
            reasons.append(f"For reference, the starting settings scored {_fmt(objective, d_h)} ({name}) on {where(hold_name, hold_span)}.")
    elif any(v is None or not np.isfinite(v) for v in (d_t, b_t, d_h, b_h)):
        level, headline = "mixed", f"Could not judge: the {name} is undefined on part of the data"
        reasons.append(f"A stretch with no trades or no losing days has no such ratio, so the winner cannot be compared with the starting settings on {hold_name}. Try another measure, or a longer history.")
    else:
        gain_t, gain_h = b_t - d_t, b_h - d_h
        tiny = 0.005 if OBJECTIVES[objective]["unit"] == "pct" else 0.05
        reasons.append(f"On {where(tuning_name, tuning_span)} the {name} went from {_fmt(objective, d_t)} to {_fmt(objective, b_t)} ({_gain(objective, gain_t)}). That is the part the search could see, so some of it is luck.")
        reasons.append(f"On {where(hold_name, hold_span)} it went from {_fmt(objective, d_h)} to {_fmt(objective, b_h)} ({_gain(objective, gain_h)}). That is the honest number: no setting was chosen using {hold_these}.")
        if gain_h <= 0:
            level, headline = "bad", "Probably overfitted: better on the data it was tuned on, worse on the data it never saw"
            reasons.append("Settings that only help on the data they were chosen on are usually noise. Keep the starting settings, or try again with a longer history or fewer settings ticked.")
        elif gain_h < tiny:
            level, headline = "mixed", f"No real difference on {hold_name}"
            reasons.append(f"The improvement vanished outside {tuning_name}. The starting settings are as good, and simpler to defend.")
        elif b_h <= 0:
            level, headline = "mixed", f"Better than the starting settings, but still losing money on {hold_name}"
            reasons.append("Tuning made a poor strategy less poor. It did not make it a good one.")
        elif gain_t > 0 and gain_h >= 0.5 * gain_t:
            level, headline = "good", f"The improvement held up on {hold_name}"
            reasons.append(f"About {min(100, round(100 * gain_h / gain_t))}% of the gain survived. Expect the real gain from now on to be smaller than either number: the future is a third window.")
        else:
            level, headline = "mixed", "Only part of the improvement held up"
            reasons.append(f"Less than half of the gain survived outside {tuning_name}, so most of it was probably luck. Treat the new settings with caution.")
    rw = (selection or {}).get("romano_wolf_p_best")
    total = (selection or {}).get("n_trials_counted", tried)
    weak: list[str] = []
    if level == "good" and not same:
        if hold_t is not None and np.isfinite(hold_t) and hold_t < HOLD_T_MIN:
            weak.append(f"The gain on {hold_these} is small beside how much those days wobble: a paired test of the two return streams gives t = {hold_t:.1f}, and at least {HOLD_T_MIN:.1f} is asked for. "
                        f"A gain this big shows up by chance in about {_normal_tail(hold_t):.0%} of stretches like this.")
        if rw is not None and np.isfinite(rw) and rw >= RW_MAX:
            weak.append(f"Of the {total} settings tried, one at least this good would turn up by luck alone about {rw:.0%} of the time (the Romano-Wolf test, which allows for how many were tried and how "
                        f"alike they are; under {RW_MAX:.0%} is asked for).")
    walk_bad = bool(walk and not same and walk["of"] >= 3 and walk["gain"] is not None and walk["gain"] <= 0 and 2 * walk["beat"] < walk["of"])
    if walk and not same and walk["tuned_value"] is not None:
        reasons.append(f"Re-choosing the settings on the past alone, {walk['of']} times, from the {walk['candidates']} tries drawn before the search had learned anything, and using each choice on the stretch that "
                       f"followed scored {_fmt(objective, walk['tuned_value'])} ({name}), against {_fmt(objective, walk['start_value'])} for the starting settings on the same days; it was ahead in {walk['beat']} of "
                       f"{walk['of']} stretches.")
    if level == "good" and walk_bad:
        weak.append("Choosing again and again on the past did not beat the starting settings on what came next, so the improvement may not repeat.")
    if weak:
        level, headline = "mixed", f"Better on {hold_name}, but not by enough to rule out luck"
        reasons.extend(weak)
    held = (benchmark or {}).get("holdout", {}).get(objective)
    shown = start if same else best
    mine = shown["holdout"].get(objective)
    if held is not None and mine is not None and np.isfinite(held) and np.isfinite(mine):
        verdict = "" if not differs(objective, mine, held) else (": even so, the strategy has not beaten simply holding the tickers" if mine < held else ": the strategy is ahead of it")
        reasons.append(f"For comparison, holding every ticker in equal amounts scored {_fmt(objective, held)} on {hold_these}, against {_fmt(objective, mine)} for "
                       f"{'the starting settings' if same else 'the tuned settings'}{verdict}.")
    if selection and selection.get("deflated_sharpe_probability") is not None:
        eff = selection.get("effective_trials")
        alike = f", about {eff:.0f} of them independent, since settings that sit near each other trade alike" if eff and not prior and eff < 0.8 * total else ""
        reasons.append(f"After {total} tries{' (counting ' + str(prior) + ' earlier ones on these tickers)' if prior else ''}{alike}, the chance that the winner's Sharpe ratio on {tuning_name} is more than luck is "
                       f"{selection['deflated_sharpe_probability'] * 100:.0f}% (95% or more is convincing).")
    if edges:
        reasons.append(f"The best value of {_join(edges)} sits at the end of the range that was tried, so the real best may lie beyond it. Widen the range and tune again to find out.")
    return {"level": level, "headline": headline, "reasons": reasons}


def differs(objective: str, mine: float, other: float) -> bool:
    """Whether two scores are different enough to say which is ahead (one hundredth of a ratio, a tenth of a point of growth)."""
    return abs(mine - other) >= (0.001 if OBJECTIVES[objective]["unit"] == "pct" else 0.01)


def _join(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def paired_t(a: pd.Series, b: pd.Series) -> float | None:
    """The t statistic of the average daily difference ``a - b`` over the days both streams have. A paired test: two settings of one strategy move together, so the difference between them is far
    steadier than either stream, and it is the difference that a hold-out is asked about. ``None`` for fewer than 20 shared days or a difference that never changes."""
    d = (pd.Series(a, dtype=float) - pd.Series(b, dtype=float)).replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 20 or not d.std(ddof=1) > 1e-15:
        return None
    return float(d.mean() / (d.std(ddof=1) / math.sqrt(len(d))))


def _normal_tail(t: float) -> float:
    """The chance that a standard normal variable is at least ``t``: how often a gain this big appears by luck where there is nothing to find."""
    return 0.5 * math.erfc(t / math.sqrt(2.0))


# ------------------------------------------------------------------------------------------------------------------ the walk-forward check
def walk_forward_check(kept: dict, candidates: list[int], win: Windows, objective: str, stable: bool, folds: int = WALK_FOLDS) -> dict | None:
    """Had the settings been re-chosen on the past alone, again and again, would they have beaten the starting settings on what came next?

    The tuning window's second half is cut into ``folds`` stretches. For each, the best of the starting settings and the ``candidates`` on the days before it is used on the days of the stretch; the
    stretches are laid end to end and compared with the starting settings on the same days. Only the tries drawn before the search learned anything take part (a space-filling start does not depend on
    the data); a try the search proposed after it had seen the whole window would carry knowledge of the stretches it is then tested on. Needs at least three candidates."""
    if 0 not in kept or len(candidates) < 3:
        return None
    index = kept[0]["tuning"].index
    n = len(index)
    if n < 2 * 120:
        return None
    edges = np.linspace(n * 0.5, n, folds + 1).astype(int)
    pool = [0] + [c for c in candidates if c in kept]
    fn = OBJECTIVES[objective]["fn"]
    tuned_parts, start_parts, rows = [], [], []
    for k, lo in enumerate(edges[:-1]):
        hi = edges[k + 1]
        if hi - lo < 20:
            continue
        first_test, end = index[lo], (index[hi] if hi < n else None)
        scores = {}
        for c in pool:
            r = kept[c]["tuning"]
            value = score(r[r.index < first_test], objective, stable)
            if np.isfinite(value):
                scores[c] = value
        if not scores:
            continue
        pick = max(scores, key=lambda c: (scores[c], -c))

        def window(r, first_test=first_test, end=end):
            return r[(r.index >= first_test) & ((r.index < end) if end is not None else True)]

        t_part, s_part = window(kept[pick]["tuning"]), window(kept[0]["tuning"])
        tuned_parts.append(t_part)
        start_parts.append(s_part)
        rows.append({"from": str(first_test.date()), "to": str(((end - pd.Timedelta(days=1)) if end is not None else index[-1]).date()), "chosen": int(pick), "chose_start": pick == 0,
                     "tuned": fn(t_part) if len(t_part) > 1 else None, "start": fn(s_part) if len(s_part) > 1 else None})
    if len(rows) < 2:
        return None
    tuned, start = pd.concat(tuned_parts), pd.concat(start_parts)
    tv, sv = fn(tuned), fn(start)
    beat = sum(1 for x in rows if x["tuned"] is not None and x["start"] is not None and np.isfinite(x["tuned"]) and np.isfinite(x["start"]) and x["tuned"] > x["start"] + 1e-9)
    ok = tv is not None and sv is not None and np.isfinite(tv) and np.isfinite(sv)
    return {"of": len(rows), "candidates": len(pool) - 1, "beat": int(beat), "tuned_value": float(tv) if ok else None, "start_value": float(sv) if ok else None, "gain": float(tv - sv) if ok else None,
            "folds": rows, "kept_start": sum(1 for x in rows if x["chose_start"])}


# ------------------------------------------------------------------------------------------------------------------ the search
def tune(run: Callable[[dict], Run], settings: list[Setting], base: dict | None = None, objective: str = "sharpe", holdout: float = 0.3, trials: int = 50, seconds: float = 300.0,
         stable: bool = True, seed: int = 0, progress: Callable[[str], None] | None = None, stop: Callable[[], bool] | None = None, assets: pd.DataFrame | None = None,
         prior_trials: int = 0, clock: Callable[[], float] = time.monotonic, search: str = SEARCH, search_options: dict | None = None, select: str = SELECT,
         inspect: dict | None = None) -> dict:
    """Search ``settings`` (the ticked ones, each with ``low``/``high``/``scale``/``choices`` and a ``start`` value, ``default`` when ``None``) of a strategy.

    ``run(params)`` runs the strategy with ``base`` (the settings that stay as they are) and these values, and returns a :class:`Run`. The first run is the starting settings; it fixes the window and the
    split. Every later run is scored on the tuning window only. The search ends after ``trials`` runs, after ``seconds`` once at least ``MIN_TRIALS`` have run, when ``stop()`` says so, or when no new
    combination is left. ``prior_trials`` are earlier looks at the same prices (they count against the deflated Sharpe ratio).

    ``search`` is ``tpe``, ``bayes`` or ``random`` (``ops.hpo.Study``) and ``search_options`` its keywords. ``select`` says which try is the winner: ``best`` (the single highest score) or ``plateau``
    (the try whose neighbourhood scored best, ``Study.plateau``; the starting settings stay the winner unless that try also beats them on the tuning window). ``inspect``, when a dict is given, is
    filled with the study, the kept return streams and the windows, for an experiment that wants to try other rules on the same tries.
    """
    if objective not in OBJECTIVES:
        raise TuningError(f"the measure must be one of {', '.join(OBJECTIVES)}")
    if search not in SEARCHES:
        raise TuningError(f"the search must be one of {', '.join(SEARCHES)}")
    if select not in SELECTS:
        raise TuningError(f"the way of choosing the winner must be one of {', '.join(SELECTS)}")
    if not settings:
        raise TuningError("tick at least one setting to tune")
    if not 0.1 <= holdout <= 0.5:
        raise TuningError("the hold-out share must be between 10% and 50% of the days")
    trials = int(trials)
    if not 2 <= trials <= MAX_TRIALS:
        raise TuningError(f"the number of tries must be between 2 and {MAX_TRIALS}")
    for s in settings:
        if s.kind in ("int", "float") and not (s.low is not None and s.high is not None and s.low < s.high):
            raise TuningError(f"{s.label}: the lowest value to try must be below the highest")
        if s.kind == "float" and s.scale == "log" and s.low <= 0:
            raise TuningError(f"{s.label}: the lowest value must be above zero")
    base = dict(base or {})
    started = clock()
    say = progress or (lambda text: None)
    starts = {s.name: (s.default if s.start is None else s.start) for s in settings}

    say("running the starting settings")
    first = run({**base, **starts})
    first_seconds = clock() - started
    r0 = _clean(first.returns)
    win = make_windows(r0.index, holdout)
    kept: dict[int, dict] = {}
    best_weights = {"score": -np.inf, "weights": None}
    state = {"stopped": None}

    study = Study({s.name: s.space() for s in settings}, "maximize", seed)

    def objective_fn(params: dict):
        number = len(study.trials)
        res = first if number == 0 else run({**base, **params})
        r = _clean(res.returns)
        part = r[(r.index >= win.start) & (r.index < win.split)]
        if len(part) < max(MIN_TUNE_DAYS // 2, int(0.5 * win.n_tune)):
            raise TuningError(f"it traded on only {len(part)} of the {win.n_tune} days of the tuning window")
        value = score(part, objective, stable)
        if not np.isfinite(value):
            raise TuningError("it has no score (no trades, or no losing days)")
        kept[number] = {"full": r, "tuning": part, "holdout": r[r.index >= win.split]}
        if value > best_weights["score"]:
            best_weights.update(score=value, weights=res.weights, number=number)
        return value, part

    study._run(objective_fn, dict(starts))
    if study.trials[0].error:
        raise TuningError(f"the starting settings could not be scored: {study.trials[0].error.split(': ', 1)[-1]}")

    def callback(st: Study, trial) -> bool:
        n = len(st.trials)
        ok = [t.value for t in st.trials if t.error is None and np.isfinite(t.value)]
        elapsed = clock() - started
        per = max(elapsed / n, 1e-3)
        left = max(0.0, min(trials - n, (seconds - elapsed) / per)) * per
        say(f"try {n} of up to {trials} · best score so far {max(ok):.2f}" + (f" · about {left / 60:.0f} min left" if left >= 90 else f" · {elapsed:.0f} s so far"))
        if n == MIN_TRIALS - 1 and len(ok) <= 1:
            errors = Counter(str(t.error).split(": ", 1)[-1] for t in st.trials if t.error)
            raise TuningError("almost every setting tried failed, so there is nothing to search: " + "; ".join(f"{m} (x{c})" for m, c in errors.most_common(3)))
        if stop is not None and stop():
            state["stopped"] = "user"
            return True
        if n >= MIN_TRIALS and elapsed >= seconds:
            state["stopped"] = "time"
            return True
        return False

    n_startup = max(5, min(12, trials // 4))
    if search == "bayes":
        n_startup = max(8, n_startup)                                                # a surface fitted to fewer points than that says little
    options = {"n_startup": n_startup, **({"quasi_start": True} if search in ("tpe", "bayes") else {}), **(search_options or {})}
    study.optimize(objective_fn, n_trials=trials - 1, method=search, callback=callback, unique=True, **options)
    if state["stopped"] is None:
        state["stopped"] = "limit" if len(study.trials) >= trials else "exhausted"
    ok = [t for t in study.trials if t.error is None and np.isfinite(t.value)]
    if len(ok) < 2:
        errors = Counter(str(t.error).split(": ", 1)[-1] for t in study.trials if t.error)
        raise TuningError("fewer than two settings could be scored: " + "; ".join(f"{m} (x{c})" for m, c in errors.most_common(3)))
    raw_best = study.best
    best, rule = raw_best, "best"
    if select == "plateau":
        middle = study.plateau()
        if middle.number != raw_best.number and middle.value > study.trials[0].value:       # the middle of the best region, if it also beats the starting settings on the tuning window
            best, rule = middle, "plateau"

    def pack(number: int) -> dict:
        k = kept[number]
        return {"number": number, "params": dict(study.trials[number].params), "score": study.trials[number].value, "tuning": metrics(k["tuning"]), "holdout": metrics(k["holdout"])}

    start_pack, best_pack = pack(0), pack(best.number)
    same = best.number == 0
    choice = {"rule": rule, "raw_best": None if rule == "best" else {"number": raw_best.number, "score": float(raw_best.value), "params": dict(raw_best.params)}}
    walk = walk_forward_check(kept, [t.number for t in study.trials[1:n_startup] if t.number in kept], win, objective, stable)
    hold_t = None if same else paired_t(kept[best.number]["holdout"], kept[0]["holdout"])
    if inspect is not None:
        inspect.update(study=study, kept=kept, win=win, first=first, start_count=n_startup, run=run, base=base)

    # selection: how much of the winner is the choosing?
    selection: dict | None
    try:
        selection = study.selection_report(n_boot=200, winner=best.number)
        mat = study.returns_matrix().dropna()
        if prior_trials and mat.shape[1] >= 2:
            from ..stats.inference import deflated_sharpe_ratio

            dsr = deflated_sharpe_ratio(mat[best.number].to_numpy(), n_trials=selection["effective_trials"] + prior_trials)           # earlier looks have no streams here: each counts as independent
            selection["deflated_sharpe_probability"], selection["deflated_benchmark_sharpe_annual"] = dsr["probability"], float(dsr["benchmark"] * np.sqrt(252))
        selection["n_trials_counted"] = study.n_trials + prior_trials
    except (ValueError, KeyError, np.linalg.LinAlgError, IndexError) as error:
        selection = {"error": str(error)}

    # the settings, one by one: where the best sits, how much each mattered, and every try as a point
    rows = []
    values_by = {s.name: [t.params[s.name] for t in ok] for s in settings}
    scores = [t.value for t in ok]
    for s in settings:
        vals = values_by[s.name]
        imp = _importance(s, vals, scores)
        row = {**s.view(), "start": starts[s.name], "best": best.params[s.name], "importance": imp, "sweet": None, "edge": None, "points": [], "choices_summary": []}
        if s.kind in ("choice", "bool"):
            for c in s.choices:
                sel = [sc for v, sc in zip(vals, scores) if v == c]
                row["choices_summary"].append({"choice": c, "n": len(sel), "mean": float(np.mean(sel)) if sel else None, "best": float(np.max(sel)) if sel else None})
        else:
            row["points"] = sorted([[float(v), float(sc)] for v, sc in zip(vals, scores)])
            if len(scores) >= 8:
                u = s.space().to_unit(best.params[s.name])
                row["edge"] = "low" if u < EDGE else "high" if u > 1.0 - EDGE else None
                top = np.argsort(scores)[::-1][:max(3, len(scores) // 4)]
                row["sweet"] = [min(vals[i] for i in top), max(vals[i] for i in top)]
        rows.append(row)
    total_imp = sum(r["importance"] or 0.0 for r in rows)
    for r in rows:
        r["share"] = None if r["importance"] is None else (r["importance"] / total_imp if total_imp > 0 else 0.0)

    # tries in order, the best so far, and the table of the best
    history, top_so_far = [], -np.inf
    for t in study.trials:
        good = t.error is None and np.isfinite(t.value)
        top_so_far = max(top_so_far, t.value) if good else top_so_far
        history.append({"number": t.number, "score": float(t.value) if good else None, "best": float(top_so_far) if np.isfinite(top_so_far) else None, "params": dict(t.params)})
    table = []
    shown = sorted(ok, key=lambda t: -t.value)[:12]
    for t in shown + [x for x in (study.trials[0], best) if x not in shown]:                  # the twelve best scores, the starting settings and the winner (which the plateau rule can pick from further down)
        table.append({"number": t.number, "params": dict(t.params), "score": float(t.value), "tuning": metrics(kept[t.number]["tuning"]), "is_start": t.number == 0, "is_best": t.number == best.number})
    failures = Counter(str(t.error).split(": ", 1)[-1] for t in study.trials if t.error)

    # growth of one dollar, tuning window and hold-out, with the split marked; and what each ticker contributed
    full_start, full_best = kept[0]["full"].loc[win.start:], kept[best.number]["full"].loc[win.start:]
    dates = full_start.index
    growth_start, growth_best = cumulative_returns(full_start), cumulative_returns(full_best.reindex(dates))
    pick = _thin(dates)
    curves = {"dates": [str(dates[i].date()) for i in pick], "start": [float(growth_start.iloc[i]) for i in pick], "best": [float(growth_best.iloc[i]) for i in pick], "split": str(win.split.date())}
    benchmark, tickers = None, []
    if assets is not None and len(assets):
        every = assets.reindex(dates).mean(axis=1).fillna(0.0)
        benchmark = {"label": "Hold every ticker in equal amounts", "tuning": metrics(every[every.index < win.split]), "holdout": metrics(every[every.index >= win.split])}
        if first.weights is not None and best_weights["weights"] is not None:
            weights = best_weights["weights"] if best.number == best_weights.get("number") else run({**base, **study.trials[best.number].params}).weights     # the plateau try's weights were not kept
            if weights is not None:
                tickers = _by_ticker(first.weights, weights, assets, win)
    return {
        "objective": {"id": objective, "label": OBJECTIVES[objective]["label"], "short": OBJECTIVES[objective]["short"], "unit": OBJECTIVES[objective]["unit"]},
        "windows": {"tuning": [str(win.start.date()), str(win.tuning_end.date())], "holdout": [str(win.split.date()), str(win.end.date())], "tuning_days": win.n_tune, "holdout_days": win.n_hold,
                    "holdout_share": holdout},
        "search": {"tried": study.n_trials, "scored": len(ok), "failed": study.n_trials - len(ok), "budget": trials, "seconds": round(clock() - started, 1), "stopped": state["stopped"], "method": search, "select": select,
                   "stable": bool(stable), "prior_trials": int(prior_trials), "first_run_seconds": round(first_seconds, 2)},
        "start": start_pack, "best": best_pack, "same_as_start": same, "benchmark": benchmark, "settings": rows, "trials": table, "history": history,
        "failures": [{"reason": m, "count": c} for m, c in failures.most_common(5)], "curves": curves, "selection": selection, "tickers": tickers,
        "choice": choice, "walk_forward": walk, "holdout_t": hold_t,
        "verdict": judge(objective, start_pack, best_pack, *spans(win), same, selection, study.n_trials, prior_trials, [r["label"] for r in rows if r["edge"] and not same], benchmark, walk=walk, hold_t=hold_t),
    }


def _by_ticker(start_weights: pd.DataFrame, best_weights: pd.DataFrame, assets: pd.DataFrame, win: Windows) -> list[dict]:
    """What each ticker added to the strategy's return (percentage points of capital, summed over the days) on the tuning window and on the hold-out, with the starting settings and with the winner.
    The position held on a day is the one chosen the day before."""
    def added(weights: pd.DataFrame, lo, hi) -> pd.Series:
        held = weights.shift(1).reindex(index=assets.index, columns=assets.columns)
        mask = (assets.index >= lo) & (assets.index < hi) if hi is not None else (assets.index >= lo)
        return 100.0 * (held[mask].fillna(0.0) * assets[mask].fillna(0.0)).sum()

    start_t, best_t = added(start_weights, win.start, win.split), added(best_weights, win.start, win.split)
    start_h, best_h = added(start_weights, win.split, None), added(best_weights, win.split, None)
    rows = [{"ticker": str(t), "start_tuning": float(start_t[t]), "best_tuning": float(best_t[t]), "start_holdout": float(start_h[t]), "best_holdout": float(best_h[t])} for t in assets.columns]
    return sorted(rows, key=lambda r: -r["best_holdout"])
