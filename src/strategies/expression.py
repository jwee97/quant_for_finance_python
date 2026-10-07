"""A strategy written as a one-line formula, with no Python to write.

``expression`` evaluates a formula over the price table and uses the result as the score (higher = expect a higher return). The formula is parsed, never executed:
only a fixed vocabulary of operators and functions is accepted, every function looks backwards only, and a formula that reaches for anything else is rejected
with a message saying what is allowed. That is what makes it safe to take from a web form and what keeps it causal: the framework's causality check passes for any formula.

Examples:  ``mom(126, 21)`` (12-1 style momentum), ``-zscore(20)`` (fade a 20-day stretch), ``rank(mom(63)) - rank(vol(63))`` (momentum among the calm),
``where(close > sma(200), 1, -1)`` (trend filter). ``FUNCTIONS`` is the whole vocabulary.
"""

from __future__ import annotations

import ast
import operator

import numpy as np
import pandas as pd

from ..framework.forecasting import ForecastModel
from ..framework.registry import register_model

MAX_LENGTH, MAX_NODES, MAX_WINDOW = 400, 120, 1000

# name -> (signature, what it computes); the dashboard's formula reference is built from this table
FUNCTIONS: dict[str, tuple[str, str]] = {
    "close": ("close", "adjusted closing prices"),
    "ret": ("ret(n)", "return over the last n days"),
    "mom": ("mom(n, skip=0)", "return from n days ago to `skip` days ago (skip=21 gives the classic 12-1 momentum)"),
    "sma": ("sma(n)", "n-day simple moving average of the price"),
    "ema": ("ema(n)", "n-day exponential moving average of the price (span n)"),
    "vol": ("vol(n)", "annualised volatility of daily returns over n days"),
    "zscore": ("zscore(n)", "how many n-day standard deviations the price is above its n-day mean"),
    "rsi": ("rsi(n)", "relative strength index over n days, scaled to 0-100"),
    "hi": ("hi(n)", "highest price of the last n days"),
    "lo": ("lo(n)", "lowest price of the last n days"),
    "lag": ("lag(x, n)", "x as it was n days ago (n >= 0)"),
    "rollmean": ("rollmean(x, n)", "n-day rolling mean of x"),
    "rollstd": ("rollstd(x, n)", "n-day rolling standard deviation of x"),
    "rollmax": ("rollmax(x, n)", "n-day rolling maximum of x"),
    "rollmin": ("rollmin(x, n)", "n-day rolling minimum of x"),
    "ts_z": ("ts_z(x, n)", "x minus its n-day mean, over its n-day standard deviation"),
    "rank": ("rank(x)", "each day, rank of x across assets scaled to 0-1"),
    "zs": ("zs(x)", "each day, x minus the cross-sectional mean over the cross-sectional standard deviation"),
    "demean": ("demean(x)", "each day, x minus the average across assets"),
    "sign": ("sign(x)", "-1, 0 or +1"),
    "abs": ("abs(x)", "absolute value"),
    "log": ("log(x)", "natural log (non-positive values become missing)"),
    "tanh": ("tanh(x)", "squash to (-1, 1)"),
    "clip": ("clip(x, lo, hi)", "limit x to [lo, hi]"),
    "where": ("where(cond, a, b)", "a where the condition holds, b elsewhere (combine conditions with & | ~)"),
    "macro": ('macro("VIX")', "a macro series known on each date (point in time), the same value for every asset; needs a bundle with that series"),
}
EXAMPLES = [
    ("12-1 momentum", "mom(252, 21)", "cross_sectional"),
    ("Short-term reversal", "-ret(5)", "cross_sectional"),
    ("Momentum among the calm", "rank(mom(126, 21)) - rank(vol(63))", "cross_sectional"),
    ("Trend filter", "where(close > sma(200), 1, -1)", "time_series"),
    ("Stretch fade", "-tanh(zscore(20) / 2)", "time_series"),
    ("Near the 52-week high", "close / hi(252) - 1", "cross_sectional"),
    ("Golden cross", "sign(sma(50) - sma(200))", "time_series"),
    ("Fear gauge tilt", 'tanh(-ts_z(macro("VIX"), 252)) * where(close > sma(100), 1, 0)', "time_series"),
]
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow,
        ast.BitAnd: operator.and_, ast.BitOr: operator.or_}
_CMP = {ast.Gt: operator.gt, ast.Lt: operator.lt, ast.GtE: operator.ge, ast.LtE: operator.le, ast.Eq: operator.eq, ast.NotEq: operator.ne}


class FormulaError(ValueError):
    """The formula is not allowed or cannot be evaluated; the message says why."""


def _window(value, name: str, minimum: int = 1) -> int:
    if isinstance(value, (pd.DataFrame, pd.Series)) or not float(value).is_integer() or not minimum <= value <= MAX_WINDOW:
        raise FormulaError(f"{name}: the window must be a whole number from {minimum} to {MAX_WINDOW}")
    return int(value)


def _rsi(close: pd.DataFrame, n: int) -> pd.DataFrame:
    change = close.diff()
    gain, loss = change.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean(), (-change.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        return 100.0 - 100.0 / (1.0 + gain / loss)


def _mom(close: pd.DataFrame, n, skip) -> pd.DataFrame:
    n, skip = _window(n, "mom"), _window(skip, "mom", 0)
    if n <= skip:
        raise FormulaError("mom: n must be larger than skip")
    return close.shift(skip) / close.shift(n) - 1.0


def _functions(data) -> dict:
    close = data.prices

    def need(x, name):
        if not isinstance(x, pd.DataFrame):
            raise FormulaError(f"{name}: the first argument must be a table (a price, a return, or another function), not a plain number")
        return x

    def macro(name):
        if not isinstance(name, str) or name not in data.macro.columns or data.macro[name].dropna().empty:
            raise FormulaError(f"macro: this bundle has no series '{name}' (available: {sorted(map(str, data.macro.columns))[:12]})")
        return pd.DataFrame({a: data.macro[name] for a in close.columns})

    return {
        "ret": lambda n: close / close.shift(_window(n, "ret")) - 1.0,
        "mom": lambda n, skip=0: _mom(close, n, skip),
        "sma": lambda n: close.rolling(_window(n, "sma")).mean(),
        "ema": lambda n: close.ewm(span=_window(n, "ema"), adjust=False, min_periods=_window(n, "ema")).mean(),
        "vol": lambda n: close.pct_change().rolling(_window(n, "vol", 2)).std() * np.sqrt(252.0),
        "zscore": lambda n: (close - close.rolling(_window(n, "zscore", 2)).mean()) / close.rolling(_window(n, "zscore", 2)).std().replace(0.0, np.nan),
        "rsi": lambda n: _rsi(close, _window(n, "rsi", 2)),
        "hi": lambda n: close.rolling(_window(n, "hi")).max(),
        "lo": lambda n: close.rolling(_window(n, "lo")).min(),
        "lag": lambda x, n: need(x, "lag").shift(_window(n, "lag", 0)),
        "rollmean": lambda x, n: need(x, "rollmean").rolling(_window(n, "rollmean")).mean(),
        "rollstd": lambda x, n: need(x, "rollstd").rolling(_window(n, "rollstd", 2)).std(),
        "rollmax": lambda x, n: need(x, "rollmax").rolling(_window(n, "rollmax")).max(),
        "rollmin": lambda x, n: need(x, "rollmin").rolling(_window(n, "rollmin")).min(),
        "ts_z": lambda x, n: (need(x, "ts_z") - x.rolling(_window(n, "ts_z", 2)).mean()) / x.rolling(_window(n, "ts_z", 2)).std().replace(0.0, np.nan),
        "rank": lambda x: need(x, "rank").rank(axis=1, pct=True),
        "zs": lambda x: need(x, "zs").sub(x.mean(axis=1), axis=0).div(x.std(axis=1).replace(0.0, np.nan), axis=0),
        "demean": lambda x: need(x, "demean").sub(x.mean(axis=1), axis=0),
        "sign": lambda x: np.sign(x),
        "abs": lambda x: x.abs() if hasattr(x, "abs") else abs(x),
        "log": lambda x: np.log(x.where(x > 0)) if hasattr(x, "where") else np.log(x),
        "tanh": lambda x: np.tanh(x),
        "clip": lambda x, lo, hi: x.clip(lo, hi) if hasattr(x, "clip") else min(max(x, lo), hi),
        "where": lambda cond, a, b: _where(cond, a, b, close),
        "macro": macro,
    }


def _where(cond, a, b, close):
    if not isinstance(cond, pd.DataFrame):
        raise FormulaError("where: the condition must be a comparison of tables, e.g. close > sma(200)")
    fill = lambda v: v if isinstance(v, pd.DataFrame) else pd.DataFrame(float(v), index=close.index, columns=close.columns)      # noqa: E731
    return fill(a).where(cond, fill(b))


def parse(formula: str) -> ast.Expression:
    """Parse and vet a formula without evaluating it. Raises ``FormulaError`` with a human-readable reason."""
    if not isinstance(formula, str) or not formula.strip():
        raise FormulaError("the formula is empty")
    if len(formula) > MAX_LENGTH:
        raise FormulaError(f"the formula is longer than {MAX_LENGTH} characters")
    try:
        tree = ast.parse(formula.strip(), mode="eval")
    except SyntaxError as error:
        raise FormulaError(f"syntax error: {error.msg} (column {error.offset})") from None
    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_NODES:
        raise FormulaError("the formula is too long: simplify it or split the idea into two models")
    for node in nodes:
        if isinstance(node, (ast.Expression, ast.Load, ast.operator, ast.unaryop, ast.cmpop)):
            if isinstance(node, ast.operator) and type(node) not in _BIN:
                raise FormulaError(f"operator {type(node).__name__} is not allowed; use + - * / ** & |")
            continue
        if isinstance(node, ast.BinOp | ast.UnaryOp | ast.Compare | ast.Call | ast.Tuple):
            if isinstance(node, ast.UnaryOp) and not isinstance(node.op, ast.USub | ast.UAdd | ast.Invert):
                raise FormulaError("only unary - + ~ are allowed (use ~ to negate a condition)")
            if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name) or node.keywords):
                raise FormulaError("call functions by name with positional arguments only, e.g. sma(50)")
            if isinstance(node, ast.Compare) and len(node.ops) != 1:
                raise FormulaError("chain comparisons with &, e.g. (a > b) & (b > c)")
            continue
        if isinstance(node, ast.Name):
            if node.id not in FUNCTIONS:
                raise FormulaError(f"unknown name '{node.id}'; allowed: {', '.join(sorted(FUNCTIONS))}")
            continue
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, int | float | str):
                raise FormulaError("only numbers and quoted series names are allowed as constants")
            continue
        raise FormulaError(f"'{type(node).__name__}' is not allowed in a formula (no attributes, subscripts, lambdas, comprehensions or statements)")
    return tree


def evaluate(formula: str, data) -> pd.DataFrame:
    """The formula's value as a dates x assets table (infinities become missing)."""
    tree = parse(formula)
    functions = _functions(data)

    def run(node):
        if isinstance(node, ast.Expression):
            return run(node.body)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id == "close":
                return data.prices
            if node.id in functions:
                raise FormulaError(f"'{node.id}' is a function: call it, e.g. {FUNCTIONS[node.id][0]}")
            raise FormulaError(f"unknown name '{node.id}'")
        if isinstance(node, ast.Call):
            fn = functions.get(node.func.id)
            if fn is None:
                raise FormulaError(f"unknown function '{node.func.id}'; allowed: {', '.join(sorted(k for k in FUNCTIONS if k != 'close'))}")
            try:
                return fn(*[run(a) for a in node.args])
            except TypeError as error:
                raise FormulaError(f"{node.func.id}: wrong arguments; expected {FUNCTIONS[node.func.id][0]}") from error
        if isinstance(node, ast.BinOp):
            left, right = run(node.left), run(node.right)
            if isinstance(node.op, ast.Pow) and not (isinstance(right, int | float) and abs(right) <= 4):
                raise FormulaError("the exponent must be a plain number between -4 and 4")
            if isinstance(node.op, ast.BitAnd | ast.BitOr) and not (isinstance(left, pd.DataFrame) and isinstance(right, pd.DataFrame)):
                raise FormulaError("& and | combine two conditions (comparisons of tables)")
            return _BIN[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp):
            value = run(node.operand)
            return -value if isinstance(node.op, ast.USub) else ~value if isinstance(node.op, ast.Invert) else +value
        if isinstance(node, ast.Compare):
            return _CMP[type(node.ops[0])](run(node.left), run(node.comparators[0]))
        raise FormulaError(f"cannot evaluate {type(node).__name__}")

    value = run(tree)
    if not isinstance(value, pd.DataFrame):
        raise FormulaError("the formula must produce a table of scores (it produced a plain number); start from close, ret(n), mom(n) ...")
    if value.dtypes.eq(bool).all():
        value = value.astype(float)
    return value.replace([np.inf, -np.inf], np.nan).astype(float)


@register_model("expression", "custom", "A strategy written as a one-line formula over prices (for example rank(mom(126, 21)) - rank(vol(63))); parsed safely, backward-looking only")
class Expression(ForecastModel):
    """A formula is a hypothesis in one line: ``mom(252, 21)`` bets that last year's winners keep winning. Higher score = expect a higher return."""

    name, family = "expression", "custom"

    def __init__(self, expr: str = "mom(252, 21)", mode: str = "cross_sectional"):
        if mode not in ("cross_sectional", "time_series"):
            raise ValueError("mode must be 'cross_sectional' (rank assets against each other) or 'time_series' (each asset on its own)")
        parse(expr)
        self.expr, self.mode = expr.strip(), mode
        self.position_mode = mode

    def score(self, data):
        return evaluate(self.expr, data).reindex(index=data.index, columns=data.assets).where(data.investable)
