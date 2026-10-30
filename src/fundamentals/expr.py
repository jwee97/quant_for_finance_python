"""Expression trees: a factor as a small, typed, machine-readable formula that can be stored, shown, hashed, parsed from text and evaluated, without ever running arbitrary code.

A node is one of three plain dictionaries (so an expression is JSON and nothing else)::

    {"f": "operating_income"}                                  a field: a statement figure, a derived metric or a market quantity
    {"c": 2.0}                                                 a constant
    {"op": "div", "args": [A, B]}                              an operator applied to arguments; time-series operators carry a lag or window  "k"

Operators: ``add sub mul div`` (``div`` gives a missing value unless the denominator is positive), ``neg abs log``, the time-series operators ``delta growth accel rolling_mean rolling_std`` over
``k`` rebalance periods, and the cross-sectional ones ``rank sector_rank industry_rank zscore`` that work on one date at a time. Every field has a *kind* (an amount in the reporting currency, an
amount in the trading currency, a pure ratio, a count) and the evaluator knows what each operator makes of kinds: an amount divided by an amount is a ratio, a reporting-currency amount set against a
trading-currency one is converted at the exchange rate of the date first, and adding an amount to a ratio is refused. The discovery search only builds well-typed trees, so it does not waste its
tests on formulas that mean nothing.

The **canonical form** puts commutative arguments in order, removes double negations and repeated rank or z-score, and rejects degenerate formulas (``x - x``). Its hash is the identity of an idea
in the library and the graveyard. The **signal key** goes one step further for the purpose of finding duplicates: a rank, a z-score or a sign on the outside of a formula does not change which
companies it prefers, so ``rank(x)``, ``zscore(x)`` and ``neg(x)`` are the same idea as ``x``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import factors as factor_library
from . import transforms
from .errors import FundamentalsError
from .research import ResearchContext

BINARY = ("add", "sub", "mul", "div")
UNARY = ("neg", "abs", "log")
TIME_SERIES = ("delta", "growth", "accel", "rolling_mean", "rolling_std")
CROSS_SECTIONAL = ("rank", "sector_rank", "industry_rank", "zscore")
OPERATORS = BINARY + UNARY + TIME_SERIES + CROSS_SECTIONAL
COMMUTATIVE = ("add", "mul")
MAX_DEPTH = 6
MAX_NODES = 31
KINDS = ("amount_rep", "amount_trade", "ratio", "count", "other")
LAGS = {"W": (4, 13, 26, 52), "M": (3, 6, 12, 24), "Q": (1, 2, 4, 8)}
WINDOWS = {"W": (26, 52, 104, 156), "M": (12, 24, 36, 60), "Q": (4, 8, 12, 20)}


class ExprError(FundamentalsError):
    """A formula that cannot be built, parsed or evaluated; the message says why."""


# ------------------------------------------------------------------------------------------------------------------ building
def field(name: str) -> dict:
    return {"f": str(name)}


def const(value: float) -> dict:
    return {"c": float(value)}


def op(name: str, *args: dict, k: int | None = None) -> dict:
    node: dict = {"op": name, "args": list(args)}
    if k is not None:
        node["k"] = int(k)
    return node


def is_field(n: dict) -> bool:
    return "f" in n


def is_const(n: dict) -> bool:
    return "c" in n


def children(n: dict) -> list[dict]:
    return n.get("args", []) if "op" in n else []


def size(n: dict) -> int:
    """The number of nodes: the measure of complexity."""
    return 1 + sum(size(c) for c in children(n))


def depth(n: dict) -> int:
    return 1 + max((depth(c) for c in children(n)), default=0)


def fields_used(n: dict) -> list[str]:
    if is_field(n):
        return [n["f"]]
    return [f for c in children(n) for f in fields_used(c)]


def validate(n: dict) -> dict:
    """The expression if it is well formed, else :class:`ExprError`."""
    if not isinstance(n, dict):
        raise ExprError("a formula node must be a dictionary")
    if depth(n) > MAX_DEPTH or size(n) > MAX_NODES:
        raise ExprError(f"formula too large: at most {MAX_NODES} nodes and {MAX_DEPTH} levels")
    if "f" in n:
        if not isinstance(n["f"], str) or not re.fullmatch(r"[a-z][a-z0-9_]*", n["f"]):
            raise ExprError(f"bad field name {n['f']!r}")
        return n
    if "c" in n:
        if not isinstance(n["c"], (int, float)) or not math.isfinite(float(n["c"])):
            raise ExprError("a constant must be a finite number")
        return n
    name = n.get("op")
    if name not in OPERATORS:
        raise ExprError(f"unknown operator {name!r}; choose from {', '.join(OPERATORS)}")
    args = n.get("args")
    if not isinstance(args, list):
        raise ExprError(f"{name} needs a list of arguments")
    want = 2 if name in BINARY else 1
    if len(args) != want:
        raise ExprError(f"{name} takes {want} argument{'s' if want > 1 else ''}, not {len(args)}")
    if name in TIME_SERIES:
        k = n.get("k")
        if not isinstance(k, int) or not 1 <= k <= 260:
            raise ExprError(f"{name} needs a lag or window k between 1 and 260 periods")
    elif "k" in n:
        raise ExprError(f"{name} takes no parameter")
    for a in args:
        validate(a)
    return n


# ------------------------------------------------------------------------------------------------------------------ text form
def to_string(n: dict) -> str:
    if is_field(n):
        return n["f"]
    if is_const(n):
        v = float(n["c"])
        return str(int(v)) if v == int(v) and abs(v) < 1e9 else repr(round(v, 8))
    inner = ",".join(to_string(a) for a in n["args"])
    return f"{n['op']}({inner}{',' + str(n['k']) if 'k' in n else ''})"


_TOKEN = re.compile(r"\s*(?:(?P<num>-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)|(?P<name>[A-Za-z_][A-Za-z0-9_]*)|(?P<sym>[(),]))")


def parse(text: str) -> dict:
    """The expression a text such as ``sub(rank(fcf_yield), rank(delta(roic,12)))`` stands for. Only the operators and fields of this module are understood; nothing is ever evaluated as Python."""
    tokens, pos = [], 0
    text = str(text)
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m:
            if text[pos:].strip() == "":
                break
            raise ExprError(f"cannot read the formula at position {pos}: {text[pos:pos + 12]!r}")
        pos = m.end()
        kind = m.lastgroup
        tokens.append((kind, m.group(kind)))
    index = 0

    def peek():
        return tokens[index] if index < len(tokens) else (None, None)

    def take(expected=None):
        nonlocal index
        if index >= len(tokens):
            raise ExprError("the formula ends too soon")
        token = tokens[index]
        index += 1
        if expected and token[1] != expected:
            raise ExprError(f"expected {expected!r}, found {token[1]!r}")
        return token

    def node():
        kind, value = take()
        if kind == "num":
            return const(float(value))
        if kind != "name":
            raise ExprError(f"unexpected {value!r}")
        if peek()[1] != "(":
            return field(value)
        if value not in OPERATORS:
            raise ExprError(f"unknown operator {value!r}; choose from {', '.join(OPERATORS)}")
        take("(")
        args, k = [], None
        while True:
            if value in TIME_SERIES and len(args) == 1 and peek()[0] == "num":
                k = int(float(take()[1]))
            else:
                args.append(node())
            nxt = take()
            if nxt[1] == ")":
                break
            if nxt[1] != ",":
                raise ExprError(f"expected ',' or ')', found {nxt[1]!r}")
        return op(value, *args, k=k)

    out = node()
    if index != len(tokens):
        raise ExprError(f"unexpected {tokens[index][1]!r} after the formula")
    return validate(out)


# ------------------------------------------------------------------------------------------------------------------ canonical form and identity
def canonical(n: dict) -> dict:
    """The same formula in its one canonical spelling (see the module docstring). Raises :class:`ExprError` for a degenerate formula."""
    if is_field(n) or is_const(n):
        return dict(n)
    name = n["op"]
    args = [canonical(a) for a in n["args"]]
    k = n.get("k")
    if name == "neg" and args[0].get("op") == "neg":
        return args[0]["args"][0]
    if name in ("rank", "zscore", "sector_rank", "industry_rank", "abs") and args[0].get("op") == name:
        return args[0]
    if name == "add" and args[1].get("op") == "neg":
        return canonical(op("sub", args[0], args[1]["args"][0]))
    if name == "add" and args[0].get("op") == "neg":
        return canonical(op("sub", args[1], args[0]["args"][0]))
    if name in COMMUTATIVE:
        args.sort(key=to_string)
    if name in ("sub", "div") and to_string(args[0]) == to_string(args[1]):
        raise ExprError(f"{name}(x, x) is degenerate")
    if all(is_const(a) for a in args):
        raise ExprError("a formula of constants only has nothing to rank")
    return op(name, *args, k=k)


def canonical_string(n: dict) -> str:
    return to_string(canonical(n))


def expression_hash(n: dict) -> str:
    return hashlib.sha256(canonical_string(n).encode()).hexdigest()[:16]


def signal_key(n: dict) -> str:
    """Identity for the purpose of finding the same idea twice: a rank, a z-score or a sign on the outside does not change which companies the formula prefers."""
    c = canonical(n)
    while c.get("op") in ("rank", "zscore", "neg"):
        c = canonical(c["args"][0])
    return hashlib.sha256(to_string(c).encode()).hexdigest()[:16]


def dumps(n: dict) -> str:
    return json.dumps(canonical(n), sort_keys=True, separators=(",", ":"))


def loads(text: str) -> dict:
    try:
        return validate(json.loads(text))
    except ValueError as error:
        raise ExprError(f"not a valid formula: {error}") from None


# ------------------------------------------------------------------------------------------------------------------ the fields
@dataclass(frozen=True)
class Field:
    name: str
    kind: str
    label: str
    source: str            # statement | metric | market | factor


STATEMENT_FLOWS = ("revenue", "cost_of_revenue", "gross_profit", "operating_income", "ebitda", "net_income", "operating_cash_flow", "capital_expenditure", "depreciation_amortization",
                   "research_and_development", "selling_general_admin", "interest_expense", "income_tax", "dividends_paid", "share_repurchases", "stock_based_compensation", "acquisitions")
STATEMENT_STOCKS = ("total_assets", "shareholders_equity", "total_debt", "cash", "current_assets", "current_liabilities", "inventory", "accounts_receivable", "accounts_payable", "ppe",
                    "total_liabilities", "goodwill", "intangible_assets")
MARKET_AMOUNTS = ("market_cap", "enterprise_value")
MARKET_RATIOS = ("mom_12_1", "mom_6_1", "mom_3_1", "reversal_1m", "high_52w", "volatility_63", "volatility_252", "beta", "idiosyncratic_volatility")


def catalogue(ctx: ResearchContext) -> dict[str, Field]:
    """The fields a formula may use on this data: statement figures and metrics that exist in the tables, market quantities, and the price factors."""
    if "expr_fields" in ctx._cache:
        return ctx._cache["expr_fields"]
    from . import metrics as metric_module
    out: dict[str, Field] = {}
    for name in STATEMENT_FLOWS + STATEMENT_STOCKS:
        if ctx.has(name):
            out[name] = Field(name, "amount_rep", name.replace("_", " "), "statement")
    if ctx.has("shares_outstanding"):
        out["shares_outstanding"] = Field("shares_outstanding", "count", "shares outstanding", "statement")
    for m in metric_module.catalogue():
        if m["unit"] == "per_share":                                                   # an amount per share is not scale-free and has no price to be set against here: the price-based factors use it
            continue
        if ctx.has(m["name"]) and m["name"] not in out:
            out[m["name"]] = Field(m["name"], "amount_rep" if m["unit"] == "currency" else "ratio", m["label"], "metric")
    for name in MARKET_AMOUNTS:
        out[name] = Field(name, "amount_trade", name.replace("_", " "), "market")
    for name in MARKET_RATIOS:
        out[name] = Field(name, "ratio", factor_library.get(name).label, "market")
    ctx._cache["expr_fields"] = out
    return out


def _field_panel(ctx: ResearchContext, name: str) -> pd.DataFrame:
    if name == "market_cap":
        return ctx.market_cap()
    if name == "enterprise_value":
        return ctx.enterprise_value()
    if name in MARKET_RATIOS:
        return factor_library.compute(ctx, name)
    if ctx.has(name):
        return ctx.panel(name)
    raise ExprError(f"unknown field {name!r}: this universe's database has no such figure")


# ------------------------------------------------------------------------------------------------------------------ kinds
def result_kind(name: str, kinds: list[str]) -> str:
    """The kind an operator makes of its arguments' kinds, or ``other`` when the combination has no clear meaning."""
    a = kinds[0]
    if name in ("add", "sub"):
        b = kinds[1]
        if a == b:
            return a
        if {a, b} == {"amount_rep", "amount_trade"}:
            return "amount_trade"
        return "other"
    if name == "mul":
        b = kinds[1]
        if a == "ratio":
            return b
        if b == "ratio":
            return a
        return "other"
    if name == "div":
        b = kinds[1]
        amounts = ("amount_rep", "amount_trade")
        if a in amounts and b in amounts:
            return "ratio"
        if a == b == "ratio" or a == b == "count":
            return "ratio"
        if a in amounts and b == "ratio":
            return a
        return "other"
    if name in ("neg", "abs", "delta", "rolling_mean", "rolling_std"):
        return a
    if name == "log":
        return "ratio"
    return "ratio"                                                                          # growth, accel and every cross-sectional operator are scale-free


def scale_tag(n: dict) -> str:
    """Which scale a formula lives on, so that two things added or subtracted are on the same one: a rank or a z-score is on the ``norm`` scale; a time-series operator keeps the scale of what it is
    applied to (a change in ROIC is on ROIC's scale); a ratio of two things has its own scale."""
    if is_const(n):
        return "const"
    if is_field(n):
        return n["f"]
    name, args = n["op"], n["args"]
    if name in CROSS_SECTIONAL:
        return "norm"
    if name in ("growth", "accel"):
        return "pct"
    if name == "log":
        return f"log({scale_tag(args[0])})"
    if name in ("neg", "abs", "delta", "rolling_mean", "rolling_std"):
        return scale_tag(args[0])
    a, b = scale_tag(args[0]), scale_tag(args[1])
    if name in ("add", "sub"):
        return a if a == b else "mixed"
    return f"({a}{'/' if name == 'div' else '*'}{b})"


def infer_kind(n: dict, kinds: dict[str, str]) -> str:
    if is_const(n):
        return "ratio"
    if is_field(n):
        if n["f"] not in kinds:
            raise ExprError(f"unknown field {n['f']!r}")
        return kinds[n["f"]]
    return result_kind(n["op"], [infer_kind(a, kinds) for a in n["args"]])


def well_typed(n: dict, kinds: dict[str, str]) -> tuple[bool, str]:
    """Whether every operator in the formula is applied to kinds it makes sense for, and why not."""
    if is_const(n):
        return True, ""
    if is_field(n):
        return (n["f"] in kinds), ("" if n["f"] in kinds else f"unknown field {n['f']!r}")
    for a in n["args"]:
        ok, why = well_typed(a, kinds)
        if not ok:
            return False, why
    ks = [infer_kind(a, kinds) for a in n["args"]]
    name = n["op"]
    if name in ("add", "sub"):
        a, b = ks
        if is_const(n["args"][0]) or is_const(n["args"][1]):
            return False, f"{name} with a bare constant says nothing about the ranking"
        if a != b and {a, b} != {"amount_rep", "amount_trade"}:
            return False, f"cannot {name} a {a} and a {b}"
        if a == b == "ratio" and scale_tag(n["args"][0]) != scale_tag(n["args"][1]):
            return False, f"{name} of two ratios on different scales ({scale_tag(n['args'][0])} and {scale_tag(n['args'][1])}): rank or standardise each first"
    if name == "div" and result_kind("div", ks) == "other":
        return False, f"dividing a {ks[0]} by a {ks[1]} has no clear meaning"
    if name == "mul" and result_kind("mul", ks) == "other":
        return False, f"multiplying a {ks[0]} by a {ks[1]} has no clear meaning"
    if name == "log" and ks[0] not in ("amount_rep", "amount_trade", "count", "ratio"):
        return False, "log of an undefined quantity"
    return True, ""


# ------------------------------------------------------------------------------------------------------------------ evaluation
def _positive(x: pd.DataFrame) -> pd.DataFrame:
    return x.where(x > 0)


def evaluate(n: dict, ctx: ResearchContext, cache: dict | None = None) -> tuple[pd.DataFrame, str]:
    """``(panel, kind)``: the formula's value for every date and security, as known on the date. Sub-formulas are remembered in ``cache`` (by canonical form) so that a search never computes the
    same piece twice."""
    cache = ctx._cache.setdefault("expr", {}) if cache is None else cache
    return _eval(canonical(n), ctx, cache)


def _eval(n: dict, ctx: ResearchContext, cache: dict) -> tuple[pd.DataFrame, str]:
    key = to_string(n)
    if key in cache:
        return cache[key]
    if is_const(n):
        raise ExprError("a constant has no panel of its own; use it inside add, sub, mul or div")
    if is_field(n):
        kinds = catalogue(ctx)
        if n["f"] not in kinds:
            raise ExprError(f"unknown field {n['f']!r}: this universe's database has no such figure")
        out = (_field_panel(ctx, n["f"]), kinds[n["f"]].kind)
    else:
        name, args = n["op"], n["args"]
        if name in BINARY and any(is_const(a) for a in args):
            const_side = 0 if is_const(args[0]) else 1
            other, kind = _eval(args[1 - const_side], ctx, cache)
            c = float(args[const_side]["c"])
            panel = {"add": other + c, "mul": other * c, "sub": (c - other) if const_side == 0 else (other - c),
                     "div": (c / _positive(other)) if const_side == 0 else (other / c if c > 0 else other.where(False))}[name]
            out = (panel, kind if name != "div" or const_side == 1 else "ratio")
        else:
            evaluated = [_eval(a, ctx, cache) for a in args]
            kinds = [k for _, k in evaluated]
            panels = [p for p, _ in evaluated]
            rep, trade = "amount_rep", "amount_trade"
            if name in BINARY and {kinds[0], kinds[1]} == {rep, trade}:               # an amount in the reporting currency set against one in the trading currency: convert first
                panels = [ctx.to_trading(p) if k == rep else p for p, k in zip(panels, kinds)]
            kind = result_kind(name, kinds)
            x = panels[0]
            if name == "add":
                p = x + panels[1]
            elif name == "sub":
                p = x - panels[1]
            elif name == "mul":
                p = x * panels[1]
            elif name == "div":
                p = x / _positive(panels[1])
            elif name == "neg":
                p = -x
            elif name == "abs":
                p = x.abs()
            elif name == "log":
                p = np.log(_positive(x))
            elif name == "delta":
                p = transforms.delta(x, n["k"])
            elif name == "growth":
                p = transforms.pct_change(x, n["k"])
            elif name == "accel":
                g = transforms.pct_change(x, n["k"])
                p = g - g.shift(n["k"])
            elif name == "rolling_mean":
                p = x.rolling(n["k"], min_periods=max(n["k"] // 2, 2)).mean()
            elif name == "rolling_std":
                p = x.rolling(n["k"], min_periods=max(n["k"] // 2, 3)).std()
            elif name == "rank":
                p = transforms.rank(x.where(ctx.investable().reindex_like(x).fillna(False)), pct=True)
            elif name == "sector_rank":
                p = transforms.group_rank(x.where(ctx.investable().reindex_like(x).fillna(False)), ctx.group("sector"))
            elif name == "industry_rank":
                p = transforms.group_rank(x.where(ctx.investable().reindex_like(x).fillna(False)), ctx.group("industry"))
            elif name == "zscore":
                p = transforms.apply(x, [{"op": "winsorize", "pct": 1.0}, {"op": "zscore"}], ctx)
            else:                                                                           # validate() has already refused anything else
                raise ExprError(f"unknown operator {name!r}")
            out = (p.replace([np.inf, -np.inf], np.nan), kind)
    cache[key] = out
    return out


def evaluate_panel(n: dict, ctx: ResearchContext) -> pd.DataFrame:
    return evaluate(n, ctx)[0]


def describe(n: dict, ctx: ResearchContext) -> dict:
    """What a formula is, for display: its text, hash, size, fields, kind and whether it is well typed on this data."""
    cat = catalogue(ctx)
    kinds = {k: v.kind for k, v in cat.items()}
    ok, why = well_typed(n, kinds)
    c = canonical(n)
    return {"text": to_string(c), "hash": expression_hash(n), "signal_key": signal_key(n), "nodes": size(c), "depth": depth(c), "fields": sorted(set(fields_used(c))), "well_typed": ok, "problem": why,
            "kind": infer_kind(c, kinds) if ok else None}


# ------------------------------------------------------------------------------------------------------------------ making formulas
def _paths(n: dict, prefix: tuple = ()):
    yield prefix, n
    for i, c in enumerate(children(n)):
        yield from _paths(c, prefix + (i,))


def _get(n: dict, path: tuple) -> dict:
    for i in path:
        n = n["args"][i]
    return n


def _put(n: dict, path: tuple, new: dict) -> dict:
    if not path:
        return new
    args = list(n["args"])
    args[path[0]] = _put(args[path[0]], path[1:], new)
    return {**n, "args": args}


class Generator:
    """Builds well-typed formulas from the fields of one data set: random ones, mutations of a formula and crossings of two. Everything follows from the random generator passed in, so a search
    can be repeated or resumed exactly."""

    def __init__(self, ctx: ResearchContext, rng: np.random.Generator, max_nodes: int = 15, only_fields: list[str] | None = None):
        cat = catalogue(ctx)
        self.kinds = {k: v.kind for k, v in cat.items() if only_fields is None or k in only_fields}
        self.rng, self.max_nodes = rng, max_nodes
        self.flows = [f for f in list(STATEMENT_FLOWS) + ["fcf"] if f in self.kinds]
        self.stocks = [f for f in STATEMENT_STOCKS if f in self.kinds]
        self.market = [f for f in MARKET_AMOUNTS if f in self.kinds]
        self.ratios = [f for f, v in cat.items() if f in self.kinds and v.kind == "ratio" and v.source in ("metric", "market")]
        self.lags, self.windows = LAGS[ctx.freq], WINDOWS[ctx.freq]
        if not (self.flows and (self.stocks or self.market)) and not self.ratios:
            raise ExprError("this universe has too few fields to build formulas from")

    def _pick(self, items):
        return items[int(self.rng.integers(len(items)))]

    def base(self) -> dict:
        r = self.rng.random()
        if r < 0.4 and self.flows and (self.stocks or self.market):
            return op("div", field(self._pick(self.flows)), field(self._pick(self.stocks + self.market)))
        if r < 0.5 and len(self.stocks) >= 2:
            a, b = self._pick(self.stocks), self._pick(self.stocks + self.market)
            return op("div", field(a), field(b)) if a != b else field(self._pick(self.ratios or self.stocks))
        if self.ratios:
            return field(self._pick(self.ratios))
        return op("div", field(self._pick(self.flows)), field(self._pick(self.stocks)))

    def _time_series(self, b: dict) -> dict:
        r = self.rng.random()
        if r < 0.35:
            return op("delta", b, k=self._pick(self.lags))
        if r < 0.6:
            return op("growth", b, k=self._pick(self.lags))
        if r < 0.75:
            return op("accel", b, k=self._pick(self.lags))
        if r < 0.9:
            return op("sub", b, op("rolling_mean", b, k=self._pick(self.windows)))
        return op("neg", op("rolling_std", b, k=self._pick(self.windows)))

    def random(self) -> dict:
        for _ in range(60):
            r = self.rng.random()
            b = self.base()
            if r < 0.28:
                e = b
            elif r < 0.62:
                e = self._time_series(b)
            elif r < 0.9:
                other = self.base() if self.rng.random() < 0.6 else self._time_series(self.base())
                e = op(self._pick(["add", "sub"]), op("rank", self._time_series(b) if self.rng.random() < 0.4 else b), op("rank", other))
            else:
                e = op("rank", self._time_series(b))
            if self.rng.random() < 0.3 and e.get("op") not in ("rank", "sector_rank"):
                e = op("sector_rank", e)
            if self._ok(e):
                return canonical(e)
        raise ExprError("could not build a well-typed formula from these fields")

    def _ok(self, e: dict) -> bool:
        try:
            if size(e) > self.max_nodes or depth(e) > MAX_DEPTH:
                return False
            ok, _ = well_typed(canonical(e), self.kinds)
            return ok
        except ExprError:
            return False

    def _replacement(self, node: dict, path: tuple) -> dict | None:
        r = self.rng.random()
        if is_const(node):
            return None
        if is_field(node):
            same = [f for f, k in self.kinds.items() if k == self.kinds.get(node["f"]) and f != node["f"]]
            return field(self._pick(same)) if same else None
        name = node["op"]
        if name in TIME_SERIES:
            pool = self.windows if name.startswith("rolling") else self.lags
            return op(name, *node["args"], k=self._pick([k for k in pool if k != node["k"]] or list(pool)))
        if name in ("add", "sub") and r < 0.5:
            return op("sub" if name == "add" else "add", *node["args"])
        if name == "sector_rank":
            return node["args"][0]
        try:
            kind = infer_kind(node, self.kinds)
        except ExprError:
            return None
        if r < 0.3 and path and kind == "ratio":
            return op("sector_rank", node)
        if kind == "ratio":
            return self._time_series(self.base()) if r < 0.65 else self.base()
        if kind == "amount_rep":
            return field(self._pick(self.flows + self.stocks))
        return None

    def mutate(self, e: dict) -> dict:
        """A small change to a formula: another field of the same kind, another lag, a sector rank put on or taken off, or a new part in place of a subtree."""
        for _ in range(40):
            nodes = list(_paths(e))
            path, node = nodes[int(self.rng.integers(len(nodes)))]
            new = self._replacement(node, path)
            if new is None:
                continue
            candidate = _put(e, path, new)
            if self._ok(candidate):
                try:
                    if expression_hash(candidate) != expression_hash(e):
                        return canonical(candidate)
                except ExprError:
                    continue
        return self.random()

    def cross(self, a: dict, b: dict) -> dict:
        """Replace a subtree of ``a`` by a subtree of ``b`` that has the same kind."""
        for _ in range(40):
            pa, na = list(_paths(a))[int(self.rng.integers(size(a)))]
            candidates = [nb for _, nb in _paths(b) if not is_const(nb)]
            nb = candidates[int(self.rng.integers(len(candidates)))]
            try:
                same_kind = infer_kind(na, self.kinds) == infer_kind(nb, self.kinds)
            except ExprError:
                continue
            if not same_kind:
                continue
            child = _put(a, pa, nb)
            if self._ok(child):
                try:
                    if expression_hash(child) not in (expression_hash(a), expression_hash(b)):
                        return canonical(child)
                except ExprError:
                    continue
        return self.mutate(a)
