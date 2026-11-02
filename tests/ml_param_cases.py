"""Values to try against the settings of a model, made from the schema itself, so that a model added to the catalogue is covered without anyone listing its settings here.

``valid_values`` are inside the range (its bounds too, where they are allowed), ``invalid_values`` are not: out of range, of the wrong type, or malformed. ``request_params`` is what a client would send."""

from __future__ import annotations

from src.ml import param_schema as ps


def _inside(s: ps.Setting, x: float) -> bool:
    if s.low is not None and (x < s.low or (s.low_open and x == s.low)):
        return False
    return not (s.high is not None and (x > s.high or (s.high_open and x == s.high)))


def _span(s: ps.Setting) -> float:
    return (s.high - s.low) if s.low is not None and s.high is not None else 1.0


def valid_values(s: ps.Setting) -> list:
    """A handful of values the setting accepts: its default, each bound that is allowed (or the nearest value inside an excluded one), and a value in between."""
    out: list = []
    if s.type in (ps.INT, ps.FLOAT):
        whole = s.type == ps.INT
        eps = 1 if whole else max(_span(s) * 1e-3, 1e-6)
        picks = [s.default]
        if s.low is not None:
            picks.append(s.low + (eps if s.low_open else 0))
        if s.high is not None:
            picks.append(s.high - (eps if s.high_open else 0))
        if s.low is not None and s.high is not None:
            picks.append((s.low + s.high) / 2)
        else:
            picks.append((s.default * 2 if s.default else (s.low or 0) + 2))
        for x in picks:
            x = int(round(x)) if whole else float(x)
            if _inside(s, x) and x not in out:
                out.append(x)
    elif s.type == ps.BOOL:
        out = [0, 1, True, False]
    elif s.type == ps.CHOICE:
        out = list(s.choices)
    elif s.type == ps.LIST:
        out = [[], ["ret_1"], "ret_1,ret_lag_2", ["ret_1", "ret_lag_2", "mom_21"]]
    else:
        out = {"ints": ["1", "4,2,1", " 3 , 2 "], "members": ["ridge,lasso", "ridge,lasso[alpha=5]", "ridge[alpha=3],elasticnet[alpha=0.1;l1_ratio=0.3],linear"]}.get(s.grammar or "", ["", "abc"])
        if s.default and s.default not in out:
            out.append(s.default)
    return out


def invalid_values(s: ps.Setting) -> list:
    """Values the setting refuses, each of a different kind of wrong: out of range at each bound, the wrong type, not a number, malformed text."""
    out: list = []
    if s.type in (ps.INT, ps.FLOAT):
        step = 1 if s.type == ps.INT else max(_span(s) * 1e-2, 1e-3)
        if s.low is not None:
            out.append(s.low if s.low_open else s.low - step)
        if s.high is not None:
            out.append(s.high if s.high_open else s.high + step)
        out += ["abc", True, [1], {"a": 1}, float("nan"), float("inf"), -float("inf"), 1e19]
        if s.type == ps.INT:
            out.append(2.5)
    elif s.type == ps.BOOL:
        out = ["yes", 2, 0.5, [1], {}, -1]
    elif s.type == ps.CHOICE:
        out = ["no_such_choice", 7, ["x"], True]
    elif s.type == ps.LIST:
        out = [5, [1], ["bad name!"], "a b", {"a"}, [["ret_1"]], ["x" * 61]]
    else:
        out = [5, True, ["a"], "x\x00y", "a" * (ps.MAX_TEXT + 1)]
        out += {"ints": ["a,b", "0", "1,,2", "1001", "-1", "1.5"], "members": ["", "ridge", "ridge,ppo", "ridge,nope", "ridge,model_ensemble", "ridge[bogus=1],lasso", "ridge[alpha],lasso", "ridge,lasso[alpha=-1]",
                                                                                 "ridge[alpha=1;alpha=2;beta=3],lasso", ",".join(["ridge"] * 9), "Ridge,lasso"]}.get(s.grammar or "", [])
    return out


def request_params(kind: str, defaults: dict | None = None) -> dict:
    """The params of a request that sends every setting the way the form used to: switches as 1 or 0, numbers as JSON numbers (an integral one without a decimal point), text as typed."""
    from src.ml.models import _INFO

    out = {}
    for name, value in (defaults if defaults is not None else _INFO[kind].params).items():
        if isinstance(value, float) and value.is_integer():
            value = int(value)                                                 # JSON.stringify(10) is "10"
        out[name] = list(value) if isinstance(value, (list, tuple)) else value
    return out
