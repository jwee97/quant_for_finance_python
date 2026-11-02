"""Feature and strategy discovery with a finite budget, full logging and selection controls (spec v3, Release 8, P3-07). Optional: nothing is searched unless it is asked for.

Searching many expressions for one that works is the surest way to find something that does not. The module lets the search happen and makes its limits impossible to miss.

* **A closed grammar that cannot read the future.** An expression is parsed (never ``eval``-ed) and accepted only if every node is one of: a number, a feature name from the panel it is given, ``+ - * /``,
  unary minus, or one of ``lag`` ``delta`` ``rolling_mean`` ``rolling_std`` ``rank`` ``zscore`` ``abs`` ``sign``. The time functions look back only (``lag(x, k)`` with ``k >= 0``); the cross-sectional
  ones use the same date. A name that is not a feature (the label, a forward return, a sealed outcome) is refused, so an expression cannot read an outcome. Beyond the parser, ``causal_check`` evaluates
  an expression on the data and on the data with everything after a date scrambled and requires the values up to that date to be identical.
* **A finite budget, all of it logged.** A search has a maximum number of candidates; every candidate that was generated is in the log, evaluated or rejected, with its reason. The budget cannot be
  exceeded by asking for more.
* **Only development rows.** Candidates are scored by their rank IC on dates before ``sealed_from``; later dates are masked whatever the target holds, and a search whose development window reaches the
  sealed dates is refused.
* **Selection controls, and what they do not do.** Every evaluated candidate gets a two-sided p-value from a Newey-West error (lag at least ``horizon - 1``). The p-values are adjusted for the number of
  candidates tested by Benjamini-Hochberg (the false discovery rate) and Holm (the family-wise error rate), and the best candidate's p-value is also reported as ``min(1, N p)``. These controls bound the
  chance of *false discoveries among the candidates tested*; they do not make the best mined rule independent evidence, they assume the tests are valid, and they are only as good as the count of what was
  really tried. The one independent test is ``confirm``: a single pre-named rule scored on development rows the search never saw.
"""

from __future__ import annotations

import ast

import numpy as np
import pandas as pd

from . import alpha as al
from . import drift as dr
from . import inference as inf

VERSION = "discovery-v1"
FUNCS = {"lag": 2, "delta": 2, "rolling_mean": 2, "rolling_std": 2, "rank": 1, "zscore": 1, "abs": 1, "sign": 1}
TIME_FUNCS = ("lag", "delta", "rolling_mean", "rolling_std")
MAX_K = 252


class DiscoveryError(ValueError):
    """An expression or a search the procedure refuses."""


def _int_constant(node) -> int | None:
    """A whole-number constant written in the expression (a leading minus is allowed so that a negative window is recognised and refused with its reason)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return int(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _int_constant(node.operand)
        return None if inner is None else -inner
    return None


def validate(expr: str, features: set | list) -> dict:
    """Parse ``expr`` and accept it only if it is built from the allowed nodes. Returns the features it reads, the longest look-back in sessions and the expression's canonical text."""
    names = set(features)
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as exc:
        raise DiscoveryError(f"not an expression: {exc.msg}") from None
    used = set()

    def walk(node) -> int:                                                                         # returns the look-back (sessions) of the value of this node
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise DiscoveryError("only numbers are allowed as constants")
            return 0
        if isinstance(node, ast.Name):
            if node.id not in names:
                raise DiscoveryError(f"'{node.id}' is not a feature of the panel: an expression can read only the features it is given, never a label, a forward return or a sealed outcome")
            used.add(node.id)
            return 0
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return walk(node.operand)
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            return max(walk(node.left), walk(node.right))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in FUNCS or node.keywords:
                raise DiscoveryError("only the functions " + ", ".join(sorted(FUNCS)) + " are allowed, without keyword arguments")
            f = node.func.id
            if len(node.args) != FUNCS[f]:
                raise DiscoveryError(f"{f} takes {FUNCS[f]} argument(s)")
            inner = walk(node.args[0])
            if f in TIME_FUNCS:
                k = _int_constant(node.args[1])
                if k is None:
                    raise DiscoveryError(f"the window of {f} is a whole-number constant")
                if k < 0:
                    raise DiscoveryError(f"{f} with a negative window would read the future")
                if not (1 <= k <= MAX_K) and not (f == "lag" and k == 0):
                    raise DiscoveryError(f"the window of {f} is between 1 and {MAX_K} sessions")
                return inner + (k if f in ("lag", "delta") else k - 1)
            return inner
        raise DiscoveryError(f"{type(node).__name__} is not allowed in an expression")

    depth = walk(tree.body)
    if not used:
        raise DiscoveryError("an expression reads at least one feature")
    return {"features": sorted(used), "lookback": int(depth), "canonical": ast.unparse(tree.body)}


def evaluate(expr: str, panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """The expression's value for every date and asset. Every operator looks back or sits on one date, so a value at ``t`` is a function of data up to ``t``."""
    validate(expr, set(panel))
    tree = ast.parse(expr.strip(), mode="eval").body

    def ev(node):
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, ast.Name):
            return panel[node.id]
        if isinstance(node, ast.UnaryOp):
            return -ev(node.operand)
        if isinstance(node, ast.BinOp):
            a, b = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Mult):
                return a * b
            out = a / b
            return out.replace([np.inf, -np.inf], np.nan) if isinstance(out, pd.DataFrame) else out
        f = node.func.id
        x = ev(node.args[0])
        if f == "rank":
            return x.rank(axis=1, pct=True)
        if f == "zscore":
            sd = x.std(axis=1, ddof=0).replace(0.0, np.nan)
            return x.sub(x.mean(axis=1), axis=0).div(sd, axis=0)
        if f == "abs":
            return x.abs()
        if f == "sign":
            return np.sign(x)
        k = _int_constant(node.args[1])
        if f == "lag":
            return x.shift(k)
        if f == "delta":
            return x - x.shift(k)
        if f == "rolling_mean":
            return x.rolling(k, min_periods=k).mean()
        return x.rolling(k, min_periods=k).std(ddof=1)

    out = ev(tree)
    if not isinstance(out, pd.DataFrame):
        raise DiscoveryError("an expression that is a constant has no value per asset")
    return out


def causal_check(expr: str, panel: dict[str, pd.DataFrame], at: int | None = None, seed: int = 0) -> dict:
    """Evaluate on the panel and on a copy whose rows after position ``at`` are replaced by noise; the values up to ``at`` must be identical."""
    rng = np.random.default_rng(seed)
    n = len(next(iter(panel.values())))
    at = n // 2 if at is None else int(at)
    base = evaluate(expr, panel)
    scrambled = {k: v.copy() for k, v in panel.items()}
    for v in scrambled.values():
        v.iloc[at + 1:] = rng.normal(size=v.iloc[at + 1:].shape) * 10.0
    other = evaluate(expr, scrambled)
    a, b = base.iloc[: at + 1].to_numpy(), other.iloc[: at + 1].to_numpy()
    same = bool(np.allclose(a, b, equal_nan=True, atol=0.0, rtol=0.0))
    return {"causal": same, "checked_through": str(base.index[at]), "differences": int((~np.isclose(a, b, equal_nan=True, atol=0.0, rtol=0.0)).sum())}


def generate(features: list[str], n: int, seed: int = 0, depth: int = 2) -> list[str]:
    """Up to ``n`` distinct random expressions from the grammar (fewer if the grammar runs out at this depth). The budget is applied here: asking for more returns ``n`` at most."""
    rng = np.random.default_rng(seed)
    ks = [1, 2, 3, 5, 10, 21]
    unary = ["rank", "zscore", "abs", "sign"]

    def node(d: int) -> str:
        if d <= 0 or rng.random() < 0.3:
            return str(rng.choice(features))
        r = rng.random()
        if r < 0.35:
            return f"{rng.choice(unary)}({node(d - 1)})"
        if r < 0.7:
            return f"{rng.choice(['lag', 'delta', 'rolling_mean', 'rolling_std'])}({node(d - 1)}, {int(rng.choice(ks))})"
        return f"({node(d - 1)} {rng.choice(['+', '-', '*'])} {node(d - 1)})"

    seen, out, tries = set(), [], 0
    while len(out) < n and tries < 50 * n:
        tries += 1
        e = node(depth)
        if "(" not in e and e in features and len(features) > n:
            continue
        key = ast.unparse(ast.parse(e, mode="eval").body)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def adjust_bh(p: list[float]) -> list[float]:
    return dr.adjust(list(p), "bh")


def search(panel: dict[str, pd.DataFrame], target: pd.DataFrame, *, horizon: int, budget: int, sealed_from, dev_start=None, seed: int = 0, q: float = 0.1, min_assets: int = 5, expressions: list[str] | None = None,
           depth: int = 2) -> dict:
    """Score up to ``budget`` expressions by rank IC on development dates and apply the selection controls. ``target`` is the forward return over ``horizon`` sessions; dates from ``sealed_from`` on are ignored
    whatever it holds. Returns the full log of every candidate, the controls and the discoveries at false discovery rate ``q``."""
    if budget < 1:
        raise DiscoveryError("the budget is at least one candidate")
    sealed = pd.Timestamp(sealed_from)
    idx = target.index
    dev = (idx < sealed) & (idx >= (pd.Timestamp(dev_start) if dev_start is not None else idx[0]))
    if dev.sum() < 60:
        raise DiscoveryError(f"{int(dev.sum())} development dates before {sealed.date()}; at least 60 are needed")
    cands = list(expressions) if expressions is not None else generate(list(panel), budget, seed, depth)
    truncated = len(cands) > budget
    cands = cands[:budget]
    log, p_idx = [], []
    dev_s = pd.Series(dev, index=idx)
    tgt = target.where(dev_s, axis=0)                                                               # the sealed rows of the label are not visible to any score
    for i, e in enumerate(cands):
        row = {"id": i, "expression": e, "status": "evaluated"}
        try:
            info = validate(e, set(panel))
            val = evaluate(e, panel)
        except DiscoveryError as exc:
            log.append({**row, "status": "rejected", "reason": str(exc)})
            continue
        ic = al.ic_table(val.where(dev_s, axis=0), tgt, min_assets)["ic"].dropna()
        if len(ic) < 30:
            log.append({**row, "status": "rejected", "reason": f"only {len(ic)} development dates had an IC", "lookback": info["lookback"]})
            continue
        res = inf.hac_mean(ic, horizon=int(horizon), label=f"rank IC of {e}").as_dict()
        if res["eligibility"] != "applied" or res.get("p_value") is None:
            log.append({**row, "status": "rejected", "reason": "the IC could not be tested: " + str(res.get("reason") or res["eligibility"])})
            continue
        row.update(ic_mean=res["value"], p_value=res["p_value"], n_dates=int(len(ic)), lookback=info["lookback"], features=info["features"])
        log.append(row)
        p_idx.append(len(log) - 1)
    ps = [log[i]["p_value"] for i in p_idx]
    bh, holm = adjust_bh(ps), dr.adjust(ps, "holm")
    for i, a, b in zip(p_idx, bh, holm):
        log[i].update(p_bh=a, p_holm=b, discovered=bool(a is not None and a <= q))
    ev = [log[i] for i in p_idx]
    best = min(ev, key=lambda r: r["p_value"], default=None)
    n_tested = len(ev)
    return {"version": VERSION, "budget": int(budget), "generated": len(cands), "truncated_to_budget": bool(truncated), "tested": n_tested, "rejected": len(log) - n_tested, "q": q, "log": log,
            "discoveries": [r["id"] for r in ev if r["discovered"]], "development": {"first": str(idx[dev][0].date()), "last": str(idx[dev][-1].date()), "dates": int(dev.sum()), "sealed_from": str(sealed.date())},
            "best": None if best is None else {"id": best["id"], "expression": best["expression"], "ic_mean": best["ic_mean"], "p_value": best["p_value"], "p_best_of_n": min(1.0, n_tested * best["p_value"]),
                                                 "p_bh": best["p_bh"], "p_holm": best["p_holm"]},
            "controls": {"false_discovery_rate": "Benjamini-Hochberg over every tested candidate", "family_wise": "Holm over every tested candidate", "best_of_n": "min(1, N * p) for the smallest p",
                         "tests": n_tested, "expected_false_discoveries_if_nothing_works": float(q * 1.0)},
            "limitations": ["the controls bound false discoveries among the candidates that were tested; they do not make the best mined rule independent evidence", "they assume each test is valid and count only what was tried here: "
                            "earlier searches on the same data are not in N", "the adjusted p-values are not corrected for the generation of expressions that were never scored (rejected ones) or for candidates run elsewhere",
                            "the only independent evidence is `confirm`, on development rows the search never saw"]}


def confirm(expr: str, panel: dict[str, pd.DataFrame], target: pd.DataFrame, *, horizon: int, start, end, sealed_from, min_assets: int = 5) -> dict:
    """The single pre-named test: one expression on development dates the search never saw (``start`` must be after the search's last development date, and before ``sealed_from``)."""
    sealed = pd.Timestamp(sealed_from)
    if pd.Timestamp(end) >= sealed:
        raise DiscoveryError("the confirmation window reaches the sealed dates")
    idx = target.index
    m = (idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))
    if m.sum() < 30:
        raise DiscoveryError(f"{int(m.sum())} dates in the confirmation window; at least 30 are needed")
    val = evaluate(expr, panel)
    ic = al.ic_table(val.loc[m], target.loc[m], min_assets)["ic"].dropna()
    res = inf.hac_mean(ic, horizon=int(horizon), label=f"confirmation rank IC of {expr}").as_dict()
    return {"expression": expr, "window": [str(pd.Timestamp(start).date()), str(pd.Timestamp(end).date())], "dates": int(len(ic)), "ic": res,
            "note": "one rule, named before it was scored, on rows the search never used: the only part of this procedure that is independent evidence (and it is development data, not the sealed segments)"}


def pairs_budget(n_features: int, depth: int = 2) -> int:
    """A rough count of how many distinct expressions the grammar offers at a depth, to size a budget sensibly: it grows very fast, which is the reason for the budget."""
    unary = 4 + 4 * 6
    layer = n_features
    for _ in range(depth):
        layer = n_features + unary * layer + 3 * layer * layer
    return int(layer)

