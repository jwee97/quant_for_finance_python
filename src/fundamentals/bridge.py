"""BACKTEST THIS ALPHA: a signal becomes a book, the book goes through the platform's own backtest engine, and the result comes back with the context a Sharpe ratio needs.

**The signal** can be one library factor, a technical (price-only) signal, a library alpha, a formula, a combination of alphas by a rule, a machine-learning model fitted walk-forward to several of
them, or a weighted composite of any of these (:func:`resolve`). Higher means expect more.

**The book** (:func:`build_book`) is decided on each rebalance date from what was known on it: the companies that are investable, large enough and liquid enough are ranked; the top ``n`` (or top
percentile) are bought and, for a long-short book, the bottom ``n`` are sold; each leg is weighted equally, by how far the signal clears the cutoff, or by inverse volatility; then the limits are
applied. A position, a sector, an industry or a country is limited as a share of the account, the net beta to the market is held inside a band, and a name is never held above a share of its own
average daily dollar volume. The limits are met by the nearest book that respects them (a linear program finds how much of each leg can be invested, a quadratic program finds the weights nearest the
wanted ones); if the limits cannot all be met, both legs are scaled down together and the rest stays in cash, and the report says how often. A turnover limit (the most bought or sold at one rebalance, as a share of the book) trades only part of the way to the
new book, so the first book is built over several dates. Too few eligible companies means cash for that date, never a smaller, riskier book.

**The run** hands the weights, stamped on the decision date, to :class:`backtest.engine.BacktestEngine`, the engine every other strategy on the platform uses: the book is struck ``entry_lag`` days
after the decision (the same lag the factor tests use, so the two measure the same trade), drifts with prices between rebalances, and pays a linear cost on what it trades; a short pays a borrow
fee on its notional. A company that stops trading is carried at its last price. The result is set against holding every eligible company equally, split by the dates the alpha was found on when the
library knows them, and deflated for the number of ideas that were tried to find it.

Nothing here makes an alpha valid. It shows what trading it would have returned, net of costs, on this data.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from ..backtest.costs import LinearCostModel, cost_sensitivity
from ..backtest.engine import BacktestEngine
from ..backtest.metrics import deflated_sharpe_ratio, performance_summary
from ..equity import alpha_model
from ..equity.qp import solve_qp, solve_qp_ipm
from . import combine as combiner
from . import expr as X
from . import factors as factor_library
from .errors import FundamentalsError
from .evaluate import EvalSpec
from .evaluate import build_signal as build_factor_signal
from .jsonutil import clean, series_points
from .library import AlphaLibrary
from .library import build_signal as build_alpha_signal
from .research import ResearchContext
from .schema import now_iso

SIDES = ("long_only", "long_short")
SELECTIONS = ("top_n", "percentile")
WEIGHTINGS = ("equal", "signal", "inverse_vol")
TECHNICAL_FAMILIES = ("price_momentum", "risk")
KINDS = ("factor", "technical", "alpha", "expression", "combination", "ml")
MAX_PARTS, MAX_DEPTH = 20, 3
TOL = 1e-9


# ------------------------------------------------------------------------------------------------------------------ what to trade
def _bounded(name: str, value, low: float, high: float, optional: bool = False, open_low: bool = False):
    """``value`` as a float inside the range (the lower end excluded when ``open_low``); ``None`` is allowed when ``optional``."""
    if value is None:
        if optional:
            return None
        raise FundamentalsError(f"{name} is required")
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise FundamentalsError(f"{name} must be a number") from None
    if not np.isfinite(x) or x > high or x < low or (open_low and x <= low):
        raise FundamentalsError(f"{name} must be above {low:g} and at most {high:g}" if open_low else f"{name} must be between {low:g} and {high:g}")
    return x


@dataclass(frozen=True)
class PortfolioSpec:
    """How a signal becomes a book. Every limit is a share of the account (1.0 is the whole account); ``None`` means no limit."""

    side: str = "long_only"                      # long_only | long_short (the short leg has the same exposure as the long)
    selection: str = "top_n"                     # top_n | percentile
    n: int = 30                                  # names per leg when selecting the top n
    percentile: float = 20.0                     # share of the eligible names per leg when selecting by percentile
    weighting: str = "equal"                     # equal | signal (how far the signal clears the cutoff) | inverse_vol
    leg_exposure: float = 1.0                    # what each leg adds up to
    max_position: float = 0.10                   # one name
    max_sector: float | None = None              # net exposure to one sector (long minus short)
    max_industry: float | None = None
    max_country: float | None = None
    max_beta: float | None = None                # the net beta of the book to the universe, both signs
    max_turnover: float | None = None            # the most bought, or sold, at one rebalance as a share of the book (a whole new book is 1.0)
    min_market_cap_usd: float = 0.0
    min_dollar_volume_usd: float = 0.0
    max_adv_share: float | None = None           # a position may be at most this share of the name's average daily dollar volume, at ``capital_usd``
    capital_usd: float = 1_000_000.0
    min_names: int = 5                           # fewer eligible names per leg than this and the date is cash
    cost_bps: float = 10.0                       # one way, on the notional traded
    borrow_bps: float = 50.0                     # a year, on the short notional
    start: str | None = None
    end: str | None = None

    def __post_init__(self):
        if self.side not in SIDES:
            raise FundamentalsError(f"side must be one of {SIDES}")
        if self.selection not in SELECTIONS:
            raise FundamentalsError(f"selection must be one of {SELECTIONS}")
        if self.weighting not in WEIGHTINGS:
            raise FundamentalsError(f"weighting must be one of {WEIGHTINGS}")
        for key, low, high in (("n", 1, 5000), ("min_names", 1, 5000)):
            v = getattr(self, key)
            if isinstance(v, bool) or int(v) != v or not low <= v <= high:
                raise FundamentalsError(f"{key} must be a whole number from {low} to {high}")
            object.__setattr__(self, key, int(v))
        _bounded("percentile", self.percentile, 0.0, 50.0 if self.side == "long_short" else 100.0, open_low=True)
        _bounded("leg_exposure", self.leg_exposure, 0.0, 3.0, open_low=True)
        _bounded("max_position", self.max_position, 0.0, 1.0, open_low=True)
        for key in ("max_sector", "max_industry", "max_country", "max_turnover", "max_adv_share"):
            _bounded(key, getattr(self, key), 0.0, 1.0, optional=True, open_low=True)
        _bounded("max_beta", self.max_beta, 0.0, 5.0, optional=True)
        for key, high in (("min_market_cap_usd", 1e15), ("min_dollar_volume_usd", 1e15), ("cost_bps", 1000.0), ("borrow_bps", 5000.0)):
            _bounded(key, getattr(self, key), 0.0, high)
        _bounded("capital_usd", self.capital_usd, 0.0, 1e15, open_low=True)

    def describe(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict | None) -> "PortfolioSpec":
        d = dict(d or {})
        unknown = sorted(set(d) - set(PortfolioSpec.__dataclass_fields__))
        if unknown:
            raise FundamentalsError(f"unknown portfolio setting(s) {unknown}; the settings are {sorted(PortfolioSpec.__dataclass_fields__)}")
        return PortfolioSpec(**d)

    @property
    def min_leg(self) -> int:
        return min(self.min_names, self.n) if self.selection == "top_n" else self.min_names


def technical_names() -> list[str]:
    """The price-only factors: momentum and risk measured from the price history alone."""
    return [n for n, f in factor_library.FACTORS.items() if f.family in TECHNICAL_FAMILIES]


def resolve(ctx: ResearchContext, spec, library: AlphaLibrary | None = None, depth: int = 0) -> tuple[pd.DataFrame, dict]:
    """A signal panel (dates by securities, higher means expect more) from a description, with a record of what it is made of.

    ``{"kind": "factor", "name": ..., "steps": ..., "direction": ...}`` or ``"technical"`` (the same, limited to price-only factors); ``{"kind": "alpha", "alpha_id": ...}`` from the library;
    ``{"kind": "expression", "expression": {...}, "direction": 1}``; ``{"kind": "combination", "method": ..., "of": [...]}`` by a rule of :mod:`combine`; ``{"kind": "ml", "model": ..., "of": [...]}``
    a model fitted walk-forward; and ``{"parts": [..., {"weight": 2.0}]}`` a weighted composite of any of these (standardised on each date; a company missing a part has no view on it).
    """
    if depth > MAX_DEPTH:
        raise FundamentalsError(f"a signal can be built from other signals at most {MAX_DEPTH} levels deep")
    if isinstance(spec, list):
        spec = {"parts": spec}
    if not isinstance(spec, dict):
        raise FundamentalsError("describe the signal as a dictionary (one signal) or a list of them (a composite)")
    if "parts" in spec:
        return _composite(ctx, spec["parts"], library, depth)
    kind = spec.get("kind")
    if kind not in KINDS:
        raise FundamentalsError(f"a signal's kind must be one of {KINDS}")
    info: dict = {"kind": kind, "warnings": [], "alphas": [], "n_trials": 1}
    if kind in ("factor", "technical"):
        name = spec.get("name")
        if kind == "technical" and name not in technical_names():
            raise FundamentalsError(f"a technical signal is one of {technical_names()}")
        direction = spec.get("direction", "theory")
        if direction not in ("theory", 1, -1):
            raise FundamentalsError("a factor's direction is 'theory' (its own), 1 or -1; choosing the sign from the data is a trial to be counted, not a setting")
        es = EvalSpec(factor=name, direction=direction)
        if spec.get("steps") is not None:
            es.steps = list(spec["steps"])
        signal = build_factor_signal(ctx, es)
        info["label"] = f"{name}" + ("" if es.direction == "theory" else f" ({'+' if int(es.direction) > 0 else '-'})")
    elif kind == "alpha":
        if library is None:
            raise FundamentalsError("an alpha comes from the library; none was given")
        rec = library.get(str(spec.get("alpha_id")))
        signal = build_alpha_signal(ctx, X.loads(rec["expression"]), rec["settings"])
        n_tested = int(rec["metrics"].get("n_tested") or 1)
        info.update(label=rec["text"], n_trials=max(n_tested, 1), alphas=[{"alpha_id": rec["alpha_id"], "status": rec["status"], "n_tested": n_tested, "settings": rec["settings"],
                                                                         "text": rec["text"]}])
        if rec["status"] in ("Candidate", "Rejected", "Retired", "Decaying"):
            info["warnings"].append(f"{rec['text']}: its status is {rec['status']}, " + {"Candidate": "found by a search and not yet validated, so a good backtest here is what a search is expected to produce",
                                                                                   "Rejected": "it failed its tests", "Retired": "it has been withdrawn",
                                                                                   "Decaying": "its recent evidence is weaker than its past"}[rec["status"]])
    elif kind == "expression":
        tree = spec.get("expression")
        if not isinstance(tree, dict):
            raise FundamentalsError("an expression signal needs an 'expression' tree")
        signal = build_alpha_signal(ctx, tree, {"steps": spec.get("steps", [{"op": "winsorize", "pct": 1.0}, {"op": "zscore"}]), "direction": spec.get("direction", 1)})
        info["label"] = X.to_string(X.canonical(tree))
    else:
        members = spec.get("of")
        if not isinstance(members, list) or not 2 <= len(members) <= MAX_PARTS:
            raise FundamentalsError(f"'of' lists between 2 and {MAX_PARTS} signals to put together")
        panels, infos = {}, []
        for j, member in enumerate(members):
            panel, sub = resolve(ctx, member, library, depth + 1)
            panels[f"{j + 1}:{sub['label']}"] = panel
            infos.append(sub)
        horizon = spec.get("horizon", "1m")
        if kind == "combination":
            method = spec.get("method", "icir")
            signal = combiner.combine(ctx, panels, method, horizon, **_only(spec, ("window", "min_obs", "shrink", "orthogonalize", "cut"))).signal
            info["label"] = f"{method} combination of {len(panels)}"
        else:
            model = spec.get("model", "ridge")
            result = combiner.ml_combine(ctx, panels, model, horizon, permutation_repeats=0,
                                         **_only(spec, ("min_train", "retrain_every", "window", "embargo", "tune", "seed", "params")))
            signal = result.signal
            info["label"] = f"{model} model of {len(panels)}"
            info["ml"] = {"model": model, "folds": len(result.folds), "first_prediction": result.folds[0]["predict_from"], "notes": result.notes}
        _merge(info, infos)
    return signal.reindex(index=ctx.dates, columns=ctx.ids), info


def _only(spec: dict, keys: tuple) -> dict:
    return {k: spec[k] for k in keys if k in spec}


def _merge(info: dict, infos: list[dict]) -> None:
    for sub in infos:
        info["warnings"] += [w for w in sub["warnings"] if w not in info["warnings"]]
        info["alphas"] += sub["alphas"]
        info["n_trials"] = max(info["n_trials"], sub["n_trials"])


def _composite(ctx, parts, library, depth) -> tuple[pd.DataFrame, dict]:
    if not isinstance(parts, list) or not 1 <= len(parts) <= MAX_PARTS:
        raise FundamentalsError(f"a composite is made of between 1 and {MAX_PARTS} parts")
    panels, weights, infos = {}, [], []
    for j, part in enumerate(parts):
        panel, sub = resolve(ctx, part, library, depth + 1)
        weight = float(part.get("weight", 1.0)) if isinstance(part, dict) else 1.0
        if not np.isfinite(weight):
            raise FundamentalsError("a part's weight must be a number")
        panels[f"{j + 1}:{sub['label']}"] = panel
        weights.append(weight)
        infos.append(sub)
    if not any(weights):
        raise FundamentalsError("at least one part needs a weight other than zero")
    info: dict = {"kind": "composite", "warnings": [], "alphas": [], "n_trials": 1, "parts": [{"label": s["label"], "weight": w} for s, w in zip(infos, weights)]}
    _merge(info, infos)
    if len(panels) == 1:
        (key, panel), = panels.items()
        info["label"] = infos[0]["label"]
        return panel * (1.0 if weights[0] >= 0 else -1.0), info
    standard = combiner.standardise(panels, ctx)
    info["label"] = " + ".join(f"{w:g} x {s['label']}" for s, w in zip(infos, weights))
    return alpha_model.combine_factors(standard, pd.Series(weights, index=list(standard))).reindex(index=ctx.dates, columns=ctx.ids), info


# ------------------------------------------------------------------------------------------------------------------ the book
def eligibility(ctx: ResearchContext, spec: PortfolioSpec) -> pd.DataFrame:
    """Where a company may be bought on a date: investable in the context, and at least as large and as liquid as this portfolio asks (a missing size or volume fails a test that is asked)."""
    ok = ctx.investable().copy()
    if spec.min_market_cap_usd > 0:
        ok &= ctx.market_cap_usd() >= spec.min_market_cap_usd
    if spec.min_dollar_volume_usd > 0 or spec.max_adv_share:
        ok &= ctx.dollar_volume_usd() >= max(spec.min_dollar_volume_usd, 1e-9)
    return ok.fillna(False).astype(bool)


@dataclass
class Book:
    weights: pd.DataFrame            # decision dates by securities; zeros where nothing is held, a whole row of zeros is a date in cash
    diagnostics: pd.DataFrame        # one row per decision date
    notes: list = field(default_factory=list)
    limits: dict = field(default_factory=dict)


def _project(w0: np.ndarray, signs: np.ndarray, cap: np.ndarray, rows: list, limits: list, exposure: float, two: bool) -> tuple[np.ndarray, float, str]:
    """The nearest book to ``w0`` that respects every limit. A linear program finds the largest fraction ``t`` of the wanted exposure that fits (both legs together, so a dollar-neutral book stays
    neutral), then a quadratic program finds the weights nearest ``t * w0``. Returns ``(weights, t, how)``."""
    k = len(w0)
    lb, ub = np.where(signs > 0, 0.0, -cap), np.where(signs > 0, cap, 0.0)
    long_row, short_row = (signs > 0).astype(float), (signs < 0).astype(float)
    a_ub = np.array([np.r_[sign * v, 0.0] for v in rows for sign in (1.0, -1.0)]) if rows else None
    b_ub = np.array([lim for lim in limits for _ in (0, 1)]) if rows else None
    a_eq = [np.r_[long_row, -exposure]] + ([np.r_[short_row, exposure]] if two else [])
    lp = linprog(np.r_[np.zeros(k), -1.0], A_ub=a_ub, b_ub=b_ub, A_eq=np.array(a_eq), b_eq=np.zeros(len(a_eq)), bounds=list(zip(lb, ub)) + [(0.0, 1.0)], method="highs")
    if lp.status != 0:
        return np.zeros(k), 0.0, "no book respects the limits"
    t = float(lp.x[-1])
    if t <= 1e-9:
        return np.zeros(k), 0.0, "the limits leave no room"
    t_use = 1.0 if t >= 1.0 - 1e-9 else t * (1.0 - 1e-6)
    a = np.vstack([long_row] + ([short_row] if two else []) + rows)
    lo = np.r_[[t_use * exposure] + ([-t_use * exposure] if two else []), [-lim for lim in limits]]
    hi = np.r_[[t_use * exposure] + ([-t_use * exposure] if two else []), limits]
    for solver in ((solve_qp_ipm, solve_qp) if k <= 300 else (solve_qp, solve_qp_ipm)):          # the interior-point method is exact and quick up to a few hundred names; ADMM scales better beyond
        sol = solver(np.eye(k), -t_use * w0, a, lo, hi, lb, ub)
        if sol.ok and np.isfinite(sol.x).all():
            return np.clip(sol.x, lb, ub), t_use, "projected"
    return np.clip(lp.x[:k], lb, ub), t_use, "linear program"


def build_book(ctx: ResearchContext, signal: pd.DataFrame, spec: PortfolioSpec) -> Book:
    """Target weights for every rebalance date from the signal and the portfolio rules; each row uses only what was known on its date."""
    dates, ids = ctx.dates, ctx.ids
    n_ids = len(ids)
    s_all = signal.reindex(index=dates, columns=ids).to_numpy(float)
    elig = eligibility(ctx, spec).reindex(index=dates, columns=ids).fillna(False).to_numpy(bool)
    two = spec.side == "long_short"
    vol = ctx.volatility(63).reindex(index=dates, columns=ids).to_numpy(float) if spec.weighting == "inverse_vol" else None
    beta = ctx.beta(252).reindex(index=dates, columns=ids).to_numpy(float) if spec.max_beta is not None else None
    adv = ctx.dollar_volume_usd().reindex(index=dates, columns=ids).to_numpy(float) if spec.max_adv_share else None
    group_limits = []
    for kind, limit in (("sector", spec.max_sector), ("industry", spec.max_industry), ("country", spec.max_country)):
        if limit is not None:
            labels = ctx.group(kind).reindex(ids).astype(str).to_numpy()
            known = labels != "unknown"
            codes = np.full(n_ids, -1)
            codes[known] = pd.factorize(labels[known])[0]
            group_limits.append((kind, codes, float(limit), int(known.sum())))
    carried = ctx.adj.ffill().to_numpy(float)
    start = pd.Timestamp(spec.start) if spec.start else dates[0]
    end = pd.Timestamp(spec.end) if spec.end else dates[-1]
    weights = np.zeros((len(dates), n_ids))
    live = np.zeros(len(dates), bool)
    rows, notes = [], []
    previous: tuple[int, np.ndarray] | None = None
    cap_all = np.full(n_ids, spec.max_position)
    for i, day in enumerate(dates):
        if day < start or day > end:
            continue
        live[i] = True
        ok = elig[i] & np.isfinite(s_all[i])
        idx = np.flatnonzero(ok)
        m = len(idx)
        info = {"date": day, "eligible": m, "reason": "", "projected": False, "fraction": 1.0}
        k = min(spec.n if spec.selection == "top_n" else int(np.ceil(m * spec.percentile / 100.0)), m // 2 if two else m)
        row = np.zeros(n_ids)
        if k < max(spec.min_leg, 1):
            info["reason"] = f"only {m} eligible companies with a signal, {'two legs need' if two else 'a leg needs'} at least {spec.min_leg} each: held in cash"
        else:
            order = idx[np.argsort(-s_all[i][idx], kind="stable")]
            longs, shorts = order[:k], (order[::-1][:k] if two else order[:0])
            cut_l = order[k] if m > k else -1
            cut_s = order[m - k - 1] if two and m > k else -1
            z = np.full(n_ids, np.nan)
            if spec.weighting == "signal":
                sd = float(np.std(s_all[i][idx], ddof=1)) if m > 1 else 0.0
                z[idx] = (s_all[i][idx] - s_all[i][idx].mean()) / sd if sd > 0 else 0.0

            def leg(names: np.ndarray, cut: int, direction: float) -> np.ndarray:
                if spec.weighting == "equal":
                    raw = np.ones(len(names))
                elif spec.weighting == "signal":
                    strength = direction * z[names]
                    ref = direction * z[cut] if cut >= 0 else float(strength.min()) - max(float(strength.std()), 1e-6) * 0.25
                    raw = np.maximum(strength - ref, 1e-9)
                else:
                    v = vol[i][names]
                    typical = np.nanmedian(v) if np.isfinite(v).any() else np.nan
                    v = np.where(np.isfinite(v) & (v > 0), v, typical)
                    raw = np.ones(len(names)) if not np.isfinite(typical) else 1.0 / np.maximum(v, np.nanpercentile(v, 5))
                return raw / raw.sum()

            sel = np.r_[longs, shorts].astype(int)
            signs = np.r_[np.ones(len(longs)), -np.ones(len(shorts))]
            w0 = spec.leg_exposure * np.r_[leg(longs, cut_l, 1.0), -leg(shorts, cut_s, -1.0) if two else np.zeros(0)]
            cap = np.full(len(sel), spec.max_position)
            if adv is not None:
                cap = np.minimum(cap, spec.max_adv_share * np.where(np.isfinite(adv[i][sel]), adv[i][sel], 0.0) / spec.capital_usd)
            rows_, limits_, hits = [], [], []
            for kind, codes, limit, n_known in group_limits:
                code = codes[sel]
                for g in np.unique(code[code >= 0]):
                    rows_.append((code == g).astype(float))
                    limits_.append(limit)
                    hits.append(kind)
            if beta is not None:
                b = beta[i][sel]
                fallback = np.nanmedian(beta[i][ok]) if np.isfinite(beta[i][ok]).any() else 1.0
                rows_.append(np.where(np.isfinite(b), b, fallback))
                limits_.append(float(spec.max_beta))
                hits.append("beta")
            tight = (np.abs(w0) > cap + TOL).any() or any(abs(float(r @ w0)) > lim + TOL for r, lim in zip(rows_, limits_))
            if tight:
                w, t, how = _project(w0, signs, cap, rows_, limits_, spec.leg_exposure, two)
                info.update(projected=True, fraction=t, how=how)
                if t < 1.0 - 1e-6:
                    info["reason"] = "the limits could not all be met at full exposure: " + (f"{t:.0%} of it invested, the rest in cash" if t > 0 else "held in cash")
            else:
                w = w0
            w = np.where(np.abs(w) < 1e-7, 0.0, w)
            row[sel] = w
            info["at_position_cap"] = int((np.abs(w) >= cap - 1e-6).sum())
            info["at_group_limit"] = int(sum(1 for r, lim, h in zip(rows_, limits_, hits) if h != "beta" and abs(float(r @ w)) >= lim - 1e-6))
            info["beta"] = float(rows_[-1] @ w) if beta is not None else None
        # a turnover limit: trade only part of the way from the book as it has drifted to the book wanted
        if spec.max_turnover is not None:
            h = _drifted(previous, i, ctx, carried, n_ids)
            gross = max(float(np.abs(row).sum()), float(np.abs(h).sum()))
            change = row - h
            move = max(float(np.clip(change, 0.0, None).sum()), float(np.clip(-change, 0.0, None).sum())) / gross if gross > 0 else 0.0
            info["wanted_turnover"] = move
            if move > spec.max_turnover:
                row = h + spec.max_turnover / move * (row - h)
                info["limited_turnover"] = True
            row = _enforce(row, spec, cap_all, adv[i] if adv is not None else None, group_limits)
        weights[i] = row
        previous = (i, row)
        info.update(long_names=int((row > 1e-9).sum()), short_names=int((row < -1e-9).sum()), gross=float(np.abs(row).sum()), net=float(row.sum()))
        rows.append(info)
    if not live.any():
        raise FundamentalsError("no rebalance date falls inside the dates asked for")
    frame = pd.DataFrame(weights[live], index=dates[live], columns=ids)
    diag = pd.DataFrame(rows).set_index("date")
    if group_limits:
        unlabeled = [f"{kind}: {n_ids - n_known} companies have no label and are not limited" for kind, _, _, n_known in group_limits if n_known < n_ids]
        notes += unlabeled
    return Book(frame, diag, notes, {"groups": [g[0] for g in group_limits]})


def _drifted(previous, i: int, ctx: ResearchContext, carried: np.ndarray, n_ids: int) -> np.ndarray:
    """The last book as prices have moved it up to the close of date ``i`` (what is known when the next one is decided), scaled to the gross exposure it had."""
    if previous is None:
        return np.zeros(n_ids)
    j, book = previous
    a, b = min(int(ctx.pos[j]) + ctx.entry_lag, int(ctx.pos[i])), int(ctx.pos[i])
    with np.errstate(divide="ignore", invalid="ignore"):
        growth = np.where(np.isfinite(carried[a]) & (carried[a] > 0) & np.isfinite(carried[b]), carried[b] / carried[a], 1.0)
    out = book * growth
    gross, total = float(np.abs(book).sum()), float(np.abs(out).sum())
    return out * (gross / total) if total > 1e-12 else out


def _enforce(row: np.ndarray, spec: PortfolioSpec, cap_all: np.ndarray, adv_row, group_limits: list) -> np.ndarray:
    """The hard limits again, after a partial move: positions are cut to their caps and a group over its limit is cut back on the side that is over."""
    cap = cap_all.copy()
    if adv_row is not None:
        cap = np.minimum(cap, spec.max_adv_share * np.where(np.isfinite(adv_row), adv_row, 0.0) / spec.capital_usd)
    row = np.clip(row, -cap, cap)
    for _, codes, limit, _ in group_limits:
        known = codes >= 0
        net = np.bincount(codes[known], weights=row[known], minlength=int(codes.max()) + 1) if known.any() else np.zeros(0)
        for g in np.flatnonzero(np.abs(net) > limit + TOL):
            side = np.sign(net[g])
            members = (codes == g) & (row * side > 0)
            row[members] *= max(1.0 - (abs(net[g]) - limit) / float(np.abs(row[members]).sum()), 0.0)
    return row


# ------------------------------------------------------------------------------------------------------------------ the run
def engine_returns(ctx: ResearchContext) -> pd.DataFrame:
    """Daily returns for the engine: a company that stops printing keeps its last price (zero return) so a held position is neither lost nor invented, and the first day back carries the whole move."""
    return ctx.adj.ffill().pct_change(fill_method=None)


def _stats(net: pd.Series, bench: pd.Series | None = None, turnover: pd.Series | None = None, costs: pd.Series | None = None) -> dict:
    return clean(performance_summary(net, 0.0, turnover, costs, bench)) if len(net.dropna()) > 5 else {"n_obs": int(len(net.dropna()))}


def _equal_weight(ctx: ResearchContext, eligible: pd.DataFrame, returns: np.ndarray, index: pd.DatetimeIndex) -> pd.Series:
    """Holding every eligible company equally, the eligible set refreshed at each rebalance date and entered with the same lag: what owning the universe would have returned."""
    out = np.full(len(index), np.nan)
    pos, lag = ctx.pos, ctx.entry_lag
    mask = eligible.reindex(index=ctx.dates, columns=ctx.ids).fillna(False).to_numpy(bool)
    for p in range(len(pos)):
        first = pos[p] + lag + 1
        last = (pos[p + 1] + lag) if p + 1 < len(pos) else len(index) - 1
        if first > last or not mask[p].any():
            continue
        block = returns[first:last + 1][:, mask[p]]
        seen = np.isfinite(block)
        count = seen.sum(axis=1)
        out[first:last + 1] = np.where(count > 0, np.where(seen, block, 0.0).sum(axis=1) / np.maximum(count, 1), np.nan)
    return pd.Series(out, index=index).dropna()


def backtest(ctx: ResearchContext, signal: pd.DataFrame, spec: PortfolioSpec | dict | None = None, name: str = "alpha", info: dict | None = None, blocks: dict | None = None,
             n_trials: int | None = None) -> dict:
    """Run a signal through the backtest engine and report. ``blocks`` is ``{label: (start, end)}`` to split the result by (the dates an alpha was found, validated and locked on);
    ``n_trials`` is how many ideas were tried to find it (default: what the signal's description says, else one)."""
    spec = spec if isinstance(spec, PortfolioSpec) else PortfolioSpec.from_dict(spec)
    info = info or {"label": name, "warnings": [], "alphas": [], "n_trials": 1}
    book = build_book(ctx, signal, spec)
    held_cols = book.weights.columns[(book.weights.abs() > 0).any(axis=0)]
    if len(held_cols) == 0:
        reasons = book.diagnostics["reason"].replace("", np.nan).dropna()
        raise FundamentalsError("the portfolio never held anything" + (f": {reasons.iloc[0]}" if len(reasons) else ""))
    returns_all = engine_returns(ctx)
    daily = ctx.adj.index
    targets = pd.DataFrame(np.nan, index=daily, columns=held_cols)
    targets.loc[book.weights.index, held_cols] = book.weights[held_cols].to_numpy()
    engine = BacktestEngine(signal_lag=ctx.entry_lag, rebalance="daily", weight_drift=True, annualisation=252, min_assets=1, cost_model=LinearCostModel(cost_bps=spec.cost_bps))
    result = engine.run(targets, returns_all[held_cols], name)
    exposure = result.weights.abs().sum(axis=1)
    first = exposure.index[exposure.to_numpy() > 1e-9][0]
    keep = result.net_returns.index[result.net_returns.index.searchsorted(first, side="right"):]
    short = (-result.weights.clip(upper=0.0)).sum(axis=1).shift(1).reindex(keep).fillna(0.0)
    borrow = short * (spec.borrow_bps / 1e4 / 252.0)
    gross = result.gross_returns.reindex(keep)
    trades = result.trades.reindex(keep).fillna(0.0)
    trades.iloc[0] = trades.iloc[0] + result.trades.loc[first].fillna(0.0)            # the engine drops the day the first book is bought; its cost is charged on the first day of returns
    costs = result.costs.reindex(keep).fillna(0.0) + borrow
    costs.iloc[0] += float(result.costs.get(first, 0.0))
    net = gross - costs
    turnover = result.turnover.reindex(keep).fillna(0.0)
    turnover.iloc[0] += float(result.turnover.get(first, 0.0))
    eligible = eligibility(ctx, spec)
    benchmark = _equal_weight(ctx, eligible, returns_all.to_numpy(float), daily).reindex(keep).dropna()
    summary = _stats(net, benchmark, turnover, costs)
    sensitivity = cost_sensitivity(gross - borrow, trades, (0.0, 5.0, 10.0, 25.0, 50.0, 100.0))
    exposure_k = result.weights.reindex(keep)
    n_trials = int(n_trials if n_trials is not None else info.get("n_trials", 1))
    dsr = None
    if len(net) > 30 and np.isfinite(summary.get("sharpe", np.nan)):
        dsr = clean({"probability": deflated_sharpe_ratio(float(summary["sharpe"]), max(n_trials, 1), len(net), float(summary.get("skew") or 0.0), float(summary.get("excess_kurtosis") or 0.0) + 3.0),
                     "n_trials": n_trials, "sharpe": summary["sharpe"]})
    splits = {}
    for label, (lo, hi) in (blocks or {}).items():
        part = net.loc[pd.Timestamp(lo):pd.Timestamp(hi)]
        splits[label] = {"start": str(lo), "end": str(hi), **_stats(part, benchmark.reindex(part.index))}
    last_book = book.weights.iloc[-1]
    held_now = last_book[last_book.abs() > 1e-9].sort_values(key=np.abs, ascending=False)
    labels = {kind: ctx.group(kind) for kind in ("sector", "country")}
    diag = book.diagnostics
    equity = (1.0 + net).cumprod()
    limits = {"projected_dates": int(diag["projected"].sum()), "dates_not_fully_invested": int((diag["fraction"] < 1.0 - 1e-6).sum()), "dates_in_cash": int((diag["gross"] < 1e-9).sum()),
              "dates": int(len(diag)), "turnover_limited_dates": int(diag["limited_turnover"].astype(float).fillna(0.0).sum()) if "limited_turnover" in diag else 0,
              "max_position_held": float(book.weights.abs().to_numpy().max()), "position_cap": spec.max_position,
              "average_names_long": float(diag["long_names"].mean()), "average_names_short": float(diag["short_names"].mean()),
              "max_net_beta": None if "beta" not in diag or diag["beta"].isna().all() else float(diag["beta"].abs().max()),
              "average_wanted_turnover": None if "wanted_turnover" not in diag else float(diag["wanted_turnover"].mean())}
    warnings = list(dict.fromkeys(list(info.get("warnings", [])) + list(book.notes)))
    if limits["dates_in_cash"]:
        warnings.append(f"{limits['dates_in_cash']} of {limits['dates']} rebalance dates were held in cash (too few eligible companies, or the limits left no room)")
    surv = ctx.survivorship()
    warnings.append("survivorship: " + surv["verdict"])
    if summary.get("n_obs", 0) < 3 * 252:
        warnings.append(f"only {summary.get('n_obs', 0)} trading days: too short to say much about a Sharpe ratio")
    gross_exposure = exposure_k.abs().sum(axis=1)
    out = {
        "name": name, "signal": {k: info.get(k) for k in ("label", "kind", "parts", "alphas", "ml")}, "portfolio": spec.describe(), "universe": ctx.describe(),
        "summary": summary, "gross_summary": _stats(gross), "benchmark": _stats(benchmark.reindex(keep).dropna()), "blocks": splits, "deflated_sharpe": dsr,
        "costs": {"trading_bps_a_year": float((costs - borrow).mean() * 252 * 1e4), "borrow_bps_a_year": float(borrow.mean() * 252 * 1e4),
                  "annual_turnover": float(turnover.mean() * 252), "breakeven_bps": clean(float((gross - borrow).mean() / max(float(turnover.mean()), 1e-12) * 1e4)),
                  "sensitivity": [{"cost_bps": float(b), **clean(r)} for b, r in sensitivity.iterrows()]},
        "exposure": {"gross_average": float(gross_exposure.mean()), "net_average": float(exposure_k.sum(axis=1).mean()), "cash_average": float((1.0 - exposure_k.clip(lower=0.0).sum(axis=1)).mean())},
        "limits": clean(limits), "series": {"equity": series_points(equity), "gross_equity": series_points((1.0 + gross).cumprod()), "benchmark": series_points((1.0 + benchmark).cumprod()),
                                          "drawdown": series_points(equity / equity.cummax() - 1.0), "exposure": series_points(gross_exposure), "turnover": series_points(turnover[turnover > 0])},
        "holdings": [{"security_id": int(s), "ticker": str(ctx.info.loc[s, "ticker"]), "name": str(ctx.info.loc[s, "company_name"] or ""), "weight": float(w), "sector": labels["sector"].get(s),
                      "country": labels["country"].get(s)} for s, w in held_now.head(40).items()],
        "as_of": str(book.weights.index[-1].date()), "first_trade": str(first.date()), "warnings": warnings,
        "reading": ("Net of costs this book returned %s a year at a Sharpe ratio of %s; holding every eligible company equally returned %s. This shows what trading the signal would have earned on this data, "
                    "not that it will keep doing so." % (_pct(summary.get("cagr")), _num(summary.get("sharpe")), _pct(_stats(benchmark.reindex(keep).dropna()).get("cagr")))),
    }
    out["experiment_id"] = _record(ctx, spec, info, name, out)
    return out


def _pct(x) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:+.1f}%"


def _num(x) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.2f}"


def _record(ctx: ResearchContext, spec: PortfolioSpec, info: dict, name: str, result: dict) -> str:
    """Keep what a backtest was and what it found (not its series) so it can be shown, compared and run again; the id is a hash of what defines it, the data it ran on included."""
    counts = ctx.store.counts()
    version = f"{counts['companies']}c/{counts['observations']}o/{counts['price_rows']}p"
    body = json.dumps(clean({"signal": {k: info.get(k) for k in ("label", "kind", "parts", "alphas")}, "portfolio": spec.describe(), "universe": ctx.spec.describe(), "freq": ctx.freq,
                             "entry_lag": ctx.entry_lag, "data": version}), sort_keys=True)
    key = "bt-" + hashlib.sha1(body.encode()).hexdigest()[:12]
    summary = {k: result["summary"].get(k) for k in ("cagr", "ann_vol", "sharpe", "max_drawdown", "n_obs", "ann_turnover")}
    summary["deflated_sharpe"] = (result["deflated_sharpe"] or {}).get("probability")
    with ctx.store.db.transaction() as conn:
        conn.execute("INSERT OR REPLACE INTO experiments (experiment_id, kind, created_at, spec, data_version, code_version, summary, status) VALUES (?,?,?,?,?,?,?,?)",
                     (key, "backtest", now_iso(), body, f"{version}@{now_iso()[:10]}", "fundamentals-1", json.dumps(clean(summary)), "finished"))
    return key


def alpha_blocks(library: AlphaLibrary, alpha_id: str) -> dict | None:
    """The train, validation and locked dates an alpha was found on (stored with it), as backtest blocks; ``None`` when it has none."""
    blocks = library.get(alpha_id)["settings"].get("blocks")
    return {label: tuple(dates) for label, dates in blocks.items()} if blocks else None


def tear(ctx: ResearchContext, signal_spec, portfolio: dict | None = None, library: AlphaLibrary | None = None, name: str | None = None, n_trials: int | None = None) -> dict:
    """The whole path in one call: describe the signal, build it, trade it, report. For a single library alpha the report is split by the dates it was found on."""
    signal, info = resolve(ctx, signal_spec, library)
    alphas = info.get("alphas", [])
    single = info.get("kind") == "alpha" or (info.get("kind") == "composite" and len(info.get("parts", [])) == 1)
    blocks = alpha_blocks(library, alphas[0]["alpha_id"]) if library is not None and single and len(alphas) == 1 else None
    return backtest(ctx, signal, portfolio, name or info["label"], info, blocks, n_trials)


__all__ = ["PortfolioSpec", "Book", "SIDES", "SELECTIONS", "WEIGHTINGS", "resolve", "technical_names", "eligibility", "build_book", "engine_returns", "backtest", "alpha_blocks", "tear"]
