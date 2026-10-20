"""Automated alpha discovery: generate formulas, throw out the ones that are not worth more of the computer's time as early and as cheaply as possible, keep honest count of everything tried, and test
the survivors on dates that were never used to choose them.

The pipeline, in the order of cost:

    stage 0  sanity       the formula computes, has enough companies and enough different values, and has not been tried before under the same conditions
    stage 1  fast IC      the rank correlation with next month's return, on the *training* dates, with a Newey-West t-statistic
    stage 2  stability    the IC is positive on most dates, in both halves of the training dates, and survives the removal of its best year
    stage 3  quantiles    the buckets rise in order
    stage 4  long-short   the top-minus-bottom portfolio earns, gross
    stage 5  costs        it still earns after trading costs and borrowing, at a turnover that can be traded
    stage 6  validation   the same sign, on a second block of dates that was not used for stages 1-5
    stage 7  locked test  the finalists (chosen from training and validation only) are tested once on the last block, the *out-of-sample* dates, which nothing before has looked at

The dates are cut into three consecutive blocks with a gap (an embargo) between them as wide as the forecast horizon, so that a return that overlaps two blocks is in neither. The sign of a formula
(whether high or low should earn) is learned on the training block, which makes the test two-sided, and is held fixed afterwards.

**Multiple testing.** Every formula that reaches stage 1 is a test, and the count is kept. The training p-values of all of them are put through the Benjamini-Hochberg procedure to give each finalist
a false-discovery q-value; its training-and-validation long-short Sharpe ratio is deflated for the number of formulas tried (Bailey and Lopez de Prado); and, when there are enough survivors, the
probability of backtest overfitting is estimated by combinatorially symmetric cross-validation. Nothing here is ever called validated: the best status a run can give is ``Candidate``.

**Resumable.** Every candidate is written to the database as soon as it is judged, and the formula stream is a function of the seed and the position, so a stopped or crashed run continues exactly
where it was. Rejected formulas go to the graveyard (never deleted); only the metadata of survivors goes to the library.

**Complexity.** The fitness that ranks survivors subtracts a penalty for the number of nodes in the formula; simpler is preferred when evidence is equal.
"""

from __future__ import annotations

import json
import math
import uuid
from collections import OrderedDict
from collections.abc import MutableMapping
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from ..backtest.metrics import deflated_sharpe_ratio, probabilistic_sharpe_ratio
from ..models.regression import cross_sectional_ic
from ..validation.multiple_testing import pbo_cscv
from . import expr as X
from . import transforms
from .errors import FundamentalsError
from .evaluate import PERIODS_PER_YEAR, EvalSpec, ic_stats, long_short, monotonicity, newey_west, overlap_lags, quantile_returns, rank_ic
from .jsonutil import clean
from .library import AlphaLibrary, build_signal, settings_hash
from .research import FREQ_DAYS, ResearchContext, UniverseSpec, horizon_days
from .schema import now_iso
from .store import FundamentalsStore

STAGES = ("sanity", "fast_ic", "stability", "quantiles", "long_short", "costs", "validation", "locked_test")
YEAR = {"W": 52, "M": 12, "Q": 4}
HALF = {"W": 26, "M": 6, "Q": 2}


@dataclass
class Thresholds:
    min_coverage: float = 0.40          # stage 0: share of investable companies with a value, typical date
    min_dates: int = 36                 # stage 1: dates with an IC
    t_ic: float = 1.65                  # stage 1: |t| of the mean IC on the training block
    min_abs_ic: float = 0.005
    positive_share: float = 0.55        # stage 2
    min_icir: float = 0.10
    best_year_t: float = 1.0
    monotonicity: float = 0.5           # stage 3
    ls_t: float = 1.5                   # stage 4
    ls_sharpe: float = 0.3
    net_sharpe: float = 0.15            # stage 5
    max_turnover: float = 0.7
    val_t: float = 1.0                  # stage 6
    oos_t: float = 1.0                  # stage 7
    max_q: float = 0.25                 # to be a finalist: the false-discovery q-value over everything tried in the run
    max_correlation: float = 0.9        # to a formula already accepted in the run: the same idea


@dataclass
class DiscoverySpec:
    universe: dict = field(default_factory=dict)
    freq: str = "M"
    horizon: str = "1m"
    entry_lag: int = 1
    n_candidates: int = 300
    seed: int = 7
    train_fraction: float = 0.55
    validation_fraction: float = 0.20
    steps: list = field(default_factory=lambda: [{"op": "winsorize", "pct": 1.0}, {"op": "zscore"}])
    max_nodes: int = 15
    cost_bps: float = 10.0
    borrow_bps: float = 50.0
    quantiles: int = 5
    weighting: str = "equal"
    min_assets: int = 20
    thresholds: dict = field(default_factory=dict)
    seeds: list | None = None            # formulas (text) tried first; None: the built-in classics, [] : none
    evolve: float = 0.4                  # of the candidates after the first half, the share bred from survivors instead of drawn at random
    finalists: int = 15
    only_fields: list | None = None
    fitness_weights: dict = field(default_factory=dict)

    def describe(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict | None) -> "DiscoverySpec":
        d = dict(d or {})
        return DiscoverySpec(**{k: v for k, v in d.items() if k in DiscoverySpec.__dataclass_fields__})

    def limits(self) -> Thresholds:
        return Thresholds(**{k: v for k, v in self.thresholds.items() if k in Thresholds.__dataclass_fields__})

    def conditions(self) -> dict:
        """The conditions a result depends on: what makes two results about the same question."""
        return {"universe": self.universe, "freq": self.freq, "horizon": self.horizon, "entry_lag": self.entry_lag, "steps": self.steps, "quantiles": self.quantiles, "cost_bps": self.cost_bps,
                "borrow_bps": self.borrow_bps, "weighting": self.weighting, "min_assets": self.min_assets, "train_fraction": self.train_fraction, "validation_fraction": self.validation_fraction}


FITNESS = {"ic": 0.30, "stability": 0.20, "sharpe": 0.25, "monotonic": 0.10, "decorrelation": 0.15, "turnover": -0.15, "complexity": -0.10}


class BoundedCache(MutableMapping):
    """A dictionary of evaluated sub-formulas that forgets the least recently used ones once it holds more than ``max_bytes``, so a search of thousands of formulas does not fill the memory.

    (A plain mapping around an ordered dictionary rather than a subclass of it: the standard ordered dictionary calls an overridden ``__getitem__`` from inside ``popitem`` on some versions of Python.)
    """

    def __init__(self, max_bytes: int = 600_000_000):
        self._items: OrderedDict = OrderedDict()
        self.max_bytes, self.bytes = int(max_bytes), 0

    @staticmethod
    def _size(value) -> int:
        panel = value[0] if isinstance(value, tuple) else value
        return int(getattr(panel, "size", 0)) * 8                                    # a panel of floats

    def __getitem__(self, key):
        value = self._items[key]
        self._items.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        if key in self._items:
            self.bytes -= self._size(self._items[key])
        self._items[key] = value
        self._items.move_to_end(key)
        self.bytes += self._size(value)
        while self.bytes > self.max_bytes and len(self._items) > 1:
            _, old = self._items.popitem(last=False)
            self.bytes -= self._size(old)

    def __delitem__(self, key):
        self.bytes -= self._size(self._items.pop(key))

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)

    def __contains__(self, key):
        return key in self._items


# ------------------------------------------------------------------------------------------------------------------ helpers
def default_seeds(ctx: ResearchContext) -> list[str]:
    """The classic formulas of the literature, in the expression language, kept when the data can support them."""
    y, h = YEAR[ctx.freq], HALF[ctx.freq]
    texts = ["div(fcf,market_cap)", "div(net_income,market_cap)", "div(revenue,market_cap)", "div(shareholders_equity,market_cap)", "div(operating_income,enterprise_value)", "div(gross_profit,total_assets)",
             f"neg(growth(total_assets,{y}))", f"delta(roic,{y})", f"delta(operating_margin,{y})", f"delta(gross_margin,{y})", f"accel(revenue,{y})", f"growth(revenue,{y})", "roic",
             "neg(accruals_ratio)", "neg(debt_to_equity)", "neg(share_growth)", "sector_rank(div(operating_income,total_assets))", f"sub(operating_margin,rolling_mean(operating_margin,{3 * y}))",
             f"delta(asset_turnover,{h})", f"neg(rolling_std(operating_margin,{2 * y}))"]
    kinds = {k: v.kind for k, v in X.catalogue(ctx).items()}
    return [t for t in texts if X.well_typed(X.parse(t), kinds)[0]]


def splits(ctx: ResearchContext, spec: DiscoverySpec) -> dict:
    """Three consecutive blocks of rebalance dates with an embargo between them: ``{'train': (a, b), 'validation': ..., 'oos': ..., 'embargo': n}``, as date strings."""
    h, step = horizon_days(spec.horizon), FREQ_DAYS[ctx.freq]
    usable = ctx.dates[ctx.forward(spec.horizon).notna().sum(axis=1).to_numpy() >= spec.min_assets]
    n, embargo = len(usable), overlap_lags(h, step) + 1
    a, b = int(n * spec.train_fraction), int(n * (spec.train_fraction + spec.validation_fraction))
    if min(a, b - a - embargo, n - b - embargo) < 12:
        raise FundamentalsError(f"too little history to cut into training, validation and locked test blocks ({n} usable dates): widen the sample or shorten the horizon")
    train, val, oos = usable[:a], usable[a + embargo:b], usable[b + embargo:]
    f = lambda idx: (idx[0].strftime("%Y-%m-%d"), idx[-1].strftime("%Y-%m-%d"))     # noqa: E731
    return {"train": f(train), "validation": f(val), "oos": f(oos), "embargo": embargo, "dates": {"train": len(train), "validation": len(val), "oos": len(oos)}}


def _block(series, window: tuple[str, str]):
    return series.loc[window[0]:window[1]]


def bh_qvalues(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values): the smallest false-discovery rate at which each result would be kept."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(adjusted, 0, 1)
    return out


def signal_correlation(a: pd.DataFrame, b: pd.DataFrame) -> float:
    """The average over dates of the rank correlation between two signals across companies."""
    ic = cross_sectional_ic(a, b, "spearman", 10)
    return float(ic.mean()) if len(ic) else float("nan")


def _sharpe(r: pd.Series, ppy: int) -> float:
    r = r.dropna()
    if len(r) < 6 or r.std(ddof=1) <= 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * math.sqrt(ppy))


def _t(r: pd.Series) -> float:
    r = r.dropna()
    return float(r.mean() / (r.std(ddof=1) / math.sqrt(len(r)))) if len(r) > 5 and r.std(ddof=1) > 0 else float("nan")


# ------------------------------------------------------------------------------------------------------------------ one formula through the stages
class Screen:
    """Runs one formula through stages 0-6 (and, for a finalist, 7) against a research context and a set of conditions."""

    def __init__(self, ctx: ResearchContext, spec: DiscoverySpec, blocks: dict, cache: dict):
        self.ctx, self.spec, self.blocks, self.cache = ctx, spec, blocks, cache
        self.limits = spec.limits()
        self.h, self.step, self.ppy = horizon_days(spec.horizon), FREQ_DAYS[ctx.freq], PERIODS_PER_YEAR[ctx.freq]
        self.forward = ctx.forward(spec.horizon)
        self.eval_spec = EvalSpec(horizon=spec.horizon, quantiles=spec.quantiles, weighting=spec.weighting, cost_bps=spec.cost_bps, borrow_bps=spec.borrow_bps, min_assets=spec.min_assets)

    def signal(self, expression: dict, direction: float = 1.0) -> pd.DataFrame:
        raw = X.evaluate_panel(expression, self.ctx) if self.cache is None else X.evaluate(expression, self.ctx, self.cache)[0]
        out = transforms.apply(raw, self.spec.steps, self.ctx)
        return out * direction

    def run(self, expression: dict) -> dict:
        """``{'stage': last stage reached (0-6), 'status': survivor|rejected, 'reason', 'metrics', 'direction'}``."""
        L, ctx = self.limits, self.ctx
        m: dict = {"nodes": X.size(expression)}
        try:
            sig = self.signal(expression)
        except (FundamentalsError, ValueError, KeyError, ZeroDivisionError) as error:
            return self._reject(0, f"cannot be computed: {error}", m)
        mask = ctx.investable().reindex_like(sig).fillna(False)
        train = self.blocks["train"]
        have = sig.where(mask).notna().sum(axis=1)
        cover = (have / mask.sum(axis=1).replace(0, np.nan)).loc[train[0]:train[1]]
        m["coverage"] = float(cover.median()) if cover.notna().any() else 0.0
        if m["coverage"] < L.min_coverage:
            return self._reject(0, f"only {m['coverage']:.0%} of the investable companies have a value (needs {L.min_coverage:.0%})", m)
        distinct = sig.loc[train[0]:train[1]].nunique(axis=1)
        if not (distinct > 5).any() or float(distinct[distinct > 0].median()) <= 5:
            return self._reject(0, "takes five or fewer different values on a typical date", m)
        # stage 1: the information coefficient on the training dates
        ic_all = rank_ic(sig, self.forward.reindex(sig.index), self.spec.min_assets)
        ic = _block(ic_all, train)
        if len(ic) < L.min_dates:
            return self._reject(1, f"only {len(ic)} training dates have an IC (needs {L.min_dates})", m)
        st = ic_stats(ic, self.h, self.step, self.ppy)
        m.update(ic=st["mean_ic"], icir=st["icir"], t_train=st["t_stat"], p_train=st["p_value"], n_ic=st["n_dates"], positive_share_raw=st["positive_share"])
        direction = 1.0 if st["mean_ic"] >= 0 else -1.0
        m["direction"] = direction
        if not np.isfinite(st["t_stat"]) or abs(st["t_stat"]) < L.t_ic or abs(st["mean_ic"]) < L.min_abs_ic:
            return self._reject(1, f"IC {st['mean_ic']:+.3f} (t = {st['t_stat']:.1f}) is too weak on the training dates", m, direction)
        # stage 2: stability
        icd = ic * direction
        pos = float((icd > 0).mean())
        halves = (float(icd.iloc[:len(icd) // 2].mean()), float(icd.iloc[len(icd) // 2:].mean()))
        years = icd.groupby(icd.index.year).mean()
        rest = icd[icd.index.year != years.idxmax()]
        rest_t = newey_west(rest, overlap_lags(self.h, self.step) + 1)["t"] if len(rest) > 12 else float("nan")
        m.update(positive_share=pos, half_1=halves[0], half_2=halves[1], t_without_best_year=rest_t)
        if pos < L.positive_share:
            return self._reject(2, f"the IC is positive on only {pos:.0%} of the training dates (needs {L.positive_share:.0%})", m, direction)
        if st["icir"] * direction < L.min_icir:
            return self._reject(2, f"the information ratio of the IC ({st['icir'] * direction:.2f}) is below {L.min_icir}", m, direction)
        if halves[0] <= 0 or halves[1] <= 0:
            return self._reject(2, "the IC has the wrong sign in one half of the training dates", m, direction)
        if not np.isfinite(rest_t) or rest_t < L.best_year_t:
            return self._reject(2, f"without its best year the t-statistic falls to {rest_t:.1f}", m, direction)
        # stage 3: quantiles
        oriented = sig * direction
        q = quantile_returns(oriented.loc[train[0]:train[1]], self.forward.loc[train[0]:train[1]], self.spec.quantiles)
        mono = monotonicity(q.mean())
        m.update(monotonicity=mono["spearman"], rising_pairs=mono["rising_pairs"], spread=float(q.mean().iloc[-1] - q.mean().iloc[0]))
        if not np.isfinite(mono["spearman"]) or mono["spearman"] < L.monotonicity or m["spread"] <= 0:
            return self._reject(3, f"the buckets do not rise in order (rank correlation {mono['spearman']:.2f})", m, direction)
        # stages 4-5: the portfolio
        ls = long_short(oriented, ctx, self.eval_spec)
        gross_t, net_t = ls["gross"].loc[train[0]:train[1]], ls["net"].loc[train[0]:train[1]]
        m.update(ls_t=_t(gross_t), sharpe_gross=_sharpe(gross_t, self.ppy), sharpe=_sharpe(net_t, self.ppy), turnover=float(ls["turnover"].loc[train[0]:train[1]].mean()))
        if not np.isfinite(m["ls_t"]) or m["ls_t"] < L.ls_t or m["sharpe_gross"] < L.ls_sharpe:
            return self._reject(4, f"the long-short portfolio earns too little gross (Sharpe {m['sharpe_gross']:.2f}, t = {m['ls_t']:.1f})", m, direction)
        if m["sharpe"] < L.net_sharpe:
            return self._reject(5, f"after costs the Sharpe falls to {m['sharpe']:.2f} (needs {L.net_sharpe})", m, direction)
        if m["turnover"] > L.max_turnover:
            return self._reject(5, f"turnover of {m['turnover']:.0%} a period is too high to trade", m, direction)
        # stage 6: the validation dates
        val = self.blocks["validation"]
        icv = _block(ic_all, val) * direction
        v_nw = newey_west(icv, overlap_lags(self.h, self.step) + 1) if len(icv) > 6 else {"t": float("nan"), "mean": float("nan")}
        net_v = ls["net"].loc[val[0]:val[1]]
        m.update(val_ic=float(icv.mean()) if len(icv) else float("nan"), val_t=v_nw["t"], val_sharpe=_sharpe(net_v, self.ppy), val_positive_share=float((icv > 0).mean()) if len(icv) else float("nan"))
        if not np.isfinite(v_nw["t"]) or v_nw["t"] < L.val_t or not m["val_ic"] > 0:
            return self._reject(6, f"on the validation dates the IC is {m['val_ic']:+.3f} in the chosen direction (t = {v_nw['t']:.1f})", m, direction)
        if not m["val_sharpe"] > 0:
            return self._reject(6, f"on the validation dates the long-short portfolio loses money after costs (Sharpe {m['val_sharpe']:.2f})", m, direction)
        return {"stage": 6, "status": "survivor", "reason": "passed stages 1-6", "metrics": m, "direction": direction}

    def locked_test(self, expression: dict, direction: float) -> dict:
        """The one look at the out-of-sample dates."""
        L, oos = self.limits, self.blocks["oos"]
        sig = self.signal(expression, direction)
        ic_all = rank_ic(sig, self.forward.reindex(sig.index), self.spec.min_assets)
        ic = _block(ic_all, oos)
        nw = newey_west(ic, overlap_lags(self.h, self.step) + 1) if len(ic) > 6 else {"t": float("nan"), "mean": float("nan")}
        ls = long_short(sig, self.ctx, self.eval_spec)
        net = ls["net"].loc[oos[0]:oos[1]]
        out = {"oos_ic": float(ic.mean()) if len(ic) else float("nan"), "oos_t": nw["t"], "oos_sharpe": _sharpe(net, self.ppy), "oos_positive_share": float((ic > 0).mean()) if len(ic) else float("nan"),
               "oos_dates": int(len(ic)), "oos_turnover": float(ls["turnover"].loc[oos[0]:oos[1]].mean())}
        out["oos_psr"] = probabilistic_sharpe_ratio(out["oos_sharpe"], 0.0, max(len(net.dropna()), 10), periods_per_year=self.ppy) if np.isfinite(out["oos_sharpe"]) else float("nan")
        out["passed"] = bool(np.isfinite(out["oos_t"]) and out["oos_t"] >= L.oos_t and out["oos_ic"] > 0 and out["oos_sharpe"] > 0)
        out["returns_net"] = ls["net"]
        return out

    def _reject(self, stage: int, reason: str, metrics: dict, direction: float | None = None) -> dict:
        return {"stage": stage, "status": "rejected", "reason": reason, "metrics": metrics, "direction": direction}


def fitness(m: dict, decorrelation: float = 1.0, weights: dict | None = None) -> tuple[float, dict]:
    """The multi-objective score that ranks survivors, and its parts. Every part is scaled to about 0-1; the weights are in :data:`FITNESS`. Complexity is a penalty, so that of two formulas with
    the same evidence the simpler one ranks higher."""
    w = {**FITNESS, **(weights or {})}
    val_ic = m.get("val_ic") if np.isfinite(m.get("val_ic", np.nan)) else 0.0
    parts = {"ic": float(np.clip(min(abs(m.get("ic", 0.0)), abs(val_ic) if val_ic else 1.0) / 0.05, 0, 1)),
             "stability": float(np.clip((m.get("positive_share", 0.5) - 0.5) / 0.3, 0, 1)),
             "sharpe": float(np.clip(min(m.get("sharpe", 0.0), m.get("val_sharpe", 0.0) if np.isfinite(m.get("val_sharpe", np.nan)) else 0.0) / 1.5, 0, 1)),
             "monotonic": float(np.clip(m.get("monotonicity", 0.0), 0, 1)),
             "decorrelation": float(np.clip(decorrelation, 0, 1)),
             "turnover": float(np.clip(m.get("turnover", 0.0) / 0.7, 0, 1)),
             "complexity": float(np.clip((m.get("nodes", 3) - 3) / 12.0, 0, 1))}
    return float(sum(w[k] * v for k, v in parts.items())), parts


# ------------------------------------------------------------------------------------------------------------------ a run
class Discovery:
    def __init__(self, store: FundamentalsStore, ctx: ResearchContext | None = None):
        self.store, self.ctx = store, ctx
        self.library = AlphaLibrary(store)

    # ------------------------------------------------------------------------------------------------------------ the run record
    def create(self, spec: DiscoverySpec) -> str:
        run_id = "R" + uuid.uuid4().hex[:10]
        stamp = now_iso()
        with self.store.db.transaction() as conn:
            conn.execute("INSERT INTO discovery_runs (run_id, created_at, updated_at, status, spec, stage_counts) VALUES (?,?,?,?,?,?)",
                         (run_id, stamp, stamp, "created", json.dumps(clean(spec.describe())), json.dumps({})))
        return run_id

    def get_run(self, run_id: str) -> dict:
        with self.store.db.transaction() as conn:
            row = conn.execute("SELECT * FROM discovery_runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise FundamentalsError(f"no discovery run called {run_id!r}")
        d = dict(row)
        d["spec"], d["stage_counts"] = json.loads(d["spec"]), json.loads(d["stage_counts"] or "{}")
        return d

    def runs(self, limit: int = 30) -> list[dict]:
        with self.store.db.transaction() as conn:
            rows = conn.execute("SELECT run_id, created_at, updated_at, finished_at, status, stage_counts, error FROM discovery_runs ORDER BY created_at DESC LIMIT ?", (int(limit),)).fetchall()
        return [{**dict(r), "stage_counts": json.loads(r["stage_counts"] or "{}")} for r in rows]

    def _save(self, run_id: str, status: str, counts: dict, error: str | None = None, finished: bool = False) -> None:
        with self.store.db.transaction() as conn:
            conn.execute("UPDATE discovery_runs SET status = ?, stage_counts = ?, updated_at = ?, error = ?, finished_at = COALESCE(?, finished_at) WHERE run_id = ?",
                         (status, json.dumps(clean(counts)), now_iso(), error, now_iso() if finished else None, run_id))

    def candidates(self, run_id: str, status: str | None = None, limit: int = 200) -> list[dict]:
        sql = "SELECT * FROM discovery_candidates WHERE run_id = ?" + (" AND status = ?" if status else "") + " ORDER BY rowid LIMIT ?"
        with self.store.db.transaction() as conn:
            rows = conn.execute(sql, [run_id] + ([status] if status else []) + [int(limit)]).fetchall()
        return [{**dict(r), "metrics": json.loads(r["metrics"] or "{}"), "text": X.to_string(X.loads(r["expression"]))} for r in rows]

    # ------------------------------------------------------------------------------------------------------------ the loop
    def run(self, run_id: str, progress=None, stop=None) -> dict:
        """Process the candidates of a run from where it stopped, then (when all are done) the locked test, the multiple-testing correction and the library. Safe to call again on a finished run."""
        say = progress or (lambda done, total, text: None)
        record = self.get_run(run_id)
        spec = DiscoverySpec.from_dict(record["spec"])
        ctx = self.ctx or ResearchContext(self.store, UniverseSpec.from_dict(spec.universe), freq=spec.freq, entry_lag=spec.entry_lag)
        blocks = splits(ctx, spec)
        conditions = spec.conditions()
        digest = settings_hash({**conditions, "blocks": {k: blocks[k] for k in ("train", "validation", "oos")}})
        cache = BoundedCache()
        screen = Screen(ctx, spec, blocks, cache)
        seeds = [X.parse(t) for t in (default_seeds(ctx) if spec.seeds is None else spec.seeds)]
        known = {c["expression_hash"]: c for c in self.candidates(run_id, limit=1_000_000)}
        keys = {c["metrics"].get("signal_key") for c in known.values() if c["metrics"].get("signal_key")}
        counts = {s: 0 for s in STAGES}
        counts.update(record["stage_counts"])
        counts.setdefault("generated", 0)
        counts.setdefault("skipped_seen", 0)
        counts.setdefault("duplicate_idea", 0)
        counts.setdefault("tested", 0)
        counts.setdefault("next_index", 0)
        accepted_signals: list[tuple[str, pd.DataFrame]] = []
        for c in known.values():
            if c.get("status") == "survivor" and c["metrics"].get("direction"):
                accepted_signals.append((c["text"], screen.signal(X.loads(c["expression"]), c["metrics"]["direction"])))
        self._save(run_id, "running", counts)
        gen = X.Generator(ctx, np.random.default_rng(spec.seed), spec.max_nodes, spec.only_fields)
        index = int(counts["next_index"])
        stopped = False
        while index < spec.n_candidates:
            if stop is not None and stop():
                stopped = True
                break
            expression = self._candidate(index, seeds, spec, ctx, gen, run_id)
            index += 1
            counts["next_index"], counts["generated"] = index, counts["generated"] + 1
            try:
                ehash, skey = X.expression_hash(expression), X.signal_key(expression)
            except X.ExprError:
                continue
            if ehash in known:
                continue
            seen = self.library.seen(ehash, digest)
            if seen:
                counts["skipped_seen"] += 1
                self._write(run_id, ehash, expression, 0, "duplicate", f"already in the {seen['where']} under these conditions", {"signal_key": skey})
                known[ehash] = {"metrics": {"signal_key": skey}}
                continue
            if skey in keys:
                counts["duplicate_idea"] += 1
                self._write(run_id, ehash, expression, 0, "duplicate", "the same idea as a formula already tried in this run (a rank, z-score or sign on the outside does not change it)", {"signal_key": skey})
                known[ehash] = {"metrics": {"signal_key": skey}}
                continue
            keys.add(skey)
            try:
                result = screen.run(expression)
            except Exception as error:                                                  # one bad formula must not end a search of thousands
                result = {"stage": 0, "status": "rejected", "reason": f"error while testing: {type(error).__name__}: {str(error)[:120]}", "metrics": {"nodes": X.size(expression)}, "direction": None}
            result["metrics"]["signal_key"] = skey
            if result["status"] == "survivor":
                sig = screen.signal(expression, result["direction"])
                too_alike = next((name for name, other in accepted_signals if abs(signal_correlation(sig, other)) > screen.limits.max_correlation), None)
                if too_alike:
                    result = {**result, "status": "rejected", "stage": 6, "reason": f"correlation above {screen.limits.max_correlation} with {too_alike}, an idea already accepted in this run"}
                else:
                    accepted_signals.append((X.to_string(expression), sig))
            last_passed = 6 if result["status"] == "survivor" else result["stage"] - 1
            for s_index in range(0, last_passed + 1):
                counts[STAGES[s_index]] += 1
            counts["tested"] += 1 if result["metrics"].get("p_train") is not None else 0
            self._write(run_id, ehash, expression, result["stage"], "survivor" if result["status"] == "survivor" else "rejected", result["reason"], result["metrics"])
            known[ehash] = {"metrics": result["metrics"]}
            if result["status"] == "rejected":
                self.library.add_graveyard(expression, {**conditions, "blocks": {k: blocks[k] for k in ("train", "validation", "oos")}}, result["reason"], STAGES[result["stage"]], result["metrics"], run_id)
            if index % 10 == 0:
                self._save(run_id, "running", counts)
                say(index, spec.n_candidates, f"{counts['tested']} formulas tested, {sum(1 for c in known.values() if c['metrics'].get('direction') and 'val_ic' in c['metrics'])} survivors so far")
        self._save(run_id, "stopped" if stopped else "running", counts)
        if stopped:
            return self.summary(run_id)
        self._conclude(run_id, spec, ctx, screen, blocks, conditions, digest, counts, accepted_signals)
        say(index, spec.n_candidates, "finished")
        return self.summary(run_id)

    def _candidate(self, index: int, seeds: list, spec: DiscoverySpec, ctx, gen: X.Generator, run_id: str) -> dict:
        """The index-th formula of the stream: a built-in classic first, then random ones, and in the second half a share bred from the best survivors so far. A function of the seed, the index and
        what the database holds, all of which are the same on a resumed run."""
        if index < len(seeds):
            return seeds[index]
        rng = np.random.default_rng([spec.seed, index])
        gen.rng = rng
        if spec.evolve > 0 and index >= spec.n_candidates // 2 and rng.random() < spec.evolve:
            parents = [c for c in self.candidates(run_id, limit=1_000_000) if c["metrics"].get("t_train") is not None and c["stage_reached"] >= 2]
            parents.sort(key=lambda c: (-abs(c["metrics"]["t_train"]), c["expression_hash"]))
            parents = parents[:10]
            if len(parents) >= 2 and rng.random() < 0.5:
                a, b = rng.choice(len(parents), size=2, replace=False)
                return gen.cross(X.loads(parents[int(a)]["expression"]), X.loads(parents[int(b)]["expression"]))
            if parents:
                return gen.mutate(X.loads(parents[int(rng.integers(len(parents)))]["expression"]))
        return gen.random()

    def _write(self, run_id: str, ehash: str, expression: dict, stage: int, status: str, reason: str, metrics: dict) -> None:
        with self.store.db.transaction() as conn:
            conn.execute("INSERT OR IGNORE INTO discovery_candidates (run_id, expression_hash, expression, stage_reached, status, reason, metrics) VALUES (?,?,?,?,?,?,?)",
                         (run_id, ehash, X.dumps(expression), int(stage), status, str(reason)[:300], json.dumps(clean(metrics))))

    # ------------------------------------------------------------------------------------------------------------ the end of a run
    def _conclude(self, run_id, spec, ctx, screen: Screen, blocks, conditions, digest, counts, accepted_signals) -> None:
        survivors = [c for c in self.candidates(run_id, "survivor", 1_000_000)]
        tested = [c for c in self.candidates(run_id, limit=1_000_000) if c["metrics"].get("p_train") is not None]
        n_tested = len(tested)
        p = np.array([c["metrics"]["p_train"] for c in tested if np.isfinite(c["metrics"]["p_train"])])
        q_of = {}
        if len(p):
            q = bh_qvalues(np.array([c["metrics"]["p_train"] if np.isfinite(c["metrics"]["p_train"]) else 1.0 for c in tested]))
            q_of = {c["expression_hash"]: float(q[i]) for i, c in enumerate(tested)}
        # rank survivors by fitness from training and validation evidence only, with a bonus for being unlike the others
        signals = {c["expression_hash"]: screen.signal(X.loads(c["expression"]), c["metrics"]["direction"]) for c in survivors}
        library_signals = []
        for a in self.library.list(limit=200):
            if a["status"] in ("Candidate", "Validated", "Production"):
                try:
                    library_signals.append(build_signal(ctx, X.loads(a["expression"]), a["settings"]))
                except FundamentalsError:
                    continue
        scored = []
        for c in survivors:
            others = [s for k, s in signals.items() if k != c["expression_hash"]] + library_signals[:20]
            worst = max((abs(signal_correlation(signals[c["expression_hash"]], o)) for o in others), default=0.0)
            worst = 0.0 if not np.isfinite(worst) else worst
            score, parts = fitness(c["metrics"], 1.0 - worst, spec.fitness_weights)
            scored.append((score, parts, c, worst))
        scored.sort(key=lambda t: (-t[0], t[2]["expression_hash"]))
        limit_q = spec.limits().max_q
        eligible = [t for t in scored if q_of.get(t[2]["expression_hash"], 1.0) <= limit_q]                  # allowing for the number of ideas tried comes before the one look at the locked dates
        for _, _, c, _ in scored:
            if q_of.get(c["expression_hash"], 1.0) > limit_q:
                c["metrics"]["q_value"] = q_of.get(c["expression_hash"])
                self._update_candidate(run_id, c["expression_hash"], "survivor", f"passed stages 1-6, but its false-discovery q-value of {q_of.get(c['expression_hash'], 1.0):.2f} over the "
                                                                               f"{n_tested} formulas tried is above {limit_q}: not tested on the locked dates", c["metrics"])
        finalists = eligible[:spec.finalists]
        # the locked test: once, on the finalists only
        oos_key = f"oos_looks:{spec.freq}:{spec.horizon}:{json.dumps(spec.universe, sort_keys=True)}:{blocks['oos'][0]}"
        looks = int(self.store.get_meta(oos_key, "0") or 0) + (1 if finalists else 0)
        if finalists:
            self.store.set_meta(oos_key, str(looks))
        returns, results = {}, []
        for score, parts, c, worst in finalists:
            expression = X.loads(c["expression"])
            lock = screen.locked_test(expression, c["metrics"]["direction"])
            net = lock.pop("returns_net")
            returns[c["expression_hash"]] = net
            results.append((score, parts, c, worst, lock))
        # deflated Sharpe on the returns the formulas were chosen from, and the probability of backtest overfitting among the survivors
        train_val = (blocks["train"][0], blocks["validation"][1])
        pbo = None
        stage5 = [c for c in survivors]
        if len(stage5) >= 6:
            frame = pd.concat({c["expression_hash"]: long_short(signals[c["expression_hash"]], ctx, screen.eval_spec)["net"].loc[train_val[0]:train_val[1]] for c in stage5[:30]}, axis=1).dropna()
            if len(frame) >= 32:
                try:
                    pbo = pbo_cscv(frame.to_numpy(float), blocks=min(16, (len(frame) // 2) * 2), annualise=float(screen.ppy))
                except Exception:                                                      # a failure of an optional diagnostic must not lose the run
                    pbo = None
        added, failed = [], []
        for score, parts, c, worst, lock in results:
            m = dict(c["metrics"])
            net_tv = long_short(signals[c["expression_hash"]], ctx, screen.eval_spec)["net"].loc[train_val[0]:train_val[1]].dropna()
            sr = _sharpe(net_tv, screen.ppy)
            m.update(lock)
            m.update(fitness=score, fitness_parts=parts, max_correlation=worst, q_value=q_of.get(c["expression_hash"]), n_tested=n_tested, run_id=run_id, oos_looks=looks,
                     deflated_sharpe=deflated_sharpe_ratio(sr, max(n_tested, 1), len(net_tv), float(stats.skew(net_tv, bias=False)) if len(net_tv) > 3 else 0.0,
                                                           float(stats.kurtosis(net_tv, bias=False, fisher=False)) if len(net_tv) > 3 else 3.0, periods_per_year=screen.ppy) if np.isfinite(sr) else None,
                     complexity=X.size(X.loads(c["expression"])), sharpe_train_val=sr, pbo=(pbo or {}).get("pbo") if pbo else None, blocks=blocks, direction=c["metrics"]["direction"])
            expression = X.loads(c["expression"])
            settings = {**conditions, "blocks": {k: blocks[k] for k in ("train", "validation", "oos")}, "direction": c["metrics"]["direction"]}
            if lock["passed"]:
                alpha_id = self.library.add_candidate(expression, settings, {**m, "ic": m["ic"], "icir": m["icir"], "sharpe": m["sharpe"], "turnover": m["turnover"], "oos_ic": lock["oos_ic"],
                                                                         "oos_sharpe": lock["oos_sharpe"], "oos_t": lock["oos_t"]},
                                                      description=f"found by discovery run {run_id}", reason="passed every screen, including the locked out-of-sample dates")
                added.append(alpha_id)
                self._update_candidate(run_id, c["expression_hash"], "candidate", f"added to the library as {alpha_id}", m)
            else:
                reason = f"failed the locked test: out-of-sample IC {lock['oos_ic']:+.3f} (t = {lock['oos_t']:.1f}), net Sharpe {lock['oos_sharpe']:.2f}"
                self.library.add_graveyard(expression, settings, reason, STAGES[7], m, run_id)
                failed.append(c["expression_hash"])
                self._update_candidate(run_id, c["expression_hash"], "rejected", reason, m)
        for _, _, c, _ in eligible[spec.finalists:]:
            self._update_candidate(run_id, c["expression_hash"], "survivor", "passed stages 1-6 but was not among the finalists given the one look at the locked dates", c["metrics"])
        counts[STAGES[7]] = len(added)
        counts.update(n_tested=n_tested, survivors=len(survivors), finalists=len(finalists), candidates=len(added), failed_locked_test=len(failed), pbo=(pbo or {}).get("pbo") if pbo else None, oos_looks=looks)
        self._save(run_id, "finished", counts, finished=True)
        self._experiment(run_id, spec, counts)

    def _update_candidate(self, run_id, ehash, status, reason, metrics) -> None:
        with self.store.db.transaction() as conn:
            conn.execute("UPDATE discovery_candidates SET status = ?, reason = ?, metrics = ? WHERE run_id = ? AND expression_hash = ?",
                         (status, str(reason)[:300], json.dumps(clean(metrics)), run_id, ehash))

    def _experiment(self, run_id: str, spec: DiscoverySpec, counts: dict) -> None:
        counts_now = self.store.counts()
        with self.store.db.transaction() as conn:
            conn.execute("INSERT OR REPLACE INTO experiments (experiment_id, kind, created_at, spec, data_version, code_version, summary, status) VALUES (?,?,?,?,?,?,?,?)",
                         (run_id, "discovery", now_iso(), json.dumps(clean(spec.describe())), f"{counts_now['companies']}c/{counts_now['observations']}o/{counts_now['price_rows']}p@{now_iso()[:10]}",
                          "fundamentals-1", json.dumps(clean(counts)), "finished"))

    # ------------------------------------------------------------------------------------------------------------ reporting
    def summary(self, run_id: str) -> dict:
        """Where a run stands: the funnel (how many formulas survived each stage), the multiple-testing numbers, and the leaderboard."""
        run = self.get_run(run_id)
        counts = run["stage_counts"]
        spec = DiscoverySpec.from_dict(run["spec"])
        board = [c for c in self.candidates(run_id, limit=1_000_000) if c["status"] in ("candidate", "survivor")]
        board.sort(key=lambda c: (-(c["metrics"].get("fitness") if c["metrics"].get("fitness") is not None else -1.0), -abs(c["metrics"].get("t_train") or 0.0)))
        funnel = [{"stage": s, "passed": counts.get(s, 0)} for s in STAGES]
        return {"run_id": run_id, "status": run["status"], "created_at": run["created_at"], "finished_at": run["finished_at"], "error": run["error"], "progress": {"done": counts.get("next_index", 0), "total": spec.n_candidates},
                "counts": counts, "funnel": funnel, "spec": run["spec"],
                "leaderboard": [{"expression": c["text"], "status": c["status"], "stage": c["stage_reached"], "reason": c["reason"], "ic": c["metrics"].get("ic"), "icir": c["metrics"].get("icir"),
                                 "val_ic": c["metrics"].get("val_ic"), "oos_ic": c["metrics"].get("oos_ic"), "oos_t": c["metrics"].get("oos_t"), "sharpe": c["metrics"].get("sharpe"),
                                 "oos_sharpe": c["metrics"].get("oos_sharpe"), "turnover": c["metrics"].get("turnover"), "q_value": c["metrics"].get("q_value"),
                                 "deflated_sharpe": c["metrics"].get("deflated_sharpe"), "fitness": c["metrics"].get("fitness"), "nodes": c["metrics"].get("nodes"), "direction": c["metrics"].get("direction")}
                                for c in board[:25]],
                "note": "Every formula that reached stage 1 counts as a test; significance statements are corrected for that number. Nothing here is validated: a Candidate has passed dates it was not chosen on, "
                        "and still needs the corrections in its q-value and deflated Sharpe, and time."}
