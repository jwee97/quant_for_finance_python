"""Alpha and portfolio diagnostics of a saved experiment (spec v3, Release 6; ``GET /api/ml/alpha-diagnostics`` and ``GET /api/ml/portfolio-diagnostics``).

Both read what the experiment saved (its predictions, positions and returns); nothing is refitted. Both default to the validation segment, which is development data. The protected test and forward
segments are refused unless the experiment's holdout was evaluated once under a frozen plan, the same rule the statistical diagnostics follow, because these figures drive choices (which features to keep, which
covariance estimator to trust) and a choice made on a sealed segment is a look at it.

``alpha_diagnostics``: the rank IC of the saved scores against the raw forward return of the dataset's horizon, with a dependence-aware interval whose lag is at least ``horizon - 1``; the rolling IC; the decay of
the IC with extra delay and horizon; coverage and missingness; stability across blocks of the segment; the redundancy of the feature panel (development rows only); and, given another experiment as the
existing book, the conditional IC and the paired Sharpe difference of a blend (predictive incremental value on this sample, not a causal claim and not a feature importance).

``portfolio_diagnostics``: the book the experiment held (exposure, concentration, turnover); the covariance estimators compared on the risk that book actually realised; the risk contributions of its last
book under the estimator whose forecast was closest to realised; and, only where the model's output is a return forecast in return units, how an optimal cost-aware book would move under a predeclared grid of
perturbations. A rank score is not a return forecast: for a ranking model the allocation part says so and is not run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import alpha as al
from . import allocation as alloc
from . import contracts as ct
from . import diagnostics as dx
from . import experiments as xp
from . import risk_models as rm
from .backtest import label_days
from .targets import forward_return, parse_target

VERSION = "r6-reports-v1"
RETURN_UNIT_TARGETS = ("fwd_ret", "excess_ret", "sector_ret", "resid_ret")
DEFAULT_SPECS = (rm.CovSpec("sample", window=252), rm.CovSpec("ewma", window=252, halflife=60.0), rm.CovSpec("lw_constant_correlation", window=252), rm.CovSpec("factor", window=252))
DEFAULT_GRID = {"mu_scale": [0.5, 1.0, 2.0], "cov_scale": [0.5, 1.0, 2.0], "cost_scale": [0.0, 1.0, 3.0], "position_max": [0.2, 0.35]}


def _guard(registry, exp_id: str, segment: str) -> None:
    holdout = (registry.report(exp_id).get("holdout") or {}).get("status")
    if segment in ("test", "forward") and holdout != "evaluated_under_plan":
        raise ValueError(f"the {segment} segment of this experiment is protected (holdout status: {holdout or 'not recorded'}): diagnostics that drive choices are computed on the validation segment, "
                         "or on a segment that a frozen plan evaluated once")
    if segment not in ("train", "validation", "test", "forward"):
        raise ValueError("segment must be one of train, validation, test or forward")


def _segment_mask(index: pd.DatetimeIndex, train, segment: str) -> pd.Series:
    return label_days(index, train.plan.labels()) == segment


def _rolling_summary(tab: pd.DataFrame, window: int) -> dict:
    r = al.rolling_ic(tab, window=window, min_periods=max(10, window // 3))
    mean = r["mean"].dropna()
    if mean.empty:
        return {"eligibility": "insufficient", "reason": f"no window of {window} dates has enough dates with an IC", "window": window}
    step = max(1, len(mean) // 60)
    return {"eligibility": "applied", "window": window, "last": float(mean.iloc[-1]), "min": float(mean.min()), "max": float(mean.max()), "share_positive": float((mean > 0).mean()),
            "series": [[str(d.date()), float(v)] for d, v in mean.iloc[::step].items()]}


def alpha_diagnostics(registry, exp_id: str, *, segment: str = "validation", baseline_id: str | None = None, decay_lags=(0, 1, 2, 5), decay_horizons=(1, 5, 10, 20), roll: int = 63, min_assets: int = 5,
                      n_boot: int = 400, seed: int = 0) -> dict:
    _guard(registry, exp_id, segment)
    cfg, ds, train, _ = xp.load_run(registry, exp_id)
    scores = train.scores_wide(ds)
    mask = _segment_mask(scores.index, train, segment)
    sc = scores[mask.to_numpy()].dropna(how="all")
    h, lag = int(ds.horizon), int(ds.lag)
    out: dict = {"version": VERSION, "experiment_id": exp_id, "segment": segment, "label": f"rank IC against the raw {h}-session forward return starting {lag} session(s) after the decision", "horizon": h, "lag": lag,
                 "model": cfg["model"]["kind"], "target": cfg["dataset"]["target"], "output_kind": ct.output_kind(cfg["model"]["kind"], train.task, cfg["dataset"]["target"])}
    if len(sc) == 0:
        return {**out, "eligibility": "insufficient", "reason": f"the experiment has no scores on the {segment} segment"}
    fwd = forward_return(ds.prices, h, lag).reindex(index=ds.dates, columns=ds.assets)
    tab = al.ic_table(sc, fwd.loc[sc.index], min_assets)
    out.update(eligibility="applied", ic=al.ic_inference(tab, h), rolling=_rolling_summary(tab, roll))
    out["decay"] = al.decay_grid(sc, ds.prices, base_lag=lag, extra_lags=tuple(decay_lags), horizons=tuple(decay_horizons), min_assets=min_assets)
    out["coverage"] = al.coverage_report(sc, ds.investable.reindex(sc.index))
    out["stability"] = al.stability_report(tab, blocks=4, signal=sc)
    labels = label_days(ds.dates, train.plan.labels())
    X = al.development_only(ds.X, labels)                                                     # the feature panel the diagnostics may look at: train and validation rows only
    out["redundancy"] = {**al.redundancy_report(X, every=5), "rows_scope": "train and validation rows only (a diagnostic that drives feature choice reads development data)"}
    if baseline_id:
        out["incremental"] = _incremental(registry, exp_id, baseline_id, segment, cfg, ds, train, sc, fwd, min_assets, n_boot, seed)
    return dx._jsonable(out)                                                                  # noqa: SLF001


def _incremental(registry, exp_id: str, baseline_id: str, segment: str, cfg: dict, ds, train, sc: pd.DataFrame, fwd: pd.DataFrame, min_assets: int, n_boot: int, seed: int) -> dict:
    _guard(registry, baseline_id, segment)
    bcfg, bds, btrain, _ = xp.load_run(registry, baseline_id)
    problems = []
    if (int(bds.horizon), int(bds.lag)) != (int(ds.horizon), int(ds.lag)):
        problems.append(f"the existing experiment looks {bds.horizon} sessions ahead with lag {bds.lag}; this one {ds.horizon} with lag {ds.lag}")
    if list(bds.assets) != list(ds.assets):
        problems.append("the two experiments do not trade the same assets")
    if problems:
        return {"eligibility": "not_comparable", "reason": "; ".join(problems), "baseline_id": baseline_id}
    base_scores = btrain.scores_wide(bds).reindex(index=sc.index, columns=sc.columns)
    cond = al.conditional_ic(sc, {baseline_id: base_scores}, fwd.loc[sc.index], min_assets=max(min_assets, 8))
    out = {"baseline_id": baseline_id, "conditional_ic": al.ic_inference(cond, int(ds.horizon)),
           "plain_ic": al.ic_inference(al.ic_table(sc, fwd.loc[sc.index], min_assets), int(ds.horizon)), "assumptions": ["both experiments' scores are out-of-sample predictions on the same dates and assets",
                                                                                                       "the partial rank correlation removes the cross-sectional rank of the existing signal from both the new signal and the label, date by date"]}
    ret_new = pd.read_parquet(registry.root / "experiments" / exp_id / "returns.parquet")
    ret_old = pd.read_parquet(registry.root / "experiments" / baseline_id / "returns.parquet")
    new_net = ret_new["net"][ret_new["segment"] == segment].dropna()
    old_net = ret_old["net"][ret_old["segment"] == segment].dropna()
    out["blend"] = al.blend_value(old_net, new_net, n_boot=n_boot, seed=seed, label=f"blend of {baseline_id[-8:]} and {exp_id[-8:]} against {baseline_id[-8:]} alone")
    pos_new = pd.read_parquet(registry.root / "experiments" / exp_id / "positions.parquet").reindex(new_net.index)
    pos_old = pd.read_parquet(registry.root / "experiments" / baseline_id / "positions.parquet").reindex(old_net.index)
    out["correlation"] = al.correlation_report({exp_id: sc, baseline_id: base_scores}, {exp_id: new_net, baseline_id: old_net}, {exp_id: pos_new, baseline_id: pos_old}, min_assets=min_assets)
    out["correlation"]["limitations"] = ["factor overlap is not reported: the platform has no factor returns beyond the equal-weight market, so only the overlap of the books themselves is shown"]
    out["eligibility"] = "applied"
    return out


# --------------------------------------------------------------------------------------------------------------------- portfolio
def _book_summary(positions: pd.DataFrame) -> dict:
    live = positions[positions.abs().sum(axis=1) > 1e-12]
    if live.empty:
        return {"eligibility": "insufficient", "reason": "the experiment held no positions on this segment"}
    gross, net = live.abs().sum(axis=1), live.sum(axis=1)
    share = live.abs().div(gross, axis=0)
    hhi = (share ** 2).sum(axis=1)
    turnover = positions.diff().abs().sum(axis=1).loc[live.index].iloc[1:]
    top3 = share.apply(lambda r: r.nlargest(3).sum(), axis=1)
    return {"eligibility": "applied", "days_with_a_book": int(len(live)), "gross": {"mean": float(gross.mean()), "max": float(gross.max())}, "net": {"mean": float(net.mean()), "min": float(net.min()), "max": float(net.max())},
            "concentration": {"mean_hhi": float(hhi.mean()), "effective_positions": float((1.0 / hhi).mean()), "top3_share": float(top3.mean())}, "mean_daily_turnover": float(turnover.mean()) if len(turnover) else None,
            "units": "weights as a fraction of equity; turnover is the sum of absolute daily changes in the target book"}


def portfolio_diagnostics(registry, exp_id: str, *, segment: str = "validation", horizon: int = 21, max_dates: int = 24, specs=None, grid: dict | None = None, risk_aversion: float = 5.0) -> dict:
    _guard(registry, exp_id, segment)
    cfg, ds, train, _ = xp.load_run(registry, exp_id)
    folder = registry.root / "experiments" / exp_id
    positions = pd.read_parquet(folder / "positions.parquet").reindex(ds.dates).fillna(0.0)
    seg = pd.read_parquet(folder / "returns.parquet")["segment"]                                       # the dates on which the book earned, with the segment the experiment itself gave them
    book = positions.reindex(seg[seg == segment].index).dropna(how="all").fillna(0.0)
    kind = ct.output_kind(cfg["model"]["kind"], train.task, cfg["dataset"]["target"])
    out: dict = {"version": VERSION, "experiment_id": exp_id, "segment": segment, "model": cfg["model"]["kind"], "target": cfg["dataset"]["target"], "output_kind": kind, "book": _book_summary(book)}
    if out["book"]["eligibility"] != "applied":
        return dx._jsonable({**out, "eligibility": "insufficient", "reason": out["book"]["reason"]})                     # noqa: SLF001
    live = book[book.abs().sum(axis=1) > 1e-12].index
    step = max(1, len(live) // max_dates)
    dates = list(live[::step][:max_dates])
    specs = list(specs or DEFAULT_SPECS)
    cmp = rm.compare(ds.returns, dates, specs, weights=positions, horizon=horizon, eligible=ds.investable)
    out["risk_models"] = cmp
    usable = {k: v for k, v in cmp["estimators"].items() if v["windows"] and v["risk_ratio_forecast_to_realised"] is not None}
    if usable:
        best = min(usable, key=lambda k: abs(np.log(usable[k]["risk_ratio_forecast_to_realised"])))
        spec = next(s for s in specs if s.label() == best)
        last = dates[-1]
        fit = rm.fit_asof(ds.returns, last, spec, ds.investable)
        w = positions.loc[last].reindex(fit.matrix.index).fillna(0.0)
        rc = rm.risk_contributions(w, fit.matrix)
        out["risk_contributions"] = {"estimator": best, "chosen_because": "its forecast risk of the book the experiment held was closest to the risk realised (ratio nearest one); a description of these dates, not a selection of an estimator for use",
                                     "asof": str(last.date()), "fit": fit.record(), "table": rc.assign(asset=rc.index).round(8).to_dict("records")}
    else:
        out["risk_contributions"] = {"eligibility": "insufficient", "reason": "no estimator could be fitted on any decision date"}
    out["allocation"] = _allocation_sensitivity(cfg, ds, train, positions, dates, kind, grid or DEFAULT_GRID, risk_aversion, segment)
    out["eligibility"] = "applied"
    out["caveats"] = ["covariances are per day", "the experiment's own book was not re-optimised: these are diagnostics of the book it held"]
    return dx._jsonable(out)                                                                                          # noqa: SLF001


def _allocation_sensitivity(cfg: dict, ds, train, positions: pd.DataFrame, dates: list, kind: str, grid: dict, risk_aversion: float, segment: str) -> dict:
    tkind = parse_target(cfg["dataset"]["target"])[0]
    if kind != "expected_return" or tkind not in RETURN_UNIT_TARGETS:
        return {"eligibility": "not_applicable", "reason": f"the model's output is a {kind.replace('_', ' ')} on a {tkind} target, not a return forecast in return units: an optimiser needs expected returns on the "
                                                                "same units and horizon as the covariance, and a score with no return meaning cannot be put into one"}
    last = dates[-1]
    scores = train.scores_wide(ds)
    if last not in scores.index or scores.loc[last].notna().sum() < 4:
        return {"eligibility": "insufficient", "reason": "no scores on the last decision date"}
    spec = rm.CovSpec("lw_constant_correlation", window=252)
    try:
        fit = rm.fit_asof(ds.returns, last, spec, ds.investable)
    except ValueError as exc:
        return {"eligibility": "insufficient", "reason": f"no covariance could be fitted on {last.date()}: {exc}"}
    assets = list(fit.matrix.index)
    mu = scores.loc[last].reindex(assets)
    mu = mu.where(mu.notna(), 0.0)
    days = max(1, int(cfg["signal"].get("holding_period", 1)))
    mu_w, cov_w = alloc.window_inputs(mu, fit.matrix, days)
    prev = positions.shift(1).loc[last].reindex(assets).fillna(0.0)
    sig = cfg["signal"]
    cost = (float(cfg["backtest"]["cost_bps"]) + float(cfg["backtest"]["slippage_bps"])) / 1e4
    net = float(sig.get("net", 0.0))
    aspec = alloc.AllocationSpec(risk_aversion=risk_aversion, cost=cost, position_max=float(sig.get("max_weight") or 0.35), gross_max=float(sig.get("gross", 1.0)), net_min=net - 0.05, net_max=net + 0.05)
    res = alloc.sensitivity(mu_w, cov_w, prev, aspec, grid)
    relative = tkind != "fwd_ret"
    return {"eligibility": "applied", "asof": str(last.date()), "covariance": fit.record(), "holding_days": days,
            "units": f"the score is read as a fractional return over the {ds.horizon}-session horizon and the covariance is scaled to the {days}-day holding window",
            "interpretation": ("the target is relative to a benchmark or group, so the score is a relative return and the book is held dollar-neutral (net within 5 percentage points of the experiment's net)" if relative
                               else "the score is read as an absolute expected return; the book is constrained to the experiment's net exposure within 5 percentage points"),
            "spec": aspec.content(), **res}
