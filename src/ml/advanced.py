"""The optional advanced modules of Release 8 (spec v3, P3-01 to P3-12): what exists, how each is reached and what it needs. Nothing here runs anything.

The modules are **off by default**: no experiment, search, forward test or dashboard view imports them, and none of them is scheduled. Each is reached only by a call that names it, through the API and
the command line (the ones that work on saved experiments and the registry's records) or as a library (the numerical ones, which take arrays and frames the caller supplies). All of them need only the
base install (numpy, pandas, scipy, scikit-learn): none needs a solver extra, an accelerator or a downloaded foundation model, and importing the base workflow does not import them.

``catalog`` lists them with their surface and with the latest recorded verification of their requirement where there is one. Availability here means *the code can run in this environment*; it says
nothing about whether the technique helps. The verification and traceability documents carry the evidence.
"""

from __future__ import annotations

import importlib
import importlib.util

VERSION = "advanced-v1"
PACKAGES = {"scipy": "scipy", "scikit-learn": "sklearn", "numpy": "numpy", "pandas": "pandas"}

MODULES = (
    {"id": "P3-01", "key": "drift", "name": "Drift monitoring", "module": "src.ml.drift", "surface": ["api", "cli"], "needs": ["scipy"],
     "routes": ["GET /api/ml/drift"], "cli": "ml-advanced drift",
     "text": "PSI and a dependence-aware KS test of each feature against a fixed reference window ending at the model's fit cutoff, with coverage and sample counts; drift of the settled rank IC; alert lineage. An alert is a proposal to look, not an action."},
    {"id": "P3-02", "key": "retrain", "name": "Retraining triggers and proposals", "module": "src.ml.retrain", "surface": ["api", "cli"], "needs": [],
     "routes": ["GET /api/ml/retraining", "POST /api/ml/retraining/action"], "cli": "ml-advanced retrain",
     "text": "Predefined triggers, cooldown, minimum data, duplicate suppression, an approval by a named person, nested validation, suspension after repeated failures, recovery and rollback. A retraining never promotes a challenger and never reads a protected segment. Starting a retraining is a library call (it needs a trainer)."},
    {"id": "P3-03", "key": "bma", "name": "Bayesian model averaging", "module": "src.ml.bma", "surface": ["library"], "needs": [],
     "routes": [], "cli": None,
     "text": "Posterior model weights from closed-form Bayes factors under a stated g-prior and a stated approximation, with uncertainty, next to regularised stacked weights on the same out-of-sample data. Performance weights are never called posterior probabilities."},
    {"id": "P3-04", "key": "dro", "name": "Distributionally robust allocation", "module": "src.ml.dro", "surface": ["library"], "needs": ["scipy"],
     "routes": [], "cli": None,
     "text": "A box on expected returns and a scale on the covariance, both calibrated from the history, solved through the allocation module; worst-case value, limiting cases and simpler robust baselines."},
    {"id": "P3-05", "key": "horizon_combo", "name": "Multi-horizon signal combination", "module": "src.ml.horizon_combo", "surface": ["library"], "needs": [],
     "routes": [], "cli": None,
     "text": "Short, medium and long horizon signals in per-session units, settled-IC weights, dependence reported, sleeves or one blended score book, a turnover cap, and a comparison with every single-horizon book."},
    {"id": "P3-06", "key": "transfer", "name": "Cross-asset transfer", "module": "src.ml.transfer", "surface": ["library"], "needs": ["scikit-learn"],
     "routes": [], "cli": None,
     "text": "Disjoint source and target assets, a cutoff in time, preprocessing fitted on the source only, and the target-domain out-of-sample result against training from scratch, negative transfer included."},
    {"id": "P3-07", "key": "discovery", "name": "Feature and strategy discovery", "module": "src.ml.discovery", "surface": ["library"], "needs": [],
     "routes": [], "cli": None,
     "text": "A finite, fully logged search over a whitelisted expression grammar that cannot read the future or sealed outcomes, with multiplicity control; the best mined rule is never presented as independent evidence."},
    {"id": "P3-08", "key": "offpolicy", "name": "Off-policy evaluation", "module": "src.ml.offpolicy", "surface": ["library"], "needs": [],
     "routes": [], "cli": None,
     "text": "IPS, self-normalised and doubly robust estimates only with logged behaviour probabilities and action support; effective sample size and bootstrap intervals; blocked or not applicable otherwise."},
    {"id": "P3-09", "key": "tax_lots", "name": "Tax-lot and after-tax accounting", "module": "src.ml.tax_lots", "surface": ["library"], "needs": [],
     "routes": [], "cli": None,
     "text": "FIFO, LIFO and HIFO lots, fees, a short/long-term split and loss netting under a supplied, versioned policy with provenance. It carries no jurisdiction's rules: an absent or unimplementable policy is unavailable. Pre-tax figures are never changed."},
    {"id": "P3-10", "key": "champion", "name": "Champion and challenger", "module": "src.ml.champion", "surface": ["api", "cli"], "needs": ["scipy"],
     "routes": ["GET /api/ml/champion", "POST /api/ml/champion/register", "POST /api/ml/champion/commit", "POST /api/ml/champion/look"], "cli": "ml-advanced champion",
     "text": "A pair registered before the observations it is judged on, both forecasts committed together under one cutoff, a predeclared schedule of looks with alpha spending, and no backfilling."},
    {"id": "P3-11", "key": "uncertainty_monitor", "name": "Uncertainty and disagreement monitor", "module": "src.ml.uncertainty_monitor", "surface": ["api", "cli"], "needs": ["scipy"],
     "routes": ["GET /api/ml/uncertainty-monitor"], "cli": "ml-advanced monitor",
     "text": "Calibration, prediction distribution, uncertainty and ensemble disagreement monitored separately; a missing estimate is unavailable or degraded and never read as zero uncertainty; alerts escalate with persistence."},
    {"id": "P3-12", "key": "repro_report", "name": "Reproducible reports", "module": "src.ml.repro_report", "surface": ["api", "cli"], "needs": [],
     "routes": ["GET /api/ml/reports", "GET /api/ml/report", "POST /api/ml/report/build"], "cli": "ml-advanced report",
     "text": "A report that is a pure function of a stored snapshot of artifact hashes, diagnostics and policy versions: it is rendered from the snapshot alone and does not change when models or data are updated later."},
)
KEYS = tuple(m["key"] for m in MODULES)


def availability(entry: dict) -> tuple[bool, str | None]:
    """Whether the module can be imported here (its packages are installed and the module itself loads)."""
    missing = [p for p in entry["needs"] if importlib.util.find_spec(PACKAGES.get(p, p)) is None]
    if missing:
        return False, f"needs {', '.join(missing)}, which is not installed"
    try:
        importlib.import_module(entry["module"])
    except Exception as exc:                                                                      # noqa: BLE001  (a module that fails to import is reported, not raised)
        return False, f"the module does not import here: {type(exc).__name__}: {exc}"
    return True, None


def catalog(registry=None) -> dict:
    """The twelve modules with their surface, availability and (when a registry is given) the latest recorded verification of each requirement."""
    verified = {}
    if registry is not None:
        try:
            from . import verification as vf

            verified = {r["requirement_id"]: r for r in vf.current(registry)}
        except Exception:                                                                         # noqa: BLE001  (no verification run yet, or an old registry: the column is simply empty)
            verified = {}
    rows = []
    for m in MODULES:
        ok, why = availability(m)
        v = verified.get(m["id"])
        rows.append({**m, "default_state": "off", "available": ok, "reason": why,
                     "verification": None if v is None else {"status": v.get("status"), "stale": bool(v.get("stale")), "ran_at": v.get("created")}})
    return {"version": VERSION, "modules": rows,
            "note": "all of these are off by default: nothing in the base workflow imports or schedules them, and each is reached only by a call that names it. 'library' modules take data you supply and have no route. "
                    "Availability means the code runs here, not that the technique helps."}


# ---------------------------------------------------------------------------------------------------------------------------------- services shared by the API, the command line and programmatic callers
def run_drift(registry, exp_id: str, *, spec=None, as_of=None, actor: str = "api") -> dict:
    """The drift report of a saved experiment, recorded as an analysis record (every report is a look, and the next one links to it)."""
    from . import diagnostics as dx
    from . import drift as dr

    out = dr.report(registry, exp_id, spec=spec, as_of=as_of)
    return {**out, "record_seq": dx.store(registry, "drift_report", exp_id, out, actor=actor)}


def run_monitor(registry, exp_id: str, *, as_of=None, reference_end=None, policy=None, actor: str = "api") -> dict:
    """The uncertainty monitor on the predictive distribution a saved experiment's model reported on its validation segment. A model that reports no distribution is ``not_applicable`` (said so, and
    nothing is stored): the absence of an uncertainty estimate is never a healthy reading."""
    import pandas as pd

    from . import diagnostics as dx
    from . import experiments as xp
    from . import uncertainty_monitor as um
    from .backtest import label_days

    cfg, ds, train, _ = xp.load_run(registry, exp_id)
    base = {"version": um.VERSION, "experiment_id": exp_id, "model": cfg["model"]["kind"]}
    dist = train.extras.get("distribution")
    if dist is None or len(dist) == 0:
        return {**base, "eligibility": "not_applicable", "reason": "this model reports no predictive distribution (no standard deviation or interval), so there is no stated uncertainty to monitor; that is not a clean bill of health"}
    dates = pd.DatetimeIndex(dist.index.get_level_values(0))
    labels = label_days(pd.DatetimeIndex(sorted(set(dates))), train.plan.labels()).reindex(dates).to_numpy()
    frame = dist[labels == "validation"]                                                                    # development outcomes only: the protected segments stay sealed
    if len(frame) < 50:
        return {**base, "eligibility": "insufficient", "reason": f"{len(frame)} validation-segment forecasts carry a predictive distribution; at least 50 are needed"}
    fd = pd.DatetimeIndex(frame.index.get_level_values(0))
    udates = pd.DatetimeIndex(sorted(set(fd)))
    ref_end = pd.Timestamp(reference_end) if reference_end is not None else udates[max(int(len(udates) * 0.6) - 1, 0)]
    when = pd.Timestamp(as_of) if as_of is not None else udates[-1]
    pos = ds.dates.get_indexer(fd)
    end = pos + int(ds.lag) + int(ds.horizon)
    ends = pd.Series([ds.dates[e] if p >= 0 and e < len(ds.dates) else pd.NaT for p, e in zip(pos, end)], index=frame.index)                  # a label that has not ended has no end date yet
    prev = dx.latest(registry, exp_id, "uncertainty_monitor")
    members = train.extras.get("members")                                                                   # an ensemble's calibrated member forecasts, when the experiment kept them
    gone = "this ensemble experiment was saved before its members' forecasts were kept, so the disagreement between them cannot be monitored" if members is None and cfg["model"]["kind"] == "model_ensemble" else None
    out = um.monitor(frame, ds.Y[ds.target].reindex(frame.index), ends, as_of=when, reference_end=ref_end, policy=policy, members=None if members is None else members.reindex(frame.index),
                     members_missing_reason=gone, previous=None if prev is None else prev["content"])
    out = dx._jsonable({**base, "eligibility": "applied", "scope": "settled decisions on the validation segment: a development statistic, not live monitoring",         # noqa: SLF001
                        "reference_rule": "the reference window is the first 60% of the validation dates unless reference_end is given", **out})
    return {**out, "record_seq": dx.store(registry, "uncertainty_monitor", exp_id, out, actor=actor)}


def propose_retraining(registry, exp_id: str, *, kind: str, policy, actor: str, now: float | None = None) -> dict:
    """Record a retraining trigger for a saved experiment and say whether it became a proposal. The counts the policy tests (new rows since the last fit, days of history) are read from the experiment, and a
    drift or performance trigger must be backed by an alert in the latest recorded drift report: a trigger is not something a caller can assert."""
    import pandas as pd

    from . import diagnostics as dx
    from . import experiments as xp
    from . import retrain as rt

    detail: dict = {}
    if kind in ("drift_alert", "performance_alert"):
        rec = dx.latest(registry, exp_id, "drift_report")
        if rec is None:
            raise rt.RetrainError(f"a {kind} trigger needs a recorded drift report that shows an alert: none has been computed for this experiment")
        c = rec["content"]
        if kind == "drift_alert":
            alerting = sorted(r["feature"] for r in ((c.get("inputs") or {}).get("features") or []) if r.get("state") == "alert")
            if not alerting:
                raise rt.RetrainError("the latest drift report records no alerting feature, so there is no drift alert to act on")
            detail = {"drift_record": rec["seq"], "alerting_features": alerting}
        else:
            perf = c.get("performance") or {}
            if perf.get("state") != "alert":
                raise rt.RetrainError("the latest drift report records no performance alert, so there is none to act on")
            detail = {"drift_record": rec["seq"], "performance": {k: perf.get(k) for k in ("state", "reason")}}
    _, ds, train, _ = xp.load_run(registry, exp_id)
    folds = [f for f in train.folds if not f.get("skipped") and f.get("fit_cutoff")]
    if not folds:
        raise rt.RetrainError("the experiment records no fold with a fit cutoff, so the data since its last fit cannot be counted")
    cutoff = pd.Timestamp(folds[-1]["fit_cutoff"])
    return rt.trigger(registry, exp_id, policy, kind=kind, detail=detail, rows_since_fit=int((ds.dates > cutoff).sum()), train_days=int((ds.dates <= cutoff).sum()), actor=actor, now=now)


def retrain_policy(raw: dict | None):
    """A retraining policy from a plain mapping. ``autonomous`` is not accepted here: these services record proposals and decisions, and starting a retraining is a library call that needs a trainer."""
    from . import retrain as rt

    raw = dict(raw or {})
    if raw.pop("autonomous", False):
        raise rt.RetrainError("autonomous retraining cannot be enabled here: this records proposals and decisions; a retraining is started from code with a trainer after a person approves it")
    allowed = {"name", "triggers", "cooldown_days", "min_new_rows", "min_train_days", "require_nested_validation", "max_failures"}
    extra = sorted(set(raw) - allowed)
    if extra:
        raise rt.RetrainError(f"a retraining policy has {sorted(allowed)}, not {extra}")
    if "triggers" in raw:
        raw["triggers"] = tuple(raw["triggers"])
    return rt.RetrainPolicy(**{"name": "default", **raw})
