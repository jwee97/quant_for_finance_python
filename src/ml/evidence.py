"""The research evidence for one experiment, as a checklist with the rule behind every line (Research Evidence workspace).

Nothing here is computed again: each check reads a number the run already recorded (the dataset audit, the segment Sharpe ratios, the trial count and deflated Sharpe, the robustness tests, the holdout
ledger, the execution mode, the simulation's model versions, the random-signal control if it was run) and applies a stated rule. A check that cannot be answered from the record is ``unknown``, not a pass.
There is no combined score: a strategy with ten passes and one fail is not a nine-tenths strategy, and the list says which line is the problem.
"""

from __future__ import annotations

VERSION = "evidence-v1"
STATUSES = ("pass", "warn", "fail", "unknown")


def _c(cid: str, title: str, status: str, detail: str, rule: str, source: str, value=None) -> dict:
    assert status in STATUSES
    return {"id": cid, "title": title, "status": status, "detail": detail, "rule": rule, "source": source, "value": value}


def build(report: dict, controls: dict | None = None) -> dict:
    cfg = report.get("config") or {}
    checks = []

    audit = report.get("audit")
    if audit:
        n = len(audit.get("features") or {})
        stale = audit.get("stale") or {}
        checks.append(_c("features_causal", "Features use only what was known at the time", "pass", f"{n} features, each with a declared window checked by the feature-dependency audit; no backward fill, no imputation in the feature layer.",
                         "Pass when the dataset audit is present (the build refuses a feature that reads past its declared window).", "report.audit", n))
        runs = (stale or {}).get("assets") if isinstance(stale, dict) else None
        if runs:
            checks[-1]["detail"] += f" {len(runs)} asset(s) had runs of repeated prices ({stale.get('rule', '')}; longest {stale.get('longest_run')} sessions): see the audit."
        else:
            checks[-1]["detail"] += " No run of repeated prices was found."
    else:
        checks.append(_c("features_causal", "Features use only what was known at the time", "unknown", "This run has no recorded dataset audit.", "Unknown without report.audit.", "report.audit"))

    folds = report.get("folds") or {}
    ends = [f for f in (folds.get("first"), folds.get("last")) if isinstance(f, dict) and f.get("fit_cutoff") and f.get("label_end_max")]
    if ends:
        bad = [f for f in ends if str(f["label_end_max"]) > str(f["fit_cutoff"])]
        rule = (folds.get("first") or {}).get("purge_rule", "")
        checks.append(_c("oos_lineage", "Every prediction came from a model that had not seen its date", "fail" if bad else "pass",
                         (f"{len(bad)} of the recorded refits trained on labels that reached past their fitting day." if bad else
                          f"{folds.get('n', '?')} refits ({folds.get('n_skipped', 0)} skipped); the first and last record a fitting day that the labels do not run past" + (f" ({rule})." if rule else ".")),
                         "Fail if a recorded refit's label_end_max is after its fit_cutoff. The report keeps the first and last refit; the training result checks every refit when the run is made (oos_violations).",
                         "report.folds", len(bad)))
    else:
        checks.append(_c("oos_lineage", "Every prediction came from a model that had not seen its date", "unknown", "The refits' fitting days were not recorded for this run.", "Unknown without per-fold lineage.", "report.folds"))

    hold = (report.get("holdout") or {}).get("status", "unrecorded")
    hs = {"untouched": "pass", "protected": "pass", "development": "pass", "exposed_exploratory": "warn", "unrecorded": "unknown"}.get(hold, "warn")
    checks.append(_c("holdout", "The final test history was kept out of the choices", hs, (report.get("holdout") or {}).get("note") or f"Holdout status: {hold}.",
                     "Pass for development or protected runs; warn when the protected segments were looked at and recorded as an exposure; unknown when no ledger was kept.", "report.holdout", hold))

    over = report.get("overfit") or {}
    ntr, dsp = over.get("n_trials"), over.get("deflated_sharpe_probability")
    if dsp is not None:
        st = "pass" if dsp >= 0.95 else ("warn" if dsp >= 0.5 else "fail")
        checks.append(_c("multiple_testing", "The result survives the number of things tried", st, f"The validation Sharpe ratio, deflated for {ntr} trial(s) on this dataset, beats what no-skill search would find with probability {dsp:.0%}.",
                         "Pass at 95% or more, warn from 50%, fail below.", "report.overfit.deflated_sharpe_probability", dsp))
    seg = (over.get("sharpe") or {})
    if seg:
        v, t, f = seg.get("validation"), seg.get("test"), seg.get("forward")
        neg = [k for k, x in (("test", t), ("forward", f)) if x is not None and x <= 0]
        weak = t is not None and v is not None and v > 0 and t < 0.5 * v
        st = "fail" if neg else ("warn" if weak else "pass")
        det = ", ".join(f"{k} {x:.2f}" for k, x in seg.items() if x is not None)
        checks.append(_c("out_of_sample", "It kept working on data it was not tuned on", st, f"Net Sharpe by segment: {det}.",
                         "Fail if the test or forward Sharpe is not positive; warn if the test Sharpe is under half the validation Sharpe.", "report.overfit.sharpe", {k: x for k, x in seg.items()}))

    rob = report.get("robustness") or {}
    if rob.get("tests"):
        st = "fail" if rob.get("n_fail") else ("warn" if rob.get("n_warn") else "pass")
        checks.append(_c("robustness", "It survives higher costs, delays, noise and other starts", st, f"{rob.get('n_pass', 0)} passed, {rob.get('n_warn', 0)} warned, {rob.get('n_fail', 0)} failed of {rob.get('n_scored', 0)} scored tests.",
                         "Fail if any scored test failed, warn if any warned.", "report.robustness", rob.get("score")))
    else:
        checks.append(_c("robustness", "It survives higher costs, delays, noise and other starts", "unknown", "The stress tests were not run for this experiment (turn on robustness).", "Unknown unless the tests ran.", "report.robustness"))

    ex = report.get("execution") or {}
    realistic = str(ex.get("mode", "")).startswith("realistic")
    checks.append(_c("execution", "Costs are modelled at the order level", "pass" if realistic else "warn",
                     "Realistic cash-equity execution: orders, fills, participation limits and a ledger." if realistic else "Vectorized engine: a linear cost on traded weight and a one-day lag; no order lifecycle, partial fills or capacity.",
                     "Pass for the realistic mode; warn for the vectorized engine.", "report.execution.mode", ex.get("mode")))

    sig = (cfg.get("signal") or {})
    net, method = sig.get("net"), sig.get("method")
    trading = report.get("trading") or {}
    if trading:
        a = trading.get("analytics") or {}
        n = a.get("avg_net_exposure")
        comparable = n is not None and n > 0.8
        checks.append(_c("benchmark_match", "The benchmark is the same kind of bet", "pass" if comparable else "warn",
                         f"Average net exposure {n:.0%}; the equal-weight benchmark is fully long." + ("" if comparable else " A book that holds cash or shorts is not comparable with it: read the beta-matched figures."),
                         "Pass when the average net exposure is above 80% (comparable with a long-only benchmark); warn otherwise.", "report.trading.analytics.avg_net_exposure", n))
    elif method in ("top_n_long_short", "regime_filtered") and net is not None and abs(net) < 0.2:
        checks.append(_c("benchmark_match", "The benchmark is the same kind of bet", "warn", "A dollar-neutral book is compared with a fully long equal-weight benchmark: they hold different bets, so the active return mixes skill with the market's direction.",
                         "Warn for a market-neutral rule against the long-only equal-weight benchmark.", "config.signal"))

    if controls:
        s = controls.get("share_of_random_at_least_as_good")
        ok = controls.get("replay_check", {}).get("reproduces", True)
        if s is not None:
            st = "pass" if s <= 0.05 else ("warn" if s <= 0.25 else "fail")
            checks.append(_c("random_signals", "It beats random signals at the same pace of trading", st if ok else "warn", f"{s:.0%} of {controls.get('draws')} random-signal draws did at least as well." + ("" if ok else " The replay did not reproduce the saved run."),
                             "Pass at 5% or less, warn to 25%, fail above.", "controls.json", s))
    else:
        checks.append(_c("random_signals", "It beats random signals at the same pace of trading", "unknown", "The random-signal control has not been run for this experiment (Strategy Explorer).", "Unknown until the control is run.", "controls.json"))

    sim = report.get("simulation")
    if sim:
        v = sim.get("oos_violations") or []
        comp = (sim.get("comparison") or {})
        checks.append(_c("simulation", "The retraining loop was out of sample and its choices were made on prior data", "fail" if v else "pass",
                         f"{len(sim.get('model_versions') or [])} model versions; the policy was chosen at each refit from {sim.get('selection', {}).get('n_candidates')} candidates using only earlier days."
                         + (f" Scheduled Sharpe {comp.get('scheduled', {}).get('sharpe', float('nan')):.2f}." if comp.get("scheduled") else ""),
                         "Fail if the framework found a prediction made by a model that had seen its date.", "report.simulation", len(v)))

    rep = report.get("reproducibility") or {}
    checks.append(_c("reproducible", "The run can be reproduced", "pass" if (report.get("config_hash") and report.get("dataset_key")) else "unknown",
                     f"Configuration hash {report.get('config_hash')}, dataset key {report.get('dataset_key')}" + (f", seed {cfg.get('seed')}" if cfg.get("seed") is not None else "") + ".",
                     "Pass when the configuration hash and the dataset key are recorded.", "report.config_hash", rep.get("verdict") if isinstance(rep, dict) else None))

    prov = (report.get("provenance") or {}).get("source")
    if prov:
        st = "fail" if prov == "simulated" else ("warn" if prov == "user" else "pass")
        checks.append(_c("data_source", "The data are real and traceable", st, f"Data source: {prov}." + (" Synthetic data cannot say anything about markets." if prov == "simulated" else ""), "Fail for simulated data; warn for user-supplied data (not verified here); pass for the platform's own.",
                         "report.provenance.source", prov))

    warns = report.get("warnings") or []
    checks.append(_c("warnings", "The run raised no warnings", "warn" if warns else "pass", "; ".join(str(w) for w in warns[:4]) if warns else "No warnings were recorded.", "Warn if the run recorded any warning.", "report.warnings", len(warns)))

    counts = {s: sum(1 for c in checks if c["status"] == s) for s in STATUSES}
    return {"version": VERSION, "id": report.get("id"), "name": report.get("name"), "checks": checks, "counts": counts,
            "reading": "Each line applies the stated rule to a number the run recorded. There is no combined score: a fail is the line to read first, and an unknown is something nobody has checked yet, not something that is fine."}
