"""``quant ml-advanced``: the optional advanced modules of Release 8 from the command line (spec v3, P3-01, P3-02, P3-10, P3-11, P3-12).

One command, one action word, the same service functions the API calls (``src.ml.advanced`` and the modules behind it), so the two doors enforce the same rules and say the same things. ``--json`` prints
the full machine-readable result. A decision (an approval, a rejection, a recovery, a registration, a look) needs ``--actor``; the numerical modules have no command (see ``catalog``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from . import advanced as adv
from .registry import Registry, RegistryError

ACTIONS = ("catalog", "drift", "monitor", "retrain-status", "retrain-trigger", "retrain-approve", "retrain-reject", "retrain-recover", "champion-show", "champion-register", "champion-commit", "champion-look",
           "report-build", "report-show", "report-verify")


def _dump(out) -> None:
    print(json.dumps(out, indent=1, allow_nan=False, default=str))


def _num(v, places: int = 3) -> str:
    return "n/a" if v is None or (isinstance(v, float) and v != v) else f"{v:.{places}f}"


def _need(args, *names) -> None:
    missing = [n for n in names if not getattr(args, n, None)]
    if missing:
        raise ValueError("this action needs " + ", ".join("--" + n.replace("_", "-") for n in missing))


def _json_file(path: str) -> dict:
    try:
        raw = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read {path} as JSON: {exc}") from None
    if not isinstance(raw, dict):
        raise ValueError(f"{path} holds a JSON object")
    return raw


def _catalog(args) -> int:
    root = Path(args.registry)
    out = adv.catalog(Registry(args.registry) if (root / "registry.sqlite").exists() else None)
    if args.json:
        _dump(out)
        return 0
    print(f"{'id':<6} {'module':<42} {'reached by':<10} {'available':<10} verification")
    for m in out["modules"]:
        v = m["verification"]
        print(f"{m['id']:<6} {m['name']:<42} {'+'.join(m['surface']):<10} {('yes' if m['available'] else 'no'):<10} {('-' if v is None else v['status'] + (' (stale)' if v['stale'] else ''))}")
    print(out["note"])
    return 0


def _drift(args, registry) -> int:
    from .drift import DriftSpec

    _need(args, "target")
    spec = DriftSpec(reference_days=args.reference_days, recent_days=args.recent_days)
    out = adv.run_drift(registry, args.target, spec=spec, as_of=args.as_of, actor="cli")
    if args.json:
        _dump(out)
        return 0
    if out.get("eligibility") != "applied":
        print(f"{args.target}: {out.get('reason')}")
        return 0
    inp, perf = out["inputs"], out["performance"]
    print(f"{args.target} ({out['model']}), as of {out['as_of']}: reference {inp['reference']['first']} to {inp['reference']['last']} ({inp['reference']['dates']} dates, fixed at the fit cutoff {inp['reference']['fixed_at']}), "
          f"recent {inp['recent']['first']} to {inp['recent']['last']}; features {inp['counts']}")
    print(f"  {'feature':<28} {'state':<13} {'PSI':>7} {'KS p (adj)':>11} {'n ref':>7} {'n recent':>9}")
    for r in inp["features"][:12]:
        print(f"  {r['feature']:<28} {r['state']:<13} {_num(r['psi']):>7} {_num(r.get('ks_p_adjusted')):>11} {r['n_reference']:>7} {r['n_recent']:>9}")
    print(f"  settled-IC drift: {perf['state']}: {perf.get('reason')}")
    lin = out["lineage"]
    print(f"  look {lin['looks_so_far']} (record {out['record_seq']}); new alerts {lin['new'] or 'none'}, persisting {lin['persisting'] or 'none'}, cleared {lin['cleared'] or 'none'}")
    print("  " + out["note"])
    return 0


def _monitor(args, registry) -> int:
    _need(args, "target")
    out = adv.run_monitor(registry, args.target, as_of=args.as_of, reference_end=args.reference_end, actor="cli")
    if args.json:
        _dump(out)
        return 0
    if out["eligibility"] != "applied":
        print(f"{args.target}: {out['eligibility']}: {out['reason']}")
        return 0
    print(f"{args.target} ({out['model']}), as of {out['as_of']}, reference ends {out['reference_end']}: overall {out['overall']['state']}: {out['overall']['reason']}")
    for c in out["checks"]:
        lvl = f" [{c['level']}]" if c.get("level") else ""
        print(f"  {c['check']:<13} {c['state']:<14}{lvl:<11} value {_num(c.get('value')):>7}  {c.get('reason')}")
    print("  " + out["note"])
    return 0


def _retrain(args, registry) -> int:
    from . import retrain as rt

    _need(args, "target")
    act = args.what.split("-", 1)[1]
    if act == "status":
        st, ev = rt.status(registry, args.target), rt.events(registry, args.target)
        out = {"subject": args.target, "state": st, "events": ev[-20:], "events_total": len(ev)}
    elif act == "trigger":
        policy = adv.retrain_policy(_json_file(args.policy) if args.policy else None)
        out = adv.propose_retraining(registry, args.target, kind=args.kind, policy=policy, actor=args.actor or "cli")
    elif act in ("approve", "reject"):
        _need(args, "proposal", "actor", "note")
        out = (rt.approve if act == "approve" else rt.reject)(registry, args.target, args.proposal, actor=args.actor, note=args.note)
    else:
        _need(args, "actor", "note")
        out = rt.recover(registry, args.target, actor=args.actor, note=args.note)
    if args.json:
        _dump(out)
        return 0
    if act == "status":
        st = out["state"]
        print(f"{args.target}: open proposal {st['open_proposal'] or 'none'}, consecutive failures {st['consecutive_failures']}, {'SUSPENDED' if st['suspended'] else 'not suspended'}; {out['events_total']} events")
        for e in out["events"]:
            print(f"  {e['seq']:>5} {e['event']:<10} by {e['actor']:<10} {e.get('reason') or e.get('note') or e.get('trigger') or ''}")
    else:
        print(f"{args.target}: {out.get('outcome')}" + (f" (proposal {out['proposal']})" if out.get("proposal") else "") + (f": {out['reason']}" if out.get("reason") else ""))
    return 0


def _champion(args, registry) -> int:
    from . import champion as ch

    act = args.what.split("-", 1)[1]
    if act == "register":
        _need(args, "spec", "actor")
        out = ch.register(registry, actor=args.actor, **_json_file(args.spec))
    elif act == "commit":
        _need(args, "target", "spec", "actor")
        spec = _json_file(args.spec)
        extra = sorted(set(spec) - {"decision_ts", "forecasts", "data_cutoff", "fit_cutoffs", "model_created", "weights", "label_start"})
        if extra:
            raise ValueError(f"a commitment spec has decision_ts, forecasts, data_cutoff, fit_cutoffs, model_created and optionally weights and label_start, not {extra}")
        out = ch.commit_pair(registry, args.target, actor=args.actor, **spec)
    elif act == "show":
        _need(args, "target")
        out = ch.summary(registry, args.target)
    else:
        _need(args, "target", "actor")
        out = ch.look(registry, args.target, actor=args.actor)
    if args.json:
        _dump(out)
        return 0
    if act == "commit":
        print(f"{out['pair_id']}: decision {out['decision_ts']}: champion {out['champion']['status']} ({out['champion'].get('classification', 'unchanged')}), challenger {out['challenger']['status']} "
              f"({out['challenger'].get('classification', 'unchanged')}); a replay never counts, and a commitment is never changed")
    elif act == "look":
        print(f"{out['pair_id']}: look {out['look']} of {out['of']} on {out['pairs_used']} settled pairs at level {_num(out['level'], 4)}: {out['verdict']}: {out['reason']}")
    else:
        print(f"{out['pair_id']}: {out['champion']} (champion) against {out['challenger']} (challenger), horizon {out['horizon']}, looks {out['looks']} at levels {[round(x, 4) for x in out['levels']]} ({out['rule']}); "
              f"looks left {out['looks_remaining']}" + (f"; settled pairs {out['settled_pairs']}, verdict {out['verdict']}" if "settled_pairs" in out else f"; {out['status']}"))
    return 0


def _report(args, registry) -> int:
    from . import repro_report as rr

    act = args.what.split("-", 1)[1]
    if act == "build":
        _need(args, "target")
        out = rr.build(registry, args.target, actor=args.actor or "cli")
    else:
        _need(args, "target")
        rec = rr.get(registry, args.target)
        if act == "show":
            out = {"report_id": args.target, "report_hash": rec["report_hash"], "content": rec["content"]}
            if args.markdown and not args.json:
                print(rr.to_markdown(rec["content"]))
                return 0
        else:
            out = {"report_id": args.target, "reproduction": rr.reproduce(registry, args.target), "inputs_now": rr.verify_inputs(registry, args.target)}
    if args.json:
        _dump(out)
        return 0
    if act == "verify":
        r, n = out["reproduction"], out["inputs_now"]
        print(f"{args.target}: {'reproduced' if r['reproduced'] else 'NOT reproduced'} from its stored snapshot (hash {r['stored_hash']}); inputs {'have changed since' if n['stale'] else 'are unchanged'}")
        for c in n["changed"]:
            print(f"  changed: {c['input']}")
    elif act == "show":
        print(f"{args.target} (hash {out['report_hash']}): use --markdown for the readable form or --json for the full content")
    else:
        print(f"{out['report_id']}: {out['status']} (hash {out['report_hash']})")
    return 0


def cmd_advanced(args, config) -> int:
    try:
        if args.what == "catalog":
            return _catalog(args)
        registry = Registry(args.registry)
        if args.what == "drift":
            return _drift(args, registry)
        if args.what == "monitor":
            return _monitor(args, registry)
        if args.what.startswith("retrain-"):
            return _retrain(args, registry)
        if args.what.startswith("champion-"):
            return _champion(args, registry)
        return _report(args, registry)
    except (ValueError, KeyError, TypeError, RegistryError, FileNotFoundError) as exc:
        print(f"error: {exc.args[0] if isinstance(exc, KeyError) and exc.args else exc}", file=sys.stderr)
        return 2


def register(sub) -> None:
    p = sub.add_parser("ml-advanced", help="the optional advanced modules (drift, retraining proposals, champion/challenger, uncertainty monitor, reproducible reports) and the catalogue of all twelve")
    p.add_argument("what", choices=ACTIONS, help="catalog | drift EXP | monitor EXP | retrain-status|trigger|approve|reject|recover EXP | champion-show|register|commit|look PAIR | report-build EXP | report-show|verify REPORT")
    p.add_argument("target", nargs="?", help="the experiment, pair or report the action is about")
    p.add_argument("--registry", default="artifacts/ml", help="where experiments and the registry live (default artifacts/ml)")
    p.add_argument("--as-of", dest="as_of"); p.add_argument("--reference-end", dest="reference_end"); p.add_argument("--reference-days", dest="reference_days", type=int, default=504)
    p.add_argument("--recent-days", dest="recent_days", type=int, default=63)
    p.add_argument("--kind", default="manual", help="a retraining trigger: drift_alert, performance_alert, schedule or manual"); p.add_argument("--policy", help="JSON file with a retraining policy")
    p.add_argument("--proposal", type=int); p.add_argument("--actor"); p.add_argument("--note"); p.add_argument("--spec", help="JSON file with a champion/challenger declaration, or with the forecasts of one decision (champion-commit)")
    p.add_argument("--markdown", action="store_true"); p.add_argument("--json", action="store_true"); p.set_defaults(func=cmd_advanced)
