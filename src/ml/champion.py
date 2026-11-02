"""Champion/challenger evidence with a predeclared schedule of looks (spec v3, Release 8, P3-10). Optional; it adds nothing to the base workflow.

A challenger is only comparable with the champion if the comparison was fixed before the observations it is judged on. A **pair** is registered with: the two strategy versions, the universe and horizon, the
assumptions (execution, risk and the data cutoff) they share, a **schedule of looks** (the numbers of settled decisions at which the comparison may be made), a total alpha and the spending rule that
splits it over the looks. The server's clock stamps the registration and nothing earlier can belong to the pair.

* **Both forecasts are committed together**, for the same decision timestamp, universe, horizon and data cutoff, through the prospective ledger (immutable commitments, replays never counted). A decision
  that predates the registration is refused: prospective evidence is never backfilled.
* **Looks are spent, not repeated.** A look is allowed only once the schedule's number of settled decision pairs exists, is taken once, and tests the paired difference in the daily rank IC (challenger
  minus champion) with a Newey-West error at the level of that look. The levels are the increments of an alpha-spending function (O'Brien-Fleming type by default, Pocock type or equal shares as
  alternatives); by the union bound the chance of a false claim over all the looks is at most the total alpha. The correlation between looks is not exploited, so this is conservative.
* **No look, no verdict.** Looking at the pair outside the schedule is a refusal, and the result of a look is a statement about that look (``challenger_better``, ``no_difference`` or
  ``champion_better``), never a promotion: nothing here changes a stage.
"""

from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import pandas as pd
from scipy import stats

from . import diagnostics as dx
from . import inference as inf
from . import prospective as pr

VERSION = "champion-v1"
KIND = "champion_challenger"
SPENDING = ("obrien_fleming", "pocock", "equal")


class ChampionError(ValueError):
    """A request the champion/challenger procedure refuses."""


def spending(rule: str, fractions, alpha: float) -> list[float]:
    """The cumulative alpha spent by each information fraction ``t`` in (0, 1]: O'Brien-Fleming type ``2 - 2 Phi(z / sqrt(t))``, Pocock type ``alpha ln(1 + (e - 1) t)``, or equal shares."""
    t = np.asarray(fractions, dtype=float)
    if rule == "obrien_fleming":
        z = stats.norm.ppf(1 - alpha / 2)
        return [float(2 - 2 * stats.norm.cdf(z / np.sqrt(x))) for x in t]
    if rule == "pocock":
        return [float(alpha * np.log(1 + (np.e - 1) * x)) for x in t]
    if rule == "equal":
        return [float(alpha * x) for x in t]
    raise ChampionError(f"the spending rule is one of {list(SPENDING)}")


def levels(rule: str, looks: list[int], alpha: float) -> list[float]:
    """The nominal level of each look: the increment of the cumulative spending between consecutive information fractions (the last look is at fraction 1)."""
    cum = spending(rule, [n / looks[-1] for n in looks], alpha)
    return [float(c - p) for c, p in zip(cum, [0.0] + cum[:-1])]


def _epoch(now: float | None) -> float:
    return time.time() if now is None else float(now)


def _records(registry, pair_id: str) -> list[dict]:
    with registry._conn() as c:                                                                  # noqa: SLF001
        rows = c.execute("SELECT seq, content, created FROM analysis_records WHERE kind=? AND subject_id=? ORDER BY seq", (KIND, pair_id)).fetchall()
    return [{"seq": r["seq"], "created": r["created"], **json.loads(r["content"])} for r in rows]


def get(registry, pair_id: str) -> dict:
    """The pair as registered, with the looks taken so far."""
    recs = _records(registry, pair_id)
    if not recs or recs[0].get("event") != "registered":
        raise ChampionError(f"unknown champion/challenger pair '{pair_id}'")
    reg = recs[0]
    taken = [r for r in recs if r.get("event") == "look"]
    return {**reg, "pair_id": pair_id, "looks_taken": taken, "looks_remaining": len(reg["looks"]) - len(taken)}


SHARED = ("execution", "backtest")                 # the settings of a strategy version that a fair comparison holds equal (with the risk limits); the model, signal, dataset and split are what is under test


def assumption_check(registry, champion: str, challenger: str) -> dict:
    """Compare what the two strategy versions recorded about how they are executed and limited. When both ids are strategy versions registered with governance, their execution and cost/schedule settings and
    their risk limits must be identical (the digests are compared); what differs among the dataset, model, split and signal is listed as the difference under test. When either is not a registered version,
    nothing can be compared: the result says so, and the pair's shared assumptions stay a declaration."""
    from . import governance as gv

    found = {}
    for role, vid in (("champion", champion), ("challenger", challenger)):
        try:
            found[role] = gv.get_version(registry, vid)["content"]
        except gv.GovernanceError:
            found[role] = None
    missing = [r for r, v in found.items() if v is None]
    if missing:
        who = " and ".join(f"the {r}" for r in missing)
        what = "is not a registered strategy version" if len(missing) == 1 else "are not registered strategy versions"
        return {"checked": False, "reason": f"{who} {what}: the shared assumptions are declared and frozen with the pair, and nothing was compared with recorded settings"}
    a, b = found["champion"], found["challenger"]
    differing = [k for k in SHARED if a["settings"].get(k) != b["settings"].get(k)] + (["risk_limits"] if a.get("risk_limits") != b.get("risk_limits") else [])
    return {"checked": True, "compared": [*SHARED, "risk_limits"], "differing": differing, "under_test": [k for k in ("dataset", "model", "split", "signal") if a["settings"].get(k) != b["settings"].get(k)]}


def register(registry, *, champion: str, challenger: str, universe_version: str, horizon: int, looks: list[int], alpha: float = 0.05, rule: str = "obrien_fleming", assumptions: dict | None = None,
             data_cutoff=None, actor: str, now: float | None = None) -> dict:
    """Freeze a pair before the observations it will be judged on. The pair id is a hash of everything declared, so the same declaration is the same pair."""
    if not actor:
        raise ChampionError("a registration records who made it")
    if champion == challenger:
        raise ChampionError("the champion and the challenger are different strategy versions")
    ls = [int(n) for n in looks]
    if len(ls) < 1 or ls != sorted(set(ls)) or ls[0] < 10:
        raise ChampionError("looks are a strictly increasing list of numbers of settled decisions, the first at least 10")
    if not 0 < alpha < 0.5 or rule not in SPENDING or int(horizon) < 1:
        raise ChampionError(f"alpha in (0, 0.5), horizon >= 1 and a spending rule from {list(SPENDING)}")
    body = {"champion": champion, "challenger": challenger, "universe_version": universe_version, "horizon": int(horizon), "looks": ls, "alpha": float(alpha), "rule": rule,
            "assumptions": assumptions or {}, "data_cutoff": None if data_cutoff is None else str(pd.Timestamp(data_cutoff))}
    pair_id = "cc-" + hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:14]
    if _records(registry, pair_id):
        return {"pair_id": pair_id, "status": "unchanged", **get(registry, pair_id)}
    check = assumption_check(registry, champion, challenger)
    if check["checked"] and check["differing"]:
        raise ChampionError(f"the champion and the challenger do not share their recorded {' and '.join(check['differing'])} settings: a comparison is made under identical execution, cost and risk assumptions, "
                            "so register versions that share them (the model, signal, dataset and split are what may differ)")
    t = _epoch(now)
    content = {"event": "registered", "version": VERSION, **body, "assumptions_check": check, "levels": levels(rule, ls, alpha), "registered_epoch": t, "registered_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t)),
               "assumptions_hash": hashlib.sha256(json.dumps(body["assumptions"], sort_keys=True).encode()).hexdigest()[:16], "actor": actor,
               "note": "the schedule, the spending and the shared assumptions were fixed before any observation they are judged on"}
    dx.store(registry, KIND, pair_id, content, actor=actor)
    return {"pair_id": pair_id, "status": "registered", **get(registry, pair_id)}


def commit_pair(registry, pair_id: str, *, decision_ts, forecasts: dict, data_cutoff, fit_cutoffs: dict, model_created: dict, weights: dict | None = None, label_start=None, actor: str, now: float | None = None) -> dict:
    """Commit the champion's and the challenger's forecasts for one decision timestamp, together. ``forecasts`` maps ``champion`` and ``challenger`` to ``{asset: forecast}``; ``fit_cutoffs`` and
    ``model_created`` are given for each. The data cutoff is one value for both, and must equal the pair's if it declared one."""
    pair = get(registry, pair_id)
    if set(forecasts) != {"champion", "challenger"}:
        raise ChampionError("both forecasts are committed together: forecasts has exactly 'champion' and 'challenger'")
    dts = pd.Timestamp(decision_ts)
    if pair["data_cutoff"] is not None and pd.Timestamp(data_cutoff) != pd.Timestamp(pair["data_cutoff"]):
        raise ChampionError(f"the data cutoff {pd.Timestamp(data_cutoff)} differs from the pair's declared {pair['data_cutoff']}: both models are judged on the same information")
    if dts.timestamp() < pair["registered_epoch"]:
        raise ChampionError(f"the decision {dts} predates the registration ({pair['registered_at']}): prospective evidence is never backfilled")
    out = {}
    for role in ("champion", "challenger"):
        out[role] = pr.commit(registry, strategy_version=pair[role], decision_ts=dts, universe_version=pair["universe_version"], horizon=pair["horizon"], forecasts=forecasts[role],
                              weights=(weights or {}).get(role), data_cutoff=data_cutoff, fit_cutoff=fit_cutoffs[role], model_created=model_created[role], label_start=label_start, actor=actor, now=now)
    return {"pair_id": pair_id, "decision_ts": str(dts), **out}


def paired_series(registry, pair: dict) -> pd.DataFrame:
    """The daily rank IC of each strategy on the decisions both have settled as prospective commitments, aligned by decision timestamp (first-observed outcomes)."""
    ic = {}
    for role in ("champion", "challenger"):
        rows = pr._rows(registry, pair[role], "prospective")                                       # noqa: SLF001
        sett = pr._settlements(registry, [r["commitment_id"] for r in rows])                      # noqa: SLF001
        ic[role] = {r["decision_ts"]: pr._ic(json.loads(r["forecasts"]), json.loads(sett[r["commitment_id"]][0]["outcomes"])) for r in rows if sett[r["commitment_id"]]}   # noqa: SLF001
    both = pd.DataFrame(ic).dropna()
    both.index = pd.to_datetime(both.index)
    return both.sort_index()


def look(registry, pair_id: str, *, actor: str, now: float | None = None) -> dict:
    """Take the next predeclared look, if the schedule allows it. The paired difference of the daily rank IC is tested at the look's own level; the result is recorded and the look is spent."""
    pair = get(registry, pair_id)
    if pair["looks_remaining"] <= 0:
        raise ChampionError("no unspent look: the schedule is exhausted, and a comparison outside it is not part of this design")
    k = len(pair["looks_taken"])
    need = pair["looks"][k]
    series = paired_series(registry, pair)
    if len(series) < need:
        raise ChampionError(f"look {k + 1} of {len(pair['looks'])} needs {need} settled decision pairs; there are {len(series)}")
    use = series.iloc[:need]                                                                       # exactly the declared number: more data arriving earlier does not move the look
    diff = use["challenger"] - use["champion"]
    level = pair["levels"][k]
    res = inf.hac_mean(diff, horizon=int(pair["horizon"]), level=1 - level, label="challenger minus champion daily rank IC", scope=f"look {k + 1} of {len(pair['looks'])}, {pair['rule']} spending")
    d = res.as_dict()
    if d["eligibility"] != "applied" or d.get("p_value") is None:
        verdict, why = "no_difference", "the difference could not be estimated at this look (" + str(d.get("reason") or d["eligibility"]) + ")"
    elif d["p_value"] < level and d["value"] > 0:
        verdict, why = "challenger_better", f"the challenger's IC is higher by {d['value']:+.4f}, p = {d['p_value']:.3g} below this look's level {level:.4g}"
    elif d["p_value"] < level and d["value"] < 0:
        verdict, why = "champion_better", f"the champion's IC is higher by {-d['value']:.4f}, p = {d['p_value']:.3g} below this look's level {level:.4g}"
    else:
        verdict, why = "no_difference", f"p = {d['p_value']:.3g} is not below this look's level {level:.4g}"
    content = {"event": "look", "version": VERSION, "look": k + 1, "of": len(pair["looks"]), "pairs_used": int(need), "level": level, "difference": d, "verdict": verdict, "reason": why,
               "champion_ic": float(use["champion"].mean()), "challenger_ic": float(use["challenger"].mean()), "final": k + 1 == len(pair["looks"]), "at_epoch": _epoch(now), "actor": actor,
               "promotes": False, "note": "a look is a statement about the evidence at that look; nothing here changes a stage"}
    seq = dx.store(registry, KIND, pair_id, content, actor=actor)
    return {"pair_id": pair_id, "seq": seq, **content}


def summary(registry, pair_id: str) -> dict:
    """The pair, the evidence so far (settled pairs, spent and unspent looks) and, once the schedule is exhausted, the last look's verdict."""
    pair = get(registry, pair_id)
    series = paired_series(registry, pair)
    nxt = pair["looks"][len(pair["looks_taken"])] if pair["looks_remaining"] > 0 else None
    return {**pair, "settled_pairs": int(len(series)), "next_look_needs": nxt, "next_look_available": bool(nxt is not None and len(series) >= nxt),
            "verdict": (pair["looks_taken"][-1]["verdict"] if pair["looks_remaining"] == 0 and pair["looks_taken"] else "pending"),
            "counts": {role: pr.counts(registry, pair[role]) for role in ("champion", "challenger")}}
