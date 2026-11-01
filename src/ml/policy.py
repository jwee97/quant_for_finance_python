"""The versioned validation policy (spec v3, Release 5, section 8.1).

The platform implies no universal Sharpe, deflated-Sharpe, PBO, p-value or paper-trading-duration threshold. What counts as enough evidence for a strategy is a *decision*, made by a named person BEFORE
the evaluation and recorded with a version, so that no threshold can be chosen after seeing the result. A ``ValidationPolicy`` is that declaration:

* each criterion is optional (``None`` means this policy does not require it) and is a plain number or setting, never a hidden default;
* the policy has a content hash; changing any threshold makes a new policy with a new hash (and a stale gate decision);
* ``example_policy()`` is a documented, conservative illustration and is labelled ``illustrative``: it is *not* a recommendation and Gate B/C decisions refuse it until someone has adopted a policy
  (``adopt``) with their name and a note. Adoption is an append-only record; the latest adoption is the policy in force;
* algorithm acceptance (the ``ml-verify`` requirements) uses engineering tolerances and never these thresholds: a correct algorithm passes Gate A whatever the policy says about profit.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field

VERSION = "policy-v1"

CRITERIA = {
    # evidence length and the segment it is judged on
    "segment": ("choice", ("validation", "test", "forward"), "the segment the thresholds are judged on"),
    "min_segment_days": ("int", 20, 100_000, "fewer days than this on the segment is insufficient evidence"),
    # dependence-aware uncertainty of the Sharpe ratio
    "interval_method": ("choice", ("hac", "bootstrap"), "which dependence-aware interval the policy adopts"),
    "min_sharpe_ci_lower": ("float", -10.0, 10.0, "the lower bound of the adopted annualised Sharpe interval must exceed this"),
    "confidence_level": ("float", 0.5, 0.999, "the level of the intervals"),
    # selection bias
    "min_deflated_probability": ("float", 0.0, 1.0, "the deflated Sharpe probability must reach this"),
    "max_pbo": ("float", 0.0, 1.0, "the probability of backtest overfitting must not exceed this"),
    "max_selection_p": ("float", 0.0, 1.0, "the Romano-Wolf / SPA family-wise p-value must not exceed this"),
    # matched benchmark
    "benchmark_kind": ("choice", ("beta", "net_exposure", "sector", "risk"), "the matching that the incremental return is judged against"),
    "min_active_ci_lower": ("float", -1.0, 1.0, "the lower bound of the active mean return per period against the matched benchmark must exceed this"),
    # regimes, neighbourhood, delay, capacity, calibration
    "min_regime_days": ("int", 20, 100_000, "days a regime needs before it is read"),
    "min_regime_sharpe_upper": ("float", -10.0, 10.0, "no applied causal regime may have an annualised Sharpe interval whose UPPER bound is below this"),
    "neighborhood_min_share_positive": ("float", 0.0, 1.0, "the share of the predeclared neighbourhood with a positive metric must reach this"),
    "delay_sessions": ("int", 1, 20, "the extra decision-to-fill delay that must still work"),
    "delay_min_ratio": ("float", 0.0, 1.0, "the metric at that delay, as a share of the undelayed one, must reach this"),
    "calibration_max_ece": ("float", 0.0, 1.0, "the expected calibration error of the class probabilities must not exceed this"),
    "capacity_capital": ("float", 1.0, 1e13, "the capital at which the realistic result is judged"),
    "capacity_min_net_ratio": ("float", 0.0, 1.0, "the net metric at that capital, as a share of the smallest level, must reach this"),
    # prospective evidence (Gate C)
    "prospective_min_days": ("int", 1, 100_000, "distinct settled prospective days required"),
    "prospective_min_settled": ("int", 1, 10_000_000, "settled prospective forecasts required"),
    # operations (Gate C)
    "monitoring_max_age_days": ("int", 1, 365, "the latest monitoring check of the strategy version must be at most this many days old"),
}


class PolicyError(ValueError):
    pass


@dataclass(frozen=True)
class ValidationPolicy:
    name: str
    version: str
    criteria: dict = field(default_factory=dict)
    illustrative: bool = False
    description: str = ""

    def __post_init__(self):
        if not self.name or not self.version:
            raise PolicyError("a policy has a name and a version")
        for k, v in self.criteria.items():
            if k not in CRITERIA:
                raise PolicyError(f"unknown criterion '{k}'; the criteria are {sorted(CRITERIA)}")
            if v is None:
                continue
            kind, a, b = CRITERIA[k][0], CRITERIA[k][1], CRITERIA[k][2] if len(CRITERIA[k]) > 3 else None
            if kind == "choice":
                if v not in a:
                    raise PolicyError(f"{k} must be one of {a}")
            elif kind in ("int", "float"):
                if not isinstance(v, (int, float)) or isinstance(v, bool) or not (a <= float(v) <= b):
                    raise PolicyError(f"{k} must be a number in [{a}, {b}]")

    def get(self, key: str, default=None):
        v = self.criteria.get(key)
        return default if v is None else v

    def required(self, key: str) -> bool:
        return self.criteria.get(key) is not None

    def content(self) -> dict:
        return {"schema": VERSION, "name": self.name, "version": self.version, "illustrative": self.illustrative, "description": self.description, "criteria": dict(sorted(self.criteria.items()))}

    def content_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.content(), sort_keys=True, default=str).encode()).hexdigest()[:16]


def from_content(content: dict) -> ValidationPolicy:
    return ValidationPolicy(content["name"], content["version"], dict(content.get("criteria") or {}), bool(content.get("illustrative")), content.get("description", ""))


def example_policy() -> ValidationPolicy:
    """A documented, conservative ILLUSTRATION of a policy: every number below is an example of the kind of statement a policy makes, not a recommendation, and none has been calibrated to any market or
    strategy. Adopt it only if you have read it and mean it; otherwise write your own."""
    return ValidationPolicy(
        "illustrative-conservative", "example-1", illustrative=True,
        description="ILLUSTRATIVE ONLY. Judged on the validation segment; asks for a year of evidence, a dependence-aware Sharpe interval clear of zero, a deflated probability of at least 0.95, a probability of "
                    "overfitting under one half, a family-wise p-value under 0.05, an active return clear of a beta-matched benchmark, no applied causal regime that is clearly losing, a neighbourhood that is mostly "
                    "positive, a result that keeps three quarters of its value with one extra session of delay, and 60 settled prospective days. These thresholds have not been calibrated to anything.",
        criteria={"segment": "validation", "min_segment_days": 252, "interval_method": "hac", "min_sharpe_ci_lower": 0.0, "confidence_level": 0.95, "min_deflated_probability": 0.95, "max_pbo": 0.5,
                  "max_selection_p": 0.05, "benchmark_kind": "beta", "min_active_ci_lower": 0.0, "min_regime_days": 60, "min_regime_sharpe_upper": 0.0, "neighborhood_min_share_positive": 0.6,
                  "delay_sessions": 1, "delay_min_ratio": 0.75, "calibration_max_ece": None, "capacity_capital": None, "capacity_min_net_ratio": None, "prospective_min_days": 60,
                  "prospective_min_settled": 60})


def stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


# ------------------------------------------------------------------------------------------------------------------------------------ the registry side
def register(registry, policy: ValidationPolicy) -> str:
    """Store the policy's content under its hash (idempotent) and return the hash. Registering is not adopting."""
    h = policy.content_hash()
    with registry.transaction() as c:
        if c.execute("SELECT 1 FROM validation_policies WHERE policy_hash=?", (h,)).fetchone() is None:
            c.execute("INSERT INTO validation_policies (policy_hash, name, version, illustrative, content, created) VALUES (?,?,?,?,?,?)",
                      (h, policy.name, policy.version, int(policy.illustrative), json.dumps(policy.content(), sort_keys=True), stamp()))
    return h


def adopt(registry, policy: ValidationPolicy, actor: str, note: str) -> dict:
    """Make ``policy`` the one in force. Needs a named person and a reason; the record is append-only and the latest adoption wins. Adopting an illustrative policy is allowed and is flagged in every
    decision that uses it."""
    if not str(actor or "").strip() or not str(note or "").strip():
        raise PolicyError("adopting a policy needs an actor and a note: it is a decision of a person, not a default")
    h = register(registry, policy)
    with registry.transaction() as c:
        cur = c.execute("INSERT INTO policy_adoptions (policy_hash, actor, note, at) VALUES (?,?,?,?)", (h, str(actor).strip(), str(note).strip(), stamp()))
    return {"policy_hash": h, "seq": int(cur.lastrowid), "actor": actor, "illustrative": policy.illustrative, "name": policy.name, "version": policy.version}


def adopted(registry) -> tuple[ValidationPolicy, dict] | None:
    """``(policy, adoption record)`` for the policy in force, or ``None`` when no policy has been adopted (then no threshold is implied and Gates B and C are blocked)."""
    with registry._conn() as c:                                                                    # noqa: SLF001
        row = c.execute("SELECT a.seq, a.policy_hash, a.actor, a.note, a.at, p.content FROM policy_adoptions a JOIN validation_policies p ON p.policy_hash=a.policy_hash ORDER BY a.seq DESC LIMIT 1").fetchone()
    if row is None:
        return None
    return from_content(json.loads(row["content"])), {"seq": row["seq"], "policy_hash": row["policy_hash"], "actor": row["actor"], "note": row["note"], "at": row["at"]}


def history(registry, limit: int = 50) -> list[dict]:
    with registry._conn() as c:                                                                    # noqa: SLF001
        return [dict(r) for r in c.execute("SELECT a.seq, a.policy_hash, a.actor, a.note, a.at, p.name, p.version, p.illustrative FROM policy_adoptions a JOIN validation_policies p ON p.policy_hash=a.policy_hash "
                                           "ORDER BY a.seq DESC LIMIT ?", (int(limit),))]
