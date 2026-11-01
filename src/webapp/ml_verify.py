"""The API's view of the verification evidence: thin wrappers that turn the verification module's errors into request errors."""

from __future__ import annotations

from ..ml import verification as vf
from .api import ApiError


def validate(requirements) -> None:
    unknown = [r for r in (requirements or []) if r not in vf.REQUIREMENTS]
    if unknown:
        raise ApiError(f"unknown requirement(s) {unknown}; known: {sorted(vf.REQUIREMENTS)}")


def report(registry, requirement_id: str | None) -> dict:
    if requirement_id and requirement_id not in vf.REQUIREMENTS:
        raise ApiError(f"unknown requirement '{requirement_id}'; known: {sorted(vf.REQUIREMENTS)}")
    rows = vf.current(registry, requirement_id) if registry is not None else [{"requirement_id": r, "status": "not_run", "stale": False, "reason": "never run", "created": None} for r in vf.REQUIREMENTS
                                                                               if not requirement_id or r == requirement_id]
    return {"schema": 1, "checks": rows, "statuses": ["pass", "fail", "not_run", "blocked", "not_applicable"],
            "note": "A status is written only by a run of the tests; 'blocked' means tests were skipped (a missing package) and what they check was not verified; 'stale' means the source changed after the run."}


def run(registry, requirements, progress):
    return vf.run_verification(registry, requirements, progress)
