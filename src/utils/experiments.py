"""Experiment registry (spec §45).

Engineering addition. The registry is the project's defence against the most
insidious of the common quant traps (Ch. 22 §22.2.4): silently discarding the
experiments that did not work and reporting only the survivor. Every recorded
experiment carries its hypothesis, data version, parameters, sample windows,
cost assumption, result and an explicit decision -- including REJECT.

Records are appended to ``experiments/registry.jsonl`` (append-only) and
rendered to ``experiments/registry.md`` for reading.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

DECISIONS = ("retain", "reject", "investigate", "record")


def _jsonable(value: Any) -> Any:
    """Convert numpy / pandas scalars and containers into JSON-safe values."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "item") and getattr(value, "shape", ()) == ():
        return _jsonable(value.item())
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "to_dict"):
        try:
            return _jsonable(value.to_dict())
        except Exception:  # pragma: no cover - defensive
            return str(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


@dataclass
class Experiment:
    """One recorded experiment."""

    experiment_id: str
    hypothesis: str
    stage: str = ""
    date: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    data_version: str = ""
    config_fingerprint: str = ""
    parameters: dict = field(default_factory=dict)
    train_period: str = ""
    test_period: str = ""
    cost_bps: float | None = None
    results: dict = field(default_factory=dict)
    decision: str = "record"
    notes: str = ""

    def to_json(self) -> dict:
        payload = asdict(self)
        payload["parameters"] = _jsonable(self.parameters)
        payload["results"] = _jsonable(self.results)
        return payload


class ExperimentRegistry:
    """Append-only experiment log."""

    def __init__(self, path: str | Path | None = None, data_version: str = "", config_fingerprint: str = ""):
        root = Path(__file__).resolve().parents[2]
        self.path = Path(path) if path is not None else root / "experiments" / "registry.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data_version = data_version
        self.config_fingerprint = config_fingerprint

    # -- read -------------------------------------------------------------
    def records(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out

    def next_id(self) -> str:
        used = []
        for record in self.records():
            ident = record.get("experiment_id", "")
            if ident.startswith("EXP-") and ident[4:].isdigit():
                used.append(int(ident[4:]))
        return f"EXP-{(max(used) + 1) if used else 1:03d}"

    # -- write ------------------------------------------------------------
    def log(
        self,
        hypothesis: str,
        *,
        stage: str = "",
        parameters: dict | None = None,
        results: dict | None = None,
        decision: str = "record",
        notes: str = "",
        train_period: str = "",
        test_period: str = "",
        cost_bps: float | None = None,
        experiment_id: str | None = None,
    ) -> Experiment:
        if decision not in DECISIONS:
            raise ValueError(f"decision must be one of {DECISIONS}, got '{decision}'")
        experiment = Experiment(
            experiment_id=experiment_id or self.next_id(),
            hypothesis=hypothesis,
            stage=stage,
            data_version=self.data_version,
            config_fingerprint=self.config_fingerprint,
            parameters=parameters or {},
            results=results or {},
            decision=decision,
            notes=notes,
            train_period=train_period,
            test_period=test_period,
            cost_bps=cost_bps,
        )
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(experiment.to_json(), sort_keys=True) + "\n")
        return experiment

    def reset(self) -> None:
        """Truncate the log. Only used by ``run_all`` when rebuilding from scratch."""
        if self.path.exists():
            os.remove(self.path)

    # -- render -----------------------------------------------------------
    def to_markdown(self, path: str | Path | None = None, records: Iterable[dict] | None = None) -> Path:
        records = list(records if records is not None else self.records())
        out = Path(path) if path is not None else self.path.with_suffix(".md")
        lines = [
            "# Experiment registry",
            "",
            "Append-only log of every experiment run by the pipeline, including the",
            "ones that were rejected. Generated by `src/utils/experiments.py`.",
            "",
            f"Total experiments: **{len(records)}**",
            "",
            "| ID | Stage | Hypothesis | Decision | Headline result |",
            "|----|-------|------------|----------|-----------------|",
        ]
        for record in records:
            results = record.get("results", {}) or {}
            headline = "; ".join(
                f"{k}={v:.4g}" if isinstance(v, (int, float)) and not isinstance(v, bool) else f"{k}={v}"
                for k, v in list(results.items())[:3]
            )
            lines.append(
                "| {id} | {stage} | {hyp} | **{dec}** | {head} |".format(
                    id=record.get("experiment_id", ""),
                    stage=record.get("stage", ""),
                    hyp=str(record.get("hypothesis", "")).replace("|", "/"),
                    dec=str(record.get("decision", "")).upper(),
                    head=headline.replace("|", "/"),
                )
            )
        lines += ["", "## Detail", ""]
        for record in records:
            lines += [
                f"### {record.get('experiment_id','')} - {record.get('hypothesis','')}",
                "",
                f"- **Stage**: {record.get('stage','')}",
                f"- **Date**: {record.get('date','')}",
                f"- **Data version**: `{record.get('data_version','')}`",
                f"- **Config fingerprint**: `{record.get('config_fingerprint','')}`",
                f"- **Train period**: {record.get('train_period','') or 'n/a'}",
                f"- **Test period**: {record.get('test_period','') or 'n/a'}",
                f"- **Cost assumption**: {record.get('cost_bps','n/a')} bps",
                f"- **Parameters**: `{json.dumps(record.get('parameters', {}), sort_keys=True)}`",
                f"- **Results**: `{json.dumps(record.get('results', {}), sort_keys=True)}`",
                f"- **Decision**: **{str(record.get('decision','')).upper()}**",
                f"- **Notes**: {record.get('notes','')}",
                "",
            ]
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return out
