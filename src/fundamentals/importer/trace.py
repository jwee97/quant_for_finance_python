"""Structured diagnostic events (JSON lines) for the extraction pipeline.

Every stage of the pipeline can say what it decided and why: table detection, statement classification, column resolution, candidate mapping, ranking, rejection, period and unit inference, reconciliation,
fallback. Events are plain dictionaries with a ``stage`` and the fields of the decision. They are kept in memory (bounded) and, when tracing is on, written to the ``fundamentals.importer`` logger as
one JSON object per line. Tracing is on when the environment variable ``FUNDAMENTALS_IMPORT_TRACE`` is set or the import hint ``trace`` is true; it is off by default and costs one boolean test per event.
Nothing sensitive is ever put in an event: file names and figures from the document only.
"""

from __future__ import annotations

import json
import logging
import os

LOG = logging.getLogger("fundamentals.importer")
MAX_EVENTS = 4000


class Trace:
    def __init__(self, enabled: bool | None = None):
        self.enabled = bool(os.environ.get("FUNDAMENTALS_IMPORT_TRACE")) if enabled is None else bool(enabled)
        self.events: list[dict] = []

    def emit(self, stage: str, **fields) -> None:
        if not self.enabled:
            return
        ev = {"stage": stage, **fields}
        if len(self.events) < MAX_EVENTS:
            self.events.append(ev)
        if LOG.isEnabledFor(logging.DEBUG):
            LOG.debug(json.dumps(ev, default=str, ensure_ascii=False))

    def of(self, stage: str) -> list[dict]:
        return [e for e in self.events if e["stage"] == stage]


__all__ = ["Trace"]
