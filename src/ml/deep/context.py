"""Where the running job's training checkpoints go, without putting it in the experiment's configuration.

A run's checkpoint folder and its resume flag describe HOW it is executed, not what it learns: putting a different folder in every experiment's saved parameters would give two identical experiments different
fingerprints. The job sets them here for the thread it runs on; the trainer reads them when the model's own parameters name no folder.
"""

from __future__ import annotations

import contextlib
import contextvars


class TrainingHalted(BaseException):
    """The user stopped or paused the run from the Training monitor. Raised after the epoch's checkpoint is committed, so that a walk-forward experiment ends instead of going on to the next fit with untrained
    networks and saving a result that looks complete. The run can be resumed; ``action`` is ``stop`` or ``pause``. A ``BaseException`` like the job's cancellation, so that the walk-forward's per-fit ``except Exception`` cannot swallow it and carry on."""

    def __init__(self, action: str, epoch: int):
        super().__init__(f"training {'stopped' if action == 'stop' else 'paused'} by request after epoch {epoch}: nothing was saved as an experiment; resume it from the Training monitor to continue from its checkpoint")
        self.action, self.epoch = action, epoch


class CheckpointIncompatible(Exception):
    """The saved state does not belong to this model or this data; ``reasons`` says what differs. Deliberately NOT a ``ValueError``: a walk-forward skips a fit that raises one and carries on, which would turn a refused resume into a silently missing fold."""

    def __init__(self, reasons: list[str]):
        super().__init__("; ".join(reasons))
        self.reasons = list(reasons)


_current: contextvars.ContextVar[dict | None] = contextvars.ContextVar("training_context", default=None)


def current() -> dict | None:
    return _current.get()


@contextlib.contextmanager
def use(folder: str, resume: bool = False, warm: bool = False):
    token = _current.set({"dir": str(folder), "resume": bool(resume), "warm": bool(warm)})
    try:
        yield
    finally:
        _current.reset(token)
