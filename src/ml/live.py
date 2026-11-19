"""What a training is doing right now, for the page that is watching it.

A network is trained by a loop that can run for minutes. Until now the only thing it said was which fold it was on; this module lets the loop say what each epoch did (the loss on the training rows and on the
validation tail, the best so far and when, the learning rate, the gradient size, how long it took, how many epochs of patience are used up) and how far through the epoch it is. The loop does not know who is listening:
it calls the functions below, which hand the event to the :class:`Telemetry` installed for the running job on this thread (:func:`use`), or do nothing when none is installed (a script, a test, a command-line run).

A :class:`Telemetry` keeps a bounded history of the current fit and a summary of the fits before it, so a job's ``view()`` can send the page one small object that draws the live panel, and it turns each epoch into
the job's progress message. It can also stop the job: ``check`` is called on every event and may raise (the job's own cancellation), which ends the training between two batches instead of after the whole fit.
"""

from __future__ import annotations

import contextlib
import math
import threading
import time
from collections import deque

MAX_POINTS = 600                      # epochs kept per fit (a longer fit keeps the last ones)
MAX_DONE = 60                         # summaries of earlier fits kept
BATCH_GAP = 0.4                       # seconds between two batch updates (the loop can run thousands of batches a second)
_local = threading.local()


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


class Telemetry:
    def __init__(self, on_message=None, check=None, store=None):
        self.on_message = on_message
        self.check = check
        self.store = store                                   # an object with start(run), epoch(row), touch(), finish(summary), abort(status): the database's copy of what is said here (src/store/appdb.py)
        self._lock = threading.Lock()
        self.context: dict = {}
        self.run: dict | None = None
        self.done: deque = deque(maxlen=MAX_DONE)
        self.n_fits = 0
        self.updated = 0.0
        self._batch_at = 0.0

    # ------------------------------------------------------------------------------------------------ events
    def set_context(self, **labels) -> None:
        with self._lock:
            self.context = {k: v for k, v in labels.items() if v is not None}

    def start(self, **info) -> None:
        self._guard()
        with self._lock:
            self.n_fits += 1
            self.run = {"n": self.n_fits, "context": dict(self.context), "t0": time.time(), "history": {k: [] for k in ("epoch", "train", "val", "lr", "grad", "seconds")}, "current": None, "batch": None,
                        **{k: info.get(k) for k in ("epochs", "patience", "device", "n_params", "n_train", "n_val", "batch_size", "name")}}
            self.updated = time.time()
            started = dict(self.run)
        self._persist("start", started)
        self._say("starting " + self._label() + (f": {info['n_params']:,} parameters on {info['device']}" if info.get("n_params") and info.get("device") else ""))

    def epoch(self, epoch: int, train, val=None, lr=None, grad=None, seconds=None, best=None, best_epoch=None, bad: int = 0, improved: bool = False, **extra) -> None:
        self._guard()
        with self._lock:
            if self.run is None:                                                         # a loop that never announced itself (another trainer): make a run of it
                self.run = {"n": self.n_fits + 1, "context": dict(self.context), "t0": time.time(), "history": {k: [] for k in ("epoch", "train", "val", "lr", "grad", "seconds")}, "current": None, "batch": None,
                            "epochs": extra.get("epochs"), "patience": extra.get("patience"), "device": None, "n_params": None, "n_train": None, "n_val": None, "batch_size": None, "name": None}
                self.n_fits += 1
            r, h = self.run, self.run["history"]
            for key, value in (("epoch", epoch), ("train", _num(train)), ("val", _num(val)), ("lr", _num(lr)), ("grad", _num(grad)), ("seconds", _num(seconds))):
                h[key].append(value)
                if len(h[key]) > MAX_POINTS:
                    del h[key][0]
            done = len(h["epoch"])
            spent = sum(s for s in h["seconds"] if s) or (time.time() - r["t0"])
            left = None if not r.get("epochs") or done == 0 else max(0.0, (r["epochs"] - (epoch + 1)) * spent / done)
            r["current"] = {"epoch": int(epoch) + 1, "epochs": r.get("epochs"), "train": _num(train), "val": _num(val), "best": _num(best), "best_epoch": None if best_epoch is None else int(best_epoch) + 1, "lr": _num(lr),
                            "grad": _num(grad), "seconds": _num(seconds), "bad": int(bad), "patience": r.get("patience"), "improved": bool(improved), "eta_seconds": left, **{k: v for k, v in extra.items() if k in ("peak_mb",)}}
            r["batch"] = None
            self.updated = time.time()
            cur = r["current"]
        self._persist("epoch", {"epoch": int(epoch), "train": _num(train), "val": _num(val), "lr": _num(lr), "grad": _num(grad), "seconds": _num(seconds), "best": _num(best), "bad": int(bad), "improved": bool(improved)})
        text = f"{self._label()} · epoch {cur['epoch']}" + (f" of {cur['epochs']}" if cur["epochs"] else "") + f" · train {_fmt(cur['train'])}" + ("" if cur["val"] is None else f" · val {_fmt(cur['val'])}")
        if cur["best"] is not None:
            text += f" · best {_fmt(cur['best'])} (epoch {cur['best_epoch']})"
        if cur["patience"]:
            text += f" · patience {cur['bad']}/{cur['patience']}"
        self._say(text)

    def batch(self, index: int, of: int, loss, epoch: int) -> None:
        now = time.time()
        if now - self._batch_at < BATCH_GAP:
            return
        self._batch_at = now
        self._guard()
        with self._lock:
            if self.run is not None:
                self.run["batch"] = {"epoch": int(epoch) + 1, "index": int(index), "of": int(of), "loss": _num(loss)}
                self.updated = now
        self._persist("touch", None)

    def finish(self, **summary) -> None:
        with self._lock:
            if self.run is None:
                return
            r = self.run
            row = {"n": r["n"], "context": r["context"], "epochs_run": len(r["history"]["epoch"]), "epochs": r.get("epochs"), "seconds": round(time.time() - r["t0"], 1), **{k: summary.get(k) for k in ("best_epoch", "best_val", "stopped_early", "final_train", "peak_mb")}}
            self.done.append(row)
            r["finished"] = row
            r["batch"] = None
            self.updated = time.time()
        self._persist("finish", row)

    # ------------------------------------------------------------------------------------------------ reading
    def has_data(self) -> bool:
        return self.run is not None or bool(self.done)

    def snapshot(self) -> dict | None:
        with self._lock:
            if self.run is None and not self.done:
                return None
            r = self.run
            run = None if r is None else {**{k: v for k, v in r.items() if k != "history"}, "history": {k: list(v) for k, v in r["history"].items()}}
            return {"context": dict(self.context), "run": run, "done": list(self.done), "fits": self.n_fits, "updated": self.updated}

    def abort(self, status: str = "cancelled") -> None:
        """The job ended while a fit was running (cancelled, or failed): say so in the database instead of leaving it 'running'."""
        self._persist("abort", status)

    # ------------------------------------------------------------------------------------------------ helpers
    def _persist(self, what: str, payload) -> None:
        if self.store is None:
            return
        try:                                                 # a log that cannot be written must never stop a training
            if what == "start":
                self.store.start(payload)
            elif what == "epoch":
                self.store.epoch(payload)
            elif what == "touch":
                self.store.touch()
            elif what == "finish":
                self.store.finish(payload)
            elif what == "abort":
                self.store.abort(payload)
        except Exception:
            pass

    def _guard(self) -> None:
        if self.check is not None:
            self.check()

    def _label(self) -> str:
        c = self.context
        bits = []
        if c.get("model"):
            bits.append(str(c["model"]))
        if c.get("fold") and c.get("folds"):
            bits.append(f"fold {c['fold']} of {c['folds']}")
        elif c.get("label"):
            bits.append(str(c["label"]))
        return " · ".join(bits) or "training"

    def _say(self, text: str) -> None:
        if self.on_message is not None:
            self.on_message(text)


def _fmt(v) -> str:
    return "n/a" if v is None else f"{v:.4g}"


# ---------------------------------------------------------------------------------------------------- the thread's current listener
@contextlib.contextmanager
def use(telemetry: Telemetry | None):
    """Install ``telemetry`` as this thread's listener for the duration of the block."""
    previous = getattr(_local, "sink", None)
    _local.sink = telemetry
    try:
        yield telemetry
    finally:
        _local.sink = previous


def current() -> Telemetry | None:
    return getattr(_local, "sink", None)


def set_context(**labels) -> None:
    t = current()
    if t is not None:
        t.set_context(**labels)


def start(**info) -> None:
    t = current()
    if t is not None:
        t.start(**info)


def epoch(*args, **kwargs) -> None:
    t = current()
    if t is not None:
        t.epoch(*args, **kwargs)


def batch(index: int, of: int, loss, epoch_index: int) -> None:
    t = current()
    if t is not None:
        t.batch(index, of, loss, epoch_index)


def finish(**summary) -> None:
    t = current()
    if t is not None:
        t.finish(**summary)
