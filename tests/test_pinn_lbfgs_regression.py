"""Regression for the chunked, checkpointable L-BFGS polish of the physics-informed networks against the loop it replaced.

``tests/pinn_reference.py`` keeps the original ``_train`` (commit 20a1cbe): Adam, then ONE ``LBFGS.step`` call for the whole polish. The current ``_train`` runs the polish in chunks of ``LBFGS_CHUNK`` iterations so that a
checkpoint can be committed at each optimizer boundary. This module trains the Black-Scholes-Merton, Vasicek and Heston networks both ways, in the same process on the same machine, and requires:

* the fitted prices, every loss in the history and every network parameter to be IDENTICAL (tolerance 0: the chunks are the iterations of one run with the optimizer's memory carried across, and the evaluation budget
  is applied to the whole polish; identical arithmetic in identical order cannot differ, and any difference means the mathematics changed). Chunk sizes 3, 10 and one chunk for the whole polish are all tested;
* the optimizer phases recorded for the Training monitor to run Adam units first, then L-BFGS units, with the cumulative L-BFGS iteration count rising to the requested number and never beyond;
* a run interrupted at a chunk boundary or inside a line search, and resumed, to end with the identical network, and the interruption to be reported as recomputed work;
* the pricing accuracy tests of the models (``tests/test_generative_pinn.py``) to stay untouched: nothing here relaxes them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.ml.deep import context                                       # noqa: E402
from src.models import pinn                                           # noqa: E402
from tests.pinn_reference import reference_train                      # noqa: E402

LBFGS = {"bsm": 120, "vasicek": 100, "heston": 80}
ADAM = {"bsm": 400, "vasicek": 400, "heston": 300}


def build(kind: str, lbfgs_chunk: int | None = None):
    if kind == "bsm":
        return pinn.BSMPINN(0.03, 0.01, 0.25, 1.0, steps=ADAM[kind], lbfgs_steps=LBFGS[kind], hidden=16, layers=2, batch=256, seed=2)
    if kind == "vasicek":
        return pinn.VasicekPINN(0.8, 0.05, 0.02, maturity=5.0, steps=ADAM[kind], lbfgs_steps=LBFGS[kind], hidden=16, layers=2, batch=256, seed=2)
    return pinn.HestonPINN(0.03, 0.0, 2.0, 0.04, 0.3, -0.7, steps=ADAM[kind], lbfgs_steps=LBFGS[kind], hidden=16, layers=2, batch=256, seed=2)


def prices(kind: str, model):
    if kind == "bsm":
        return model.price(np.array([80.0, 90.0, 100.0, 110.0, 120.0]), 100.0, 0.7)
    if kind == "vasicek":
        return model.price(np.array([0.0, 0.02, 0.05, 0.08, 0.10]), 3.0)
    return model.price(np.array([90.0, 100.0, 110.0]), 100.0, 1.0, np.array([0.03, 0.04, 0.05]))


def params(model) -> np.ndarray:
    return np.concatenate([p.detach().numpy().ravel() for p in model.net.parameters()])


def fit_with(monkeypatch, trainer, kind: str):
    monkeypatch.setattr(pinn, "_train", trainer)
    model = build(kind).fit()
    return model


@pytest.fixture(scope="module", params=sorted(LBFGS))
def original(request):
    """The model trained with the original loop. (A module-scoped fixture cannot use ``monkeypatch``: patch by hand.)"""
    kind = request.param
    real = pinn._train
    pinn._train = reference_train
    try:
        model = build(kind).fit()
    finally:
        pinn._train = real
    return kind, {"prices": prices(kind, model), "history": list(model.history), "params": params(model)}


@pytest.mark.parametrize("chunk", [3, 10, 1000], ids=["chunk-3", "chunk-10", "one-chunk"])
def test_chunked_polish_is_identical_to_the_original_single_call(original, chunk, monkeypatch):
    kind, ref = original
    monkeypatch.setattr(pinn, "LBFGS_CHUNK", chunk)
    model = build(kind).fit()
    np.testing.assert_array_equal(prices(kind, model), ref["prices"])                       # tolerance 0, see the module's docstring
    assert list(model.history) == ref["history"]
    np.testing.assert_array_equal(params(model), ref["params"])


def test_a_run_with_a_folder_is_identical_too_and_records_the_phases(original, tmp_path):
    kind, ref = original
    folder = tmp_path / "run"
    with context.use(str(folder), resume=False, grad_level="standard"):
        model = build(kind).fit()
    np.testing.assert_array_equal(prices(kind, model), ref["prices"])                       # checkpointing and monitoring change nothing
    np.testing.assert_array_equal(params(model), ref["params"])
    fit = next(folder.rglob("metrics.jsonl")).parent
    log = [json.loads(x) for x in (fit / "metrics.jsonl").read_text().splitlines()]
    epochs = [r for r in log if r["type"] == "epoch"]
    phases = [r["phase"] for r in epochs]
    adam_units = -(-ADAM[kind] // 200)
    lbfgs_units = -(-LBFGS[kind] // pinn.LBFGS_CHUNK)
    assert phases == ["adam"] * adam_units + ["lbfgs"] * lbfgs_units, phases                   # Adam first, then L-BFGS, no interleaving
    iters = [r["metrics"].get("lbfgs_iterations") for r in epochs if r["phase"] == "lbfgs"]
    assert all(b is not None for b in iters) and iters == sorted(iters) and 0 < iters[-1] <= LBFGS[kind], iters
    assert [r["epoch"] for r in epochs] == list(range(1, len(epochs) + 1)) and log[-1]["type"] == "finish"
    assert all(r["committed"] for r in epochs)
    assert json.loads((fit / "commit.json").read_text())["state"] == "applied"
    gr = [json.loads(x) for x in (fit / "gradients.jsonl").read_text().splitlines()]
    assert {g["optimizer"] for g in gr} == {"adam", "lbfgs"}, "both optimizers are monitored, and only in their own phase"
    assert [g["epoch"] for g in gr if g["optimizer"] == "adam"] == list(range(adam_units))
    assert [g["epoch"] for g in gr if g["optimizer"] == "lbfgs"] == list(range(adam_units, adam_units + lbfgs_units))


@pytest.mark.parametrize("where", ["chunk-boundary", "inside-line-search"])
def test_an_interrupted_polish_resumes_to_the_identical_network(original, where, tmp_path, monkeypatch):
    kind, ref = original
    folder = tmp_path / "run"
    real_step = torch.optim.LBFGS.step
    calls = {"step": 0}

    class Killed(BaseException):
        """Stands for the process ending: not an ``Exception``, so nothing in the training loop can catch it."""

    def step(self, closure):
        calls["step"] += 1
        if calls["step"] == 4:
            if where == "chunk-boundary":
                raise Killed()                                                                # the 4th chunk never starts
            seen = {"n": 0}

            def counted():
                seen["n"] += 1
                if seen["n"] == 3:
                    raise Killed()                                                            # inside the 4th chunk, at its third loss evaluation
                return closure()
            return real_step(self, counted)
        return real_step(self, closure)

    with monkeypatch.context() as m:
        m.setattr(torch.optim.LBFGS, "step", step)
        with pytest.raises(Killed):
            with context.use(str(folder), resume=False, grad_level="standard"):
                build(kind).fit()
    fit = next(folder.rglob("metrics.jsonl")).parent
    assert (fit / "inflight.json").exists(), "the unit that was running left its marker"
    with context.use(str(folder), resume=True, grad_level="standard"):
        model = build(kind).fit()
    np.testing.assert_array_equal(prices(kind, model), ref["prices"])
    np.testing.assert_array_equal(params(model), ref["params"])
    assert list(model.history) == ref["history"]
    log = [json.loads(x) for x in (fit / "metrics.jsonl").read_text().splitlines()]
    adam_units = -(-ADAM[kind] // 200)
    resume = [r for r in log if r["type"] == "resume"][-1]
    assert resume["level"] == "exact" and resume["epoch"] == adam_units + 3, resume              # three polish chunks were committed before the interruption
    lost = [r for r in log if r["type"] == "recomputed_work"]
    assert len(lost) == 1 and lost[0]["units"] == [adam_units + 4] and lost[0]["phase"] == "lbfgs", lost
    epochs = [r["epoch"] for r in log if r["type"] == "epoch"]
    assert epochs == list(range(1, len(epochs) + 1)) and log[-1]["type"] == "finish"
