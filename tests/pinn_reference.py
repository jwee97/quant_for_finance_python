"""The physics-informed networks' training loop exactly as it was before checkpointing existed (commit 20a1cbe, ``src/models/pinn.py::_train``): Adam with a cosine decay on fresh collocation points, then ONE call of
L-BFGS for the whole polish. It is kept here, unchanged, as the reference the chunked, checkpointable loop is compared with (``tests/test_pinn_lbfgs_regression.py``). Do not edit its arithmetic: the extra keyword
arguments (``identity``, ``lbfgs_chunk``) exist only so that the models can call it in place of the real loop."""

from __future__ import annotations


def reference_train(loss_fn, params, steps: int, lr: float, seed: int, report: int = 200, lbfgs_steps: int = 0, identity=None, lbfgs_chunk=None):
    """Adam with a cosine learning-rate decay on freshly drawn collocation points, then optionally ``lbfgs_steps`` iterations of L-BFGS on one fixed draw (which polishes the fit far more than more Adam steps do)."""
    import torch

    params = list(params)
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(steps, 1), eta_min=lr * 0.02)
    history = []
    for step in range(steps):
        loss = loss_fn()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % report == 0 or step == steps - 1:
            history.append(float(loss.detach()))
    if lbfgs_steps:
        torch.manual_seed(seed + 1)
        state = torch.get_rng_state()
        lb = torch.optim.LBFGS(params, lr=1.0, max_iter=lbfgs_steps, history_size=30, line_search_fn="strong_wolfe", tolerance_grad=1e-12, tolerance_change=1e-14)

        def closure():
            torch.set_rng_state(state)                                                                  # the same collocation points at every evaluation
            lb.zero_grad()
            value = loss_fn()
            value.backward()
            return value

        lb.step(closure)
        history.append(float(closure().detach()))
    return history
