"""Run with the SB3 Python: ``python -m tests.sb3_compare <checkpoint a> <checkpoint b>`` compares the environments two SB3 checkpoints carry (unpickling them needs gymnasium and the platform's environment class) and
prints ``same`` or what differs. Monitor's wall-clock timings are not training state and are ignored."""

from __future__ import annotations

import io
import pickle
import sys

import numpy as np
import torch


def diff(a, b, path="env"):
    if isinstance(a, np.random.Generator):
        return diff(a.bit_generator.state, b.bit_generator.state, path + ".state")
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return f"{path}: keys differ"
        for k in a:
            d = diff(a[k], b[k], f"{path}.{k}")
            if d:
                return d
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return f"{path}: length differs"
        for i, (x, y) in enumerate(zip(a, b)):
            d = diff(x, y, f"{path}[{i}]")
            if d:
                return d
    elif isinstance(a, np.ndarray):
        if not np.array_equal(a, b):
            return f"{path}: arrays differ"
    elif hasattr(a, "__dict__") and not isinstance(a, type):
        return diff(vars(a), vars(b), path)
    elif a != b:
        return f"{path}: {a!r} != {b!r}"
    return None


def main() -> None:
    envs = []
    for path in sys.argv[1:3]:
        blob = torch.load(io.BytesIO(open(path, "rb").read()), map_location="cpu", weights_only=False)
        envs.append(pickle.loads(blob["history"]["extra"]["sb3"]["env"]))
    a, b = envs
    problems = [diff({k: v for k, v in vars(a.unwrapped).items() if k not in ("np_random", "_np_random")}, {k: v for k, v in vars(b.unwrapped).items() if k not in ("np_random", "_np_random")}),
                diff(a.unwrapped.np_random.bit_generator.state, b.unwrapped.np_random.bit_generator.state, "np_random"), diff(list(a.get_episode_rewards()), list(b.get_episode_rewards()), "episode_rewards")]
    print("same" if not any(problems) else "; ".join(p for p in problems if p))


if __name__ == "__main__":
    main()
