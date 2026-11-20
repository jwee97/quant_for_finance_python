"""Universe screens, computed only from the window of data that ends at the decision date.

A screen narrows the eligible names before a portfolio is built. Each takes the trailing returns window (and the dollar volume, where there is any) and returns the names it keeps. None looks at a future
return; the walk-forward engine passes in only rows that were known. ``top`` is a share of the names that pass the data requirements, never fewer than ``MIN_NAMES``.

Kinds: ``all``; ``liquidity`` (average daily dollar volume at least ``min_dollar_volume``); ``size`` (the largest ``top`` share by average dollar volume: a proxy, since shares outstanding are not in the
data, and it is labelled as one); ``low_vol`` (the ``top`` share with the lowest trailing volatility); ``momentum`` (the ``top`` share by the return of the window up to one month ago, the 12-1 convention);
``sector`` (names whose group is in ``groups``); ``manual`` (``names``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_NAMES = 3
KINDS = ("all", "liquidity", "size", "low_vol", "momentum", "sector", "manual")
LABELS = {"all": "All eligible names", "liquidity": "Liquid names only", "size": "Largest by dollar volume (a size proxy)", "low_vol": "Lowest volatility", "momentum": "Highest 12-1 momentum",
          "sector": "Chosen sectors", "manual": "Chosen names"}


class ScreenError(ValueError):
    pass


def validate(spec: dict) -> dict:
    kind = spec.get("kind", "all")
    if kind not in KINDS:
        raise ScreenError(f"screen kind must be one of {list(KINDS)}")
    out = {"kind": kind}
    if kind in ("size", "low_vol", "momentum"):
        top = float(spec.get("top", 0.5))
        if not 0.05 <= top <= 1.0:
            raise ScreenError("top is a share between 0.05 and 1")
        out["top"] = top
    if kind == "liquidity":
        v = float(spec.get("min_dollar_volume", 1e6))
        if v < 0:
            raise ScreenError("min_dollar_volume is not negative")
        out["min_dollar_volume"] = v
    if kind == "sector":
        g = [str(x) for x in spec.get("groups") or []]
        if not g:
            raise ScreenError("a sector screen needs at least one group")
        out["groups"] = g
    if kind == "manual":
        n = [str(x).upper() for x in spec.get("names") or []]
        if len(n) < MIN_NAMES:
            raise ScreenError(f"a manual screen needs at least {MIN_NAMES} names")
        out["names"] = n
    return out


def label(spec: dict) -> str:
    k = spec["kind"]
    if k in ("size", "low_vol", "momentum"):
        return f"{LABELS[k]} (top {spec['top']:.0%})"
    if k == "liquidity":
        return f"{LABELS[k]} (>= {spec['min_dollar_volume']:,.0f} a day)"
    if k == "sector":
        return f"{LABELS[k]}: {', '.join(spec['groups'])}"
    if k == "manual":
        return f"{LABELS[k]}: {', '.join(spec['names'])}"
    return LABELS[k]


def select(window: pd.DataFrame, dollar_volume: pd.DataFrame | None, spec: dict, groups: dict | None = None) -> list[str]:
    """The names of ``window`` (trailing daily returns, one column per name, no gaps) the screen keeps."""
    names = list(window.columns)
    k = spec["kind"]
    if k == "all":
        return names
    if k == "manual":
        return [n for n in names if n in set(spec["names"])]
    if k == "sector":
        g = set(spec["groups"])
        return [n for n in names if (groups or {}).get(n) in g]
    if k == "liquidity":
        if dollar_volume is None:
            raise ScreenError("a liquidity screen needs volume data, which this universe does not have")
        adv = dollar_volume.reindex(columns=names).tail(len(window)).mean()
        return [n for n in names if np.isfinite(adv[n]) and adv[n] >= spec["min_dollar_volume"]]
    top = spec["top"]
    keep = max(MIN_NAMES, int(np.ceil(top * len(names))))
    if k == "size":
        if dollar_volume is None:
            raise ScreenError("a size screen needs volume data, which this universe does not have")
        score = dollar_volume.reindex(columns=names).tail(len(window)).mean()
    elif k == "low_vol":
        score = -window.std()
    else:                                                                                  # momentum, 12-1: the whole window's return up to one month before its end
        skip = min(21, max(1, len(window) // 12))
        score = (1.0 + window.iloc[:-skip]).prod() - 1.0
    ranked = score.replace([np.inf, -np.inf], np.nan).dropna().sort_values(ascending=False)
    return list(ranked.index[:keep])
