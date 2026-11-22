"""Dated option-chain snapshots, so that a model's parameters can be followed through time.

Calibration stability (do the parameters move smoothly from day to day?) needs one chain per date. Free sources give only today's chain, so a history exists only if it is kept: each call to ``save`` stores the
chain as it is, one file per underlying and date, under ``data/user/options_history/``. A simulated chain is refused (its parameters are set by the platform, so their path says nothing about a market)
unless the caller asks for it, and then the record says so. Nothing here fabricates dates: the date of a snapshot is the date of the quotes.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

NAME = re.compile(r"^[A-Za-z0-9_.:-]{1,40}$")


class History:
    def __init__(self, root: Path):
        self.dir = Path(root) / "data" / "user" / "options_history"

    def _folder(self, underlying: str) -> Path:
        if not NAME.match(underlying):
            raise ValueError("invalid underlying name")
        return self.dir / underlying.replace(":", "_")

    def save(self, underlying: str, chain: pd.DataFrame, as_of, source: str) -> dict:
        d = self._folder(underlying)
        d.mkdir(parents=True, exist_ok=True)
        day = str(pd.Timestamp(as_of).date())
        chain.to_pickle(d / f"{day}.pkl")
        (d / f"{day}.json").write_text(json.dumps({"underlying": underlying, "date": day, "source": source, "rows": int(len(chain))}))
        return {"underlying": underlying, "date": day, "source": source, "rows": int(len(chain)), "dates_kept": len(self.dates(underlying))}

    def dates(self, underlying: str) -> list[str]:
        d = self._folder(underlying)
        return sorted(p.stem for p in d.glob("*.pkl")) if d.is_dir() else []

    def meta(self, underlying: str) -> list[dict]:
        d = self._folder(underlying)
        return [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))] if d.is_dir() else []

    def load(self, underlying: str, day: str) -> pd.DataFrame:
        p = self._folder(underlying) / f"{day}.pkl"
        if not p.is_file():
            raise KeyError(day)
        return pd.read_pickle(p)
