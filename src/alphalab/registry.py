"""A registry of signals: each version kept, each with the information lag it carries.

A signal panel (dates x assets) is registered under a name. If the panel is unchanged the same version is returned; if it changed, a new version is made and the old one stays. Each entry records
``lag`` (periods between the date on a row and the moment the information was known: a quarterly earnings figure dated at quarter end is known weeks later), the horizon the signal was built for, a
description, a content hash, the date range and the time of registration. :meth:`SignalRegistry.known` returns the panel as it could have been used, with the lag applied, and
:func:`pit_violations` counts the cells whose recorded information time is later than the date they sit on.

Files live under one directory: ``<name>/v<k>.parquet`` and one ``index.json``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

import pandas as pd

NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,63}$")


class RegistryError(ValueError):
    pass


def fingerprint(panel: pd.DataFrame) -> str:
    """A hash of the values, the dates and the asset names (column order does not matter)."""
    p = panel.sort_index(axis=1)
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(p, index=True).to_numpy().tobytes())
    h.update("|".join(map(str, p.columns)).encode())
    return h.hexdigest()[:16]


def pit_violations(panel: pd.DataFrame, info_times: pd.DataFrame) -> dict:
    """Cells where the information became known AFTER the date of the row (``info_times`` holds when each value was first known, same shape as ``panel``). A clean signal has none."""
    known = info_times.reindex_like(panel)
    dates = pd.Series(panel.index, index=panel.index)
    late = known.apply(lambda col: pd.to_datetime(col) > dates) & panel.notna()
    n = int(late.to_numpy().sum())
    return {"cells": int(panel.notna().to_numpy().sum()), "violations": n, "share": n / max(int(panel.notna().to_numpy().sum()), 1), "ok": n == 0}


class SignalRegistry:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._index = self.root / "index.json"

    # --------------------------------------------------------------------------------------------- storage
    def _read(self) -> dict:
        if not self._index.exists():
            return {}
        try:
            return json.loads(self._index.read_text())
        except (OSError, ValueError):
            return {}

    def _write(self, data: dict) -> None:
        tmp = self._index.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1, sort_keys=True))
        os.replace(tmp, self._index)

    # --------------------------------------------------------------------------------------------- API
    def register(self, name: str, panel: pd.DataFrame, lag: int = 0, horizon: int = 1, description: str = "", source: str = "") -> dict:
        if not NAME.match(str(name)):
            raise RegistryError("a signal name is letters, digits, '_', '-' or '.', up to 64 characters, starting with a letter or digit")
        if not isinstance(panel, pd.DataFrame) or panel.empty or not isinstance(panel.index, pd.DatetimeIndex):
            raise RegistryError("a signal is a non-empty frame of dates (rows) by assets (columns)")
        if int(lag) < 0 or int(horizon) < 1:
            raise RegistryError("lag must be 0 or more and horizon 1 or more")
        if not panel.index.is_monotonic_increasing or panel.index.has_duplicates:
            raise RegistryError("the dates must be increasing and unique")
        digest = fingerprint(panel)
        index = self._read()
        versions = index.setdefault(name, {"versions": []})["versions"]
        for v in versions:
            if v["hash"] == digest and v["lag"] == int(lag) and v["horizon"] == int(horizon):
                return dict(v, name=name, new=False)
        entry = {"version": len(versions) + 1, "hash": digest, "lag": int(lag), "horizon": int(horizon), "description": str(description)[:500], "source": str(source)[:200],
                 "first": str(panel.index[0].date()), "last": str(panel.index[-1].date()), "dates": int(len(panel)), "assets": [str(c) for c in panel.columns],
                 "coverage": float(panel.notna().to_numpy().mean()), "registered": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        d = self.root / name
        d.mkdir(exist_ok=True)
        panel.to_parquet(d / f"v{entry['version']}.parquet")
        versions.append(entry)
        self._write(index)
        return dict(entry, name=name, new=True)

    def names(self) -> list[str]:
        return sorted(self._read())

    def versions(self, name: str) -> list[dict]:
        data = self._read()
        if name not in data:
            raise RegistryError(f"no signal named {name!r}")
        return [dict(v, name=name) for v in data[name]["versions"]]

    def meta(self, name: str, version: int | None = None) -> dict:
        vs = self.versions(name)
        if version is None:
            return vs[-1]
        for v in vs:
            if v["version"] == int(version):
                return v
        raise RegistryError(f"{name!r} has no version {version}")

    def load(self, name: str, version: int | None = None) -> pd.DataFrame:
        m = self.meta(name, version)
        return pd.read_parquet(self.root / name / f"v{m['version']}.parquet")

    def known(self, name: str, version: int | None = None) -> pd.DataFrame:
        """The signal as it could have been used: shifted down by its information lag, so the row of date ``t`` holds only what was known at ``t``."""
        m = self.meta(name, version)
        panel = self.load(name, m["version"])
        return panel.shift(m["lag"]) if m["lag"] else panel

    def as_of(self, name: str, date, version: int | None = None) -> pd.Series:
        """The cross-section known at ``date`` (the last row at or before it, lag applied)."""
        k = self.known(name, version)
        k = k.loc[:pd.Timestamp(date)]
        if k.empty:
            raise RegistryError(f"{name!r} has nothing known at {date}")
        return k.iloc[-1]

    def listing(self) -> list[dict]:
        out = []
        for name in self.names():
            last = self.versions(name)[-1]
            out.append({k: last[k] for k in ("name", "version", "lag", "horizon", "first", "last", "dates", "coverage", "description", "registered")} | {"n_assets": len(last["assets"]), "versions": len(self.versions(name))})
        return out

    def delete(self, name: str) -> None:
        index = self._read()
        if name not in index:
            raise RegistryError(f"no signal named {name!r}")
        for v in index[name]["versions"]:
            (self.root / name / f"v{v['version']}.parquet").unlink(missing_ok=True)
        try:
            (self.root / name).rmdir()
        except OSError:
            pass
        del index[name]
        self._write(index)
