"""Loader for the cached Kenneth French daily factors (Generation 5, Stage 40). Values are converted from percent to decimal returns."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _read(path: Path) -> pd.DataFrame:
    lines = path.read_text(encoding="latin-1").splitlines()
    start = next(i for i, line in enumerate(lines) if line.split(",")[0].strip() == "" and len([f for f in line.split(",") if f.strip()]) >= 1 and "," in line)
    rows = []
    for line in lines[start + 1:]:
        parts = [p.strip() for p in line.split(",")]
        if not parts[0].isdigit() or len(parts[0]) != 8:
            if rows:
                break                                  # the footer after the data
            continue
        rows.append(parts)
    header = [p.strip() for p in lines[start].split(",")][1:]
    frame = pd.DataFrame(rows, columns=["date"] + header)
    frame["date"] = pd.to_datetime(frame["date"], format="%Y%m%d")
    return frame.set_index("date").astype(float) / 100.0


def load_factors(directory: str | Path) -> pd.DataFrame:
    """Mkt-RF, SMB, HML, RMW, CMA, RF and Mom as daily decimal returns, indexed by date."""
    d = Path(directory)
    five = _read(d / "F-F_Research_Data_5_Factors_2x3_daily.csv")
    mom = _read(d / "F-F_Momentum_Factor_daily.csv")
    mom.columns = ["Mom"]
    return five.join(mom, how="inner")
