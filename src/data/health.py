"""Data health: what a dataset covers, whether it can be trusted, and whether research may use it.

Three questions, kept apart because they have different answers:

* **Coverage**: what does the dataset span, and which expected periods have no observation? Missing periods are listed as runs, per series, against the calendar the series belongs to (business days for equities and
  futures, every day for crypto), never filled.
* **Integrity**: is what is there self-consistent? Duplicates, non-positive or invalid OHLC, stale runs, a quote older than its allowed age, a crossed or locked market, a row after its contract expired, a provider
  symbol with no internal mapping, a figure stamped available before the period it describes ended, a corporate action that the prices do not show (or show without a record). Each finding names the check, the series,
  the date and a severity; nothing is deleted or repaired (a repair is a separate, logged correction: see :class:`CorrectionLog`).
* **Research readiness**: given what a study needs (a minimum history, a longest tolerable gap, no blocking findings), can this series be used, used with stated caveats, or not used? The answer lists its reasons, so a
  blocked series says why.

Raw observations are never altered here. A correction is appended to a log that keeps the raw value, the corrected value, who or what made the change and why; applying it is the reader's choice, and the raw data stay.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

SEVERITY_ORDER = {"info": 0, "warning": 1, "error": 2}


@dataclass
class Finding:
    check: str
    series: str
    severity: str
    detail: str
    date: str | None = None
    value: float | None = None

    def __post_init__(self):
        if self.date is not None:
            self.date = str(pd.Timestamp(self.date).date())

    def as_dict(self) -> dict:
        return {"check": self.check, "series": self.series, "severity": self.severity, "detail": self.detail, "date": self.date, "value": self.value}


def calendar_for(asset_class: str) -> str:
    """``D`` (every day) for crypto, ``B`` (business days) for everything else."""
    return "D" if str(asset_class).lower() in ("crypto", "crypto_spot", "crypto_perp") else "B"


def missing_runs(index: pd.DatetimeIndex, freq: str = "B", min_run: int = 1, first=None, last=None, expected: pd.DatetimeIndex | None = None) -> list[dict]:
    """Runs of expected periods inside ``first..last`` (default: the series' own span) that have no observation. The expected periods are the calendar ``freq`` or, when given, the dates of ``expected`` (the calendar
    of a panel the series belongs to). A weekday holiday is a run of length 1: pass ``min_run`` to ignore single holidays."""
    idx = pd.DatetimeIndex(pd.to_datetime(index)).normalize().unique().sort_values()
    if len(idx) < 2:
        return []
    lo, hi = (pd.Timestamp(first) if first is not None else idx[0]), (pd.Timestamp(last) if last is not None else idx[-1])
    if expected is not None:
        exp = pd.DatetimeIndex(pd.to_datetime(expected)).normalize().unique().sort_values()
        exp = exp[(exp >= lo) & (exp <= hi)]
        gone = exp.difference(idx)
        if not len(gone):
            return []
        pos = {d: i for i, d in enumerate(exp)}
        runs, cur = [], [gone[0]]
        for d in gone[1:]:
            if pos[d] == pos[cur[-1]] + 1:
                cur.append(d)
            else:
                runs.append(cur)
                cur = [d]
        runs.append(cur)
        return [{"start": str(r[0].date()), "end": str(r[-1].date()), "periods": len(r)} for r in runs if len(r) >= min_run]
    gone = pd.date_range(lo, hi, freq=freq).difference(idx)
    if not len(gone):
        return []
    runs, cur = [], [gone[0]]
    for d in gone[1:]:
        step = pd.Timedelta(days=3) if (freq == "B" and cur[-1].dayofweek == 4) else pd.Timedelta(days=1)       # consecutive expected periods: Friday is followed by Monday
        if d - cur[-1] <= step:
            cur.append(d)
        else:
            runs.append(cur)
            cur = [d]
    runs.append(cur)
    return [{"start": str(r[0].date()), "end": str(r[-1].date()), "periods": len(r)} for r in runs if len(r) >= min_run]


def coverage(frames: dict[str, pd.DataFrame], asset_class: dict | str = "equity", min_run: int = 1, expected_index: pd.DatetimeIndex | None = None) -> list[dict]:
    """One row per series: first and last observation, how many there are, how many were expected on its calendar, the completeness, and its missing runs."""
    rows = []
    for name, df in frames.items():
        if df is None or not len(df):
            rows.append({"series": name, "first": None, "last": None, "observations": 0, "expected": 0, "completeness": 0.0, "missing_runs": [], "missing_periods": 0, "calendar": "n/a"})
            continue
        ac = asset_class.get(name, "equity") if isinstance(asset_class, dict) else asset_class
        freq = calendar_for(ac)
        idx = pd.DatetimeIndex(pd.to_datetime(df.index)).normalize().unique().sort_values()
        if expected_index is not None:
            exp_idx = pd.DatetimeIndex(pd.to_datetime(expected_index)).normalize().unique()
            expected = int(((exp_idx >= idx[0]) & (exp_idx <= idx[-1])).sum())
        else:
            expected = len(pd.date_range(idx[0], idx[-1], freq=freq))
        runs = missing_runs(idx, freq, min_run, expected=expected_index)
        rows.append({"series": name, "first": str(idx[0].date()), "last": str(idx[-1].date()), "observations": int(len(idx)), "expected": int(expected), "completeness": float(len(idx) / expected) if expected else 1.0,
                     "missing_runs": runs, "missing_periods": int(sum(r["periods"] for r in runs)), "calendar": "the dataset's own dates" if expected_index is not None else ("every day" if freq == "D" else "business days")})
    return rows


# ------------------------------------------------------------------------------------------------------ integrity
def check_series(df: pd.DataFrame, name: str, *, asset_class: str = "equity", jump: float = 0.4, stale_run: int = 5, max_gap_days: int | None = None, as_of=None, max_age_days: float | None = None,
                 expiry=None, expected_index: pd.DatetimeIndex | None = None) -> list[Finding]:
    """The self-consistency checks of one dated series (a close column, or OHLCV). Nothing is modified."""
    out: list[Finding] = []

    def add(check, sev, detail, date=None, value=None):
        out.append(Finding(check, name, sev, detail, None if date is None else str(pd.Timestamp(date).date()), None if value is None else float(value)))
    if df is None or not len(df):
        add("empty", "error", "the series has no observations")
        return out
    idx = pd.DatetimeIndex(pd.to_datetime(df.index))
    dup = idx[idx.duplicated(keep=False)]
    if len(dup):
        add("duplicate_dates", "error", f"{len(dup.unique())} dates appear more than once (first {dup[0].date()})", dup[0], len(dup.unique()))
    if not idx.is_monotonic_increasing:
        add("unsorted", "error", "the dates are not in increasing order")
    price_cols = [c for c in ("open", "high", "low", "close") if c in df.columns] or [c for c in df.columns[:1]]
    px = df[price_cols].apply(pd.to_numeric, errors="coerce")
    bad = px.le(0).any(axis=1)
    if bad.any():
        add("non_positive_price", "error", f"{int(bad.sum())} observations have a price at or below zero (first {idx[bad.to_numpy()][0].date()})", idx[bad.to_numpy()][0], int(bad.sum()))
    nan = px.isna().all(axis=1)
    if nan.any():
        add("missing_price", "warning", f"{int(nan.sum())} observations have no price", idx[nan.to_numpy()][0], int(nan.sum()))
    if {"high", "low"} <= set(df.columns):
        hl = pd.to_numeric(df["high"], errors="coerce") < pd.to_numeric(df["low"], errors="coerce")
        if hl.any():
            add("high_below_low", "error", f"{int(hl.sum())} bars have a high below the low (first {idx[hl.to_numpy()][0].date()})", idx[hl.to_numpy()][0], int(hl.sum()))
    if {"high", "low", "close"} <= set(df.columns):
        c, h, l = (pd.to_numeric(df[k], errors="coerce") for k in ("close", "high", "low"))
        oc = (c > h * (1 + 1e-9)) | (c < l * (1 - 1e-9))
        if oc.any():
            add("close_outside_range", "warning", f"{int(oc.sum())} bars close outside their high-low range", idx[oc.to_numpy()][0], int(oc.sum()))
    if "volume" in df.columns and (pd.to_numeric(df["volume"], errors="coerce") < 0).any():
        add("negative_volume", "error", "some observations have a negative volume")
    close = pd.to_numeric(df["close"] if "close" in df.columns else df.iloc[:, 0], errors="coerce")
    close.index = idx
    close = close[~close.index.duplicated(keep="first")].sort_index()
    pos = close.where(close > 0)
    if len(pos) > 1:
        r = np.log(pos).diff().abs()
        big = r[r > np.log1p(jump)]
        if len(big):
            add("large_move", "warning", f"{len(big)} one-period moves above {jump:.0%} (first {big.index[0].date()}): a split, a bad print or a real move", big.index[0], len(big))
    if len(close) >= stale_run:
        same = (close.diff() == 0).astype(int)
        run = same.groupby((same == 0).cumsum()).cumsum()
        if run.max() >= stale_run - 1:
            end = run.idxmax()
            add("stale_prices", "warning", f"the close did not change for {int(run.max()) + 1} observations in a row (ending {end.date()})", end, int(run.max()) + 1)
    freq = calendar_for(asset_class)
    runs = missing_runs(close.index, freq, 1, expected=expected_index)
    long = [r for r in runs if max_gap_days is not None and r["periods"] > max_gap_days]
    if long:
        add("long_gap", "error", f"{len(long)} gaps longer than {max_gap_days} expected periods (longest {max(r['periods'] for r in long)}, first {long[0]['start']})", long[0]["start"], len(long))
    elif runs:
        add("missing_periods", "info", f"{sum(r['periods'] for r in runs)} expected periods have no observation in {len(runs)} runs (first {runs[0]['start']})", runs[0]["start"], len(runs))
    if as_of is not None and max_age_days is not None and len(close):
        age = (pd.Timestamp(as_of).normalize() - close.index[-1].normalize()).days
        if age > max_age_days:
            add("stale_series", "warning", f"the last observation is {age} days old (allowed {max_age_days:g})", close.index[-1], age)
    if expiry is not None and len(close):
        after = close.index[close.index > pd.Timestamp(expiry)]
        if len(after):
            add("observations_after_expiry", "error", f"{len(after)} observations after the contract's expiry on {pd.Timestamp(expiry).date()} (first {after[0].date()})", after[0], len(after))
    return out


def check_quote(bid: float, ask: float, quote_ts, as_of, *, name: str = "quote", max_age_s: float = 5.0) -> list[Finding]:
    """A market quote: crossed (bid above ask), locked (equal), non-positive, or older than ``max_age_s`` at ``as_of``."""
    out: list[Finding] = []
    b, a = float(bid), float(ask)
    if not (np.isfinite(b) and np.isfinite(a)) or b <= 0 or a <= 0:
        out.append(Finding("invalid_quote", name, "error", f"bid {bid} / ask {ask} is not a positive price pair"))
    elif b > a:
        out.append(Finding("crossed_market", name, "error", f"bid {b:g} is above ask {a:g}", value=b - a))
    elif b == a:
        out.append(Finding("locked_market", name, "warning", f"bid and ask are both {b:g}"))
    age = (pd.Timestamp(as_of) - pd.Timestamp(quote_ts)).total_seconds()
    if age < 0:
        out.append(Finding("quote_from_the_future", name, "error", f"the quote is stamped {-age:.1f} s after the time it is used at", value=age))
    elif age > max_age_s:
        out.append(Finding("stale_quote", name, "warning", f"the quote is {age:.1f} s old (allowed {max_age_s:g} s)", value=age))
    return out


def check_symbol_mapping(provider_symbols, mapping: dict, name: str = "symbols") -> list[Finding]:
    """Provider symbols with no internal id, and internal ids two provider symbols map to (a silent merge)."""
    out = []
    missing = sorted({s for s in provider_symbols if s not in mapping})
    if missing:
        out.append(Finding("unmapped_symbol", name, "error", f"{len(missing)} provider symbols have no internal mapping: {', '.join(missing[:8])}" + (" ..." if len(missing) > 8 else ""), value=len(missing)))
    inv: dict = {}
    for s in provider_symbols:
        if s in mapping:
            inv.setdefault(mapping[s], []).append(s)
    clash = {k: v for k, v in inv.items() if len(v) > 1}
    if clash:
        k, v = next(iter(clash.items()))
        out.append(Finding("ambiguous_symbol", name, "error", f"{len(clash)} internal ids are the target of several provider symbols (e.g. {k}: {', '.join(v)})", value=len(clash)))
    return out


def check_point_in_time(period_end, available, name: str = "figures") -> list[Finding]:
    """Each figure's availability date against the end of the period it describes: a figure available before its period ended could not have been known, and one with no availability date cannot be used point-in-time."""
    pe = pd.to_datetime(pd.Series(period_end), errors="coerce")
    av = pd.to_datetime(pd.Series(available), errors="coerce")
    out = []
    none = av.isna()
    if none.any():
        out.append(Finding("no_availability_date", name, "error", f"{int(none.sum())} figures have no availability date, so a backtest cannot know when they became public", value=int(none.sum())))
    early = (av < pe) & ~none
    if early.any():
        out.append(Finding("available_before_period_end", name, "error", f"{int(early.sum())} figures are stamped available before the period they describe ended (look-ahead)", value=int(early.sum())))
    return out


def check_corporate_actions(close: pd.Series, adjusted: pd.Series | None, splits: pd.DataFrame | None, name: str = "prices", tolerance: float = 0.15) -> list[Finding]:
    """Consistency of the prices with the recorded corporate actions. A recorded split (``date``, ``ratio`` new shares per old) should show in the UNadjusted close as a jump of about 1/ratio and not at all in the
    adjusted one; a jump of that size in the unadjusted close with no recorded split is an unexplained discontinuity."""
    out: list[Finding] = []
    c = pd.to_numeric(close, errors="coerce")
    c.index = pd.DatetimeIndex(pd.to_datetime(c.index))
    ratio = c / c.shift(1)
    recorded = {}
    if splits is not None and len(splits):
        for _, r in splits.iterrows():
            recorded[pd.Timestamp(r["date"]).normalize()] = float(r["ratio"])
    for d, ratio_split in recorded.items():
        if d in ratio.index and np.isfinite(ratio.loc[d]):
            expect = 1.0 / ratio_split
            if abs(ratio.loc[d] / expect - 1.0) > tolerance:
                out.append(Finding("split_not_in_prices", name, "warning", f"a {ratio_split:g}:1 split is recorded on {d.date()} but the close moved by x{ratio.loc[d]:.3f} (expected about x{expect:.3f})", d, float(ratio.loc[d])))
    for d, x in ratio.items():
        if not np.isfinite(x) or d in recorded:
            continue
        for k in (2, 3, 4, 5, 10, 20):
            if abs(x * k - 1.0) < 0.03 or abs(x / k - 1.0) < 0.03:
                out.append(Finding("unexplained_discontinuity", name, "warning", f"the close moved by x{x:.3f} on {d.date()}, close to a {k}:1 split, and no split is recorded", d, float(x)))
                break
    if adjusted is not None:
        a = pd.to_numeric(adjusted, errors="coerce")
        a.index = pd.DatetimeIndex(pd.to_datetime(a.index))
        f = (a / c).dropna()
        if len(f) > 1:
            step = (f / f.shift(1)).dropna()
            jumps = step[(step - 1).abs() > 1e-3]
            for d in list(jumps.index)[:20]:
                if d not in recorded and not any(abs((d - r).days) <= 1 for r in recorded):
                    out.append(Finding("adjustment_without_record", name, "info", f"the adjustment factor changed by x{step.loc[d]:.4f} on {d.date()} with no recorded action (a dividend, a split or a provider revision)", d, float(step.loc[d])))
    return out


def fingerprint(frames: dict[str, pd.DataFrame]) -> str:
    """A content hash of the data (names, dates and values), so that two runs on 'the same dataset' can be shown to be, or not to be, the same data."""
    h = hashlib.sha256()
    for k in sorted(frames):
        df = frames[k]
        h.update(k.encode())
        if df is not None and len(df):
            h.update(pd.util.hash_pandas_object(df.round(10), index=True).to_numpy().tobytes())
    return h.hexdigest()[:16]


# ------------------------------------------------------------------------------------------------------ readiness
@dataclass
class Requirements:
    min_observations: int = 250
    max_gap_periods: int = 5
    min_completeness: float = 0.95
    max_age_days: float | None = None
    allow_warnings: bool = True


def readiness(cov: dict, findings: list[Finding], req: Requirements | None = None, as_of=None) -> dict:
    """``ready``, ``ready_with_caveats`` or ``blocked`` for one series, with every reason. A blocking reason is an error finding, too few observations, too long a gap, too incomplete or too old; a caveat is a warning or
    a short gap."""
    req = req or Requirements()
    block: list[str] = []
    caveat: list[str] = []
    if cov["observations"] < req.min_observations:
        block.append(f"only {cov['observations']} observations; {req.min_observations} are needed")
    if cov["completeness"] < req.min_completeness and cov["observations"]:
        block.append(f"{cov['completeness']:.1%} of the expected periods are present; {req.min_completeness:.0%} are needed")
    longest = max((r["periods"] for r in cov["missing_runs"]), default=0)
    if longest > req.max_gap_periods:
        block.append(f"a gap of {longest} periods (more than {req.max_gap_periods} allowed)")
    elif longest:
        caveat.append(f"gaps up to {longest} periods")
    if req.max_age_days is not None and cov["last"] and as_of is not None:
        age = (pd.Timestamp(as_of).normalize() - pd.Timestamp(cov["last"])).days
        if age > req.max_age_days:
            block.append(f"the last observation is {age} days old (limit {req.max_age_days:g})")
    for f in findings:
        if f.severity == "error":
            block.append(f"{f.check}: {f.detail}")
        elif f.severity == "warning":
            (caveat if req.allow_warnings else block).append(f"{f.check}: {f.detail}")
    status = "blocked" if block else ("ready_with_caveats" if caveat else "ready")
    return {"series": cov["series"], "status": status, "blocking": block, "caveats": caveat}


# ------------------------------------------------------------------------------------------------------ corrections
class CorrectionLog:
    """An append-only JSON-lines record of every correction: the raw value is kept, the corrected value is stated, and nothing is applied to the stored data by writing here."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._lock = threading.Lock()

    def record(self, series: str, date, field_name: str, raw, corrected, reason: str, by: str = "user") -> dict:
        if not reason.strip():
            raise ValueError("a correction needs a reason")
        entry = {"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "series": series, "date": str(pd.Timestamp(date).date()), "field": field_name, "raw": raw, "corrected": corrected, "reason": reason.strip(), "by": by}
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, default=str) + "\n")
        return entry

    def entries(self, series: str | None = None) -> list[dict]:
        if not self.path.is_file():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if series is None or e.get("series") == series:
                out.append(e)
        return out

    def apply(self, df: pd.DataFrame, series: str) -> pd.DataFrame:
        """A COPY of ``df`` with this series' corrections applied (the raw frame is untouched); later entries win."""
        out = df.copy()
        for e in self.entries(series):
            d = pd.Timestamp(e["date"])
            if d in out.index and e["field"] in out.columns:
                out.loc[d, e["field"]] = e["corrected"]
        return out


@dataclass
class HealthReport:
    dataset: str
    generated_at: str
    fingerprint: str
    coverage: list[dict] = field(default_factory=list)
    findings: list[dict] = field(default_factory=list)
    readiness: list[dict] = field(default_factory=list)

    def counts(self) -> dict:
        c = {"ready": 0, "ready_with_caveats": 0, "blocked": 0}
        for r in self.readiness:
            c[r["status"]] += 1
        return c

    def as_dict(self) -> dict:
        return {"dataset": self.dataset, "generated_at": self.generated_at, "fingerprint": self.fingerprint, "coverage": self.coverage, "findings": self.findings, "readiness": self.readiness, "counts": self.counts()}


def report(dataset: str, frames: dict[str, pd.DataFrame], asset_class: dict | str = "equity", req: Requirements | None = None, as_of=None, expected_index: pd.DatetimeIndex | None = None, **check_kw) -> HealthReport:
    """Coverage, integrity and readiness of every series of a dataset in one pass. ``expected_index`` is the calendar of a panel whose series share dates (the platform dataset); without it each series is held to the calendar of its asset class."""
    cov = coverage(frames, asset_class, expected_index=expected_index)
    findings: list[Finding] = []
    ready = []
    for row in cov:
        name = row["series"]
        ac = asset_class.get(name, "equity") if isinstance(asset_class, dict) else asset_class
        fs = check_series(frames[name], name, asset_class=ac, as_of=as_of, max_gap_days=(req.max_gap_periods if req else Requirements().max_gap_periods), max_age_days=(req.max_age_days if req else None), expected_index=expected_index, **check_kw)
        findings += fs
        ready.append(readiness(row, fs, req, as_of))
    return HealthReport(dataset, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), fingerprint(frames), cov, [f.as_dict() for f in findings], ready)


def findings_csv(rep: dict) -> str:
    import csv
    import io

    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["dataset", rep["dataset"], "fingerprint", rep["fingerprint"], "generated_at", rep["generated_at"]])
    w.writerow(["series", "status", "check", "severity", "date", "detail"])
    status = {r["series"]: r["status"] for r in rep["readiness"]}
    for f in rep["findings"]:
        w.writerow([f["series"], status.get(f["series"], ""), f["check"], f["severity"], f["date"] or "", f["detail"]])
    for r in rep["readiness"]:
        if not any(f["series"] == r["series"] for f in rep["findings"]):
            w.writerow([r["series"], r["status"], "", "", "", "no findings"])
    return buf.getvalue()
