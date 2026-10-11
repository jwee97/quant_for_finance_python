"""The dataset builder: a market bundle in, a leak-free panel of features and targets out.

A ``Dataset`` is indexed by ``(date, asset)``. A row stamped ``t`` holds the features known at the close of ``t`` and the targets measured from ``t + lag`` onwards (``targets.py``), so a row can be used to
train only once its label is complete and to predict at ``t``. Features are normalised across the assets of the same date only (winsorised at the tails, then z-scored or ranked), which uses nothing
from any other date; market-wide features are left alone and standardised inside each training fold by the model. The last ``lag + horizon`` dates have features but no label: they are kept, because the
live prediction is made on exactly those rows.

Universes: ``etf`` is the platform's 15 ETFs with point-in-time macro and implied-volatility indices, ``crypto`` the Deribit BTC and ETH data and their delta-neutral carry indices, ``futures`` the SIMULATED
futures roots of the Futures workspace, and a list of tickers is resolved by the caller (the dashboard's ticker store, or a price file).
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .features import FEATURES, compute_features, inputs_from_bundle
from .targets import make_targets, parse_target

UNIVERSES = ("etf", "crypto", "futures")


def bundle_fingerprint(bundle) -> str:
    """A short hash of the bundle's shape, date range, assets and the last year of prices: two bundles with the same fingerprint carry the same data for our purposes."""
    digest = int(pd.util.hash_pandas_object(bundle.prices.tail(250).fillna(0.0), index=True).sum() % (2 ** 62))
    text = json.dumps([list(bundle.prices.columns), str(bundle.prices.index[0].date()), str(bundle.prices.index[-1].date()), bundle.prices.shape, digest])
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def resolve_bundle(config, universe, builder=None, as_of=None):
    """``(bundle, info)`` for a universe name or a list of tickers. ``info`` says where the data came from (``source``: platform / simulated / user) so the page can label it."""
    if isinstance(universe, str):
        if universe == "etf":
            from ..framework import load_default_bundle

            return (builder.default_bundle() if builder is not None else load_default_bundle(config)), {"source": "platform", "label": "the platform's 15 ETFs", "universe": "etf"}
        if universe == "crypto":
            from ..framework.crypto_bundle import load_crypto_bundle

            return load_crypto_bundle(config), {"source": "platform", "label": "Deribit BTC and ETH with their carry indices", "universe": "crypto"}
        if universe == "futures":
            from ..assets.futures_analytics import ROOTS, futures_bundle_for

            return futures_bundle_for(list(ROOTS), as_of), {"source": "simulated", "label": "simulated futures roots (Schwartz-Smith model)", "universe": "futures"}
        raise ValueError(f"unknown universe '{universe}'; choose from {list(UNIVERSES)} or give a list of tickers")
    tickers = [str(t).upper() for t in universe]
    if builder is None:
        raise ValueError("a list of tickers needs the dashboard's ticker store; use a universe name or a price file")
    bundle, info = builder.resolve(tickers, None, None, False, None)
    return bundle, {"source": "user", "label": f"{len(tickers)} tickers", "universe": tickers, **({"info": info} if isinstance(info, dict) else {})}


# ------------------------------------------------------------------------------------------------------------------ normalisation
def _row_quantiles(values: np.ndarray, q: float) -> np.ndarray:
    """The ``q`` quantile (linear interpolation, like ``numpy.quantile``) of each row, ignoring missing values; NaN for an empty row. Vectorised: ``DataFrame.quantile(axis=1)`` loops in Python."""
    srt = np.sort(values, axis=1)                                            # NaN sorts last
    n = np.isfinite(values).sum(axis=1)
    pos = q * np.maximum(n - 1, 0)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, np.maximum(n - 1, 0))
    rows = np.arange(len(values))
    frac = pos - lo
    out = srt[rows, lo] * (1.0 - frac) + srt[rows, hi] * frac
    return np.where(n > 0, out, np.nan)


def normalise_cross_section(frame: pd.DataFrame, how: str, winsor: float = 0.0, min_assets: int = 3) -> pd.DataFrame:
    """Per date: clip each tail at ``winsor`` (quantiles of that date's cross-section), then ``cross_sectional_z`` (mean 0, standard deviation 1) or ``rank`` (uniform in [-1, 1]). A date with fewer than
    ``min_assets`` valid values is left missing: there is no cross-section to speak of."""
    if how == "none":
        return frame
    vals = frame.to_numpy(dtype=np.float64)
    n = np.isfinite(vals).sum(axis=1)
    if winsor > 0:
        lo, hi = _row_quantiles(vals, winsor), _row_quantiles(vals, 1.0 - winsor)
        vals = np.clip(vals, lo[:, None], hi[:, None])
    if how == "cross_sectional_z":
        with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)                         # an all-missing date has no mean: it stays missing
            mean = np.nanmean(vals, axis=1, keepdims=True)
            sd = np.nanstd(vals, axis=1, keepdims=True)
            out = (vals - mean) / np.where(sd > 0, sd, np.nan)
    elif how == "rank":
        out = pd.DataFrame(vals, index=frame.index, columns=frame.columns).rank(axis=1, pct=True).to_numpy() * 2.0 - 1.0
    else:
        raise ValueError("normalize must be none, cross_sectional_z or rank")
    out = np.where((n >= min_assets)[:, None], out, np.nan)
    return pd.DataFrame(out, index=frame.index, columns=frame.columns)


def _flatten(frame: pd.DataFrame) -> np.ndarray:
    return frame.to_numpy(dtype=np.float32).reshape(-1)


# ------------------------------------------------------------------------------------------------------------------ the dataset
@dataclass
class Dataset:
    name: str
    X: pd.DataFrame                      # (date, asset) x features
    Y: pd.DataFrame                      # (date, asset) x targets
    prices: pd.DataFrame                 # wide, adjusted
    returns: pd.DataFrame                # wide simple returns
    investable: pd.DataFrame
    meta: dict = field(default_factory=dict)
    asset_class: dict = field(default_factory=dict)
    group: dict = field(default_factory=dict)
    _pos: np.ndarray | None = field(default=None, repr=False)

    # -- structure
    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.prices.index)

    @property
    def assets(self) -> list[str]:
        return list(self.prices.columns)

    @property
    def features(self) -> list[str]:
        return list(self.X.columns)

    @property
    def target(self) -> str:
        return self.meta["target"]

    @property
    def lag(self) -> int:
        return int(self.meta["lag"])

    @property
    def horizon(self) -> int:
        return int(self.meta["horizon"])

    def positions(self) -> np.ndarray:
        """The position in ``dates`` of every row's date."""
        if self._pos is None:
            self._pos = self.dates.get_indexer(self.X.index.get_level_values(0))
        return self._pos

    def rows(self, date_positions: np.ndarray | range | slice) -> np.ndarray:
        """Boolean mask of the rows whose date position is in ``date_positions``."""
        pos = self.positions()
        if isinstance(date_positions, slice):
            lo, hi = date_positions.start or 0, date_positions.stop if date_positions.stop is not None else len(self.dates)
            return (pos >= lo) & (pos < hi)
        arr = np.asarray(date_positions)
        if len(arr) and arr[-1] - arr[0] + 1 == len(arr):                                           # contiguous: two comparisons instead of a set lookup
            return (pos >= arr[0]) & (pos <= arr[-1])
        return np.isin(pos, arr)

    def xy(self, date_positions, target: str | None = None) -> tuple[pd.DataFrame, pd.Series]:
        """Features and the (non-missing) target of the rows on those dates."""
        t = target or self.target
        mask = self.rows(date_positions)
        y = self.Y[t].iloc[np.flatnonzero(mask)]
        keep = y.notna().to_numpy()
        return self.X.iloc[np.flatnonzero(mask)[keep]], y[keep]

    def latest_rows(self) -> pd.DataFrame:
        """The features on the last date: what a live prediction uses."""
        last = self.dates[-1]
        return self.X.xs(last, level=0, drop_level=False)

    def summary(self) -> dict:
        n = len(self.X)
        return {"name": self.name, "n_rows": int(n), "n_dates": int(len(self.dates)), "n_assets": len(self.assets), "n_features": len(self.features), "start": str(self.dates[0].date()),
                "end": str(self.dates[-1].date()), "missing_share": float(self.X.isna().to_numpy().mean()) if n else float("nan"), "target": self.target, **{k: self.meta[k] for k in ("horizon", "lag", "universe", "source") if k in self.meta}}

    # -- persistence
    def save(self, directory: str | Path) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.X.to_parquet(d / "features.parquet")
        self.Y.to_parquet(d / "targets.parquet")
        self.prices.to_parquet(d / "prices.parquet")
        self.returns.to_parquet(d / "returns.parquet")
        self.investable.to_parquet(d / "investable.parquet")
        (d / "meta.json").write_text(json.dumps({**self.meta, "asset_class": self.asset_class, "group": self.group, "name": self.name}, indent=1, default=str))
        return d

    @classmethod
    def load(cls, directory: str | Path) -> "Dataset":
        d = Path(directory)
        meta = json.loads((d / "meta.json").read_text())
        name, asset_class, group = meta.pop("name"), dict(meta.pop("asset_class", {})), dict(meta.pop("group", {}))
        return cls(name, pd.read_parquet(d / "features.parquet"), pd.read_parquet(d / "targets.parquet"), pd.read_parquet(d / "prices.parquet"), pd.read_parquet(d / "returns.parquet"),
                   pd.read_parquet(d / "investable.parquet"), meta, asset_class, group)


def build_dataset(bundle, dataset_cfg: dict, name: str = "dataset", extra_targets: tuple | list = (), source: dict | None = None) -> Dataset:
    """Build the panel for ``bundle`` from the ``dataset`` section of an experiment configuration (``config.experiment_config``). ``extra_targets`` are built alongside the primary target."""
    c = dataset_cfg
    target, lag = str(c["target"]), int(c["lag"])
    kind, horizon = parse_target(target)
    if horizon != int(c["horizon"]):
        raise ValueError(f"the target {target} looks {horizon} days ahead but dataset.horizon is {c['horizon']}: they must agree")
    start, end = c.get("start"), c.get("end")
    if start or end:
        sl = slice(pd.Timestamp(start) if start else None, pd.Timestamp(end) if end else None)
        bundle = _slice_bundle(bundle, sl)
    x = inputs_from_bundle(bundle)
    fs = compute_features(x, list(c["feature_groups"]))
    if c.get("regime_detector"):
        fs.frames["regime_state"] = _regime_frame(bundle, str(c["regime_detector"]), x)
        fs.broadcast.add("regime_state")
    if not fs.frames:
        raise ValueError("no feature could be computed from this data: " + "; ".join(sorted(set(fs.skipped.values()))[:3]))
    names = [target] + [t for t in extra_targets if t != target]
    targets = make_targets(names, x.prices, x.returns, lag, str(c.get("benchmark", "equal_weight")))
    idx = pd.MultiIndex.from_product([x.prices.index, x.assets], names=["date", "asset"])
    norm = str(c.get("normalize", "cross_sectional_z"))
    cols = {}
    for fname, frame in fs.frames.items():
        cols[fname] = _flatten(frame if fname in fs.broadcast else normalise_cross_section(frame, norm, float(c.get("winsor", 0.0))))
    X = pd.DataFrame(cols, index=idx)
    Y = pd.DataFrame({t: _flatten(f) for t, f in targets.items()}, index=idx)
    seen = x.prices.notna().cumsum()
    ok = (bundle.investable.reindex_like(x.prices).fillna(False).astype(bool) & (seen >= int(c.get("min_history", 252))) & x.prices.notna()).to_numpy().reshape(-1)
    coverage = X.notna().mean(axis=1).to_numpy() >= 0.5
    keep = ok & coverage
    X, Y = X[keep], Y[keep]
    if X.empty:
        raise ValueError("no row survives the investability, history and coverage filters: the data is too short for this configuration")
    meta = {"universe": c.get("universe"), "target": target, "kind": kind, "horizon": horizon, "lag": lag, "feature_groups": list(c["feature_groups"]), "normalize": norm, "winsor": float(c.get("winsor", 0.0)),
            "benchmark": str(c.get("benchmark", "equal_weight")), "skipped": fs.skipped, "broadcast": sorted(fs.broadcast), "targets": names, "data_fingerprint": bundle_fingerprint(bundle),
            "bundle": getattr(bundle, "name", "bundle"), "min_history": int(c.get("min_history", 252)), **(source or {})}
    return Dataset(name, X, Y, x.prices, x.returns, bundle.investable.reindex_like(x.prices).fillna(False).astype(bool), meta, dict(bundle.asset_class or {}), dict(bundle.group or {}))


def _slice_bundle(bundle, sl: slice):
    """The bundle restricted to a date range (its data only; nothing is recomputed)."""
    from dataclasses import replace

    pick = lambda f: None if f is None else f.loc[sl]                                  # noqa: E731
    return replace(bundle, prices=bundle.prices.loc[sl], returns=bundle.returns.loc[sl], investable=bundle.investable.loc[sl], high=pick(bundle.high), low=pick(bundle.low), volume=pick(bundle.volume),
                   macro=bundle.macro.loc[sl] if len(bundle.macro) else bundle.macro, market=None)


def _regime_frame(bundle, detector: str, x) -> pd.DataFrame:
    """The market's regime state (0, 1, 2 ... in the detector's own order, missing before it has enough history) on every date, for every asset: from one of the platform's walk-forward regime detectors."""
    from ..framework import DETECTORS, load_library

    load_library()
    if detector not in DETECTORS:
        raise ValueError(f"unknown regime detector '{detector}'")
    probs = DETECTORS.create(detector).detect(bundle).probabilities
    codes = pd.Series(np.argmax(probs.fillna(-1.0).to_numpy(), axis=1), index=probs.index, dtype=float).where(probs.notna().any(axis=1))
    return pd.DataFrame(np.repeat(codes.reindex(x.prices.index).to_numpy()[:, None], len(x.assets), axis=1).astype(np.float32), index=x.prices.index, columns=x.assets)


def feature_catalog() -> list[dict]:
    """The features that exist, with their groups and requirements (for the dataset builder's picker)."""
    return [f.info() for f in FEATURES.values()]
