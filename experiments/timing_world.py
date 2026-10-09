"""Factor timing on simulated worlds with a known truth: does a model that forecasts a factor's premium from the calendar, the macroeconomy or the market's state find the timing that was planted, and does it
leave a factor alone when there is none to find?

Everything here is simulated, so the question is whether the timing tools recover what was planted and what the choice of state costs, not whether any timing rule works in a market. Nothing is tuned.

    python -m experiments.timing_world          # about two minutes
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

from src.equity import alpha_model as am
from src.equity.synthetic import simulate_earnings_world, simulate_timing_world
from src.framework import MODELS, bundle_from_prices, load_library
from src.strategies import factor_timing as ft
from src.strategies.factor_models import _grid
from src.utils.dates import rebalance_dates


def bundle_of(idx, prices, **kw):
    return bundle_from_prices(pd.DataFrame(prices, index=idx, columns=[f"S{i:02d}" for i in range(prices.shape[1])]), min_history=60, name="timing", **kw)


def monthly_ic(bundle, score, skip: int = 48, months=None) -> dict:
    grid = rebalance_dates(bundle.index, "monthly")
    px = bundle.prices.loc[grid]
    ic = am.information_coefficients({"x": score.loc[grid]}, px.shift(-1) / px - 1.0, "spearman", 15)["x"].iloc[skip:]
    if months is not None:
        ic = ic[pd.Series((ic.index.month % 12) + 1, index=ic.index).isin(months)]                 # keep the months the return is earned in
    return {"mean IC": float(ic.mean()), "t": float(ic.mean() / ic.std() * np.sqrt(len(ic)))}


def forecast_skill(model, bundle, skip: int = 48) -> dict:
    """The mean of (forecast premium x the premium actually earned) per month, in basis points of return per standard deviation squared, and its t-statistic: what acting on the premium forecast earns."""
    grid, forward = _grid(bundle)
    f = model.premia(bundle)
    gains = [f[n].to_numpy() * ft.factor_slopes(z, forward.to_numpy()) for n, z in model.exposures(bundle, grid).items()]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        product = np.nanmean(np.vstack(gains), axis=0)[skip:]
    product = product[np.isfinite(product)]
    return {"forecast x premium (bp)": 1e4 * float(product.mean()), "t": float(product.mean() / product.std() * np.sqrt(len(product)))}


# ------------------------------------------------------------------------------------------------------------------ 1. calendar
def calendar_table(seeds=(0, 1, 2, 3)) -> pd.DataFrame:
    """Momentum loses 2% a month per standard deviation in January and earns 0.4% in the other months; a second world pays the same every month. Each state, with the plain factor for comparison."""
    rows = {}
    for world, premium in (("January effect planted", lambda month, st: -0.02 if month == 1 else 0.004), ("no seasonality", lambda month, st: 0.004)):
        for label, kw in (("plain factor (no state)", {"state": "january", "shrink": 1e9}), ("state: january", {"state": "january"}), ("state: month of the year", {"state": "month"}),
                          ("state: month within the quarter", {"state": "quarter"}), ("state: November-April", {"state": "halloween"})):
            ics, skills, jan = [], [], []
            for seed in seeds:
                idx, prices, _ = simulate_timing_world(seed, premium=premium)
                bundle = bundle_of(idx, prices)
                model = MODELS.create("calendar_factor_timing", factor="mom", **kw)
                score = model.score(bundle)
                ics.append(monthly_ic(bundle, score)["mean IC"])
                jan.append(monthly_ic(bundle, score, months=[1])["mean IC"])
                skills.append(forecast_skill(model, bundle)["forecast x premium (bp)"])
            rows[(world, label)] = {"mean IC": np.mean(ics), "IC in the Januaries": np.mean(jan), "forecast x premium (bp)": np.mean(skills), "worlds": len(seeds)}
    out = pd.DataFrame(rows).T
    out.index.names = ["world", "model"]
    return out


# ------------------------------------------------------------------------------------------------------------------ 2. macro
def macro_table(seeds=(0, 1, 2, 3)) -> pd.DataFrame:
    """Momentum is paid 1.2% a month per standard deviation after up markets and loses as much after down markets (state: the market), or pays when policy tightens and loses when it eases (state: Fed)."""
    rows = {}
    up = lambda s, ret: int(np.prod(1 + ret[s - 251:s + 1].mean(axis=1)) - 1 > 0)
    for seed in seeds:
        # market state
        idx, prices, _ = simulate_timing_world(10 + seed, years=18, premium=lambda month, st: 0.012 if st == 1 else -0.012, state_of=up, market_regimes=True)
        bundle = bundle_of(idx, prices)
        for label, kw in (("plain factor (no state)", {"shrink": 1e9}), ("state: market up or down", {})):
            model = MODELS.create("macro_factor_timing", factor="mom", state="market", **kw)
            rows.setdefault(("market-state momentum", label), []).append({**monthly_ic(bundle, model.score(bundle), skip=60), **forecast_skill(model, bundle, skip=60)})
        # Fed state
        idx = pd.bdate_range("2004-01-05", periods=16 * 252)
        macro = pd.DataFrame({"DFF": 3.0 + 1.5 * np.sin(2 * np.pi * (np.arange(len(idx)) / 21.0) / 28.0)}, index=idx)

        class Shell:
            def __init__(self):
                self.macro, self.index = macro, idx
        shell = Shell()
        idx2, prices, _ = simulate_timing_world(20 + seed, years=16, premium=lambda month, st: {0: -0.012, 1: 0.0, 2: 0.012}.get(st, 0.0), state_of=lambda s, ret: int(ft.macro_state(shell, "fed", idx[[s]])[0]))
        bundle = bundle_of(idx2, prices, macro=macro)
        for label, kw in (("plain factor (no state)", {"shrink": 1e9}), ("state: Fed easing, flat or tightening", {})):
            model = MODELS.create("macro_factor_timing", factor="mom", state="fed", **kw)
            rows.setdefault(("Fed-policy momentum", label), []).append({**monthly_ic(bundle, model.score(bundle), skip=60), **forecast_skill(model, bundle, skip=60)})
    out = pd.DataFrame({k: pd.DataFrame(v).mean() for k, v in rows.items()}).T
    out.index.names = ["world", "model"]
    out["worlds"] = len(seeds)
    return out[["mean IC", "t", "forecast x premium (bp)", "worlds"]]


# ------------------------------------------------------------------------------------------------------------------ 3. earnings announcements
def earnings_table(seeds=(0, 1, 2, 3)) -> pd.DataFrame:
    """Companies that report in the same month of every quarter earn 2% more in the months they report (or nothing, in the control). Announcement dates from a file, and inferred from volume spikes."""
    rows = {}
    for world, premium in (("2% a month more in reporting months", 0.02), ("no premium (control)", 0.0)):
        for source in ("file", "volume"):
            res = []
            for seed in seeds:
                prices, volume, events = simulate_earnings_world(seed, premium=premium)
                bundle = bundle_from_prices(prices, volume=volume, min_history=60, name="earnings")
                model = MODELS.create("earnings_season_premium", source=source, path="/nonexistent/earnings.csv")
                if source == "file":
                    import tempfile
                    from pathlib import Path
                    with tempfile.TemporaryDirectory() as d:
                        path = Path(d) / "dates.csv"
                        rows_ = [(events.index[i].strftime("%Y-%m-%d"), events.columns[j]) for i, j in zip(*np.nonzero(events.to_numpy()))]
                        pd.DataFrame(rows_, columns=["date", "ticker"]).to_csv(path, index=False)
                        model = MODELS.create("earnings_season_premium", source="file", path=str(path))
                        res.append(monthly_ic(bundle, model.score(bundle), skip=36))
                else:
                    found = ft.announcement_events(bundle, "/nonexistent/earnings.csv", "volume")
                    hits = float((found.to_numpy() * events.to_numpy()).sum())
                    res.append({**monthly_ic(bundle, model.score(bundle), skip=36), "recall": hits / events.to_numpy().sum(), "precision": hits / max(found.to_numpy().sum(), 1.0)})
            rows[(world, "dates from a file" if source == "file" else "inferred from volume spikes")] = pd.DataFrame(res).mean()
    out = pd.DataFrame(rows).T
    out.index.names = ["world", "announcements"]
    return out


# ------------------------------------------------------------------------------------------------------------------ 4. the platform's 15 ETFs
def etf_table() -> pd.DataFrame:
    """The timing models on the platform's 15 ETFs (real prices and point-in-time macro series, about twenty years): the factor's own rank IC with and without the state, and what acting on the premium
    forecast earns. Fifteen assets and a few hundred months have little power: read it as a check that the machinery runs on real data and as a statement of how little such a sample can show."""
    from src.framework.data import load_default_bundle
    from src.utils.config import load_config

    bundle = load_default_bundle(load_config())
    rows = {}
    runs = [("calendar", "january", {}), ("calendar", "month", {}), ("calendar", "quarter", {}), ("calendar", "halloween", {})]
    runs += [("macro", s, {}) for s in ("market", "fed", "m1", "gdp", "inflation", "ppi")]
    for family, state, kw in runs:
        name = "calendar_factor_timing" if family == "calendar" else "macro_factor_timing"
        for label, extra in (("with the state", {}), ("plain factor", {"shrink": 1e9})):
            model = MODELS.create(name, factor="all", state=state, **extra)
            score = model.score(bundle)
            rows.setdefault(f"{family}: {state}", {}).update({f"IC {label}": monthly_ic(bundle, score, skip=60)["mean IC"], f"t {label}": monthly_ic(bundle, score, skip=60)["t"],
                                                              f"forecast x premium {label} (bp)": forecast_skill(model, bundle, skip=60)["forecast x premium (bp)"]})
    out = pd.DataFrame(rows).T
    return out[["IC plain factor", "t plain factor", "IC with the state", "t with the state", "forecast x premium plain factor (bp)", "forecast x premium with the state (bp)"]]


def run() -> dict:
    load_library()
    warnings.simplefilter("ignore")
    return {"calendar": calendar_table(), "macro": macro_table(), "earnings": earnings_table(), "etfs": etf_table()}


def main() -> None:
    pd.options.display.float_format = "{:.4f}".format
    pd.options.display.width = 220
    t0 = time.time()
    for name, table in run().items():
        print(f"== {name} ==")
        print(table.to_string(), "\n")
    print(f"{time.time() - t0:.0f} seconds")


if __name__ == "__main__":
    main()
