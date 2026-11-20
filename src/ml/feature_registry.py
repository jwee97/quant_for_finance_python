"""The feature registry and its evidence (research-backed feature engineering, phase 6).

Every feature the platform can build is listed with where its data comes from, *when it becomes known* (so a feature is never read before its value existed), how it is calculated, and the hypothesis that
makes it worth testing, with the reference that proposed it. The hypothesis is a reason to look, not a result: ``feature_table`` then measures each feature on the dataset it is asked about (its rank
information coefficient against the target, how stable that is across years) and ``ablation`` measures what the model loses when a whole group is taken away, against a control in which every feature is noise.

A feature with a high in-sample information coefficient is a candidate; one whose coefficient keeps its sign in most years is a stronger one; whether the model *uses* it out of sample is what the ablation
answers. None of them is evidence of a tradeable edge on its own.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pandas as pd

from . import evaluation as ev
from .features import FEATURES, GROUPS

VERSION = "feature-registry-v1"

# where a feature's data comes from, by what it needs (``FeatureSpec.needs``); a feature with no need uses adjusted closes
SOURCES = {"": "adjusted daily closes of the asset", "volume": "daily share or contract volume", "ohlc": "daily open, high, low and close", "raw_close": "unadjusted closes",
           "macro": "macro and rate series (the platform's point-in-time macro dataset)", "carry": "futures curve and carry data", "iv": "an implied-volatility index of the underlying",
           "crypto": "funding, basis and stablecoin data"}

# when a value is known, by group: the rule the platform applies so that a feature at the close of day t uses nothing after t
AVAILABILITY = {"price": "known at the close of day t (closes through t)", "momentum": "known at the close of day t (closes through t; the most recent month is skipped by construction)",
                "mean_reversion": "known at the close of day t (closes through t)", "volatility": "known at the close of day t (returns through t)",
                "volume": "known at the close of day t (end-of-day volume through t)", "relative": "known at the close of day t (all assets' closes through t)",
                "macro": "known at the close of day t + the series' publication delay (each macro series is shifted by the platform's recorded lag)",
                "futures": "known at the settlement of day t", "options": "known at the close of day t (the index's close)", "crypto": "known at the end of the day's funding interval",
                "fracdiff": "known at the close of day t (a fixed-width window ending at t)"}

# why the group is worth testing, and who proposed it
HYPOTHESES = {
    "price": ("Prices carry information about the state of an asset's trend and its distance from recent extremes.", "Brock, Lakonishok and LeBaron (1992), Simple technical trading rules and the stochastic properties of stock returns"),
    "momentum": ("Assets that did well over the past three to twelve months tend to keep doing well for a few months (underreaction), apart from the most recent month.",
                 "Jegadeesh and Titman (1993), Returns to buying winners and selling losers; Moskowitz, Ooi and Pedersen (2012), Time series momentum"),
    "mean_reversion": ("Over a few days to a month, past losers tend to rebound and winners to give back (liquidity provision, overreaction).", "Jegadeesh (1990), Evidence of predictable behavior of security returns; Lehmann (1990), Fads, martingales and market efficiency"),
    "volatility": ("Assets with high recent volatility have not been paid more for it; low-volatility assets have had higher risk-adjusted returns, and volatility itself clusters.",
                   "Ang, Hodrick, Xing and Zhang (2006), The cross-section of volatility and expected returns; Engle (1982), ARCH"),
    "volume": ("Unusual volume marks information arriving or liquidity changing; illiquid assets earn a premium and high-volume ones show short-run return patterns.",
               "Amihud (2002), Illiquidity and stock returns; Gervais, Kaniel and Mingelgrin (2001), The high-volume return premium"),
    "relative": ("An asset's return and risk relative to its peers separates asset-specific moves from market-wide ones, which carry different persistence.", "Moskowitz and Grinblatt (1999), Do industries explain momentum?"),
    "macro": ("Rates, spreads and the volatility index describe the regime in which other signals work or fail; they are the same for every asset on a date, so they condition, they do not rank.",
              "Fama and French (1989), Business conditions and expected returns"),
    "futures": ("The slope of a futures curve (carry and roll yield) compensates holders and predicts returns across commodities and rates.", "Koijen, Moskowitz, Pedersen and Vrugt (2018), Carry"),
    "options": ("Option-implied volatility, skew and term structure summarise the market's expectation of risk and jump fear, and have predicted returns in the cross-section.", "Xing, Zhang and Zhao (2010), What does the individual option volatility smirk tell us about future equity returns?"),
    "crypto": ("Funding and basis measure leveraged positioning in perpetual futures; extremes have preceded reversals.", "No settled academic reference; a practitioner hypothesis that this platform tests like any other."),
    "fracdiff": ("A fractional difference of the log price keeps part of the level's memory while making the series closer to stationary, which a model can use where returns discard it.",
                 "Lopez de Prado (2018), Advances in Financial Machine Learning, ch. 5"),
}


def registry(groups: list[str] | None = None) -> list[dict]:
    """One entry per feature: name, group, source, availability, calculation, lookback, memory, hypothesis and reference."""
    out = []
    for name, spec in FEATURES.items():
        if groups is not None and spec.group not in groups:
            continue
        needs = tuple(spec.needs) or ("",)
        hyp, ref = HYPOTHESES.get(spec.group, ("", ""))
        out.append({"name": name, "group": spec.group, "source": "; ".join(SOURCES.get(n, n) for n in needs), "availability": AVAILABILITY.get(spec.group, "known at the close of day t"),
                    "calculation": spec.description, "lookback_days": int(spec.lookback), "memory": spec.memory, "broadcast": bool(getattr(spec, "broadcast", False)), "hypothesis": hyp, "reference": ref})
    return out


def groups_summary() -> list[dict]:
    rows = []
    for g in GROUPS:
        feats = [n for n, s in FEATURES.items() if s.group == g]
        hyp, ref = HYPOTHESES.get(g, ("", ""))
        rows.append({"group": g, "n_features": len(feats), "hypothesis": hyp, "reference": ref, "availability": AVAILABILITY.get(g, "")})
    return rows


def feature_table(ds, target: str | None = None, min_dates: int = 60) -> list[dict]:
    """Each feature of the dataset against the target: the mean rank information coefficient across dates, its t-statistic, the share of years in which it kept its overall sign, and the coverage.

    The coefficient is measured on the dataset as given (in-sample for anything the model trains on): it ranks candidates, it does not validate them."""
    target = target or ds.target
    y = ds.Y[target].unstack().reindex(columns=ds.assets)
    reg = {r["name"]: r for r in registry()}
    rows = []
    for f in ds.features:
        x = ds.X[f].unstack().reindex(index=y.index, columns=ds.assets)
        ic = ev.rank_ic_wide(x, y).dropna()
        if len(ic) < min_dates:
            rows.append({"feature": f, "group": reg.get(f, {}).get("group"), "n_dates": int(len(ic)), "note": "too few dates with enough assets (market-wide features have no cross-section to rank)"})
            continue
        mean, sd = float(ic.mean()), float(ic.std(ddof=1))
        by_year = ic.groupby(ic.index.year).mean()
        same = float((np.sign(by_year) == np.sign(mean)).mean()) if mean != 0 else float("nan")
        rows.append({"feature": f, "group": reg.get(f, {}).get("group"), "n_dates": int(len(ic)), "mean_ic": mean, "ic_ir": mean / sd if sd > 0 else float("nan"),
                     "t_stat": mean / sd * np.sqrt(len(ic)) if sd > 0 else float("nan"), "years_same_sign": same, "n_years": int(len(by_year)), "coverage": float(x.notna().mean().mean())})
    return rows


def with_features(ds, names: list[str]):
    """The dataset restricted to ``names`` (same rows, same targets): the unit of an ablation."""
    miss = [n for n in names if n not in ds.X.columns]
    if miss:
        raise ValueError(f"not features of this dataset: {miss}")
    meta = {**ds.meta, "feature_subset": hashlib.sha256(",".join(names).encode()).hexdigest()[:10]}
    return replace(ds, X=ds.X[list(names)], meta=meta, _pos=None)


def noise_dataset(ds, n_features: int = 12, seed: int = 0):
    """A dataset whose features are standard normal noise, same index and targets: what a model scores when there is nothing to find. The reference for every ablation."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((len(ds.X), n_features)).astype(np.float32), index=ds.X.index, columns=[f"noise_{i}" for i in range(n_features)])
    return replace(ds, X=X, meta={**ds.meta, "feature_subset": "noise"}, _pos=None)


def ablation(ds, cfg: dict, info: dict, groups: list[str] | None = None, *, progress=None) -> dict:
    """Train the same model on the same splits with all features, without each group, and on noise; report rank IC and net Sharpe (out of sample) for each, and the change from the full set.

    A group whose removal changes nothing (within the noise reference's spread) is one the model does not need; a group whose removal helps was adding noise. The noise run says what a model with nothing
    to find scores here."""
    from .experiments import evaluate_run
    from .training import walk_forward_train

    progress = progress or (lambda m: None)
    present = {}
    for f in ds.features:
        g = FEATURES[f].group if f in FEATURES else None
        if g:
            present.setdefault(g, []).append(f)
    groups = [g for g in (groups or list(present)) if g in present]
    if not groups:
        raise ValueError("the dataset has none of the requested groups")

    def run(name: str, data) -> dict:
        progress(f"training: {name}")
        tr = walk_forward_train(data, cfg["model"], cfg["split"], cfg["seed"])
        e = evaluate_run(cfg, data, tr, info, None, None, False, None, False, False)
        oos = e.metrics.get("oos", {})
        return {"variant": name, "n_features": len(data.features), "rank_ic": float(e.report["ic"]["all"]["mean"]), "sharpe": oos.get("sharpe"), "cagr": oos.get("cagr")}

    base = run("all features", ds)
    rows = [base]
    for g in groups:
        rest = [f for f in ds.features if f not in present[g]]
        if not rest:
            continue
        r = run(f"without {g}", with_features(ds, rest))
        r.update({"group": g, "ic_change": r["rank_ic"] - base["rank_ic"], "sharpe_change": None if r["sharpe"] is None or base["sharpe"] is None else r["sharpe"] - base["sharpe"]})
        rows.append(r)
    noise = run("noise features only", noise_dataset(ds, max(3, min(12, len(ds.features)))))
    return {"rows": rows, "noise": noise, "base": base,
            "reading": ("A change in rank IC or Sharpe smaller than the gap between the full model and the noise run is hard to tell from nothing. Removing a group cannot create information: a model that does better "
                        "without a group was being hurt by its noise. One run per variant on one history: the differences are not tested for significance.")}
