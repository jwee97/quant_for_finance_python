"""Run the equity tools on a simulated world with statements and a planted truth, and print what they find.

The world (``src.equity.synthetic.simulate_fundamental_world``) has companies with consistent quarterly statements, analyst fields and daily prices, in which six drivers were given a known premium:
value (log book to price), quality (return on net operating assets), investment (an inverted U in abnormal capital expenditure), momentum, reversal and estimate revisions. Because the truth is known,
the question this script answers is whether the tools recover it, not whether it would make money: a real market pays what it pays. Nothing is tuned; the seed and size are the arguments.

    python -m experiments.equity_world               # 80 companies, 14 years, seed 0 (about half a minute)
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

from src.equity import alpha_model as am
from src.equity import contextual as cx
from src.equity import factors as F
from src.equity.fundamentals import FactorInputs, Fundamentals
from src.equity.synthetic import DEFAULT_PREMIA, simulate_fundamental_world
from src.utils.dates import rebalance_dates


def _ic_summary(frame: pd.DataFrame, forward: pd.DataFrame, skip: int) -> pd.Series:
    ic = am.information_coefficients({"x": frame}, forward, "spearman", 20)["x"].iloc[skip:]
    return pd.Series({"mean_ic": ic.mean(), "t": ic.mean() / ic.std() * np.sqrt(len(ic)), "months": len(ic)})


def run(n_assets: int = 80, n_years: int = 14, seed: int = 0, skip: int = 36) -> dict:
    """Every table the guides quote, for one world."""
    warnings.simplefilter("ignore")
    world = simulate_fundamental_world(n_assets=n_assets, n_years=n_years, seed=seed)
    x = FactorInputs(world.prices, Fundamentals.from_table(world.table))
    grid = rebalance_dates(world.prices.index, "monthly")
    px = world.prices.loc[grid]
    forward = px.shift(-1) / px - 1.0
    raw, _ = F.compute_all(x)
    on_grid = {k: v.loc[grid] for k, v in raw.items()}

    ic = pd.DataFrame({k: _ic_summary(v, forward, skip) for k, v in on_grid.items()}).T
    ic.insert(0, "style", [F.FACTORS[k].style for k in ic.index])
    ic.insert(1, "prior_sign", [F.FACTORS[k].sign for k in ic.index])

    # which of the value yields carries the information once the others are held fixed (every yield shares the market value in its denominator)
    realized = px / px.shift(1) - 1.0
    marginal = am.marginal_contributions({k: on_grid[k] for k in ("b2p", "earnings_yield", "cfo2ev", "s2ev")}, realized)

    # the alpha model on one factor per style, with and without learning the weights, and made orthogonal or not
    styles = {"value": ["b2p", "earnings_yield", "cfo2ev"], "quality": ["rnoa", "cfroi"], "momentum": ["ret9", "adj_ret9"], "reversal": ["ret1"], "revisions": ["earn_rev9"]}
    composite = {}
    for style, members in styles.items():
        z = [am.standardize(on_grid[m]) * F.FACTORS[m].sign for m in members]
        composite[style] = am.standardize(sum(z) / len(z))
    rows = {}
    for label, orth, weights in (("equal weights", False, "equal"), ("optimal weights", False, "optimal"), ("optimal, Gram-Schmidt", "gram_schmidt", "optimal"), ("optimal, symmetric", "symmetric", "optimal")):
        if weights == "equal":
            alpha = am.combine_factors(composite, pd.Series(1.0 / len(composite), index=list(composite)))
        else:
            alpha = am.optimal_alpha(composite, forward, orthogonalize=orth).alpha
        rows[label] = _ic_summary(alpha, forward, skip)
    for style, frame in composite.items():
        rows[f"{style} alone"] = _ic_summary(frame, forward, skip)
    alpha_table = pd.DataFrame(rows).T

    # a U-shape: capital expenditure
    features = cx.nonlinear_features({"icapx": on_grid["icapx"], "b2p": on_grid["b2p"], "rnoa": on_grid["rnoa"]}, [("quadratic", "icapx")])
    curved = cx.fama_macbeth_forecast(features, forward, window=60, min_obs=24)
    straight = cx.fama_macbeth_forecast({k: v for k, v in features.items() if k != "icapx^2"}, forward, window=60, min_obs=24)
    nonlinear = pd.DataFrame({"linear terms only": _ic_summary(straight.forecast, forward, skip), "with icapx squared": _ic_summary(curved.forecast, forward, skip)}).T

    return {"world": world, "ic": ic.sort_values("t", ascending=False), "marginal": marginal, "alpha": alpha_table, "nonlinear": nonlinear, "slopes": curved.average}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--assets", type=int, default=80)
    parser.add_argument("--years", type=int, default=14)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = run(args.assets, args.years, args.seed)
    print(f"Planted premia per month per standard deviation: {DEFAULT_PREMIA}\n")
    pd.options.display.float_format = "{:.3f}".format
    for title in ("ic", "marginal", "alpha", "nonlinear", "slopes"):
        print(f"== {title} ==")
        print(out[title].to_string(), "\n")


if __name__ == "__main__":
    main()
