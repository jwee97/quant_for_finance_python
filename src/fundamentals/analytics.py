"""Company analytics built on the derived metrics: the DuPont decomposition of a change in return on equity, the Piotroski F-score, and an accounting-anomaly score of the Beneish kind.

The two scores are *screens*, not verdicts. The F-score counts nine yes-or-no signs of improving financial health (Piotroski 2000). The anomaly score is Beneish's (1999) eight-variable model of
how unusual a company's accounting changes look next to companies that were later found to have manipulated earnings; it was fitted on United States manufacturers of the 1980s and early 1990s, a
high value means "look closer", and the application never calls anything fraud on its strength. Both are missing, not guessed, whenever any input is missing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import Ctx, Derived, ratio, register_scores
from .pit import PeriodTables

DUPONT_STEPS = (("tax_burden", "Tax burden"), ("interest_burden", "Interest burden"), ("ebit_margin", "Operating margin"), ("asset_turnover", "Asset turnover"),
                ("equity_multiplier", "Equity multiplier"))
BENEISH_THRESHOLD = -1.78           # Beneish's cut-off at a 20:1 cost of missing a manipulator against flagging an honest company; the model's own sample, not a law


# ------------------------------------------------------------------------------------------------------------------ DuPont
def dupont(tables: PeriodTables, basis: str = "fy") -> pd.DataFrame:
    """The five-step DuPont identity for every period, and what each step contributed to the change in ROE from a year earlier.

    ``roe = tax_burden * interest_burden * operating_margin * asset_turnover * equity_multiplier`` (all on closing balances, so the identity is exact). The contributions use the logarithmic-mean
    Divisia method: ``contribution_i = L(roe, roe_prior) * ln(step_i / step_i_prior)`` with ``L(a, b) = (a - b) / ln(a / b)``, which splits the change exactly (the contributions add up to
    ``d_roe``) and does not depend on the order of the steps. It needs every step positive in both years; otherwise the contributions are missing and ``method`` says why.
    """
    table = tables.table(basis)
    ctx = Ctx(tables, basis)
    op = ctx.first("operating_income", "ebit")
    steps = {"tax_burden": ratio(ctx["net_income"], ctx["income_before_tax"], positive=False), "interest_burden": ratio(ctx["income_before_tax"], op, positive=False),
             "ebit_margin": ratio(op, ctx["revenue"]), "asset_turnover": ratio(ctx["revenue"], ctx["total_assets"]), "equity_multiplier": ratio(ctx["total_assets"], ctx["shareholders_equity"])}
    out = pd.DataFrame(steps, index=table.index)
    out["roe"] = ratio(ctx["net_income"], ctx["shareholders_equity"], positive=False).where(ctx["shareholders_equity"] > 0)
    out["roe_from_steps"] = out[list(steps)].prod(axis=1, skipna=False)
    prior = {name: tables.prior_of(basis, series) for name, series in steps.items()}                    # each step as it was a year before, from that row's own figures
    out["roe_prior"] = tables.prior_of(basis, out["roe"])
    out["d_roe"] = out["roe"] - out["roe_prior"]
    positive = (out["roe"] > 0) & (out["roe_prior"] > 0)
    for name in steps:
        positive &= (steps[name] > 0) & (prior[name] > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_mean = np.where(np.isclose(out["roe"], out["roe_prior"]), out["roe"], (out["roe"] - out["roe_prior"]) / np.log(out["roe"] / out["roe_prior"]))
        for name in steps:
            out[f"c_{name}"] = np.where(positive, log_mean * np.log(steps[name] / prior[name]), np.nan)
    out["method"] = np.where(positive, "log-mean (exact)", np.where(out["roe_prior"].notna(), "needs every step positive in both years", "no prior year"))
    out["c_total"] = out[[f"c_{k}" for k in steps]].sum(axis=1, skipna=False)
    return out


# ------------------------------------------------------------------------------------------------------------------ Piotroski
PIOTROSKI = (("pio_roa_positive", "Return on assets is positive"), ("pio_cfo_positive", "Operating cash flow is positive"), ("pio_roa_improved", "Return on assets improved"),
             ("pio_cash_beats_profit", "Operating cash flow exceeds net income"), ("pio_leverage_fell", "Leverage fell"), ("pio_liquidity_rose", "Current ratio rose"),
             ("pio_no_dilution", "No new shares were issued"), ("pio_gross_margin_rose", "Gross margin rose"), ("pio_turnover_rose", "Asset turnover rose"))


def _flag(condition: pd.Series, *inputs: pd.Series) -> pd.Series:
    """1.0 where the condition holds, 0.0 where it does not, missing where any input is missing."""
    known = pd.concat(inputs, axis=1).notna().all(axis=1)
    return pd.Series(np.where(known, condition.astype(float), np.nan), index=condition.index)


def _signal(name: str, label: str, fn):
    return Derived(name, label, "quality", "score", f"Piotroski signal: {label.lower()} (1 yes, 0 no, missing when an input is).", fn, stage=3)


def _pio_funcs() -> list[Derived]:
    return [
        _signal("pio_roa_positive", "Return on assets is positive", lambda c: _flag(c["roa"] > 0, c["roa"])),
        _signal("pio_cfo_positive", "Operating cash flow is positive", lambda c: _flag(c["operating_cash_flow"] > 0, c["operating_cash_flow"])),
        _signal("pio_roa_improved", "Return on assets improved", lambda c: _flag(c["d_roa"] > 0, c["d_roa"])),
        _signal("pio_cash_beats_profit", "Operating cash flow exceeds net income", lambda c: _flag(c["operating_cash_flow"] > c["net_income"], c["operating_cash_flow"], c["net_income"])),
        _signal("pio_leverage_fell", "Leverage fell", lambda c: _flag(c["debt_to_assets"] < c.prior("debt_to_assets"), c["debt_to_assets"], c.prior("debt_to_assets"))),
        _signal("pio_liquidity_rose", "Current ratio rose", lambda c: _flag(c["d_current_ratio"] > 0, c["d_current_ratio"])),
        _signal("pio_no_dilution", "No new shares were issued", lambda c: _flag(c["share_growth"] <= 0.0, c["share_growth"])),
        _signal("pio_gross_margin_rose", "Gross margin rose", lambda c: _flag(c["d_gross_margin"] > 0, c["d_gross_margin"])),
        _signal("pio_turnover_rose", "Asset turnover rose", lambda c: _flag(c["d_asset_turnover"] > 0, c["d_asset_turnover"])),
    ]


def _f_score(c: Ctx) -> pd.Series:
    signals = pd.concat([c[name] for name, _ in PIOTROSKI], axis=1)
    return signals.sum(axis=1, skipna=False)


# ------------------------------------------------------------------------------------------------------------------ Beneish
def _beneish_parts(c: Ctx) -> dict[str, pd.Series]:
    rev, rev0 = c["revenue"], c.prior("revenue")
    rec, rec0 = c["accounts_receivable"], c.prior("accounts_receivable")
    gm, gm0 = ratio(c["gross_profit"], rev), ratio(c.prior("gross_profit"), rev0)
    ta, ta0 = c["total_assets"], c.prior("total_assets")
    ca, ca0 = c["current_assets"], c.prior("current_assets")
    ppe, ppe0 = c["ppe"], c.prior("ppe")
    dep, dep0 = c["depreciation_amortization"], c.prior("depreciation_amortization")
    sga, sga0 = c["selling_general_admin"], c.prior("selling_general_admin")
    cl, cl0 = c["current_liabilities"], c.prior("current_liabilities")
    ltd, ltd0 = c["long_term_debt"], c.prior("long_term_debt")
    return {
        "dsri": ratio(ratio(rec, rev), ratio(rec0, rev0)),
        "gmi": ratio(gm0, gm),
        "aqi": ratio(1.0 - ratio(ca + ppe, ta, positive=True), 1.0 - ratio(ca0 + ppe0, ta0, positive=True)),
        "sgi": ratio(rev, rev0),
        "depi": ratio(ratio(dep0, dep0 + ppe0), ratio(dep, dep + ppe)),
        "sgai": ratio(ratio(sga, rev), ratio(sga0, rev0)),
        "lvgi": ratio(ratio(cl + ltd, ta), ratio(cl0 + ltd0, ta0)),
        "tata": ratio(c["net_income"] - c["operating_cash_flow"], ta),
    }


BENEISH_WEIGHTS = {"dsri": 0.920, "gmi": 0.528, "aqi": 0.404, "sgi": 0.892, "depi": 0.115, "sgai": -0.172, "tata": 4.679, "lvgi": -0.327}


def _beneish_m(c: Ctx) -> pd.Series:
    parts = _beneish_parts(c)
    total = pd.Series(-4.84, index=c.t.index)
    for name, weight in BENEISH_WEIGHTS.items():
        total = total + weight * parts[name]                                       # a missing variable leaves the score missing
    return total


def _anomaly_flag(c: Ctx) -> pd.Series:
    m = c["accounting_anomaly_score"]
    return pd.Series(np.where(m.notna(), (m > BENEISH_THRESHOLD).astype(float), np.nan), index=c.t.index)


def _definitions() -> list[Derived]:
    return [*_pio_funcs(),
            Derived("piotroski_f", "Piotroski F-score", "quality", "score", "The number of the nine Piotroski signs of improving financial health that hold (0-9); missing if any sign cannot be judged.", _f_score, stage=4),
            Derived("accounting_anomaly_score", "Accounting anomaly score", "quality", "score",
                    "Beneish's eight-variable score of how unusual the year's accounting changes look (receivables, margins, asset quality, growth, depreciation, SG&A, leverage, accruals). "
                    "Higher is more unusual. A screen to look closer, not evidence of fraud.", _beneish_m, stage=1),
            Derived("accounting_anomaly_flag", "Accounting anomaly flag", "quality", "score", f"1 where the anomaly score is above {BENEISH_THRESHOLD}, the model's own cut-off.", _anomaly_flag, stage=2)]


register_scores(_definitions())


def signals(tables: PeriodTables, basis: str = "fy") -> pd.DataFrame:
    """The nine Piotroski signs and the score for every period (the derived metrics must have been computed: :func:`src.fundamentals.metrics.compute`)."""
    table = tables.table(basis)
    keep = [name for name, _ in PIOTROSKI] + ["piotroski_f"]
    return table[[c for c in keep if c in table.columns]].copy()


def anomaly_parts(tables: PeriodTables, basis: str = "fy") -> pd.DataFrame:
    """The eight Beneish indices for every period, so the Data Manager and the company page can show *what* looks unusual."""
    ctx = Ctx(tables, basis)
    return pd.DataFrame(_beneish_parts(ctx), index=tables.table(basis).index)


def interpret_anomaly(score: float) -> str:
    if score is None or not np.isfinite(score):
        return "not available: an input is missing"
    if score > BENEISH_THRESHOLD:
        return "unusual accounting changes next to the model's sample: look closer (this is a screen, not evidence of manipulation)"
    return "within the range of the model's ordinary companies"


__all__ = ["dupont", "signals", "anomaly_parts", "interpret_anomaly", "PIOTROSKI", "DUPONT_STEPS", "BENEISH_THRESHOLD"]
