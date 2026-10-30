"""Scores and yardsticks from the books on reading financial statements, computed from the same point-in-time tables as every other derived metric.

* **Distress.** Altman's Z-score family (Altman 1968; the Z' model for private firms and the Z'' model for non-manufacturers and emerging markets, Altman, Hartzell and Peck 1995). Each is a weighted sum
  of a few ratios of the balance sheet and the income statement, fitted to separate firms that failed from firms that did not; a zone says which side of the fitted cut-offs a company is on.
* **Return on capital** of Greenblatt's Magic Formula (2005): operating profit over the capital the business needs to run (working capital without cash and debt, plus fixed assets).
* **Owner earnings** (Buffett 1986): reported earnings plus the non-cash charges, less what has to be spent to stay where the business is.
* **Sustainable and internal growth** (Higgins 1977): how fast a company can grow with the profit it keeps, without selling shares and, for the internal rate, without borrowing more either.
* **Graham's defensive-investor tests** (Graham, *The Intelligent Investor*, chapter 14): the Graham number, net current asset value and five tests of the balance sheet, the earnings record and the dividends.
* **Net operating assets** (Hirshleifer, Hou, Teoh and Zhang 2004): how much of the balance sheet is operations financed by investors rather than cash, scaled by last year's assets.

Every rule is written out here once, in terms of the canonical figures, and follows the conventions of :mod:`metrics`: a ratio whose denominator cannot divide is missing, a missing input is missing and
never zero (the three places a missing line is read as zero are named where they occur: short-term debt, dividends and preferred stock), and a test that needs more years of history than the company has is
not judged. The scores are *screens*: they say where a company stands next to the rule's own sample, not what will happen to it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import MIN_INVESTED_CAPITAL, Ctx, _cash, ratio, metric

# ------------------------------------------------------------------------------------------------------------------ Altman
ALTMAN_ORIGINAL = (1.2, 1.4, 3.3, 0.6, 1.0)             # working capital, retained earnings, EBIT over assets; MARKET equity over liabilities; sales over assets (public manufacturers, 1968)
ALTMAN_PRIME = (0.717, 0.847, 3.107, 0.420, 0.998)        # the same five, with BOOK equity over liabilities (private firms)
ALTMAN_DOUBLE_PRIME = (6.56, 3.26, 6.72, 1.05)            # the first four only, book equity (non-manufacturers and emerging markets; no sales to assets, to take out the effect of the industry)
ALTMAN_ZONES = {"original": (1.81, 2.99), "prime": (1.23, 2.90), "double_prime": (1.10, 2.60)}      # (distress below, safe above); between the two is the grey zone
ALTMAN_PARTS = (("x1", "Working capital to assets"), ("x2", "Retained earnings to assets"), ("x3", "EBIT to assets"), ("x4", "Equity to liabilities"), ("x5", "Sales to assets"))
ALTMAN_MODELS = {"double_prime": ("altman_z_double_prime", "Altman Z″ (any industry)", ALTMAN_DOUBLE_PRIME), "prime": ("altman_z_prime", "Altman Z′ (manufacturers)", ALTMAN_PRIME)}


def altman_parts(c: Ctx) -> dict[str, pd.Series]:
    """The five ratios of the Z-score family, each over total assets except the fourth (book equity over total liabilities). A company with a negative equity keeps its (negative) fourth ratio."""
    assets = c["total_assets"]
    return {"x1": ratio(c["current_assets"] - c["current_liabilities"], assets), "x2": ratio(c["retained_earnings"], assets), "x3": ratio(c.first("operating_income", "ebit"), assets),
            "x4": ratio(c["shareholders_equity"], c["total_liabilities"]), "x5": ratio(c["revenue"], assets)}


def _weighted(parts: dict[str, pd.Series], weights: tuple) -> pd.Series:
    total = None
    for weight, key in zip(weights, ("x1", "x2", "x3", "x4", "x5")):
        term = weight * parts[key]
        total = term if total is None else total + term
    return total


def altman_zone(score: float | None, model: str) -> str | None:
    """``distress``, ``grey`` or ``safe`` for a score of ``model`` (``original``, ``prime`` or ``double_prime``); ``None`` where there is no score."""
    if score is None or not np.isfinite(score):
        return None
    low, high = ALTMAN_ZONES[model]
    return "distress" if score < low else "safe" if score > high else "grey"


@metric("retained_earnings_to_assets", "Retained earnings to assets", "leverage", "ratio", "Retained earnings over total assets: how much of the balance sheet was paid for by profit kept in the business (Altman's second ratio).")
def _re_to_assets(c): return ratio(c["retained_earnings"], c["total_assets"])


@metric("ebit_to_assets", "EBIT to assets", "profitability", "ratio", "Operating income over total assets: the return the assets earn before interest and tax (Altman's third ratio).")
def _ebit_to_assets(c): return ratio(c.first("operating_income", "ebit"), c["total_assets"])


@metric("equity_to_liabilities", "Equity to liabilities", "leverage", "ratio", "Book equity over total liabilities (Altman's fourth ratio, in the book-value models); negative when the company owes more than it owns.")
def _equity_to_liabilities(c): return ratio(c["shareholders_equity"], c["total_liabilities"])


@metric("altman_z_double_prime", "Altman Z″-score", "leverage", "score",
        "6.56 working capital over assets + 3.26 retained earnings over assets + 6.72 EBIT over assets + 1.05 book equity over liabilities (Altman, Hartzell and Peck 1995). Below 1.10 is the distress zone, above 2.60 the safe zone. "
        "Built for non-manufacturers and emerging markets; a screen next to the sample it was fitted on, not a forecast of failure.")
def _z_double_prime(c): return _weighted(altman_parts(c), ALTMAN_DOUBLE_PRIME)


@metric("altman_z_prime", "Altman Z′-score", "leverage", "score",
        "0.717 working capital over assets + 0.847 retained earnings over assets + 3.107 EBIT over assets + 0.420 book equity over liabilities + 0.998 sales over assets (Altman 1983, in Altman 2000). Below 1.23 is the distress zone, "
        "above 2.90 the safe zone. Fitted on manufacturers; it leans on asset turnover, so it flatters retailers and penalises asset-heavy firms.")
def _z_prime(c): return _weighted(altman_parts(c), ALTMAN_PRIME)


# ------------------------------------------------------------------------------------------------------------------ Greenblatt
def greenblatt_capital(c: Ctx) -> pd.Series:
    """The capital Greenblatt sets the profit against: working capital without cash and without borrowings, plus net property, plant and equipment, floored at 5% of total assets so that a company that
    is paid in advance (negative working capital) does not get an infinite return. A line of short-term debt that is not reported is read as none."""
    working = (c["current_assets"] - _cash(c)) - (c["current_liabilities"] - c["short_term_debt"].fillna(0.0))
    capital = working + c["ppe"]
    return np.maximum(capital, MIN_INVESTED_CAPITAL * c["total_assets"]).where(c["total_assets"] > 0)


@metric("greenblatt_roc", "Return on capital (Magic Formula)", "profitability", "ratio",
        "Operating income over net working capital (current assets less cash, less current liabilities less short-term debt) plus net property, plant and equipment, with the capital floored at 5% of assets (Greenblatt 2005).")
def _greenblatt_roc(c): return ratio(c.first("operating_income", "ebit"), greenblatt_capital(c))


# ------------------------------------------------------------------------------------------------------------------ owner earnings
@metric("owner_earnings", "Owner earnings", "cashflow", "currency",
        "Net income plus depreciation and amortisation less capital expenditure (Buffett 1986). All spending is treated as the spending needed to keep the business where it is, and the change in working capital is left out: a rough proxy.")
def _owner_earnings(c): return c["net_income"] + c["depreciation_amortization"] - c["capital_expenditure"]


@metric("owner_earnings_margin", "Owner earnings margin", "cashflow", "ratio", "Owner earnings over revenue.")
def _owner_earnings_margin(c): return ratio(_owner_earnings(c), c["revenue"])


# ------------------------------------------------------------------------------------------------------------------ growth a company can finance
def _dividends(c: Ctx) -> pd.Series:
    """Dividends paid, where a cash-flow statement with no dividend line is read as none paid (companies that do not pay do not report the line)."""
    paid = c["dividends_paid"]
    return paid.fillna(0.0).where(paid.notna() | c["operating_cash_flow"].notna())


@metric("retention_ratio", "Retention ratio", "growth", "ratio", "One less the dividend payout: the share of profit kept in the business. Missing when net income is not positive. A cash-flow statement without a dividend line counts as no dividend.")
def _retention(c): return 1.0 - ratio(_dividends(c), c["net_income"])


def _compounded(rate: pd.Series) -> pd.Series:
    """``k / (1 - k)`` for ``k < 1``: the rate at which the base grows when the profit it earns (``k`` of it) is added to the base the profit was earned on."""
    return (rate / (1.0 - rate)).where(rate < 1.0)


@metric("sustainable_growth", "Sustainable growth rate", "growth", "ratio",
        "Retention ratio times return on equity, over one less that product (Higgins 1977): the growth in equity, and in sales at a constant leverage and margin, that the profit kept can pay for with no new shares.")
def _sustainable(c): return _compounded(_retention(c) * ratio(c["net_income"], c["shareholders_equity"]))


@metric("internal_growth", "Internal growth rate", "growth", "ratio",
        "Retention ratio times return on assets, over one less that product: the growth the profit kept can pay for with no new shares and no new borrowing at all.")
def _internal(c): return _compounded(_retention(c) * ratio(c["net_income"], c["total_assets"]))


@metric("eps_cagr_3y", "Earnings per share growth, 3-year", "growth", "ratio", "Compound annual growth in diluted earnings per share over three years; missing unless both ends are positive.")
def _eps_cagr(c):
    now, then = c["eps_diluted"], c.prior("eps_diluted", 3)
    return (ratio(now, then).where(now > 0)) ** (1.0 / 3.0) - 1.0


# ------------------------------------------------------------------------------------------------------------------ Graham
GRAHAM_MULTIPLIER = 22.5            # 15 times earnings times 1.5 times book value
GRAHAM_CURRENT_RATIO = 2.0
GRAHAM_MIN_YEARS = 5                # the earnings and dividend tests are judged on the years held, and not at all on fewer than this many
GRAHAM_WINDOW = 10                  # ... looking back at most this many
GRAHAM_GROWTH = 4.0 / 3.0           # "an increase of at least one-third" in earnings per share
GRAHAM_GAP = 7                      # three-year averages at the two ends of ten years: the windows end seven years apart
GRAHAM_NCAV_BUY = 2.0 / 3.0         # Graham bought net-nets at two-thirds of net current asset value or less
GRAHAM_TESTS = (("graham_current_ratio_ok", "Current ratio of at least 2"), ("graham_debt_ok", "Long-term debt within working capital"), ("graham_earnings_ok", "Earnings in every year held"),
                ("graham_dividends_ok", "A dividend in every year held"), ("graham_growth_ok", "Earnings per share up by a third over ten years"))
GRAHAM_SHORT = {"graham_current_ratio_ok": "Current ratio ≥ 2", "graham_debt_ok": "Debt ≤ working capital", "graham_earnings_ok": "Profit every year", "graham_dividends_ok": "Dividend every year",
                "graham_growth_ok": "EPS up by a third"}


def _shares(c: Ctx) -> pd.Series:
    return c.first("shares_outstanding", "weighted_avg_shares_diluted")


def _history(c: Ctx, name: str, years: int = GRAHAM_WINDOW) -> pd.DataFrame:
    """This year's figure and each of the years before it, one column per year back (missing where the company has no row about that many years earlier)."""
    return pd.concat([c[name]] + [c.prior(name, k) for k in range(1, years)], axis=1)


def _flag(condition: pd.Series, *inputs: pd.Series) -> pd.Series:
    known = pd.concat(inputs, axis=1).notna().all(axis=1)
    return pd.Series(np.where(known, condition.astype(float), np.nan), index=condition.index)


@metric("eps_avg_3y", "Average earnings per share, 3 years", "value", "per_share", "The mean of diluted earnings per share over this year and the two before it (Graham's yardstick: one year's earnings can be a peak or a trough).")
def _eps_avg(c): return (c["eps_diluted"] + c.prior("eps_diluted", 1) + c.prior("eps_diluted", 2)) / 3.0


@metric("book_value_per_share", "Book value per share", "value", "per_share", "Shareholders' equity over the shares in issue (the diluted weighted average where the count at the year end is not reported).")
def _bvps(c): return ratio(c["shareholders_equity"], _shares(c))


@metric("ncav", "Net current asset value", "value", "currency", "Current assets less all liabilities and preferred stock (Graham): what shareholders would be left with if only the current assets were worth anything. A missing preferred line counts as none.")
def _ncav(c): return c["current_assets"] - c["total_liabilities"] - c["preferred_equity"].fillna(0.0)


@metric("ncav_per_share", "Net current asset value per share", "value", "per_share", "Net current asset value over the shares in issue. Graham bought when the price was two-thirds of this or less (Oppenheimer 1986 tested the rule).",
        stage=2)
def _ncav_per_share(c): return ratio(c["ncav"], _shares(c))


@metric("graham_number", "Graham number", "value", "per_share", "The square root of 22.5 times the three-year average earnings per share times book value per share: the price at which the share is on Graham's limit of 15 times earnings and 1.5 times book. Missing unless both are positive.",
        stage=2)
def _graham_number(c):
    product = GRAHAM_MULTIPLIER * c["eps_avg_3y"] * c["book_value_per_share"]
    return np.sqrt(product.where(product > 0))


@metric("graham_current_ratio_ok", "Graham: current ratio of at least 2", "quality", "score", "1 where current assets are at least twice current liabilities (the defensive investor's test of financial strength), 0 where not.", stage=2)
def _t_current(c): return _flag(c["current_ratio"] >= GRAHAM_CURRENT_RATIO, c["current_ratio"])


@metric("graham_debt_ok", "Graham: long-term debt within working capital", "quality", "score", "1 where long-term debt does not exceed working capital (current assets less current liabilities), 0 where it does.", stage=2)
def _t_debt(c):
    long_term = c["long_term_debt"].fillna(c["total_debt"] - c["short_term_debt"].fillna(0.0))
    return _flag(long_term <= c["working_capital"], long_term, c["working_capital"])


@metric("graham_earnings_ok", "Graham: earnings in every year held", "quality", "score",
        f"1 where net income was positive in every one of the last {GRAHAM_WINDOW} years the database holds, 0 where any was not; not judged on fewer than {GRAHAM_MIN_YEARS} years (Graham asked for ten).", stage=2)
def _t_earnings(c):
    history = _history(c, "net_income")
    return pd.Series(np.where(history.notna().sum(axis=1) >= GRAHAM_MIN_YEARS, (history.fillna(1.0) > 0).all(axis=1).astype(float), np.nan), index=c.t.index)


@metric("graham_dividends_ok", "Graham: a dividend in every year held", "quality", "score",
        f"1 where a dividend was paid in every one of the last {GRAHAM_WINDOW} years with a cash-flow statement, 0 where one was missed; not judged on fewer than {GRAHAM_MIN_YEARS} years (Graham asked for twenty).", stage=2)
def _t_dividends(c):
    paid, present = (_history(c, "dividends_paid").fillna(0.0) > 0).to_numpy(), _history(c, "operating_cash_flow").notna().to_numpy()          # arrays: the two frames' columns are named differently
    return pd.Series(np.where(present.sum(axis=1) >= GRAHAM_MIN_YEARS, (paid | ~present).all(axis=1).astype(float), np.nan), index=c.t.index)


@metric("graham_growth_ok", "Graham: earnings per share up by a third", "quality", "score",
        "1 where the mean earnings per share of the last three years is at least four-thirds of the mean of the three years that ended seven years before, 0 where it is not; missing unless all six years are held and the earlier mean is positive.", stage=2)
def _t_growth(c):
    history = _history(c, "eps_diluted", GRAHAM_GAP + 3)
    now, then = history.iloc[:, :3].mean(axis=1, skipna=False), history.iloc[:, GRAHAM_GAP:].mean(axis=1, skipna=False)
    return pd.Series(np.where((then > 0) & now.notna(), (now >= GRAHAM_GROWTH * then).astype(float), np.nan), index=c.t.index)


@metric("graham_criteria_judged", "Graham: tests that could be judged", "quality", "score", "How many of the five balance-sheet and record tests the history held is long enough to judge.", stage=3)
def _judged(c): return pd.concat([c[name] for name, _ in GRAHAM_TESTS], axis=1).notna().sum(axis=1).astype(float)


@metric("graham_criteria_met", "Graham: tests met", "quality", "score", "How many of the five tests are passed (read it with the number judged: a company that could be judged on two tests and passed both has met two).", stage=3)
def _met(c): return pd.concat([c[name] for name, _ in GRAHAM_TESTS], axis=1).sum(axis=1, skipna=True)


# ------------------------------------------------------------------------------------------------------------------ net operating assets
@metric("net_operating_assets", "Net operating assets", "investment", "currency", "Debt plus equity plus minority interest less cash: operating assets less operating liabilities (the same sum that funds the business, Hirshleifer, Hou, Teoh and Zhang 2004).")
def _noa(c): return c["total_debt"] + c["shareholders_equity"] + c["minority_interest"].fillna(0.0) - _cash(c)


@metric("noa_to_assets", "Net operating assets to assets", "investment", "ratio", "Net operating assets over last year's total assets. Companies with a high value, whose books carry more operations than cash flow can pay for, have tended to earn less afterwards.", stage=2)
def _noa_to_assets(c): return ratio(c["net_operating_assets"], c.prior("total_assets"))


@metric("noa_change", "Change in net operating assets", "investment", "ratio", "The year's change in net operating assets over last year's total assets: the balance-sheet accrual that profit not yet received in cash leaves behind.", stage=2)
def _noa_change(c): return ratio(c["net_operating_assets"] - c.prior("net_operating_assets"), c.prior("total_assets"))


# ------------------------------------------------------------------------------------------------------------------ what the company page shows
GRAHAM_RULES = {"current_ratio": GRAHAM_CURRENT_RATIO, "min_years": GRAHAM_MIN_YEARS, "window": GRAHAM_WINDOW, "growth": GRAHAM_GROWTH, "gap": GRAHAM_GAP, "multiplier": GRAHAM_MULTIPLIER, "ncav_buy": GRAHAM_NCAV_BUY}


def company_view(tables, rows: pd.DataFrame) -> dict:
    """The distress scores and the Graham tests of one company, year by year. ``rows`` are its fiscal years in order, taken from the annual table with the derived metrics attached.

    For each Altman model: the score, its zone, the ratios it is built from and what each ratio added to the score (weight times ratio), so the page can say what drives it.
    """
    ctx = Ctx(tables, "fy")
    parts = pd.DataFrame(altman_parts(ctx))
    names = dict(ALTMAN_PARTS)
    models = []
    for key, (metric_name, label, weights) in ALTMAN_MODELS.items():
        used = [k for k, _ in ALTMAN_PARTS][:len(weights)]
        low, high = ALTMAN_ZONES[key]
        series = []
        for index, year in zip(rows.index, rows["fiscal_year"]):
            score = float(rows.at[index, metric_name]) if metric_name in rows.columns else float("nan")
            series.append({"fiscal_year": int(year), "score": score, "zone": altman_zone(score, key), "ratios": {k: float(parts.at[index, k]) for k in used},
                           "contributions": {k: float(w * parts.at[index, k]) for k, w in zip(used, weights)}})
        models.append({"key": key, "metric": metric_name, "label": label, "zones": {"distress": low, "safe": high}, "weights": [{"key": k, "label": names[k], "weight": w} for k, w in zip(used, weights)],
                       "series": series})
    graham = {"tests": [{"key": k, "label": label, "short": GRAHAM_SHORT[k]} for k, label in GRAHAM_TESTS], "rules": GRAHAM_RULES,
              "rows": [{"fiscal_year": int(year), **{k: float(rows.at[index, k]) if k in rows.columns else float("nan") for k, _ in GRAHAM_TESTS},
                        "judged": float(rows.at[index, "graham_criteria_judged"]), "met": float(rows.at[index, "graham_criteria_met"])} for index, year in zip(rows.index, rows["fiscal_year"])]}
    return {"distress": {"models": models}, "graham": graham}
