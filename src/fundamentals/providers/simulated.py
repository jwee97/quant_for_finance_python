"""A simulated world of listed companies whose truth is known: statements, filing dates, prices, and a handful of planted alphas.

This is not a data source for decisions; it is what lets the whole research pipeline be run, demonstrated and *tested* without a key. Everything it returns is labelled ``simulated``.

What is planted, so that tests can check the machinery recovers it and does not invent more:

* **Change in ROIC** (``delta_roic``): the cross-sectional z-score of the year-on-year change in the return on invested capital, known from the filing date, adds to the next months' returns with a
  half-life of six months. Part of it (``anticipation``) leaks into prices in the three months before the filing, which is what makes a look-ahead backtest (using the period end as the date the figures
  became known) look better than the honest one.
* **FCF yield** (``fcf_yield``): free cash flow over market capitalisation, known at each month end, adds to the next month's return.
* **Asset growth** (``asset_growth``): the z-score of the year-on-year growth in assets, with a negative sign (the investment effect).
* **Gross profitability in Japan only** (``gp_japan``): a country-specific alpha, so the geographic analysis has something to find.
* Everything else (revenue growth, current ratio, margins, leverage, ...) has **no** effect on returns, which is what the false-discovery tests rely on.

Imperfections are included on purpose so the data-quality engine has something to find: a few periods without a filing date, one company whose statements are reported in a different currency from
the one it trades in, one with a unit error, a share of companies that delist during the sample (with their final losses), and :meth:`SimulatedProvider.restate` to simulate a later restatement.

The world is built once from a seed with numpy only (a few hundred companies take about a second) and is identical every time.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..store import CompanyProfile, PeriodStatement
from .base import FinancialsResult, FundamentalDataProvider

# market: (country, currency, annual filing lag days, quarterly lag days, weight, fiscal-year-end months, fx scale (units of local currency per USD at the start))
MARKETS = {
    "NYSE": ("US", "USD", 55, 35, 0.30, (12,), 1.0), "LSE": ("GB", "GBP", 90, 45, 0.15, (12, 3), 0.65), "TSE": ("JP", "JPY", 50, 40, 0.20, (3, 12), 110.0),
    "SGX": ("SG", "SGD", 60, 40, 0.10, (12, 6), 1.4), "ASX": ("AU", "AUD", 60, 45, 0.15, (6, 12), 1.3), "XETRA": ("DE", "EUR", 100, 50, 0.10, (12,), 0.8),
}
# sector: growth, operating margin, opex/gross-margin gap, asset turnover, capex intensity, leverage, receivables, inventory, payout, R&D intensity (None: not reported), seasonality (Q1..Q4 weights)
SECTORS = {
    "Technology": (0.09, 0.19, 0.28, 0.70, 0.04, 0.10, 0.17, 0.05, 0.15, 0.12, (0.23, 0.25, 0.25, 0.27)),
    "Health Care": (0.07, 0.16, 0.30, 0.65, 0.05, 0.15, 0.16, 0.08, 0.20, 0.14, (0.24, 0.25, 0.25, 0.26)),
    "Industrials": (0.04, 0.10, 0.18, 0.85, 0.05, 0.25, 0.17, 0.12, 0.35, 0.02, (0.23, 0.26, 0.25, 0.26)),
    "Consumer Discretionary": (0.05, 0.09, 0.25, 1.05, 0.04, 0.22, 0.08, 0.15, 0.30, None, (0.21, 0.23, 0.24, 0.32)),
    "Consumer Staples": (0.03, 0.11, 0.22, 1.10, 0.04, 0.24, 0.09, 0.11, 0.55, None, (0.25, 0.25, 0.25, 0.25)),
    "Energy": (0.02, 0.12, 0.10, 0.60, 0.12, 0.28, 0.12, 0.05, 0.45, None, (0.24, 0.26, 0.26, 0.24)),
    "Materials": (0.03, 0.13, 0.12, 0.70, 0.09, 0.26, 0.12, 0.13, 0.35, None, (0.24, 0.26, 0.26, 0.24)),
    "Utilities": (0.02, 0.17, 0.10, 0.28, 0.14, 0.40, 0.11, 0.03, 0.65, None, (0.26, 0.24, 0.24, 0.26)),
    "Communication": (0.04, 0.16, 0.30, 0.45, 0.10, 0.30, 0.12, 0.01, 0.40, None, (0.25, 0.25, 0.25, 0.25)),
}
INDUSTRIES = {s: [f"{s} A", f"{s} B"] for s in SECTORS}
ADJ = ["Alder", "Boreal", "Cobalt", "Delta", "Ember", "Fjord", "Garnet", "Harbor", "Indigo", "Juniper", "Kestrel", "Lumen", "Meridian", "Nimbus", "Onyx", "Pioneer", "Quarry", "Raven", "Summit", "Tundra"]
NOUN = ["Systems", "Holdings", "Industries", "Partners", "Group", "Works", "Dynamics", "Resources", "Networks", "Labs", "Energy", "Foods", "Materials", "Capital", "Devices"]
PLANTED = {"delta_roic": 0.0045, "fcf_yield": 0.0027, "asset_growth": 0.0018, "gp_japan": 0.0040, "anticipation": 0.5}


@dataclass
class SimConfig:
    n_companies: int = 120
    start_year: int = 2009
    end_year: int = 2022
    seed: int = 7
    markets: tuple = tuple(MARKETS)
    quarterly: bool = True
    planted: dict = field(default_factory=lambda: dict(PLANTED))
    delist_share: float = 0.08
    missing_filing_share: float = 0.03
    dividend_yield: float = 0.02


def _ar1(rng, shape, phi, sigma, mean=0.0, x0=None):
    """An AR(1) path along the last axis around ``mean`` (an array broadcastable to ``shape``)."""
    n, y = shape
    out = np.empty(shape)
    mean = np.broadcast_to(np.asarray(mean, dtype=float), shape) if np.ndim(mean) else np.full(shape, float(mean))
    out[:, 0] = mean[:, 0] + (sigma / np.sqrt(1 - phi ** 2)) * rng.standard_normal(n) if x0 is None else x0
    for j in range(1, y):
        out[:, j] = mean[:, j] + phi * (out[:, j - 1] - mean[:, j - 1]) + sigma * rng.standard_normal(n)
    return out


def _zscore_cols(x: np.ndarray) -> np.ndarray:
    """Standardise each column across firms (ignoring NaN), then clip at 3."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)                          # a column that is all NaN (the first year has no change) stays NaN
        mu, sd = np.nanmean(x, axis=0), np.nanstd(x, axis=0)
    return np.clip((x - mu) / np.where(sd > 0, sd, 1.0), -3.0, 3.0)


class SimulatedWorld:
    def __init__(self, config: SimConfig | None = None, **kwargs):
        self.cfg = config or SimConfig(**kwargs)
        self.rng = np.random.default_rng(self.cfg.seed)
        self._firms()
        self._statements()
        self._calendar()
        self._prices()

    # ------------------------------------------------------------------------------------------------------------ the firms
    def _firms(self) -> None:
        c, rng = self.cfg, self.rng
        n = c.n_companies
        weights = np.array([MARKETS[m][4] for m in c.markets], dtype=float)
        self.market = np.array(c.markets)[rng.choice(len(c.markets), size=n, p=weights / weights.sum())]
        self.country = np.array([MARKETS[m][0] for m in self.market])
        self.currency = np.array([MARKETS[m][1] for m in self.market])
        self.sector_names = list(SECTORS)
        self.sector_idx = rng.integers(0, len(self.sector_names), size=n)
        self.sector = np.array([self.sector_names[i] for i in self.sector_idx])
        self.industry = np.array([INDUSTRIES[s][rng.integers(0, 2)] for s in self.sector])
        self.quality = rng.standard_normal(n)
        self.beta = np.clip(1.0 + 0.25 * rng.standard_normal(n), 0.4, 1.8)
        self.idio_vol = rng.uniform(0.055, 0.12, size=n)
        # one company in twenty reports in US dollars whatever it trades in (outside the US)
        usd_reporter = (rng.random(n) < 0.06) & (self.country != "US")
        self.reporting_currency = np.where(usd_reporter, "USD", self.currency)
        self.fiscal_month = np.array([MARKETS[m][5][rng.integers(0, len(MARKETS[m][5]))] for m in self.market])
        names, tickers, seen = [], [], set()
        for i in range(n):
            name = f"{ADJ[rng.integers(0, len(ADJ))]} {NOUN[rng.integers(0, len(NOUN))]}"
            while name in seen:
                name = f"{ADJ[rng.integers(0, len(ADJ))]} {NOUN[rng.integers(0, len(NOUN))]} {rng.integers(2, 99)}"
            seen.add(name)
            names.append(name + " (simulated)")
            tickers.append(f"S{i + 1:04d}")
        self.name, self.ticker = np.array(names), np.array(tickers)
        # the one company with a unit error in one year, and where it happens
        self.unit_error = (int(rng.integers(0, n)), c.start_year + 3)

    # ------------------------------------------------------------------------------------------------------------ the statements
    def _statements(self) -> None:
        c, rng = self.cfg, self.rng
        n, years = c.n_companies, np.arange(c.start_year - 1, c.end_year + 1)
        self.years = years
        y = len(years)
        sp = np.array([SECTORS[s][:10] if SECTORS[s][9] is not None else SECTORS[s][:9] + (0.0,) for s in self.sector], dtype=float)
        g_s, m_s, opex_s, t_s, k_s, lev_s, rec_s, inv_s, pay_s, rd_s = (sp[:, j] for j in range(10))
        macro = 0.03 * rng.standard_normal(y)[None, :]
        growth = _ar1(rng, (n, y), 0.3, 0.06, (g_s + 0.015 * self.quality)[:, None]) + macro
        revenue0 = np.exp(rng.normal(np.log(1.5e9), 1.1, size=n)) * np.where(self.reporting_currency == "USD", 1.0, np.array([MARKETS[m][6] for m in self.market]))
        revenue = revenue0[:, None] * np.exp(np.cumsum(growth, axis=1))
        margin = _ar1(rng, (n, y), 0.75, 0.016, (m_s + 0.025 * self.quality + rng.normal(0, 0.02, n))[:, None])
        turnover = np.exp(_ar1(rng, (n, y), 0.8, 0.05, (np.log(t_s) + rng.normal(0, 0.25, n))[:, None]))
        leverage = np.clip(_ar1(rng, (n, y), 0.85, 0.02, (lev_s + rng.normal(0, 0.07, n))[:, None]), 0.0, 0.55)
        cash_r = np.clip(_ar1(rng, (n, y), 0.8, 0.02, (0.10 + rng.normal(0, 0.04, n))[:, None]), 0.01, 0.35)
        capex_i = np.clip(_ar1(rng, (n, y), 0.7, 0.01, (k_s + rng.normal(0, 0.01, n))[:, None]), 0.005, 0.30)
        da_r = np.clip(0.035 + 0.5 * (capex_i - k_s[:, None]) + 0.4 * k_s[:, None] + 0.004 * rng.standard_normal((n, y)), 0.01, 0.20)
        rate = np.clip(0.035 + 0.01 * rng.standard_normal((n, 1)) + 0.004 * rng.standard_normal((n, y)), 0.005, 0.10)
        tax = np.clip(0.22 + 0.04 * rng.standard_normal((n, y)), 0.05, 0.40)
        other_liab = np.clip(0.15 + 0.03 * rng.standard_normal((n, y)), 0.05, 0.3)
        assets = revenue / turnover
        debt = leverage * assets
        cash = cash_r * assets
        liabilities = debt + other_liab * assets
        equity = np.maximum(assets - liabilities, 0.08 * assets)
        liabilities = assets - equity
        receivables = rec_s[:, None] * revenue * np.exp(0.1 * rng.standard_normal((n, y)))
        inventory = inv_s[:, None] * revenue * np.exp(0.1 * rng.standard_normal((n, y)))
        payables = 0.09 * revenue * np.exp(0.1 * rng.standard_normal((n, y)))
        current_assets = np.minimum(cash + receivables + inventory + 0.03 * assets, 0.7 * assets)
        goodwill = 0.07 * assets * np.exp(0.2 * rng.standard_normal((n, y)))
        intangibles = 0.05 * assets * np.exp(0.2 * rng.standard_normal((n, y)))
        ppe = np.maximum(assets - current_assets - goodwill - intangibles, 0.05 * assets)
        short_debt = 0.25 * debt
        current_liab = np.minimum(short_debt + payables + 0.04 * assets, 0.9 * liabilities)
        ebit = margin * revenue
        gross_margin = margin + opex_s[:, None] + 0.01 * rng.standard_normal((n, y))
        gross_profit = gross_margin * revenue
        cost_of_revenue = revenue - gross_profit
        rd = rd_s[:, None] * revenue * np.exp(0.08 * rng.standard_normal((n, y)))
        sga = np.maximum(gross_profit - ebit - rd, 0.02 * revenue)
        da = da_r * assets
        avg_debt = (debt + np.concatenate([debt[:, :1], debt[:, :-1]], axis=1)) / 2
        interest = rate * avg_debt
        ebt = ebit - interest
        income_tax = tax * np.maximum(ebt, 0)
        net_income = ebt - income_tax
        sbc = (0.012 + 0.02 * (self.sector == "Technology"))[:, None] * revenue
        nwc = receivables + inventory - payables
        d_nwc = nwc - np.concatenate([nwc[:, :1], nwc[:, :-1]], axis=1)
        cfo = net_income + da + sbc - d_nwc + 0.01 * revenue * rng.standard_normal((n, y))
        capex = capex_i * revenue
        shares0 = np.exp(rng.normal(np.log(2.5e8), 0.8, size=n))
        share_growth = rng.normal(0.004, 0.012, size=n)
        shares = shares0[:, None] * np.exp(np.outer(share_growth, np.arange(y)) + 0.004 * rng.standard_normal((n, y)).cumsum(axis=1) * 0.2)
        dividends = pay_s[:, None] * np.maximum(net_income, 0) * np.exp(0.1 * rng.standard_normal((n, y)))
        repurchases = np.maximum(0.015 * revenue * rng.standard_normal((n, y)) + 0.01 * revenue, 0)
        acquisitions = np.where(rng.random((n, y)) < 0.15, 0.04 * revenue * rng.random((n, y)), 0.0)
        a = {"revenue": revenue, "cost_of_revenue": cost_of_revenue, "gross_profit": gross_profit, "research_and_development": rd, "selling_general_admin": sga, "operating_income": ebit,
             "ebit": ebit, "ebitda": ebit + da, "interest_expense": interest, "income_before_tax": ebt, "income_tax": income_tax, "net_income": net_income,
             "depreciation_amortization": da, "stock_based_compensation": sbc, "operating_cash_flow": cfo, "capital_expenditure": capex, "free_cash_flow": cfo - capex,
             "dividends_paid": dividends, "share_repurchases": repurchases, "acquisitions": acquisitions,
             "investing_cash_flow": -(capex + acquisitions) + 0.005 * revenue * rng.standard_normal((n, y)),
             "financing_cash_flow": -(dividends + repurchases) + 0.01 * revenue * rng.standard_normal((n, y))}
        s = {"cash": cash + 0.02 * assets, "cash_and_equivalents": cash, "accounts_receivable": receivables, "inventory": inventory, "current_assets": current_assets, "total_assets": assets,
             "ppe": ppe, "goodwill": goodwill, "intangible_assets": intangibles, "accounts_payable": payables, "current_liabilities": current_liab, "short_term_debt": short_debt,
             "long_term_debt": debt - short_debt, "total_debt": debt, "total_liabilities": liabilities, "shareholders_equity": equity, "shares_outstanding": shares,
             "weighted_avg_shares_diluted": shares * 1.015, "minority_interest": 0.01 * equity, "preferred_equity": np.zeros((n, y))}
        self.annual, self.stocks = a, s
        self.latent = {"roic": (ebit * (1 - tax)) / np.maximum(debt + equity - cash, 0.05 * assets), "asset_growth": np.concatenate([np.full((n, 1), np.nan), assets[:, 1:] / assets[:, :-1] - 1], axis=1),
                       "gross_profitability": gross_profit / assets}
        self.share_path = (shares0, share_growth)
        # sectors without R&D do not report it at all (a gap, not a zero)
        self.no_rd = rd_s == 0.0

    # ------------------------------------------------------------------------------------------------------------ dates
    def _calendar(self) -> None:
        c, rng = self.cfg, self.rng
        n, y = c.n_companies, len(self.years)
        self.period_end = np.empty((n, y, 4), dtype="datetime64[D]")
        self.filing_annual = np.empty((n, y), dtype="datetime64[D]")
        self.filing_quarter = np.empty((n, y, 4), dtype="datetime64[D]")
        lag_a = np.array([MARKETS[m][2] for m in self.market])
        lag_q = np.array([MARKETS[m][3] for m in self.market])
        for i in range(n):
            for j, year in enumerate(self.years):
                fye = pd.Timestamp(year=int(year), month=int(self.fiscal_month[i]), day=1) + pd.offsets.MonthEnd(0)
                for k in range(4):
                    end = fye - pd.offsets.MonthEnd(3 * (3 - k)) if k < 3 else fye
                    self.period_end[i, j, k] = np.datetime64(end.date())
                    self.filing_quarter[i, j, k] = np.datetime64((end + pd.Timedelta(days=int(lag_q[i] + rng.integers(-5, 11)))).date())
                self.filing_annual[i, j] = np.datetime64((fye + pd.Timedelta(days=int(lag_a[i] + rng.integers(-8, 19)))).date())
        self.missing_filing = rng.random((n, y)) < self.cfg.missing_filing_share

    # ------------------------------------------------------------------------------------------------------------ prices
    def _prices(self) -> None:
        c, rng = self.cfg, self.rng
        n = c.n_companies
        first_month = pd.Timestamp(year=c.start_year + 1, month=1, day=1) + pd.offsets.MonthEnd(0)
        self.months = pd.date_range(first_month, pd.Timestamp(year=c.end_year + 1, month=12, day=31), freq=pd.offsets.MonthEnd())
        m = len(self.months)
        pl = c.planted
        fye_idx = np.arange(n)
        # signals known per fiscal year, standardised across firms within the year
        roic = self.latent["roic"]
        droic = np.concatenate([np.full((n, 1), np.nan), roic[:, 1:] - roic[:, :-1]], axis=1)
        z_droic, z_ag, z_gp = _zscore_cols(droic), _zscore_cols(self.latent["asset_growth"]), _zscore_cols(self.latent["gross_profitability"])
        self.signals = {"delta_roic": z_droic, "asset_growth": z_ag, "gross_profitability": z_gp}
        shares0, share_growth = self.share_path
        ps_star = np.exp(rng.normal(np.log(1.6), 0.35, size=n) + 0.30 * self.quality)
        revenue = self.annual["revenue"]
        # the market, country and sector shocks
        f = rng.normal(0.006, 0.04, size=m)
        country_codes = sorted(set(self.country))
        cshock = {k: rng.normal(0, 0.02, size=m) for k in country_codes}
        sshock = {k: rng.normal(0, 0.015, size=m) for k in self.sector_names}
        # start: price consistent with the first known revenue
        j0 = 1
        mcap0 = revenue[:, j0] * ps_star * np.exp(rng.normal(0, 0.25, size=n))
        logp = np.log(mcap0 / (shares0 * np.exp(share_growth * j0)))
        # when does each firm delist, if it does
        last_month = np.full(n, m, dtype=int)
        gone = rng.random(n) < c.delist_share
        last_month[gone] = rng.integers(int(m * 0.35), m - 6, size=int(gone.sum()))
        self.last_month = last_month
        month_end_np = self.months.values.astype("datetime64[D]")
        filing = np.where(self.missing_filing, np.datetime64("NaT"), self.filing_annual)
        filing_eff = np.where(np.isnat(filing), self.period_end[:, :, 3] + np.timedelta64(120, "D"), filing)    # the date the market learned the figures, whatever the data say
        shares_t = lambda t: shares0 * np.exp(share_growth * (j0 + t / 12.0))                                      # noqa: E731
        logret = np.full((m, n), np.nan)
        logprice = np.full((m, n), np.nan)
        mcap = np.full((m, n), np.nan)
        logprice[0] = logp
        kyear_all = np.empty((m, n), dtype=int)
        for t in range(m):
            d = month_end_np[t]
            kyear = (filing_eff <= d).sum(axis=1) - 1
            kyear_all[t] = kyear
            mcap[t] = np.exp(logprice[t]) * shares_t(t)
            if t == m - 1:
                break
            live = last_month > t
            ky = np.clip(kyear, 0, None)
            have = kyear >= 1
            since = ((d - filing_eff[fye_idx, ky]).astype("timedelta64[D]").astype(float)) / 30.4
            w = 0.5 ** (np.clip(since, 0, None) / 6.0)
            alpha = np.zeros(n)
            alpha += np.where(have, pl["delta_roic"] * w * z_droic[fye_idx, ky], 0.0)
            alpha += np.where(have, -pl["asset_growth"] * w * z_ag[fye_idx, ky], 0.0)
            alpha += np.where(have & (self.country == "JP"), pl["gp_japan"] * w * z_gp[fye_idx, ky], 0.0)
            fcf_known = self.annual["free_cash_flow"][fye_idx, ky]
            fy = np.where(have, fcf_known / mcap[t], np.nan)
            ok = np.isfinite(fy) & live
            if ok.sum() > 5:
                z = np.zeros(n)
                lo, hi = np.nanpercentile(fy[ok], [2, 98])
                clipped = np.clip(fy[ok], lo, hi)
                z[ok] = (clipped - clipped.mean()) / (clipped.std() or 1.0)
                alpha += pl["fcf_yield"] * z
            nxt = np.clip(kyear + 1, 0, len(self.years) - 1)
            pre = (month_end_np[t] >= self.period_end[fye_idx, nxt, 3]) & (month_end_np[t] < filing_eff[fye_idx, nxt]) & (kyear + 1 < len(self.years)) & (kyear >= 0)
            alpha += np.where(pre, pl["anticipation"] * pl["delta_roic"] * z_droic[fye_idx, nxt], 0.0)
            revenue_known = revenue[fye_idx, ky]
            reversion = 0.02 * (np.log(ps_star * revenue_known) - np.log(mcap[t]))
            drift = np.where(have, np.log(revenue[fye_idx, np.clip(ky, 0, None)] / revenue[fye_idx, np.clip(ky - 1, 0, None)]) / 12.0, 0.0)
            ctry = np.array([cshock[k][t + 1] for k in self.country])
            sect = np.array([sshock[k][t + 1] for k in self.sector])
            r = self.beta * f[t + 1] + ctry + sect + alpha + reversion + 0.3 * drift + self.idio_vol * rng.standard_normal(n)
            # a company about to delist loses money over its last quarter
            fading = (last_month - t <= 3) & (last_month < m)
            r = np.where(fading, r - 0.10, r)
            logret[t + 1] = np.where(live, r, np.nan)
            logprice[t + 1] = np.where(live, logprice[t] + r - c.dividend_yield / 12.0, np.nan)
        self.logret, self.logprice, self.mcap, self.kyear = logret, logprice, mcap, kyear_all
        self.filing_effective = filing_eff

    # ------------------------------------------------------------------------------------------------------------ outputs
    def daily_prices(self) -> pd.DataFrame:
        """Long frame ``ticker, date, close, adj_close, volume`` of business days: each month's return is spread over its days and ends exactly on the month's last business day. ``close`` is the
        price a market capitalisation is built from (after dividends), ``adj_close`` the total-return series returns are computed from."""
        c = self.cfg
        rng = np.random.default_rng(c.seed + 1)
        days = pd.bdate_range(self.months[0] + pd.Timedelta(days=1), self.months[-1])
        month_pos = pd.Series(np.arange(len(self.months)), index=self.months.to_period("M"))
        mi = month_pos.reindex(days.to_period("M")).to_numpy().astype(int)
        starts = np.flatnonzero(np.r_[True, mi[1:] != mi[:-1]])
        counts = np.diff(np.r_[starts, len(days)])
        per_day = np.repeat(counts, counts)[:, None]
        eps = rng.standard_normal((len(days), c.n_companies)) * (self.idio_vol / np.sqrt(21.0))
        eps = eps - np.repeat(np.add.reduceat(eps, starts, axis=0) / counts[:, None], counts, axis=0)      # the wiggles inside a month cancel: it still ends on its return
        r_total = np.nan_to_num(self.logret)[mi] / per_day + eps
        r_price = r_total - c.dividend_yield / 252.0
        close = np.exp(self.logprice[0] + np.cumsum(r_price, axis=0))
        adj = np.exp(np.log(100.0) + np.cumsum(r_total, axis=0))
        shares0, growth = self.share_path
        volume = np.exp(rng.normal(np.log(0.004), 0.5, size=close.shape)) * (shares0 * np.exp(growth))
        alive = mi[:, None] < self.last_month[None, :]
        di, fi = np.nonzero(alive)
        frame = pd.DataFrame({"ticker": self.ticker[fi], "date": days[di], "close": close[di, fi], "adj_close": adj[di, fi], "volume": volume[di, fi]})
        return frame.sort_values(["ticker", "date"]).reset_index(drop=True)

    def fx_frame(self) -> pd.DataFrame:
        """Monthly rates (units of local currency per US dollar) for every currency in the world."""
        rng = np.random.default_rng(self.cfg.seed + 2)
        out = []
        for code, spec in MARKETS.items():
            if code not in self.cfg.markets:
                continue
            cur, base = spec[1], spec[6]
            path = base * np.exp(np.cumsum(0.012 * rng.standard_normal(len(self.months))))
            for d, r in zip(self.months, path):
                out.append((d, "USD", cur, float(r)))
        return pd.DataFrame(out, columns=["date", "base", "quote", "rate"])

    # truth the tests check against
    @property
    def truth(self) -> dict:
        return {"planted": dict(self.cfg.planted), "noise_factors": ["revenue_growth", "current_ratio", "operating_margin", "debt_to_equity"],
                "country_specific": {"gp_japan": "JP"}, "delisted": int((self.last_month < len(self.months)).sum())}


class SimulatedProvider(FundamentalDataProvider):
    name = "simulated"
    kind = "simulated"
    as_reported = True
    has_filing_dates = True
    keyed = False

    def __init__(self, world: SimulatedWorld | None = None, **config):
        self._world = world
        self._config = config
        self._restated: dict[tuple[str, int], float] = {}

    @property
    def world(self) -> SimulatedWorld:
        if self._world is None:
            self._world = SimulatedWorld(**self._config)
        return self._world

    def describe(self) -> dict:
        d = super().describe()
        d["note"] = "A simulated world with planted alphas: for trying the application and for tests, never for decisions."
        return d

    def _index(self, symbol: str) -> int:
        w = self.world
        hit = np.flatnonzero((w.ticker == symbol.upper()) | (np.array([f"{t}.{m}" for t, m in zip(w.ticker, w.market)]) == symbol.upper()))
        if not len(hit):
            from ..errors import NotFound
            raise NotFound(f"the simulated world has no company {symbol!r}")
        return int(hit[0])

    def symbols(self) -> list[str]:
        return [str(t) for t in self.world.ticker]

    def _profile(self, i: int) -> CompanyProfile:
        w = self.world
        return CompanyProfile("simulated", str(w.ticker[i]), str(w.ticker[i]), str(w.name[i]), exchange=str(w.market[i]), market=str(w.market[i]), country=str(w.country[i]),
                              currency=str(w.currency[i]), sector=str(w.sector[i]), industry=str(w.industry[i]), active=bool(w.last_month[i] >= len(w.months)),
                              extra={"reporting_currency": str(w.reporting_currency[i])})

    def search(self, query: str, limit: int = 10) -> list[CompanyProfile]:
        w, q = self.world, query.strip().lower()
        hits = [i for i in range(len(w.ticker)) if q in str(w.name[i]).lower() or q == str(w.ticker[i]).lower()]
        return [self._profile(i) for i in hits[:limit]]

    def get_company(self, symbol: str, market: str | None = None) -> CompanyProfile:
        return self._profile(self._index(symbol))

    def offline_profile(self, symbol: str, market: str | None = None) -> CompanyProfile:
        return self._profile(self._index(symbol))                                  # the world is in memory: a complete profile for nothing

    def restate(self, symbol: str, fiscal_year: int, factor: float = 1.05) -> None:
        """Make the next answer for ``symbol`` report that fiscal year's revenue and net income ``factor`` times as large (a restatement)."""
        self._restated[(symbol.upper(), int(fiscal_year))] = float(factor)

    def get_financials(self, symbol: str, market: str | None = None, start_year: int | None = None, end_year: int | None = None, period: str = "annual",
                       limit: int | None = None) -> FinancialsResult:
        w, i = self.world, self._index(symbol)
        years = [int(y) for y in w.years if (start_year is None or y >= start_year) and (end_year is None or y <= end_year)]
        out: list[PeriodStatement] = []
        for j, year in enumerate(w.years):
            if int(year) not in years:
                continue
            if period in ("annual", "all"):
                out.append(self._statement(i, j, None))
            if period in ("quarter", "all", "quarterly") and w.cfg.quarterly:
                out.extend(self._statement(i, j, k) for k in range(4))
        # a delisted company stops reporting
        stop = w.months[min(w.last_month[i], len(w.months)) - 1] if w.last_month[i] < len(w.months) else None
        if stop is not None:
            out = [p for p in out if pd.Timestamp(p.filing_date or p.period_end) <= stop]
        out.sort(key=lambda p: (p.period_end, p.fiscal_period))
        if limit:
            out = out[-int(limit):]
        raw = [{"fiscal_period": p.fiscal_period, "period_end": p.period_end, "filing_date": p.filing_date, "values": p.values} for p in out]
        first = min(w.years)
        exhausted = start_year is not None and start_year <= first or (bool(out) and min(p.period_end for p in out)[:4] <= str(first))
        return FinancialsResult(out, raw, {"symbol": symbol, "period": period, "start_year": start_year, "end_year": end_year}, self._profile(i), exhausted=bool(exhausted), requests=1)

    def _statement(self, i: int, j: int, quarter: int | None) -> PeriodStatement:
        w = self.world
        year = int(w.years[j])
        factor = self._restated.get((str(w.ticker[i]), year), 1.0)
        if quarter is None:
            values = {k: float(v[i, j]) for k, v in w.annual.items()}
            values.update({k: float(v[i, j]) for k, v in w.stocks.items()})
            end, filing, fp = w.period_end[i, j, 3], w.filing_annual[i, j], "FY"
            start = end - np.timedelta64(364, "D")
        else:
            wq = np.array(SECTORS[str(w.sector[i])][10]) * (1 + 0.04 * np.random.default_rng(w.cfg.seed + 1000 * i + j).standard_normal(4))
            wq = wq / wq.sum()
            values = {k: float(v[i, j] * wq[quarter]) for k, v in w.annual.items()}
            prev = j - 1 if j > 0 else j
            frac = (quarter + 1) / 4.0
            for k, v in w.stocks.items():
                values[k] = float(v[i, prev] + (v[i, j] - v[i, prev]) * frac) if quarter < 3 else float(v[i, j])
            end, filing, fp = w.period_end[i, j, quarter], w.filing_quarter[i, j, quarter], f"Q{quarter + 1}"
            start = end - np.timedelta64(90, "D")
        if w.no_rd[i]:
            values.pop("research_and_development", None)
        if w.unit_error == (i, year):
            values = {k: (v * 1000.0 if k in ("revenue", "total_assets", "net_income") else v) for k, v in values.items()}
        if factor != 1.0:
            values["revenue"] *= factor
            values["net_income"] *= factor
        missing = bool(w.missing_filing[i, j]) if quarter is None else False
        if quarter is None:
            values["eps_basic"] = values["net_income"] / values["weighted_avg_shares_diluted"] * 1.015
            values["eps_diluted"] = values["net_income"] / values["weighted_avg_shares_diluted"]
        else:
            values["eps_basic"] = values["net_income"] / values["weighted_avg_shares_diluted"] * 1.015
            values["eps_diluted"] = values["net_income"] / values["weighted_avg_shares_diluted"]
        return PeriodStatement(fp, str(end), values, fiscal_year=year, period_start=str(start), filing_date=None if missing else str(filing), reporting_currency=str(w.reporting_currency[i]),
                               accounting_standard="IFRS" if w.country[i] != "US" else "US-GAAP", provider_record_id=f"sim-{w.ticker[i]}-{year}-{fp}")

    def prices(self) -> pd.DataFrame:
        """Long frame ``ticker, date, close, adj_close, volume`` for every company (for loading into the database)."""
        return self.world.daily_prices()

    def fx(self) -> pd.DataFrame:
        return self.world.fx_frame()
