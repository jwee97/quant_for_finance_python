"""Where each method comes from: the books and papers behind the strategies, allocators, forecasters, option strategies, factors and scores the application offers.

The page shows them under a method's description. A citation is written as the sources print it (authors, year, title, venue). A link is a DOI (``https://doi.org/...``) whose Crossref record was read and
matched against the citation (title, authors, year, journal), or a page of the publisher or the author's institution that was opened; a book, which has none, is cited without one. A reference is
where a method's RULE comes from. It is not evidence that the rule works on the data the application has: each result page reports what the rule did there, after costs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Ref:
    cite: str
    url: str | None = None

    def view(self) -> dict:
        return {"cite": self.cite, "url": self.url}


def paper(authors: str, year: int, title: str, venue: str, doi: str | None = None, url: str | None = None) -> Ref:
    return Ref(f"{authors} ({year}), {title}, {venue}", url or (f"https://doi.org/{doi}" if doi else None))


def book(authors: str, year: int, title: str, publisher: str) -> Ref:
    return Ref(f"{authors} ({year}), {title}, {publisher}")


# ------------------------------------------------------------------------------------------------------------------ the sources, once each
MOSKOWITZ = paper("Moskowitz, Ooi and Pedersen", 2012, "Time series momentum", "Journal of Financial Economics 104(2)", "10.1016/j.jfineco.2011.11.003")
HURST = paper("Hurst, Ooi and Pedersen", 2017, "A century of evidence on trend-following investing", "Journal of Portfolio Management 44(1)", "10.3905/jpm.2017.44.1.015")
BAZ = paper("Baz, Granger, Harvey, Le Roux and Rattray", 2015, "Dissecting investment strategies in the cross section and time series", "SSRN 2695101", "10.2139/ssrn.2695101")
CARVER_ST = book("Carver", 2015, "Systematic Trading", "Harriman House")
CARVER_LT = book("Carver", 2019, "Leveraged Trading", "Harriman House")
CARVER_AFTS = book("Carver", 2023, "Advanced Futures Trading Strategies", "Harriman House")
CLENOW_SOTM = book("Clenow", 2015, "Stocks on the Move", "Equilateral Publishing")
CLENOW_FTT = book("Clenow", 2013, "Following the Trend", "Wiley")
BALTAS_2020 = paper("Baltas and Kosowski", 2020, "Demystifying time-series momentum strategies: volatility estimators, trading rules and pairwise correlations", "in Market Momentum, Wiley", "10.1002/9781119599364.ch3")
BALTAS_2013 = paper("Baltas and Kosowski", 2013, "Momentum strategies in futures markets and trend-following funds", "SSRN 1968996", "10.2139/ssrn.1968996")
LO_MACKINLAY_88 = paper("Lo and MacKinlay", 1988, "Stock market prices do not follow random walks: evidence from a simple specification test", "Review of Financial Studies 1(1)", "10.1093/rfs/1.1.41")
LO_MACKINLAY_89 = paper("Lo and MacKinlay", 1989, "The size and power of the variance ratio test in finite samples", "Journal of Econometrics 40(2)", "10.1016/0304-4076(89)90083-3")
CLM = book("Campbell, Lo and MacKinlay", 1997, "The Econometrics of Financial Markets", "Princeton University Press")
CHAN_AT = book("Chan", 2013, "Algorithmic Trading: Winning Strategies and Their Rationale", "Wiley")
CHAN_QT = book("Chan", 2009, "Quantitative Trading", "Wiley")
CONNORS = book("Connors and Alvarez", 2009, "Short Term Trading Strategies That Work", "TradingMarkets Publishing")
AMAYA = paper("Amaya, Christoffersen, Jacobs and Vasquez", 2015, "Does realized skewness predict the cross-section of equity returns?", "Journal of Financial Economics 118(1)", "10.1016/j.jfineco.2015.02.009")
FERNANDEZ_PEREZ = paper("Fernandez-Perez, Frijns, Fuertes and Miffre", 2018, "The skewness of commodity futures returns", "Journal of Banking & Finance 86", "10.1016/j.jbankfin.2017.06.015")
BLACK_JONES = paper("Black and Jones", 1987, "Simplifying portfolio insurance", "Journal of Portfolio Management 14(1)", "10.3905/jpm.1987.409131")
BLACK_PEROLD = paper("Black and Perold", 1992, "Theory of constant proportion portfolio insurance", "Journal of Economic Dynamics and Control 16(3-4)", "10.1016/0165-1889(92)90043-e")
PEROLD_SHARPE = paper("Perold and Sharpe", 1988, "Dynamic strategies for asset allocation", "Financial Analysts Journal 44(1)", "10.2469/faj.v44.n1.16")
ESTEP_KRITZMAN = paper("Estep and Kritzman", 1988, "TIPP: insurance without complexity", "Journal of Portfolio Management 14(4)", "10.3905/jpm.1988.409172")
JEGADEESH_TITMAN = paper("Jegadeesh and Titman", 1993, "Returns to buying winners and selling losers: implications for stock market efficiency", "Journal of Finance 48(1)", "10.1111/j.1540-6261.1993.tb04702.x")
JEGADEESH_90 = paper("Jegadeesh", 1990, "Evidence of predictable behavior of security returns", "Journal of Finance 45(3)", "10.1111/j.1540-6261.1990.tb05110.x")
LEHMANN = paper("Lehmann", 1990, "Fads, martingales, and market efficiency", "Quarterly Journal of Economics 105(1)", "10.2307/2937816")
ANG = paper("Ang, Hodrick, Xing and Zhang", 2006, "The cross-section of volatility and expected returns", "Journal of Finance 61(1)", "10.1111/j.1540-6261.2006.00836.x")
FRAZZINI_PEDERSEN = paper("Frazzini and Pedersen", 2014, "Betting against beta", "Journal of Financial Economics 111(1)", "10.1016/j.jfineco.2013.10.005")
GEORGE_HWANG = paper("George and Hwang", 2004, "The 52-week high and momentum investing", "Journal of Finance 59(5)", "10.1111/j.1540-6261.2004.00695.x")
BLITZ = paper("Blitz, Huij and Martens", 2011, "Residual momentum", "Journal of Empirical Finance 18(3)", "10.1016/j.jempfin.2011.01.003")
DA_GURUN_WARACHKA = paper("Da, Gurun and Warachka", 2014, "Frog in the pan: continuous information and momentum", "Review of Financial Studies 27(7)", "10.1093/rfs/hhu003")
BALI = paper("Bali, Cakici and Whitelaw", 2011, "Maxing out: stocks as lotteries and the cross-section of expected returns", "Journal of Financial Economics 99(2)", "10.1016/j.jfineco.2010.08.014")
AMIHUD = paper("Amihud", 2002, "Illiquidity and stock returns: cross-section and time-series effects", "Journal of Financial Markets 5(1)", "10.1016/s1386-4181(01)00024-6")
ASNESS_MP = paper("Asness, Moskowitz and Pedersen", 2013, "Value and momentum everywhere", "Journal of Finance 68(3)", "10.1111/jofi.12021")
DE_BONDT = paper("De Bondt and Thaler", 1985, "Does the stock market overreact?", "Journal of Finance 40(3)", "10.1111/j.1540-6261.1985.tb05004.x")
KOIJEN = paper("Koijen, Moskowitz, Pedersen and Vrugt", 2018, "Carry", "Journal of Financial Economics 127(2)", "10.1016/j.jfineco.2017.11.002")
BOONS_PRADO = paper("Boons and Prado", 2019, "Basis-momentum", "Journal of Finance 74(1)", "10.1111/jofi.12738")
MOREIRA_MUIR = paper("Moreira and Muir", 2017, "Volatility-managed portfolios", "Journal of Finance 72(4)", "10.1111/jofi.12513")
BOLLERSLEV_86 = paper("Bollerslev", 1986, "Generalized autoregressive conditional heteroskedasticity", "Journal of Econometrics 31(3)", "10.1016/0304-4076(86)90063-1")
MCNEIL_FREY = paper("McNeil and Frey", 2000, "Estimation of tail-related risk measures for heteroscedastic financial time series: an extreme value approach", "Journal of Empirical Finance 7(3-4)", "10.1016/s0927-5398(00)00012-8")
FABER = paper("Faber", 2007, "A quantitative approach to tactical asset allocation", "Journal of Wealth Management 9(4)", "10.3905/jwm.2007.674809")
ANTONACCI = book("Antonacci", 2014, "Dual Momentum Investing", "McGraw-Hill")
KELLER_PAA = paper("Keller and Keuning", 2016, "Protective asset allocation (PAA): a simple momentum-based alternative for term deposits", "SSRN 2759734", "10.2139/ssrn.2759734")
KELLER_VAA = paper("Keller and Keuning", 2017, "Breadth momentum and vigilant asset allocation (VAA): winning more by losing less", "SSRN 3002624", "10.2139/ssrn.3002624")
KELLER_DAA = paper("Keller and Keuning", 2018, "Breadth momentum and the canary universe: defensive asset allocation (DAA)", "SSRN 3212862", "10.2139/ssrn.3212862")
BUTLER = paper("Butler, Philbrick, Gordillo and Varadi", 2012, "Adaptive asset allocation: a primer", "SSRN 2328254", "10.2139/ssrn.2328254")
LAKONISHOK_SMIDT = paper("Lakonishok and Smidt", 1988, "Are seasonal anomalies real? A ninety-year perspective", "Review of Financial Studies 1(4)", "10.1093/rfs/1.4.403")
MCCONNELL_XU = paper("McConnell and Xu", 2008, "Equity returns at the turn of the month", "Financial Analysts Journal 64(2)", "10.2469/faj.v64.n2.11")
BOUMAN_JACOBSEN = paper("Bouman and Jacobsen", 2002, "The Halloween indicator, 'Sell in May and go away': another puzzle", "American Economic Review 92(5)", "10.1257/000282802762024683")
KELOHARJU = paper("Keloharju, Linnainmaa and Nyberg", 2016, "Return seasonalities", "Journal of Finance 71(4)", "10.1111/jofi.12398")
BOLLERSLEV_TZ = paper("Bollerslev, Tauchen and Zhou", 2009, "Expected stock returns and variance risk premia", "Review of Financial Studies 22(11)", "10.1093/rfs/hhp008")
CARR_WU = paper("Carr and Wu", 2009, "Variance risk premiums", "Review of Financial Studies 22(3)", "10.1093/rfs/hhn038")
AVELLANEDA_LEE = paper("Avellaneda and Lee", 2010, "Statistical arbitrage in the US equities market", "Quantitative Finance 10(7)", "10.1080/14697680903124632")
GATEV = paper("Gatev, Goetzmann and Rouwenhorst", 2006, "Pairs trading: performance of a relative-value arbitrage rule", "Review of Financial Studies 19(3)", "10.1093/rfs/hhj020")
ENGLE_GRANGER = paper("Engle and Granger", 1987, "Co-integration and error correction: representation, estimation, and testing", "Econometrica 55(2)", "10.2307/1913236")
ELLIOTT = paper("Elliott, van der Hoek and Malcolm", 2005, "Pairs trading", "Quantitative Finance 5(3)", "10.1080/14697680500149370")
BERTRAM = paper("Bertram", 2010, "Analytic solutions for optimal statistical arbitrage trading", "Physica A 389(11)", "10.1016/j.physa.2010.01.045")
VIDYAMURTHY = book("Vidyamurthy", 2004, "Pairs Trading: Quantitative Methods and Analysis", "Wiley")
BOLLINGER = book("Bollinger", 2001, "Bollinger on Bollinger Bands", "McGraw-Hill")
FAITH = book("Faith", 2007, "Way of the Turtle", "McGraw-Hill")
KAUFMAN = book("Kaufman", 2013, "Trading Systems and Methods, 5th edition", "Wiley")
WILDER = book("Wilder", 1978, "New Concepts in Technical Trading Systems", "Trend Research")
HARVEY_KF = book("Harvey", 1989, "Forecasting, Structural Time Series Models and the Kalman Filter", "Cambridge University Press")
GU_KELLY_XIU = paper("Gu, Kelly and Xiu", 2020, "Empirical asset pricing via machine learning", "Review of Financial Studies 33(5)", "10.1093/rfs/hhaa009")
LOPEZ_AFML = book("Lopez de Prado", 2018, "Advances in Financial Machine Learning", "Wiley")
RASMUSSEN = book("Rasmussen and Williams", 2006, "Gaussian Processes for Machine Learning", "MIT Press")
NOVY_MARX = paper("Novy-Marx", 2013, "The other side of value: the gross profitability premium", "Journal of Financial Economics 108(1)", "10.1016/j.jfineco.2013.01.003")
FAMA_FRENCH_93 = paper("Fama and French", 1993, "Common risk factors in the returns on stocks and bonds", "Journal of Financial Economics 33(1)", "10.1016/0304-405x(93)90023-5")
FAMA_FRENCH_15 = paper("Fama and French", 2015, "A five-factor asset pricing model", "Journal of Financial Economics 116(1)", "10.1016/j.jfineco.2014.10.010")
PIOTROSKI = paper("Piotroski", 2000, "Value investing: the use of historical financial statement information to separate winners from losers", "Journal of Accounting Research 38(S)", "10.2307/2672906")
SLOAN = paper("Sloan", 1996, "Do stock prices fully reflect information in accruals and cash flows about future earnings?", "The Accounting Review 71(3)", "10.2308/tar-9608042309")
AFP_QMJ = paper("Asness, Frazzini and Pedersen", 2019, "Quality minus junk", "Review of Accounting Studies 24(1)", "10.1007/s11142-018-9470-2")
HOU_XUE_ZHANG = paper("Hou, Xue and Zhang", 2015, "Digesting anomalies: an investment approach", "Review of Financial Studies 28(3)", "10.1093/rfs/hhu068")
DAMODARAN = book("Damodaran", 2012, "Investment Valuation, 3rd edition", "Wiley")
LITTERMAN_SCHEINKMAN = paper("Litterman and Scheinkman", 1991, "Common factors affecting bond returns", "Journal of Fixed Income 1(1)", "10.3905/jfi.1991.692347")
ILMANEN = book("Ilmanen", 2011, "Expected Returns", "Wiley")
BESSEMBINDER = paper("Bessembinder", 1992, "Systematic risk, hedging pressure, and risk premiums in futures markets", "Review of Financial Studies 5(4)", "10.1093/rfs/5.4.637")
BROWNE = book("Browne", 1987, "Fail-Safe Investing", "St. Martin's Press")
MARKOWITZ = paper("Markowitz", 1952, "Portfolio selection", "Journal of Finance 7(1)", "10.2307/2975974")
SHARPE_64 = paper("Sharpe", 1964, "Capital asset prices: a theory of market equilibrium under conditions of risk", "Journal of Finance 19(3)", "10.1111/j.1540-6261.1964.tb02865.x")
MAILLARD = paper("Maillard, Roncalli and Teiletche", 2010, "The properties of equally weighted risk contribution portfolios", "Journal of Portfolio Management 36(4)", "10.3905/jpm.2010.36.4.060")
RONCALLI = book("Roncalli", 2013, "Introduction to Risk Parity and Budgeting", "Chapman & Hall/CRC")
LOPEZ_HRP = paper("Lopez de Prado", 2016, "Building diversified portfolios that outperform out of sample", "Journal of Portfolio Management 42(4); SSRN 2708678", "10.2139/ssrn.2708678")
RAFFINOT = paper("Raffinot", 2017, "Hierarchical clustering-based asset allocation", "Journal of Portfolio Management 44(2)", "10.3905/jpm.2018.44.2.089")
CLARKE_MINVAR = paper("Clarke, de Silva and Thorley", 2006, "Minimum-variance portfolios in the U.S. equity market", "Journal of Portfolio Management 33(1)", "10.3905/jpm.2006.661366")
CHOUEIFATY = paper("Choueifaty and Coignard", 2008, "Toward maximum diversification", "Journal of Portfolio Management 35(1)", "10.3905/jpm.2008.35.1.40")
BLACK_LITTERMAN = paper("Black and Litterman", 1992, "Global portfolio optimization", "Financial Analysts Journal 48(5)", "10.2469/faj.v48.n5.28")
KELLY = paper("Kelly", 1956, "A new interpretation of information rate", "Bell System Technical Journal 35(4)", "10.1002/j.1538-7305.1956.tb03809.x")
MACLEAN_THORP_ZIEMBA = book("MacLean, Thorp and Ziemba", 2011, "The Kelly Capital Growth Investment Criterion", "World Scientific")
ROCKAFELLAR = paper("Rockafellar and Uryasev", 2000, "Optimization of conditional value-at-risk", "Journal of Risk 2(3)", "10.21314/jor.2000.038")
ENGLE_DCC = paper("Engle", 2002, "Dynamic conditional correlation", "Journal of Business & Economic Statistics 20(3)", "10.1198/073500102288618487")
JORION = paper("Jorion", 1986, "Bayes-Stein estimation for portfolio analysis", "Journal of Financial and Quantitative Analysis 21(3)", "10.2307/2331042")
POZZI = paper("Pozzi, Di Matteo and Aste", 2013, "Spread of risk across financial markets: better to invest in the peripheries", "Scientific Reports 3", "10.1038/srep01665")
GARLEANU_PEDERSEN = paper("Garleanu and Pedersen", 2013, "Dynamic trading with predictable returns and transaction costs", "Journal of Finance 68(6)", "10.1111/jofi.12080")
LUCAS = paper("Lucas", 2014, "Ising formulations of many NP problems", "Frontiers in Physics 2", "10.3389/fphy.2014.00005")
GRINOLD_KAHN = book("Grinold and Kahn", 2000, "Active Portfolio Management, 2nd edition", "McGraw-Hill")
MICHAUD = book("Michaud", 1998, "Efficient Asset Management", "Harvard Business School Press")
SALIMANS = paper("Salimans, Ho, Chen, Sidor and Sutskever", 2017, "Evolution strategies as a scalable alternative to reinforcement learning", "arXiv 1703.03864", url="https://arxiv.org/abs/1703.03864")
SCHMELING = paper("Schmeling, Schrimpf and Todorov", 2023, "Crypto carry", "BIS Working Paper 1087", url="https://www.bis.org/publ/work1087.htm")
HE_MANELA = paper("He, Manela, Ross and von Wachter", 2022, "Fundamentals of perpetual futures", "arXiv 2212.06888", url="https://arxiv.org/abs/2212.06888")
LIU_TSYVINSKI = paper("Liu and Tsyvinski", 2021, "Risks and returns of cryptocurrency", "Review of Financial Studies 34(6)", "10.1093/rfs/hhaa113")
LIU_TSYVINSKI_WU = paper("Liu, Tsyvinski and Wu", 2022, "Common risk factors in cryptocurrency", "Journal of Finance 77(2)", "10.1111/jofi.13119")


# ------------------------------------------------------------------------------------------------------------------ method -> sources
MODELS: dict[str, tuple[Ref, ...]] = {
    "tsmom": (MOSKOWITZ, HURST), "tsmom_multi": (HURST, MOSKOWITZ), "ma_ensemble": (BAZ,), "donchian": (FAITH, CLENOW_FTT), "kama_trend": (KAUFMAN,),
    "adx_trend": (WILDER,), "kalman_trend": (HARVEY_KF,),
    "carver_ewmac": (CARVER_ST, CARVER_LT, CARVER_AFTS), "clenow_momentum": (CLENOW_SOTM,), "tstat_momentum": (BALTAS_2020, BALTAS_2013),
    "variance_ratio_switch": (LO_MACKINLAY_88, LO_MACKINLAY_89, CLM, CHAN_AT), "connors_rsi": (CONNORS,), "rsi2": (CONNORS,), "double_seven": (CONNORS,),
    "consecutive_down": (CONNORS,), "skew_premium": (AMAYA, FERNANDEZ_PEREZ), "cppi": (BLACK_JONES, PEROLD_SHARPE, BLACK_PEROLD, ESTEP_KRITZMAN),
    "momentum": (JEGADEESH_TITMAN,), "xs_momentum": (JEGADEESH_TITMAN,), "momentum_13612w": (KELLER_VAA, KELLER_DAA), "short_term_reversal": (JEGADEESH_90, LEHMANN),
    "low_volatility": (ANG,), "low_idio_vol": (ANG,), "bab": (FRAZZINI_PEDERSEN,), "defensive_beta": (FRAZZINI_PEDERSEN,), "high_52w": (GEORGE_HWANG,), "residual_momentum": (BLITZ,),
    "smooth_momentum": (DA_GURUN_WARACHKA,), "max_effect": (BALI,), "amihud_illiquidity": (AMIHUD,), "value_momentum": (ASNESS_MP,), "long_term_reversal": (DE_BONDT, ASNESS_MP),
    "value_proxy": (DE_BONDT, ASNESS_MP), "carry": (KOIJEN,), "carry_xs": (KOIJEN,), "carry_ts": (KOIJEN,), "basis_momentum": (BOONS_PRADO,), "vol_managed_long": (MOREIRA_MUIR,),
    "garch_vol_managed": (BOLLERSLEV_86, MOREIRA_MUIR), "evt_risk_managed": (MCNEIL_FREY,), "faber_gtaa": (FABER,), "dual_momentum": (ANTONACCI,), "accelerating_dual_momentum": (ANTONACCI,),
    "paa": (KELLER_PAA,), "vaa": (KELLER_VAA,), "daa": (KELLER_DAA,), "adaptive_asset_allocation": (BUTLER,), "model_portfolio": (BROWNE,), "policy_portfolio": (PEROLD_SHARPE,),
    "turn_of_month": (LAKONISHOK_SMIDT, MCCONNELL_XU), "sell_in_may": (BOUMAN_JACOBSEN,), "seasonal_rank": (KELOHARJU,), "vrp_timing": (BOLLERSLEV_TZ, CARR_WU),
    "variance_carry": (CARR_WU,), "pca_residual": (AVELLANEDA_LEE,), "cointegration_pairs": (GATEV, ENGLE_GRANGER, VIDYAMURTHY), "kalman_pairs": (ELLIOTT, CHAN_AT),
    "ou_threshold_pairs": (BERTRAM, AVELLANEDA_LEE), "ou_reversion": (BERTRAM, CHAN_AT), "bollinger_reversion": (BOLLINGER,),
    "ml_ridge": (GU_KELLY_XIU,), "ml_trees": (GU_KELLY_XIU, LOPEZ_AFML), "ml_factor_model": (GU_KELLY_XIU,), "gp_factor_model": (RASMUSSEN,),
    "fundamental_value": (FAMA_FRENCH_93, NOVY_MARX), "fundamental_quality": (NOVY_MARX, PIOTROSKI, SLOAN, AFP_QMJ), "fundamental_dcf": (DAMODARAN,),
    "fundamental_alpha": (GRINOLD_KAHN,), "curve_steepener": (LITTERMAN_SCHEINKMAN, ILMANEN), "butterfly": (LITTERMAN_SCHEINKMAN,),
    "carry_rolldown": (ILMANEN,), "duration_timing": (ILMANEN,), "cftc_positioning": (BESSEMBINDER,), "funding_carry": (SCHMELING, HE_MANELA), "basis_reversion": (SCHMELING, HE_MANELA),
}
ALLOCATORS: dict[str, tuple[Ref, ...]] = {
    "static": (MARKOWITZ, MAILLARD, RONCALLI, LOPEZ_HRP, RAFFINOT), "min_variance": (CLARKE_MINVAR,), "max_diversification": (CHOUEIFATY,), "black_litterman": (BLACK_LITTERMAN,),
    "kelly": (KELLY, MACLEAN_THORP_ZIEMBA), "tangency_cml": (MARKOWITZ, SHARPE_64), "mv_cvar": (ROCKAFELLAR,), "dynamic_cov": (ENGLE_DCC,), "bayesian": (JORION,),
    "hierarchical_sensitivity_parity": (LOPEZ_HRP,), "graph_centrality": (POZZI,), "multi_period": (GARLEANU_PEDERSEN,), "tca_mvo": (GARLEANU_PEDERSEN,), "es_policy": (SALIMANS,),
    "qubo_select": (LUCAS,), "constrained_long_short": (GRINOLD_KAHN,),
}


def references(kind: str, name: str) -> list[dict]:
    """The sources of a method as ``[{"cite", "url"}]`` (empty for a method with none on record)."""
    table = {"model": MODELS, "allocator": ALLOCATORS, **EXTRA}.get(kind, {})
    return [r.view() for r in table.get(name, ())]


EXTRA: dict[str, dict[str, tuple[Ref, ...]]] = {}                 # tables of the other tabs (forecasters, option strategies, ...), filled by their modules
