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
GORTON_ROUWENHORST = paper("Gorton and Rouwenhorst", 2006, "Facts and fantasies about commodity futures", "Financial Analysts Journal 62(2)", "10.2469/faj.v62.n2.4083")
ERB_HARVEY = paper("Erb and Harvey", 2006, "The strategic and tactical value of commodity futures", "Financial Analysts Journal 62(2)", "10.2469/faj.v62.n2.4084")
GORTON_HAYASHI_ROUWENHORST = paper("Gorton, Hayashi and Rouwenhorst", 2013, "The fundamentals of commodity futures returns", "Review of Finance 17(1)", "10.1093/rof/rfs019")
MIFFRE_RALLIS = paper("Miffre and Rallis", 2007, "Momentum strategies in commodity futures markets", "Journal of Banking & Finance 31(6)", "10.1016/j.jbankfin.2006.12.005")
SZAKMARY = paper("Szakmary, Shen and Sharma", 2010, "Trend-following trading strategies in commodity futures: a re-examination", "Journal of Banking & Finance 34(2)", "10.1016/j.jbankfin.2009.08.004")
FAMA_FRENCH_87 = paper("Fama and French", 1987, "Commodity futures prices: some evidence on forecast power, premiums, and the theory of storage", "Journal of Business 60(1)", "10.1086/296385")
KALDOR = paper("Kaldor", 1939, "Speculation and economic stability", "Review of Economic Studies 7(1)", "10.2307/2967593")
BASU_MIFFRE = paper("Basu and Miffre", 2013, "Capturing the risk premium of commodity futures: the role of hedging pressure", "Journal of Banking & Finance 37(7)", "10.1016/j.jbankfin.2013.02.031")
FUERTES = paper("Fuertes, Miffre and Rallis", 2010, "Tactical allocation in commodity futures markets: combining momentum and term structure signals", "Journal of Banking & Finance 34(10)", "10.1016/j.jbankfin.2010.04.009")
BAKSHI_GAO_ROSSI = paper("Bakshi, Gao and Rossi", 2019, "Understanding the sources of risk underlying the cross section of commodity returns", "Management Science 65(2)", "10.1287/mnsc.2017.2840")
TOLMASKY_HINDANOV = paper("Tolmasky and Hindanov", 2002, "Principal components analysis for correlated curves and seasonal commodities: the case of the petroleum market", "Journal of Futures Markets 22(11)", "10.1002/fut.10046")
ROUTLEDGE_SEPPI_SPATT = paper("Routledge, Seppi and Spatt", 2000, "Equilibrium forward curves for commodities", "Journal of Finance 55(3)", "10.1111/0022-1082.00248")
GIBSON_SCHWARTZ = paper("Gibson and Schwartz", 1990, "Stochastic convenience yield and the pricing of oil contingent claims", "Journal of Finance 45(3)", "10.1111/j.1540-6261.1990.tb05114.x")
SCHWARTZ_SMITH = paper("Schwartz and Smith", 2000, "Short-term variations and long-term dynamics in commodity prices", "Management Science 46(7)", "10.1287/mnsc.46.7.893.12034")
BROCK = paper("Brock, Lakonishok and LeBaron", 1992, "Simple technical trading rules and the stochastic properties of stock returns", "Journal of Finance 47(5)", "10.1111/j.1540-6261.1992.tb04681.x")
GERRITSEN = paper("Gerritsen, Bouri, Ramezanifar and Roubaud", 2020, "The profitability of technical trading rules in the Bitcoin market", "Finance Research Letters 34", "10.1016/j.frl.2019.08.011")
DETZEL = paper("Detzel, Liu, Strauss, Zhou and Zhu", 2021, "Learning and predictability via technical analysis: evidence from bitcoin and stocks with hard-to-value fundamentals", "Financial Management 50(1)", "10.1111/fima.12310")
MAKAROV_SCHOAR = paper("Makarov and Schoar", 2020, "Trading and arbitrage in cryptocurrency markets", "Journal of Financial Economics 135(2)", "10.1016/j.jfineco.2019.07.001")
GRIFFIN_SHAMS = paper("Griffin and Shams", 2020, "Is Bitcoin really untethered?", "Journal of Finance 75(4)", "10.1111/jofi.12903")
BURGHARDT_LANE = paper("Burghardt and Lane", 1990, "How to tell if options are cheap", "Journal of Portfolio Management 16(3)", "10.3905/jpm.1990.409259")
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


# ---- forecasting
HANSEN_LUNDE = paper("Hansen and Lunde", 2005, "A forecast comparison of volatility models: does anything beat a GARCH(1,1)?", "Journal of Applied Econometrics 20(7)", "10.1002/jae.800")
GJR = paper("Glosten, Jagannathan and Runkle", 1993, "On the relation between the expected value and the volatility of the nominal excess return on stocks", "Journal of Finance 48(5)", "10.1111/j.1540-6261.1993.tb05128.x")
NELSON = paper("Nelson", 1991, "Conditional heteroskedasticity in asset returns: a new approach", "Econometrica 59(2)", "10.2307/2938260")
BGV = paper("Barone-Adesi, Giannopoulos and Vosper", 1999, "VaR without correlations for portfolios of derivative securities", "Journal of Futures Markets 19(5)")
BARONE_ADESI = paper("Barone-Adesi, Engle and Mancini", 2008, "A GARCH option pricing model with filtered historical simulation", "Review of Financial Studies 21(3)", "10.1093/rfs/hhn031")
MCNEIL_QRM = book("McNeil, Frey and Embrechts", 2015, "Quantitative Risk Management: Concepts, Techniques and Tools, revised edition", "Princeton University Press")
POLITIS_ROMANO = paper("Politis and Romano", 1994, "The stationary bootstrap", "Journal of the American Statistical Association 89(428)", "10.1080/01621459.1994.10476870")
HULL_WHITE_98 = paper("Hull and White", 1998, "Incorporating volatility updating into the historical simulation method for value-at-risk", "Journal of Risk 1(1)", "10.21314/jor.1998.001")
BLACK_SCHOLES = paper("Black and Scholes", 1973, "The pricing of options and corporate liabilities", "Journal of Political Economy 81(3)", "10.1086/260062")
RISKMETRICS = book("J.P. Morgan and Reuters", 1996, "RiskMetrics: Technical Document, 4th edition", "J.P. Morgan")
MADAN_CARR_CHANG = paper("Madan, Carr and Chang", 1998, "The variance gamma process and option pricing", "Review of Finance 2(1)", "10.1023/a:1009703431535")
BARNDORFF_NIELSEN = paper("Barndorff-Nielsen", 1997, "Normal inverse Gaussian distributions and stochastic volatility modelling", "Scandinavian Journal of Statistics 24(1)", "10.1111/1467-9469.00045")
HESTON = paper("Heston", 1993, "A closed-form solution for options with stochastic volatility with applications to bond and currency options", "Review of Financial Studies 6(2)", "10.1093/rfs/6.2.327")
MERTON_76 = paper("Merton", 1976, "Option pricing when underlying stock returns are discontinuous", "Journal of Financial Economics 3(1-2)", "10.1016/0304-405x(76)90022-2")
HARVEY_RUIZ_SHEPHARD = paper("Harvey, Ruiz and Shephard", 1994, "Multivariate stochastic variance models", "Review of Economic Studies 61(2)")
CORSI = paper("Corsi", 2009, "A simple approximate long-memory model of realized volatility", "Journal of Financial Econometrics 7(2)", "10.1093/jjfinec/nbp001")
PARKINSON = paper("Parkinson", 1980, "The extreme value method for estimating the variance of the rate of return", "Journal of Business 53(1)", "10.1086/296071")
HAMILTON = paper("Hamilton", 1989, "A new approach to the economic analysis of nonstationary time series and the business cycle", "Econometrica 57(2)", "10.2307/1912559")
BOX_JENKINS = book("Box and Jenkins", 1970, "Time Series Analysis: Forecasting and Control", "Holden-Day")
FARMER_SIDOROWICH = paper("Farmer and Sidorowich", 1987, "Predicting chaotic time series", "Physical Review Letters 59(8)", "10.1103/physrevlett.59.845")
ASSIMAKOPOULOS = paper("Assimakopoulos and Nikolopoulos", 2000, "The theta model: a decomposition approach to forecasting", "International Journal of Forecasting 16(4)", "10.1016/s0169-2070(00)00066-2")
HYNDMAN_BILLAH = paper("Hyndman and Billah", 2003, "Unmasking the Theta method", "International Journal of Forecasting 19(2)", "10.1016/s0169-2070(01)00143-1")
FPP3 = paper("Hyndman and Athanasopoulos", 2021, "Forecasting: Principles and Practice, 3rd edition", "OTexts", url="https://otexts.com/fpp3/")
BATES_GRANGER = paper("Bates and Granger", 1969, "The combination of forecasts", "Journal of the Operational Research Society 20(4)", "10.1057/jors.1969.103")
TIMMERMANN = paper("Timmermann", 2006, "Forecast combinations", "Handbook of Economic Forecasting, vol. 1", "10.1016/s1574-0706(05)01004-9")
SMITH_WALLIS = paper("Smith and Wallis", 2009, "A simple explanation of the forecast combination puzzle", "Oxford Bulletin of Economics and Statistics 71(3)", "10.1111/j.1468-0084.2008.00541.x")
GNEITING_RAFTERY = paper("Gneiting and Raftery", 2007, "Strictly proper scoring rules, prediction, and estimation", "Journal of the American Statistical Association 102(477)", "10.1198/016214506000001437")
FERRO = paper("Ferro", 2014, "Fair scores for ensemble forecasts", "Quarterly Journal of the Royal Meteorological Society 140(683)", "10.1002/qj.2270")
DIEBOLD_MARIANO = paper("Diebold and Mariano", 1995, "Comparing predictive accuracy", "Journal of Business & Economic Statistics 13(3)", "10.1080/07350015.1995.10524599")
CLARK_WEST = paper("Clark and West", 2007, "Approximately normal tests for equal predictive accuracy in nested models", "Journal of Econometrics 138(1)", "10.1016/j.jeconom.2006.05.023")
DIEBOLD_GUNTHER_TAY = paper("Diebold, Gunther and Tay", 1998, "Evaluating density forecasts with applications to financial risk management", "International Economic Review 39(4)", "10.2307/2527342")
KUPIEC = paper("Kupiec", 1995, "Techniques for verifying the accuracy of risk measurement models", "Journal of Derivatives 3(2)", "10.3905/jod.1995.407942")
GOYAL_WELCH = paper("Goyal and Welch", 2008, "A comprehensive look at the empirical performance of equity premium prediction", "Review of Financial Studies 21(4)", "10.1093/rfs/hhm014")


# ---- options
CBOE_BENCHMARKS = paper("Cboe", 2026, "Strategy benchmark indices (PUT, BXM, CLL, PPUT, CLLZ, CNDR, BFLY)", "Cboe Global Indices web page, accessed October 2026", url="https://www.cboe.com/us/indices/benchmark_indices/")
CBOE_2015 = paper("Cboe", 2015, "CBOE introduces 10 options-based strategy performance benchmark indexes", "press release of 29 July 2015",
                  url="https://ir.cboe.com/news/news-details/2015/CBOE-Introduces-10-Options-Based-Strategy-Performance-Benchmark-Indexes-07-29-2015/default.aspx")
CBOE_VIX = paper("Cboe", 2025, "Cboe Volatility Index mathematics methodology", "Cboe Global Indices white paper", url="https://cdn.cboe.com/resources/indices/Cboe_Volatility_Index_Mathematics_Methodology.pdf")
MCMILLAN = book("McMillan", 2012, "Options as a Strategic Investment, 5th edition", "Prentice Hall Press")
NATENBERG = book("Natenberg", 2014, "Option Volatility and Pricing: Advanced Trading Strategies and Techniques, 2nd edition", "McGraw-Hill")
HULL_BOOK = book("Hull", 2018, "Options, Futures, and Other Derivatives, 10th edition", "Pearson")
COHEN = book("Cohen", 2005, "The Bible of Options Strategies: The Definitive Guide for Practical Trading Strategies", "Financial Times Prentice Hall")
SINCLAIR = book("Sinclair", 2013, "Volatility Trading, 2nd edition", "Wiley")
TALEB = book("Taleb", 1997, "Dynamic Hedging: Managing Vanilla and Exotic Options", "Wiley")
GATHERAL_BOOK = book("Gatheral", 2006, "The Volatility Surface: A Practitioner's Guide", "Wiley")
GATHERAL_JACQUIER = paper("Gatheral and Jacquier", 2014, "Arbitrage-free SVI volatility surfaces", "Quantitative Finance 14(1)", "10.1080/14697688.2013.819986")
WHALEY_BXM = paper("Whaley", 2002, "Return and risk of CBOE buy write monthly index", "Journal of Derivatives 10(2)", "10.3905/jod.2002.319194")
HILL_COVERED = paper("Hill, Balasubramanian, Gregory and Tierens", 2006, "Finding alpha via covered index writing", "Financial Analysts Journal 62(5)", "10.2469/faj.v62.n5.4281")
ISRAELOV_NIELSEN = paper("Israelov and Nielsen", 2014, "Covered call strategies: one fact and eight myths", "Financial Analysts Journal 70(6)", "10.2469/faj.v70.n6.3")
ISRAELOV_KLEIN = paper("Israelov and Klein", 2016, "Risk and return of equity index collar strategies", "Journal of Alternative Investments 19(1)", "10.3905/jai.2016.19.1.041")
COVAL_SHUMWAY = paper("Coval and Shumway", 2001, "Expected option returns", "Journal of Finance 56(3)", "10.1111/0022-1082.00352")
BAKSHI_KAPADIA = paper("Bakshi and Kapadia", 2003, "Delta-hedged gains and the negative market volatility risk premium", "Review of Financial Studies 16(2)", "10.1093/rfs/hhg002")
BONDARENKO = paper("Bondarenko", 2014, "Why are put options so expensive?", "Quarterly Journal of Finance 4(1)", "10.1142/s2010139214500153")
BREEDEN_LITZENBERGER = paper("Breeden and Litzenberger", 1978, "Prices of state-contingent claims implicit in option prices", "Journal of Business 51(4)", "10.1086/296025")
JACKWERTH_RUBINSTEIN = paper("Jackwerth and Rubinstein", 1996, "Recovering probability distributions from option prices", "Journal of Finance 51(5)", "10.1111/j.1540-6261.1996.tb05219.x")
BLISS_PANIGIRTZOGLOU = paper("Bliss and Panigirtzoglou", 2004, "Option-implied risk aversion estimates", "Journal of Finance 59(1)", "10.1111/j.1540-6261.2004.00637.x")
FIGLEWSKI = paper("Figlewski", 2010, "Estimating the implied risk-neutral density for the U.S. market portfolio", "in Volatility and Time Series Econometrics, Oxford University Press",
                  "10.1093/acprof:oso/9780199549498.003.0015")
BRITTEN_JONES = paper("Britten-Jones and Neuberger", 2000, "Option prices, implied price processes, and stochastic volatility", "Journal of Finance 55(2)", "10.1111/0022-1082.00228")
DEMETERFI = paper("Demeterfi, Derman, Kamal and Zou", 1999, "A guide to volatility and variance swaps", "Journal of Derivatives 6(4)", "10.3905/jod.1999.319129")
NEUBERGER = paper("Neuberger", 1994, "The log contract", "Journal of Portfolio Management 20(2)", "10.3905/jpm.1994.409478")
JIANG_TIAN = paper("Jiang and Tian", 2005, "The model-free implied volatility and its information content", "Review of Financial Studies 18(4)", "10.1093/rfs/hhi027")
GARLEANU_PEDERSEN_POTESHMAN = paper("Gârleanu, Pedersen and Poteshman", 2009, "Demand-based option pricing", "Review of Financial Studies 22(10)", "10.1093/rfs/hhp005")
BOLLEN_WHALEY = paper("Bollen and Whaley", 2004, "Does net buying pressure affect the shape of implied volatility functions?", "Journal of Finance 59(2)", "10.1111/j.1540-6261.2004.00647.x")
NI_PEARSON_POTESHMAN = paper("Ni, Pearson and Poteshman", 2005, "Stock price clustering on option expiration dates", "Journal of Financial Economics 78(1)", "10.1016/j.jfineco.2004.08.005")
NI_PEARSON_POTESHMAN_WHITE = paper("Ni, Pearson, Poteshman and White", 2021, "Does option trading have a pervasive impact on underlying stock prices?", "Review of Financial Studies 34(4)", "10.1093/rfs/hhaa082")
BARBON_BURASCHI = paper("Barbon and Buraschi", 2020, "Gamma fragility", "SSRN 3725454", "10.2139/ssrn.3725454")


# ------------------------------------------------------------------------------------------------------------------ method -> sources
MODELS: dict[str, tuple[Ref, ...]] = {
    "tsmom": (MOSKOWITZ, HURST), "tsmom_multi": (HURST, MOSKOWITZ), "ma_ensemble": (BAZ,), "donchian": (FAITH, CLENOW_FTT), "kama_trend": (KAUFMAN,),
    "adx_trend": (WILDER,), "kalman_trend": (HARVEY_KF,), "mayer_multiple": (BROCK, GERRITSEN, DETZEL),
    "carver_ewmac": (CARVER_ST, CARVER_LT, CARVER_AFTS), "clenow_momentum": (CLENOW_SOTM,), "tstat_momentum": (BALTAS_2020, BALTAS_2013),
    "variance_ratio_switch": (LO_MACKINLAY_88, LO_MACKINLAY_89, CLM, CHAN_AT), "connors_rsi": (CONNORS,), "rsi2": (CONNORS,), "double_seven": (CONNORS,),
    "consecutive_down": (CONNORS,), "skew_premium": (AMAYA, FERNANDEZ_PEREZ), "cppi": (BLACK_JONES, PEROLD_SHARPE, BLACK_PEROLD, ESTEP_KRITZMAN),
    "momentum": (JEGADEESH_TITMAN,), "xs_momentum": (JEGADEESH_TITMAN,), "momentum_13612w": (KELLER_VAA, KELLER_DAA), "short_term_reversal": (JEGADEESH_90, LEHMANN),
    "low_volatility": (ANG,), "low_idio_vol": (ANG,), "bab": (FRAZZINI_PEDERSEN,), "defensive_beta": (FRAZZINI_PEDERSEN,), "high_52w": (GEORGE_HWANG,), "residual_momentum": (BLITZ,),
    "smooth_momentum": (DA_GURUN_WARACHKA,), "max_effect": (BALI,), "amihud_illiquidity": (AMIHUD,), "value_momentum": (ASNESS_MP,), "long_term_reversal": (DE_BONDT, ASNESS_MP),
    "value_proxy": (DE_BONDT, ASNESS_MP), "carry": (KOIJEN,), "carry_xs": (KOIJEN,), "carry_ts": (KOIJEN,), "basis_momentum": (BOONS_PRADO,), "momentum_carry": (CARVER_ST, FUERTES, KOIJEN, MOSKOWITZ), "vol_managed_long": (MOREIRA_MUIR,),
    "garch_vol_managed": (BOLLERSLEV_86, MOREIRA_MUIR), "evt_risk_managed": (MCNEIL_FREY,), "faber_gtaa": (FABER,), "dual_momentum": (ANTONACCI,), "accelerating_dual_momentum": (ANTONACCI,),
    "paa": (KELLER_PAA,), "vaa": (KELLER_VAA,), "daa": (KELLER_DAA,), "adaptive_asset_allocation": (BUTLER,), "model_portfolio": (BROWNE,), "policy_portfolio": (PEROLD_SHARPE,),
    "turn_of_month": (LAKONISHOK_SMIDT, MCCONNELL_XU), "sell_in_may": (BOUMAN_JACOBSEN,), "seasonal_rank": (KELOHARJU,), "vrp_timing": (BOLLERSLEV_TZ, CARR_WU),
    "variance_carry": (CARR_WU,), "pca_residual": (AVELLANEDA_LEE,), "cointegration_pairs": (GATEV, ENGLE_GRANGER, VIDYAMURTHY), "kalman_pairs": (ELLIOTT, CHAN_AT),
    "ou_threshold_pairs": (BERTRAM, AVELLANEDA_LEE), "ou_reversion": (BERTRAM, CHAN_AT), "bollinger_reversion": (BOLLINGER,),
    "ml_ridge": (GU_KELLY_XIU,), "ml_trees": (GU_KELLY_XIU, LOPEZ_AFML), "ml_factor_model": (GU_KELLY_XIU,), "gp_factor_model": (RASMUSSEN,),
    "fundamental_value": (FAMA_FRENCH_93, NOVY_MARX), "fundamental_quality": (NOVY_MARX, PIOTROSKI, SLOAN, AFP_QMJ), "fundamental_dcf": (DAMODARAN,),
    "fundamental_alpha": (GRINOLD_KAHN,), "curve_steepener": (LITTERMAN_SCHEINKMAN, ILMANEN), "butterfly": (LITTERMAN_SCHEINKMAN,),
    "carry_rolldown": (ILMANEN,), "duration_timing": (ILMANEN,), "cftc_positioning": (BESSEMBINDER, BASU_MIFFRE), "funding_carry": (SCHMELING, HE_MANELA), "basis_reversion": (SCHMELING, HE_MANELA),
}
ALLOCATORS: dict[str, tuple[Ref, ...]] = {
    "static": (MARKOWITZ, MAILLARD, RONCALLI, LOPEZ_HRP, RAFFINOT), "min_variance": (CLARKE_MINVAR,), "max_diversification": (CHOUEIFATY,), "black_litterman": (BLACK_LITTERMAN,),
    "kelly": (KELLY, MACLEAN_THORP_ZIEMBA), "tangency_cml": (MARKOWITZ, SHARPE_64), "mv_cvar": (ROCKAFELLAR,), "dynamic_cov": (ENGLE_DCC,), "bayesian": (JORION,),
    "hierarchical_sensitivity_parity": (LOPEZ_HRP,), "graph_centrality": (POZZI,), "multi_period": (GARLEANU_PEDERSEN,), "tca_mvo": (GARLEANU_PEDERSEN,), "es_policy": (SALIMANS,),
    "qubo_select": (LUCAS,), "constrained_long_short": (GRINOLD_KAHN,), "risk_budget": (RONCALLI, MAILLARD), "resampled_mv": (MICHAUD,),
}


FORECASTERS: dict[str, tuple[Ref, ...]] = {
    "riskmetrics": (RISKMETRICS,), "hull_white": (HULL_WHITE_98,), "gbm": (BLACK_SCHOLES,), "bootstrap": (POLITIS_ROMANO,), "garch11": (BOLLERSLEV_86, HANSEN_LUNDE), "garch": (GJR, BOLLERSLEV_86, HANSEN_LUNDE),
    "egarch": (NELSON, HANSEN_LUNDE), "fhs": (BGV, BARONE_ADESI), "evt": (MCNEIL_FREY, MCNEIL_QRM), "garch_vg": (MADAN_CARR_CHANG,), "garch_nig": (BARNDORFF_NIELSEN,), "heston": (HESTON,), "jump": (MERTON_76,),
    "sv_kalman": (HARVEY_RUIZ_SHEPHARD,), "har": (CORSI, PARKINSON), "markov_returns": (HAMILTON,), "markov_volatility": (HAMILTON,), "hmm": (HAMILTON,), "gp_returns": (RASMUSSEN,), "gp_drift": (RASMUSSEN,),
    "gp_price": (RASMUSSEN,), "kalman": (HARVEY_KF,), "theta": (ASSIMAKOPOULOS, HYNDMAN_BILLAH, FPP3), "arma": (BOX_JENKINS,), "analog": (FARMER_SIDOROWICH,),
    "blend": (BATES_GRANGER, TIMMERMANN, SMITH_WALLIS), "blend_weighted": (BATES_GRANGER, TIMMERMANN, SMITH_WALLIS),
}
# How a forecast distribution is scored and compared (the Forecast tab's evaluation), shown once for the whole page.
FORECAST_EVALUATION: tuple[Ref, ...] = (GNEITING_RAFTERY, FERRO, DIEBOLD_MARIANO, CLARK_WEST, DIEBOLD_GUNTHER_TAY, KUPIEC, GOYAL_WELCH)


def references(kind: str, name: str) -> list[dict]:
    """The sources of a method as ``[{"cite", "url"}]`` (empty for a method with none on record)."""
    table = {"model": MODELS, "allocator": ALLOCATORS, **EXTRA}.get(kind, {})
    return [r.view() for r in table.get(name, ())]


FUTURES_VIEWS: dict[str, tuple[Ref, ...]] = {
    "curve": (GORTON_ROUWENHORST, ERB_HARVEY, FAMA_FRENCH_87, KALDOR), "convenience": (GIBSON_SCHWARTZ, SCHWARTZ_SMITH), "pca": (LITTERMAN_SCHEINKMAN, TOLMASKY_HINDANOV),
    "carry": (KOIJEN, GORTON_HAYASHI_ROUWENHORST, ERB_HARVEY), "seasonality": (ROUTLEDGE_SEPPI_SPATT,), "signals": (MOSKOWITZ, HURST, CARVER_ST, KOIJEN),
    "trend": (MOSKOWITZ, HURST, MIFFRE_RALLIS, SZAKMARY), "term_structure": (FUERTES, BAKSHI_GAO_ROSSI, BOONS_PRADO),
}
CRYPTO_VIEWS: dict[str, tuple[Ref, ...]] = {
    "trend": (BROCK, GERRITSEN, DETZEL, LIU_TSYVINSKI, LIU_TSYVINSKI_WU), "volatility": (BURGHARDT_LANE, MOREIRA_MUIR), "relative": (MAKAROV_SCHOAR, GRIFFIN_SHAMS, LIU_TSYVINSKI_WU), "carry": (SCHMELING, HE_MANELA, MAKAROV_SCHOAR),
    "signals": (LIU_TSYVINSKI, LIU_TSYVINSKI_WU, SCHMELING, MOSKOWITZ),
}
_STRATEGY_BOOKS = (MCMILLAN, NATENBERG, COHEN)                                                                      # the textbooks that catalogue the standard structures
OPTIONS: dict[str, tuple[Ref, ...]] = {
    # structures (the strategy builder) and rolling strategies (its backtest), by name
    "long_call": (HULL_BOOK, MCMILLAN, NATENBERG), "long_put": (HULL_BOOK, MCMILLAN, NATENBERG), "covered_call": (WHALEY_BXM, ISRAELOV_NIELSEN, HILL_COVERED, MCMILLAN, CBOE_BENCHMARKS),
    "protective_put": (CBOE_2015, MCMILLAN, HULL_BOOK), "bull_call_spread": (HULL_BOOK, *_STRATEGY_BOOKS), "bear_put_spread": (HULL_BOOK, *_STRATEGY_BOOKS),
    "iron_condor": (CBOE_2015, NATENBERG, COHEN, MCMILLAN), "long_straddle": (COVAL_SHUMWAY, BAKSHI_KAPADIA, HULL_BOOK, NATENBERG), "long_strangle": (COVAL_SHUMWAY, HULL_BOOK, NATENBERG),
    "short_straddle": (COVAL_SHUMWAY, BONDARENKO, CARR_WU, SINCLAIR), "short_strangle": (COVAL_SHUMWAY, BONDARENKO, CARR_WU, SINCLAIR), "calendar_spread": (HULL_BOOK, NATENBERG, MCMILLAN),
    "butterfly": (HULL_BOOK, *_STRATEGY_BOOKS), "ratio_spread": (NATENBERG, MCMILLAN), "collar": (ISRAELOV_KLEIN, CBOE_2015, CBOE_BENCHMARKS, HULL_BOOK, MCMILLAN),
    "cash_secured_put": (CBOE_BENCHMARKS, BONDARENKO, COVAL_SHUMWAY), "iron_butterfly": (CBOE_2015, NATENBERG, COHEN, MCMILLAN), "long_condor": _STRATEGY_BOOKS,
    "risk_reversal": (NATENBERG, HULL_BOOK, COHEN), "strap": (HULL_BOOK, MCMILLAN, NATENBERG), "strip": (HULL_BOOK, MCMILLAN, NATENBERG), "call_backspread": (NATENBERG, MCMILLAN, COHEN),
    "put_backspread": (NATENBERG, MCMILLAN, COHEN), "diagonal_spread": (HULL_BOOK, MCMILLAN, COHEN), "tail_hedge": (TALEB, BONDARENKO, COVAL_SHUMWAY, CBOE_2015),
    "short_put": (CBOE_BENCHMARKS, BONDARENKO, BAKSHI_KAPADIA), "delta_hedged_straddle": (BAKSHI_KAPADIA, COVAL_SHUMWAY, CARR_WU, SINCLAIR), "put_spread": (HULL_BOOK, *_STRATEGY_BOOKS),
    # what the chain says
    "distribution": (BREEDEN_LITZENBERGER, GATHERAL_JACQUIER, GATHERAL_BOOK, JACKWERTH_RUBINSTEIN, BLISS_PANIGIRTZOGLOU, FIGLEWSKI), "model_free": (BRITTEN_JONES, DEMETERFI, NEUBERGER, JIANG_TIAN, CARR_WU, CBOE_VIX),
    "positioning": (GARLEANU_PEDERSEN_POTESHMAN, BOLLEN_WHALEY, NI_PEARSON_POTESHMAN, NI_PEARSON_POTESHMAN_WHITE, BARBON_BURASCHI), "cone": (BURGHARDT_LANE, NATENBERG, SINCLAIR),
    "timing": (BOLLERSLEV_TZ, CARR_WU, SINCLAIR),
}
EXTRA: dict[str, dict[str, tuple[Ref, ...]]] = {"forecaster": FORECASTERS, "futures": FUTURES_VIEWS, "crypto": CRYPTO_VIEWS, "options": OPTIONS}   # tables of the other tabs (forecasters, futures views, option structures, ...)
