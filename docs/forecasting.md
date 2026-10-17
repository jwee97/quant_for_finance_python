# Forecasting a stock's price, and how far to trust it

Nobody knows what a stock will cost next month. What can be said is how far it is likely to move, and how that range changes with the market's mood. That is a *forecast as a distribution*: a set of possible futures, each with a probability. The **Forecast** tab makes one for a ticker with the methods a quant desk would name (Monte Carlo simulation, Markov chains and hidden Markov models, Gaussian processes, a Kalman filter, volatility models of the GARCH, Lévy and Heston kinds, analogues) and then does what most forecasting pages skip. It goes back in time, makes each method's forecast again at about two hundred past dates using only the prices before each date, and scores every one against what then happened. A method is only as good as that record.

The short version of what the record says on the platform's ETFs: the *spread* of returns can be forecast a little (on SPY by more than luck allows, on QQQ and EEM by amounts luck could explain, on bonds and gold not at all); the *average* return and the *direction* could not be forecast at all; and the most popular picture of a Gaussian-process forecast, a smooth curve drawn through recent prices and continued, was the worst method by a wide margin. The page says which of these holds for your ticker, with the numbers, every time.

## Five minutes, with SPY

1. Open the **Backtest** tab and look at the ticker list. The platform's 15 ETFs are there to begin with, and you can add symbols from Yahoo Finance or iTick. The Forecast tab uses that list, the same data source and the same start date.
2. Open **Forecast**. Under *1. Which stock, how far ahead* choose **SPY** and **1 month (21 trading days)**.
3. Under *2. Which methods* the random walk is ticked and cannot be unticked: it is the yardstick that every other method has to beat. *Recommended* ticks eleven methods (RiskMetrics, Hull-White, GJR-GARCH, filtered historical simulation, HAR, the Markov chain on volatility, the hidden Markov model, the Gaussian processes on recent moves and on the drift, the Kalman trend and the analogues); *All of them* adds the rest and an equal-weight blend; *Fast ones only* leaves out the methods whose fit takes seconds. Hover a name for what it does.
4. Under *3. How thorough* choose **Quick** (four years of past forecasts, one a month, 500 simulated futures each), **Standard** (eight years, one every two weeks, 1,000 futures) or **Thorough** (twelve years, one a week, 2,000 futures). With every method on SPY, Standard takes about a minute and a half on a laptop (the recommended eleven take about a minute); Quick takes about half a minute and Thorough about four minutes. A slower machine, a Replit workspace for one, takes longer. *Random seed* makes a run repeatable.
5. Press **Forecast and test the methods**. The page says which past forecast it is on. You can leave: the run goes on in the app's own worker and is listed under *Recent runs* when you come back. **Stop and show what it has so far** ends it after the forecast it is on and scores what was done.
6. Read the verdict and the fan (below).

## Reading the result

**The verdict** is a sentence with a symbol, never a colour alone. It answers one question: did any method describe the *spread* of the next returns better than the random walk, by more than luck allows?

| Symbol | Says | What it means |
|---|---|---|
| ✓ | *N methods described the spread of 21-day returns better than the random walk, the best by X%* | At least one method scored at least 1% better than the random walk, and the gain is more than luck allows after the correction for the number of methods compared (Holm-adjusted p below 5%). |
| ! | *Some methods did better than the random walk, but not by more than luck could explain* | The best method was ahead, but a lead that size turns up by chance often enough when this many methods are tried on this few independent forecasts. |
| ✗ | *No method forecast the spread better than the random walk* | The best was no better than a constant spread read off the last year. |

Under the sentence are the reasons, in the order a careful reader would ask them: how many forecasts were tested and how many of them are independent; what the yardstick scored; who won and by how much; who did *worse* than the random walk (a fancier method can be less honest about what it does not know); how well calibrated the best method was; whether any method forecast the average return or the direction better than the simple alternatives; and whether a complicated method added anything to the simpler method it builds on.

**The fan** is the last 120 days of the price and then, for the method you choose with the buttons, the range of where the price may go: the middle half of the forecasts in the darker band, nine in ten in the lighter band, the median as a line, and the random walk's 90% range as a grey band behind for scale. It opens on the method with the best record at your horizon. The *Table* view lists the five quantiles for the days between today and the horizon, and *Download the forecast (CSV)* saves the fans of every method.

**What each method says now** has the average return, the middle (median) return, the 5% worst case and 95% best case, the chance of a rise, the chance of losing 5% or more, and the method's record against the random walk. Methods can disagree about the average and agree about the spread, which is the first clue to where the information is: it is nearly always in the spread.

**The scoreboard** ranks the methods by *skill*: one minus the method's average CRPS divided by the random walk's. The CRPS (continuous ranked probability score) scores a whole forecast distribution: the average distance of the simulated futures from what happened, less half their average distance from each other. It rewards a forecast for being sharp *and* right, it equals the absolute error when the forecast is a single number, and an honest forecaster cannot improve it by shading the answer. The form used here is the *fair* one, so a method run with more simulated futures does not look better. Skill of +5% means the method's average score was 5% below the random walk's; a tick means the gain held up after the correction for the number of methods compared. The table view adds the skill at 1, 5 and 21 days and your horizon, how often the 90% range held the outcome, and how long each method took.

**Were the ranges the right size?** shows where the outcomes fell among each forecast's own range (the *probability integral transform*, in tenths). A flat chart means the ranges were the right size, a hump in the middle means they were too wide, a U means too narrow and a slope means the forecasts were biased. Beside it: how often the central 90% and 50% ranges held the outcome, how often the outcome fell below the method's 5% quantile (the value-at-risk check: 5% is right), and a Kolmogorov-Smirnov test of the histogram. These are tested with the forecasts one horizon apart, so that the outcomes are independent.

**Did any method forecast the average return or the direction?** compares each method's forecast of the average with the standard benchmark of the research literature, the average return of the last five years, by the out-of-sample R-squared (above zero means better) and the Clark-West test (the right test, because a model that estimates a mean the market does not have is mechanically worse than the benchmark). It compares each method's calls of up or down with *always saying up*, which is right as often as the market rose. The page asks for a p-value below 5% divided by the number of methods before it calls one useful.

**What each method assumes, and where it breaks**, **what the methods learned** (the numbers each fitted and the state each is in on the last day) and **Read before relying on it** close the page.

## The methods

Twenty-two methods (the random walk and twenty-one others) and an equal-weight blend, in nine families. Each has a technique guide with the idea, what the repository found and how it goes wrong.

| Family | Method | The idea in one line | Forecasts the average? |
|---|---|---|---|
| Baseline | **Random walk** | No view on direction; the spread of the last year. The yardstick. | no (zero) |
| Volatility | **RiskMetrics** | The spread follows an exponentially weighted average of recent squared returns (decay 0.94). | no |
| Volatility | **HAR** | Tomorrow's volatility from the volatility of the last day, week and month, using the day's high and low. [Guide](techniques/har-kalman-and-analogue-forecasts.md) | no |
| Monte Carlo | **Historical simulation** | Blocks of consecutive past days, resampled. [Guide](techniques/monte-carlo-forecasting.md) | the last five years' |
| Monte Carlo | **Hull-White** | Past returns rescaled to today's volatility, then resampled in blocks. | the last five years', scaled |
| Monte Carlo | **Brownian motion** | Normal returns with the drift and volatility of the last year. | the last year's |
| Monte Carlo | **GJR-GARCH** | Volatility that clusters, rises more after a fall, with Student-t shocks, simulated from today's volatility. | no |
| Monte Carlo | **Filtered historical simulation** | The same volatility model with the stock's own past standardised shocks resampled. | no |
| Monte Carlo | **GJR-GARCH, variance-gamma shocks** | The same volatility model with shocks from a variance-gamma law fitted to its own standardised residuals: a free skew and kurtosis and smooth tails. [Guide](techniques/stochastic-volatility-and-levy-forecasting.md) | no |
| Monte Carlo | **GJR-GARCH, NIG shocks** | The same with a normal inverse Gaussian law, a little heavier in the tails. | no |
| Monte Carlo | **Heston** | Variance that reverts to a long-run level and moves against the return, simulated with Andersen's quadratic-exponential scheme; its parameters from the autocorrelation of squared returns. | no |
| Monte Carlo | **Jump diffusion** | Small ordinary days plus rare large ones with a fitted rate and size. | no |
| Markov | **Markov chain, daily move** | Five states from a big fall to a big rise; transition probabilities counted from five years. [Guide](techniques/markov-forecasting.md) | the states' |
| Markov | **Markov chain, volatility** | Calm, ordinary or stormy by the volatility of the last month; the chain says how long storms last. | the states' |
| Markov | **Hidden Markov model** | Two unobserved regimes with their own mean and volatility; the forecast starts from today's probability of each. | the regimes' |
| Gaussian process | **On recent moves** | Learns how the next return, in units of volatility, depends on the last week's, month's and quarter's moves, on volatility and on the distance from the high. [Guide](techniques/gaussian-process-forecasting.md) | yes |
| Gaussian process | **On the drift** | The daily return as a slowly changing average plus noise, carried forward. | yes |
| Gaussian process | **On the price path** | A smooth curve through the last 120 days of log price, continued. The control. | yes |
| State space | **Kalman trend** | A level with a slowly changing slope, tracked day by day. | yes |
| State space | **Stochastic-volatility Kalman filter** | Log variance as a hidden AR(1), tracked by a Kalman filter on the log of squared returns; the forecast simulates volatility forward from the filter's uncertain state. | no |
| Time series | **ARMA(1,1)** | Whether yesterday's return and surprise predict today's. | yes |
| Pattern matching | **Analogues** | What followed the 60 past days most like today. | yes |
| Combination | **Equal-weight blend** | Draws equally from every other method you chose. [Guide](techniques/forecast-evaluation.md) | the members' |

Three of the methods are there as *controls* rather than as candidates. The Gaussian process on the price path is what people try first, and the test says how far it is from working. Hull-White is the control of the methods that learn from returns in units of volatility (the Gaussian process on recent moves and the analogues): what they add to it is what they learn about the future, and the page asks that question in words. GJR-GARCH is the control of the filtered historical simulation and the hidden Markov model for the same reason, and the natural yardstick of the three methods that vary it (variance-gamma shocks, NIG shocks and Heston); the page's control tests do not cover those three, so read their skill against GJR-GARCH's in the tables below.

## How the test works

Every number on the page comes from one procedure, the same for every method.

1. **Walk forward.** The last eight years (Standard) are divided into origins, one every ten trading days. At each origin a method is handed the prices up to that day and nothing later; it forecasts the distribution of the log return over the next 1, 5 and 21 days and your horizon, as 1,000 simulated futures; and the forecast is scored against the return that then happened. A method's model is refitted every 63 trading days (126 for the slow ones) on the last five years; between refits it brings its state up to date with each new day and keeps the fitted numbers, which is what a live system would do.
2. **Nothing from the future.** A method is only ever *given* history that ends at the origin, so there is no future in what it sees. The tests check that too, by replacing everything after an origin with noise and confirming that no forecast at that origin changes.
3. **Proper scores.** CRPS (above) and the pinball loss of five quantiles. Both are *proper*: the forecaster's best strategy is to say what it believes.
4. **Skill and the test of it.** Skill is one minus the method's average CRPS over the random walk's. Whether a difference in average CRPS is more than luck is a one-sided Diebold-Mariano test with a Newey-West (HAC) standard error: forecasts made ten days apart for a 21-day horizon share most of their outcome, so their losses are not independent, and the test is widened by the overlap (lag = the number of origins one horizon spans, less one). A Standard run of 21-day forecasts has about 200 forecasts and about 67 independent ones, and the page says so.
5. **Many methods, one winner.** With twenty-two methods one of them will look good by chance. The p-values are corrected by the Holm procedure (a step-down Bonferroni that controls the chance of *any* false winner among them), and the verdict names a method as better only if its adjusted p-value is under 5% *and* it is at least 1% better, so that a tiny but "significant" gain is not announced.
6. **Calibration.** The central 50% and 90% ranges should hold the outcome that often, the outcome should fall below the 5% quantile one time in twenty (the Kupiec check for value at risk), and the PIT values should be evenly spread. Binomial and Kolmogorov-Smirnov tests, on forecasts one horizon apart.
7. **The average and the direction.** For the methods that forecast an average, the out-of-sample R-squared against the trailing five-year average with the Clark-West test, and the share of ups and downs called right against always saying up with a Diebold-Mariano test of the two 0-1 losses.
8. **Controls and the blend.** A control comparison is a Diebold-Mariano test of a method against the simpler one it builds on. The blend pools the forecasts of the other chosen methods in equal shares (a *linear opinion pool*); because the CRPS is convex in the forecast, the pool scores no worse than the average of its members, and the tests check that.

## What it found on real prices

The numbers below are the platform's cached ETF prices, 2018 to September 2026, at the Standard setting (202 forecasts, one every ten trading days from 18 September 2018, 1,000 futures each) with every method; `python -m experiments.forecast_study` reproduces them. They are one history of five ETFs, so read the pattern and not the second decimal.

**Skill at 21 days**: how far below the random walk's CRPS each method's was. A tick: the gain held up after the Holm correction.

| Method | SPY | QQQ | EEM | TLT | GLD | Average |
|---|---:|---:|---:|---:|---:|---:|
| RiskMetrics | +2.6% | +0.6% | −0.9% | −0.2% | −3.2% | −0.2% |
| Historical simulation | +6.2% | +2.9% | +0.9% | −2.4% | +1.9% | +1.9% |
| Hull-White (volatility-scaled) | +8.5% ✓ | +4.5% | −0.6% | −1.5% | −2.8% | +1.6% |
| Brownian motion | +0.5% | +1.4% | −1.8% | −4.0% | +1.5% | −0.5% |
| GJR-GARCH | +6.5% ✓ | +2.7% | +1.7% | +0.9% | −2.0% | +1.9% |
| Filtered historical simulation | +6.7% ✓ | +2.7% | +2.0% | +0.8% | −1.9% | +2.1% |
| Jump diffusion | +1.1% | −0.1% | −0.3% | −1.2% | +0.1% | −0.1% |
| HAR volatility | +3.4% ✓ | +1.1% | +1.4% | −0.8% | −1.3% | +0.8% |
| Markov chain, daily move | +4.8% | +2.9% | +0.3% | −2.5% | +2.0% | +1.5% |
| Markov chain, volatility | +6.1% | +3.3% | +0.9% | −2.1% | +1.2% | +1.9% |
| Hidden Markov model | +4.9% | +2.0% | −1.9% | −0.6% | +0.5% | +1.0% |
| Gaussian process, recent moves | +6.6% | −0.3% | −2.3% | −8.8% | −5.1% | −2.0% |
| Gaussian process, drift | +1.3% | +0.9% | −5.1% | −8.2% | −10.7% | −4.3% |
| Gaussian process, price path | −73.3% | −81.2% | −64.7% | −62.8% | −90.3% | −74.5% |
| Kalman trend | −0.3% | −1.6% | −1.6% | −3.0% | +0.4% | −1.2% |
| ARMA(1,1) | −0.1% | −1.4% | +0.1% | −1.5% | −0.2% | −0.6% |
| Analogues | +5.9% | +3.6% | −8.7% | −9.1% | −7.6% | −3.2% |
| Equal-weight blend | +7.1% ✓ | +4.5% | +1.2% | 0.0% | +0.1% | +2.6% |

What stands out:

- **The spread can be forecast a little, and on equities.** The methods that forecast no average at all (RiskMetrics, GJR-GARCH, filtered historical simulation, HAR) isolate the spread: on SPY they were 2.6% to 6.7% better than a constant spread, and the gain held up for three of the four; on QQQ 0.6% to 2.7%, on EEM −0.9% to 2.0%, on TLT within 1% either way, on GLD worse. Only SPY got a ✓ verdict; the other four got *mixed*: some methods were ahead, but not by more than luck could explain.
- **Where the gain is largest it is partly the average.** The historical simulations and Hull-White carry the last five years' average return into a stretch in which SPY rose in about two 21-day windows out of three, and the random walk's average is zero, so they beat it a little for that reason alone. The page separates the two with the next tests, and neither the 21-day gain of the historical simulations nor Hull-White's 8.5% on SPY is shown to be a forecast of the average: against the five-year average itself none of the thirteen methods that forecast one had an out-of-sample R-squared above 4.6% on SPY (Hull-White, Clark-West p = 0.08, short of the threshold of 0.004 that thirteen methods call for), and none beat always saying up (always up was right 68% of the time on SPY, and the best caller also 68%).
- **The methods that learn from features add noise.** The Gaussian process on recent moves was 6.6% better than the random walk on SPY and 8.8% worse on TLT; against its control, Hull-White, which uses the same units without the learning, it scored between −1.7% and −7.2% on all five (p-values of 0.74 to 0.97 for the claim that it was better). The analogues behave the same way (−0.9% to −8.0% against Hull-White). A year or five of daily returns is too little signal for a flexible method to find; what the page shows is what the control already has.
- **The textbook Gaussian-process forecast is the worst by far.** A smooth curve through the last 120 days of log price, continued, was between 63% and 90% *worse* than the random walk on every ticker. A price is a sum of its own shocks, not a smooth curve plus noise, and a curve fitted to it extrapolates the shocks as if they were a trend. It is in the list as the control that shows this.
- **The equal-weight blend is the safe choice.** It was never behind the random walk on any of the five (from +7.1% on SPY down to 0.0% on TLT), it is the only method on the list of which that is true, and averaged over the five it had the best skill of all (+2.6%). It was the best, or level with the best, on only one ticker (QQQ), which is the point of a blend: it is rarely the winner and never the loser.
- **The yardstick is well calibrated in coverage and not in shape.** Its central 90% range held the outcome 88% to 93% of the time, and the outcome fell below its 5% quantile 1.5% to 5.5% of the time; yet on SPY, QQQ and GLD the test of the whole PIT histogram rejects it (p = 0.003, 0.007 and 0.005), because a constant spread is too wide in calm times and too narrow in storms. The methods that follow volatility fix that.

**By horizon, SPY.** Each column is its own run (that horizon as the headline).

| Method | 5 days | 21 days | 63 days |
|---|---:|---:|---:|
| RiskMetrics | +4.5% ✓ | +2.6% | +1.5% |
| Historical simulation | +2.5% | +6.2% | +12.8% |
| Hull-White (volatility-scaled) | +5.5% ✓ | +8.5% ✓ | +14.9% |
| Brownian motion | +1.3% | +0.5% | −2.4% |
| GJR-GARCH | +6.7% ✓ | +6.5% ✓ | +7.7% ✓ |
| Filtered historical simulation | +6.8% ✓ | +6.7% ✓ | +8.5% ✓ |
| GJR-GARCH, variance-gamma shocks | +6.6% ✓ | +6.6% ✓ | +8.1% ✓ |
| GJR-GARCH, NIG shocks | +6.3% ✓ | +6.7% ✓ | +8.6% ✓ |
| Heston (method of moments) | +5.9% ✓ | +6.7% ✓ | +7.8% ✓ |
| Jump diffusion | +1.5% | +1.1% | +0.8% |
| Stochastic-volatility Kalman filter | +4.6% ✓ | +3.0% | +1.0% |
| HAR volatility | +5.6% ✓ | +3.4% ✓ | +0.7% |
| Markov chain, daily move | +2.5% | +4.8% | +10.2% |
| Markov chain, volatility | +3.9% | +6.1% | +13.1% |
| Hidden Markov model | +3.8% | +4.9% | +9.0% |
| Gaussian process, recent moves | +5.2% ✓ | +6.6% | +2.9% |
| Gaussian process, drift | +3.7% | +1.3% | +0.1% |
| Gaussian process, price path | −43.8% | −73.3% | −49.5% |
| Kalman trend | −0.4% | −0.3% | +2.8% |
| ARMA(1,1) | −0.1% | −0.1% | +0.1% |
| Analogues | +4.4% | +5.9% | +7.9% |
| Equal-weight blend | +5.6% ✓ | +7.2% ✓ | +10.5% ✓ |

The zero-average volatility models keep their gain at three months (GJR-GARCH +7.7% ✓, filtered historical simulation +8.5% ✓, and the three variants with other shocks or other volatility dynamics, +7.8% to +8.6% ✓), while the gain of HAR fades from 5.6% at a week to 0.7% at three months, as it should, since a model of the next days' volatility says little about the next quarter's, and that of the stochastic-volatility Kalman filter from 4.6% to 1.0%. The big numbers for the historical simulations, Hull-White and the Markov chains at 63 days (+10% to +15%) are mostly the average again: SPY rose in 76% of the 63-day windows, the methods that carry a positive average score better than the zero-average random walk, and the correction for many methods and the overlap between 63-day windows (about 28 of the forecasts are independent) leaves only GJR-GARCH, the filtered historical simulation, the two shock-law variants, Heston and the blend standing.

**The average and the direction.** On SPY at 5, 21 and 63 days and on the other four at 21 days, no method beat the five-year average at the threshold the page asks for, and none beat always saying up. The best out-of-sample R-squared at 21 days was +4.6% (SPY, Hull-White), +2.0% (QQQ), +0.7% (EEM), +1.5% (TLT), +1.1% (GLD); the closest to the threshold, GLD's, has p = 0.026 against the 0.004 the page asks for. A forecast of the average is the hardest forecast in finance, and a year or five of daily prices does not make it easier.

## How it was checked

A forecasting test that cannot tell a method that works from one that does not is worth nothing, so each method was first run on a market whose truth is known (`tests/forecast_worlds.py`):

- **Volatility that clusters.** On GJR-GARCH returns RiskMetrics, GJR-GARCH and the filtered historical simulation beat the random walk by more than 2% at 1 and 5 days, with Holm-adjusted p-values below 5%, and GJR-GARCH finds the planted persistence and leverage.
- **Nothing to find.** On independent returns no method beat the random walk, every skill was within 6% of zero, and the random walk's calibration was not rejected.
- **Regimes.** On a two-regime market the hidden Markov model recovers each regime's volatility to within 30%, the persistence of each regime, and which one the market is in on the last day (above 90%); the Markov chain on volatility finds calm and stormy states that persist for days at a time and puts today in the right one.
- **A planted drift.** On a market whose average return swings slowly, the Kalman trend, the Gaussian process on the drift and the Gaussian process on recent moves have an out-of-sample R-squared above 10% against the five-year average with p below 1%, and call the direction better than always saying up; on independent returns the Gaussian process on the drift finds none.
- **A planted reversal.** Where the past month's move predicts the next days' return the other way, the Gaussian process explains part of the variance and forecasts a lower return after a rise than after a fall, and the analogues beat Hull-White's rescaled history, which has no way to see it.
- **Jumps and trends.** The jump diffusion finds the planted rate, size and calm volatility, and none where there were none; the Kalman filter finds the planted slope and a change of direction; the ARMA finds the planted serial dependence and none in independent returns; the Gaussian process on the price path continues a straight line and does worse than staying put on a random walk.
- **Shock laws and stochastic volatility.** With variance-gamma shocks the fitted law reproduces the skew and the kurtosis of the residuals and the forecast has the longer left tail; with Gaussian shocks the fitted shocks have neither. The stochastic-volatility filter finds the planted persistence and volatility of log variance, and its 90% ranges hold the outcome about nine times in ten over many forecast dates; the method of moments recovers the Heston speed, level, volatility of variance and (roughly) correlation on average over simulated samples, and says the volatility of variance is small when the returns do not cluster (`tests/test_forecast_stochvol.py`).
- **No look-ahead.** A spy method records every price it is shown; replacing everything after an origin with noise changes nothing it saw up to that origin.
- **The scores themselves.** CRPS against the closed form for a Gaussian, the fair form against the small-sample bias of the plain one, the pinball loss against its definition, the Holm adjustment against the reference implementation, the PIT as uniform for a calibrated forecast, humped for one too wide and U-shaped for one too narrow.

## What it does not do

- **It is not advice and not a strategy.** A method can describe the spread of returns well and make no money, and the other way round; how good a forecast is and how much a rule built on it earns are different questions, and the Backtest tab answers the second. The chance of a rise on the page is a probability, not a signal.
- **One ticker, one history.** The ranking is for one stock over one stretch. Another period, another stock or another horizon would rank the methods differently, and the order among the best few is within what luck could change.
- **Daily prices only.** No intraday data, order flow, earnings, news or macro series enter any method (the HAR model uses the day's high and low where the data have them).
- **Yahoo Finance and iTick data are free and unaudited.** The platform's 15 ETFs are the versioned, cleaned dataset. A ticker with fewer than about three years of prices cannot be tested (the page says how long it needs).
- **Future dates skip weekends** and ignore exchange holidays, so a date on the fan can be a day out.
- **The methods are fitted by maximum likelihood or least squares**, not by full Bayesian inference; the Gaussian processes use a few hundred points and a few kernel numbers learned from them, and the hidden Markov model has two regimes.
- **A run is not checkpointed.** If the app stops during a run (on Replit, when the workspace goes idle) the run shows as stopped and has to be started again.

## From the terminal and from Python

```bash
quant forecast --list                                           # the methods
quant forecast SPY                                              # the recommended methods, 21 days, Standard
quant forecast QQQ --horizon 5 --methods garch hmm gp_drift --effort quick
quant forecast SPY --all --json spy_forecast.json               # every method and the blend, with the whole result saved
quant forecast SPY --methods garch garch_vg garch_nig heston sv_kalman --effort quick   # the volatility models side by side
quant forecast MyTicker --prices my_prices.csv                  # a wide CSV: one date column, one column per asset
python -m experiments.forecast_study --tickers SPY QQQ          # the tables of this page (add --cache runs/ to keep each finished evaluation and resume after an interruption)
```

```python
from src.forecast import History
from src.forecast import evaluate as ev
from src.forecast.verdict import judge
from src.framework import load_default_bundle
from src.utils.config import load_config

bundle = load_default_bundle(load_config(), with_macro=False)
history = History.from_frame(bundle.prices["SPY"], bundle.high["SPY"], bundle.low["SPY"])
evaluation = ev.evaluate(history, ["garch", "hmm", "gp_drift", "blend"], horizon=21, effort="quick")      # about ten seconds
print(judge(evaluation, 21)["headline"])
now = ev.forecast_now(history, ["random_walk", "garch", "hmm", "blend"], horizon=21)                       # the fans from the last day
print({name: round(m["fan"]["0.5"][-1], 1) for name, m in now["methods"].items()})
```

The code is `src/forecast` (`base` for the interface, `scoring`, `baselines`, `montecarlo`, `stochvol`, `markov`, `gp`, `others`, `evaluate`, `verdict`, `report`); adding a method is writing a class with `fit` and `predict` and listing it in `registry.py`. The routes behind the tab are `GET /api/forecast/plan`, `POST /api/forecast/run` (a job) and `POST /api/forecast/stop`. The technique guides: [Monte Carlo forecasting](techniques/monte-carlo-forecasting.md), [Markov chains and hidden Markov models](techniques/markov-forecasting.md), [Gaussian processes](techniques/gaussian-process-forecasting.md), [HAR, Kalman, ARMA and analogues](techniques/har-kalman-and-analogue-forecasts.md), [stochastic volatility and Lévy shocks](techniques/stochastic-volatility-and-levy-forecasting.md) and [scoring forecasts honestly](techniques/forecast-evaluation.md).
