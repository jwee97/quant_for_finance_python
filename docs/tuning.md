# Tuning a strategy's settings

Every strategy has settings: the two noise levels of the Kalman trend filter, how many days a momentum rule looks back, how many assets it holds. The **Tune** tab finds good values for them on the tickers listed in the Backtest tab, and, more important, tells you whether to believe what it found. You pick a strategy, tick the settings that may change, choose how long to look, and press one button.

The reason it is more than a search: the best of many tries always looks better than a typical one, whether or not it is any good. Try fifty settings on the same prices and one of them will have done well by luck. So the tab keeps the last part of the history back, never lets the search see it, and at the end compares your starting settings with the winner on those untouched days.

## Five minutes, with the Kalman trend filter

1. Open the **Backtest** tab and choose your tickers (the platform's 15 ETFs are there to begin with). The Tune tab uses that list, the same data source and the same start date.
2. Open **Tune**. Choose **kalman_trend** under *1. What to tune*.
3. Under *2. Which settings* the settings that make sense to vary are ticked, each with a range to look in: *level noise*, *slope noise*, *vol halflife* and *scale*. *Start from* is the value your result is compared with (the strategy's own, unless you change it). The text under each ticked setting says what it does and what a higher or lower value changes.
4. Under *3. How long to look* choose **Quick** (up to 20 tries, about a minute and a half), **Standard** (50 tries, five minutes) or **Thorough** (120 tries, fifteen minutes). It stops at whichever limit it reaches first. On the 15 ETFs the Kalman filter takes about a second a try, so Quick finishes in half a minute; a strategy that fits a model every month takes much longer, and the tab says so.
5. Press **Find the best settings**. The page says which try it is on and what the best score is so far. You can leave: the search runs on the app's own worker, and it is listed under *Recent runs* when you come back. **Stop and show what it has so far** ends it after the try it is on.
6. Read the verdict (below). If you like it, **Open the Backtest tab with these settings** puts the strategy and the exact values there; press *Run backtest* to see the full result.

## Reading the result

**The verdict** is a sentence with a symbol, never a colour alone:

| Symbol | Says | What it means |
|---|---|---|
| ✓ | The improvement held up on the untouched days | At least half of the gain on the tuning window was still there on the days the search never saw, the strategy earns money there, and the gain is more than the noise of those days and more than the best of that many lucky tries would show (below). |
| ! | Better on the untouched days, but not by enough to rule out luck / Only part of the improvement held up / No real difference / better, but still losing | The gain held, but a gain that size appears by chance often enough; or some of the gain was luck, or there is no real gain, or the strategy is still a poor one. |
| ✗ | Probably overfitted | Better on the days it was tuned on, worse on the days it never saw. Keep your starting settings. |
| = | Keep what you have | Nothing tried beat the starting settings. That is a result. |

Under the headline, the reasons: the measure on the tuning window and on the untouched days before and after, how the same days look for simply holding every ticker in equal amounts, the chance that the winner's Sharpe ratio is more than luck after so many tries, which settings ended at the edge of their range (the real best may lie beyond it: widen the range and tune again), and what the checks below found.

**What a ✓ asks for.** A gain on the untouched days is one stretch of days, and settings chosen from many tries will often show one by luck: with no setting having any edge, the hold-out gain alone said "good" for about one search in five (see [how the verdict was calibrated](#how-the-verdict-was-calibrated)). So a ✓ also needs three things. (1) The gain is more than the noise of those days: a *paired* test of the tuned strategy's daily returns against the starting settings' over the same days (the two move together, so their difference is steady) must give a t statistic of at least 1. (2) The winner is more than the best of everything tried would be by luck: the *Romano-Wolf* p-value, which bootstraps the days and allows for how many settings were tried and how alike they are, must be under 30%. (3) Choosing again on the past alone must not have lost: the search's first tries, which were drawn before it learned anything, are used to re-choose the settings four times, each time from only the days before a later stretch of the tuning window, and the choices, used on those stretches, must not have fallen behind the starting settings. When a ✓ would have been given and one of these fails, the headline is *Better on the untouched days, but not by enough to rule out luck* and the reasons say which check and by how much.

**The two windows.** The days the starting settings traded are cut in two: the first 70% (the *tuning window*, where the search chooses) and the last 30% (the *hold-out*). You can keep back 20% or 40% instead, under *4. What counts as better*. At least a year is needed to tune on and a quarter to check on; a shorter history is refused with the numbers, and the fix (an earlier start date in the Backtest tab). Every try is scored on the tuning window only. The hold-out is looked at once, for the starting settings and for the winner, after the search has ended.

**The settings table** lists the starting values, the best found, the range tried and *Mattered*: the share of the differences between tries that goes with that setting, compared with the others you ticked. It is a rough number and needs a few dozen tries. ⚑ marks a best value at the end of its range.

**The charts.** *Growth of 1* shows the tuned and the starting settings with the divide between the two windows. Each setting has a chart of the score of every try against its value (a log axis where the range is logarithmic): a broad flat top is a setting that does not matter much, so the answer is safe; a sharp spike is fragile. *Every try, in order* shows whether the search was working: a drift upward after the first few random tries. *Where the new settings helped* is the return each ticker added on the untouched days with the tuned settings, minus with the starting ones. Every chart has a table view.

**How much to trust it** gives, beside the number of tries:

* *More than luck?*, the deflated Sharpe ratio: the chance that the winner's Sharpe ratio on the tuning window beats what the best of this many tries would show by luck alone (95% or more is convincing). Tries of settings near each other trade almost the same days, so they count as fewer independent tries (the text says how many); the version that compared the winner with the spread of the tries' own Sharpe ratios counted a real difference between settings as noise, and with a real edge in the search it hid it.
* *Luck alone could do this*, the Romano-Wolf p-value of the winner (lower is better; a ✓ asks for under 30%).
* *Expected on new days*: inside the tuning window, the setting that won one half of the days scored this Sharpe ratio, on average, on the other half. It is a fairer expectation than the winner's own score, which was picked for being high; on the simulations below it was close to what the winner really earns where the winner's own score was well above it.
* *Winner's rank on the other half*, the probability of backtest overfitting: how often the best setting on one half of the tuning window ranks below the middle on the other half. With a few settings that behave alike this is high even when the edge is real, because neighbours swap places, so it is shown and does not decide the verdict.
* *Choosing again and again*: in how many of the four later stretches the re-chosen settings beat the starting settings (check 3 above).

All of these use the tuning window only; the hold-out comparison is a separate, plainer check. You will often see a verdict of ✓ with a deflated Sharpe ratio well under 95%: the settings really are better than the defaults, and the strategy is still not proven.

A real example, one run, to give a sense of the numbers and nothing more: the Kalman trend on the platform's 15 ETFs, 20 tries. The Sharpe ratio went from 0.06 to 0.61 on the tuning window (2008 to early 2021) and from 0.09 to 0.67 on the untouched days (to September 2026), and the verdict was **mixed**: *better on the untouched days, but not by enough to rule out luck*. The gain was large and the checks that decide a ✓ were split: re-choosing on the past alone beat the starting settings in 4 of 4 later stretches (0.27 against 0.03), and the Romano-Wolf p-value of the winner was 3.5%, but a paired test of the two return streams on the untouched days gave t = 0.8, under the 1.0 a ✓ asks for (a gain this big shows up by chance in about one stretch in five). Before the round this run was a ✓. The page also said that holding all 15 ETFs equally scored 0.78 on those days, so tuning made the strategy better, not better than holding the tickers; that level noise and slope noise ended at the edges of their ranges; and that the chance the winner was more than luck was 84% (about five of the 20 tries independent). The strategy's volatility was about 1.3% a year: a high ratio on a very small position. Read the growth and volatility columns beside the ratio.

## What the search does

* **The first try is your starting settings.** Everything is compared with it, and "keep what you have" is a possible answer.
* **Scoring.** The score of a try is the *measure* on the tuning window. By default the window is cut into four equal parts and a try scores the mean of the measure over the parts less half of how much they differ, so a result that rests on one good stretch ranks below one that works throughout (switch it off under *4. What counts as better*). The tables show the plain measure.
* **The measure** is your choice: Sharpe ratio (return for the risk taken, the default), Sortino ratio (only losing days count as risk), Calmar ratio (growth against the worst fall), yearly growth (no allowance for risk: it favours bold settings) or the *cautious* Sharpe ratio (the Sharpe ratio less one standard error of that estimate, so a result that rests on little evidence, a short stretch or fat tails, is marked down).
* **How it proposes tries.** An evenly spread start (a quarter of the tries, between 5 and 12, from a scrambled Sobol sequence, which leaves none of the gaps and clumps that random draws do in so few tries), then a Tree-structured Parzen Estimator: it models where the good and the bad tries sat and proposes values that look like the good ones. It never runs the same combination twice, and when a small space (a few integers and a yes/no) is used up it stops and says so. The same seed gives the same search. Under *Advanced* you can swap the search for a Gaussian process (a smooth surface fitted to the scores, searched by expected improvement: strong with a few numeric settings, slower beyond about a hundred tries) or for random draws (a baseline), and choose which try wins: the highest score (the default) or the middle of the best region, the setting whose neighbours also scored well, so that a lone spike is passed over. The numbers behind those defaults are [below](#how-the-verdict-was-calibrated).
* **Every try is a trial.** Each is a backtest of these tickers, so it is counted in the trial counter in the top bar and in the deflated Sharpe ratio of the Backtest tab, under the same fingerprint a backtest of those settings has: the winner run there afterwards is not counted twice, and the dozens of looks it took to find it are not hidden. The search's own deflated Sharpe ratio counts earlier looks at the same tickers too.
* **A try that fails** (a combination the strategy refuses, a setting that leaves it almost no days) is counted, explained at the end and skipped. If almost every try fails the search stops and says why.
* **Position sizing** is the Backtest tab's automatic choice for the strategy (calibrated forecast, or independent sleeves where the strategy declares them). The allocation, regime, assets-under-management and cost settings of the Backtest tab are not part of the search; neither are settings that are lists or text.

## How the verdict was calibrated

The thresholds of a ✓ were set by simulation, not by feel. `experiments/tune_selection_study.py` builds strategies whose truth is known (no prices, no network) and runs the real `webapp.tuning.tune` on them. A *world* has one to three settings; every setting earns the same daily volatility; settings close together trade nearly the same days (a few independent noise streams, blended by distance), as the settings of a real strategy do; and one place in the settings' space has a real edge, or none does. Every verdict can then be marked right or wrong. 150 searches of 40 tries for each of twelve kinds of world (one or three settings; no edge, or a best place whose true Sharpe ratio is 0.3, 0.6 or 1.3; an edge that stops where the hold-out begins; the starting settings already on the best place):

| Kind of world | Called ✓ by the hold-out gain alone | Called ✓ now | Of the ✓ before, a real improvement (0.15 Sharpe or more) | Now |
|---|---|---|---|---|
| no edge | 21% | 5% | 0% | 0% |
| weak edge | 29% | 8% | 36% | 44% |
| moderate edge | 38% | 16% | 74% | 85% |
| strong edge | 61% | 45% | 93% | 99% |
| edge only before the split | 26% | 8% | 62% | 72% |
| start already best | 14% | 1% | 0% | 0% |

The checks cut the false alarms to a quarter or less and, at the price of some missed improvements (a hold-out of 30% of the days is one stretch, and a moderate edge shows in it only about half the time), make a ✓ more often right. A run with 100 tries a search (100 searches a world) gave 3% and 7% false ✓ for no edge in one and three settings. Three observations went into the design:

* **The probability of backtest overfitting says little about a search whose tries are alike.** It was above 50% in most worlds, strong edges included (a median of 66% to 81% with one setting), because neighbours swap places. It is shown but does not decide the verdict.
* **The deflated Sharpe ratio, as it was computed, hid real edges.** Comparing the winner with the spread of the tries' own Sharpe ratios counts a real difference between settings as noise: with a strong edge it reached 95% in 22% of one-setting searches and 8% of three-setting ones. With the effective number of independent tries and the sampling variance of one Sharpe ratio it reaches 95% in 45% and 40%, and with no edge in 0% and 1%. That is the version used.
* **A search finds higher noise peaks than a fixed list of tries would.** The Romano-Wolf p-value of the winner of a three-setting search was under 5% in 18% of searches with no edge. That is why the verdict also needs a gain on days that no try saw: the two checks fail for different reasons.

The *Expected on new days* tile was close to the truth where the winner's own score was not. The average annual Sharpe ratio that the setting winning one half of the tuning window scored on the other half, against what the winner really earns (known in a world) and the winner's own score on the whole window:

| Kind of world | The winner's own score | Expected on new days | What it really earns |
|---|---|---|---|
| no edge, one setting | 0.42 | 0.03 | 0.00 |
| weak edge, one setting | 0.55 | 0.17 | 0.17 |
| moderate edge, one setting | 0.74 | 0.40 | 0.44 |
| strong edge, one setting | 1.29 | 1.03 | 1.15 |
| no edge, three settings | 0.71 | 0.24 | 0.00 |
| moderate edge, three settings | 0.84 | 0.40 | 0.20 |
| strong edge, three settings | 1.18 | 0.89 | 0.75 |

It is a fair guide with one setting and an optimistic one with three (the half-windows share their settings' luck), and in every row far nearer the truth than the winner's own score.

**Which search.** `experiments/hpo_benchmark.py` runs every method on five noisy test functions whose best is known and reports the *regret* of the pick, measured without the noise (how far below the best the settings it finally chose really are; lower is better). Mean regret after 40 tries, 12 seeds, picking the best score:

| Function | Random | TPE | TPE, even start | TPE, joint, even start | Gaussian process |
|---|---|---|---|---|---|
| Branin (2 numbers) | 0.028 | 0.009 | 0.014 | 0.009 | **0.003** |
| Hartmann-6 (6 numbers) | 1.86 | 0.83 | 0.82 | 0.89 | **0.63** |
| Rosenbrock valley (4 numbers) | 0.162 | 0.053 | 0.051 | 0.047 | **0.038** |
| A mix: a log-scale number, a whole number, a choice, a number | 0.334 | 0.146 | 0.194 | 0.174 | **0.029** |
| A broad hill beside a narrow spike | 0.452 | **0.370** | 0.438 | 0.454 | 0.430 |

A Gaussian process is the best search on four of the five, and by a wide margin where the space mixes kinds of setting; the model-based searches beat random draws throughout. The evenly spread start helped on two of the five at 20 tries and was neutral or worse at 40; modelling the settings jointly was no better than modelling them one at a time. On the simulated strategies of the calibration above (correlated noise, a narrow best place, a noisy score) the searches could not be told apart with 24 searches each: the true Sharpe ratio of the pick at 40 tries in a four-setting world was 0.18 for random draws, 0.28 for TPE, 0.15 for TPE with the even start, 0.10 for the joint version and 0.20 for the Gaussian process, differences that 24 searches cannot separate from luck.

**On real prices** the searches could not be told apart either, and the verdicts show their limits. The same three searches ran on four library strategies over the 15 ETFs (the moving-average crossover, the MACD trend, time-series momentum and the Kalman trend; 40 tries, three seeds each). The search saw only the first 70% of each strategy's days (up to between February and June 2021, cut into its own tuning window and hold-out, as in the app) and each pick was then run on the 5.3 to 5.6 years after, which belong to no search; the starting settings are the yardstick. On the days they saw, the three reached the same score: a Sharpe ratio of 0.897 for TPE, 0.900 for TPE with the even start and 0.908 for the Gaussian process, averaged over the twelve cases (the process ahead of TPE with the even start by 0.008 ± 0.018; ahead on the four-setting Kalman trend, 0.815 against 0.793 and 0.778, and half as slow again, 97 seconds a search against 65). What the picks earned afterwards, minus what the starting settings earned, was +0.04, +0.11 and +0.09, with standard errors of 0.07 to 0.08, and it depended on the strategy far more than on the search: the Kalman trend's starting settings earned 0.01 and every pick 0.25 to 0.46; the MACD trend went from −0.26 to about zero (no pick made money); but the moving-average crossover and time-series momentum, whose starting settings are the textbook ones (a 50-day and a 200-day average, a 252-day lookback; 0.66 and 0.63), lost between 0.08 and 0.23 of Sharpe ratio in most searches. The verdict was ✓ in 58% to 67% of the searches, and a ✓ was followed by a gain over the starting settings in 3 of 8, 5 of 8 and 4 of 7 of them; the Kalman trend, the strategy whose picks did improve, was called *probably overfitted* every time because the picks lost on the last 30% of the days before the origin. Four strategies are four draws, and a Sharpe ratio measured over five and a half years has a standard error of about 0.4, so none of this separates the searches, and the simulations above, whose worlds do not change regime, say nothing about a market that does. What it says is what a ✓ is: a gain on the days kept back, one stretch of one history, not a promise about the next. TPE with the even start stays the default.

Choosing the middle of the best region (`plateau`) was no better than choosing the highest score anywhere and clearly worse on the test functions and the three-setting worlds (the true Sharpe ratio of the pick, strong edge, three settings: 0.43 against 0.73), and the middle of the best tries (`centroid`) matched the highest score: where neighbouring settings share their luck, as a strategy's do, averaging their scores does not average the luck out. The plateau is on offer for a score that jumps about between nearby values; it is not the default.

**How a try is scored.** The default scores a try by the mean of its Sharpe ratio over four equal parts of the tuning window less half their spread, so one lucky stretch counts for less. `experiments/tune_selection_study.py objective` measures what that costs and buys, by the true Sharpe ratio of the pick (an edge of 0.95 at the best place), with steady and with heavy-tailed (Student t, three degrees of freedom) returns:

| How a try is scored | Steady returns, 1 setting | 3 settings | Heavy-tailed returns, 1 setting | 3 settings |
|---|---|---|---|---|
| plain Sharpe ratio | 0.809 | 0.489 | 0.856 | 0.557 |
| block score (the default) | 0.797 | 0.462 | 0.843 | 0.485 |
| cautious Sharpe ratio | 0.809 | 0.491 | 0.855 | 0.544 |
| cautious block score | 0.797 | 0.462 | 0.842 | 0.476 |

100 searches of 40 tries in each cell; the best place earns 0.95. The block score cost between 0.01 and 0.07 of the true Sharpe ratio of the pick against the plain ratio, and differences of that size are within the noise of 100 searches: here it neither helped nor measurably hurt, and the cautious ratio changed nothing. It is kept as the default as a guard against one stretch of luck, which these worlds, with the same edge in every part of the sample, do not contain; switch it off under *4. What counts as better* if you would rather rank by the plain measure.

**Limits.** These are simulated strategies with a stated noise structure; real strategies' neighbouring settings may behave differently, and the thresholds were chosen on searches of 20 to 100 tries over one to three settings. A hold-out is one stretch of days whatever the checks say, and no search or check creates an edge that the strategy does not have.

## Choosing settings and ranges

* A strategy's **suggested settings** are ticked: those with a hand-picked range say so. Up to six are ticked at first; more than that cannot be searched well in a few dozen tries. Settings about how a model is fitted (a minimum history, a refit interval, a seed) can be ticked but are not, and are searched above their default, which is a floor the strategy insists on.
* **Ranges** for a strategy nobody has curated are a third to three times the standard value (ten times either way for a small decimal, never above 1). A rule that needs one setting below another (a fast average shorter than a slow one, an entry level below an exit level) has ranges that do not overlap, so no combination is invalid. If you untick one of a pair and tune the other, a combination can still break the order; the strategy refuses it, and the try is reported as failed.
* **Log scale.** A setting that spans orders of magnitude (the Kalman filter's noise levels) is searched on a logarithmic scale, so 0.001 to 0.01 gets as many tries as 0.1 to 1.
* **Start from and Keep at.** Type a value into *Start from* to compare against it instead of the standard one; a value outside the range widens the range and says so. An unticked setting stays at the value shown and is used in every try.
* **Redundant settings.** The Kalman filter's `clip` and `scale` only ever act through their product, so `clip` starts unticked: tuning both wastes tries.

## What it does not do

* It does not make a strategy work. It finds the best version of the idea within the ranges, and tells you how much of the improvement survived. A strategy with no edge gets a ✗, a ! or a =.
* The hold-out is used once. If you tune again after reading it, you have used it up: treat the second round as a new experiment, and give more weight to the deflated Sharpe ratio, which counts the tries.
* One strategy at a time, with one set of settings for all the tickers. It does not tune a combination of strategies, and it does not give each ticker its own settings (which would multiply the overfitting).
* It uses the tickers, data source and start date of the Backtest tab, so what it finds is about those tickers over that period.

## From a script

```python
from src.webapp import tuning

def run(params):                       # run your strategy with these settings; return net daily returns
    ...
    return tuning.Run(returns, weights)

settings = [tuning.Setting("lookback", "int", 126, 21, 378)]      # name, kind, standard value, low, high
result = tuning.tune(run, settings, objective="sharpe", holdout=0.3, trials=50, seconds=300)
print(result["verdict"]["headline"], result["best"]["params"])
```

`tuning.tune` is independent of how a strategy is run; the dashboard supplies the real pipeline (`webapp/tune_api.py`). `ops.hpo.tune_pipeline` is the lower-level search that this builds on, and `ops.hpo.Study.optimize` takes a `callback` (return `True` to stop) and `unique=True`; its methods are `random`, `grid`, `tpe` (with `quasi_start=True` for the Sobol start and `multivariate=True` to model the settings jointly), `bayes` (a Gaussian process) and `halving`, and `Study.plateau()` and `Study.centroid()` name the best neighbourhood and the middle of the best tries. `tune(..., search=, select=, inspect=)` takes the search and the rule for the winner, and fills `inspect` with the study and every try's returns for an experiment that wants to try other rules on the same tries.

The same tab is available over HTTP, with the dashboard's per-launch token in the `X-Token` header:

| Request | Does |
|---|---|
| `GET /api/tune/plan?strategy=kalman_trend` | The strategy's settings with range, scale, help text and whether each is ticked, and the offered efforts, measures and hold-out shares. |
| `POST /api/tune/run` | Starts a search as a job (`{"job": id}`, polled at `/api/job/<id>`). Body: the Backtest universe (`tickers`, `classes`, `start`, `source`), `strategy`, `tune` (`[{name, low, high, scale}]`), `values` (a setting's starting or fixed value), `allocator`, `objective` (`sharpe`, `sortino`, `calmar`, `cagr`, `cautious`), `effort` (`quick`, `standard`, `thorough`, or `custom` with `trials` and `seconds`), `holdout`, `stable`, `seed`, `search` (`tpe`, `bayes`, `random`) and `select` (`best`, `plateau`). |
| `POST /api/tune/stop` | `{"job": id}`: ends that search after the try it is on, keeping the tries so far. |

The result has the `verdict`, the two `windows`, `start` and `best` (settings, and the measures on both windows), `benchmark`, `settings` (one entry per tuned setting: its tries as points, importance, the range the best quarter sat in, an `edge` flag), `trials`, `history`, `curves`, `tickers`, `selection` (deflated Sharpe ratio and the effective number of tries it used, Romano-Wolf p-value, probability of overfitting, expected Sharpe ratio on the other half), `walk_forward` (the re-choosing check), `holdout_t` (the paired t of the gain), `choice` (which rule picked the winner, and the single best score when another did), `search` (tries, failures, seconds, why it stopped, the method and the rule), and `use`: the strategy with the settings that differ from the standard ones, as the Backtest tab would take them. It is kept with the other background runs, so a restarted app finds it again.

## The Machine learning tab

The Machine learning tab has its own tuning for its models (*Search*, then *Tune one model*): it varies a model's hyperparameters and the position rule on that tab's dataset, scores on the validation segment only, and runs the best on the test and forward segments once at the end. It works the same way: see [the machine-learning platform](ml_platform.md#tuning-a-model-for-beginners).
