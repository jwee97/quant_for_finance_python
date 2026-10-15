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
| ✓ | The improvement held up on the untouched days | At least half of the gain on the tuning window was still there on the days the search never saw, and the strategy earns money there. |
| ! | Only part of the improvement held up / No real difference / better, but still losing | Some of the gain was luck, or there is no real gain, or the strategy is still a poor one. |
| ✗ | Probably overfitted | Better on the days it was tuned on, worse on the days it never saw. Keep your starting settings. |
| = | Keep what you have | Nothing tried beat the starting settings. That is a result. |

Under the headline, the reasons: the measure on the tuning window and on the untouched days before and after, how the same days look for simply holding every ticker in equal amounts, the chance that the winner's Sharpe ratio is more than luck after so many tries, and which settings ended at the edge of their range (the real best may lie beyond it: widen the range and tune again).

**The two windows.** The days the starting settings traded are cut in two: the first 70% (the *tuning window*, where the search chooses) and the last 30% (the *hold-out*). You can keep back 20% or 40% instead, under *4. What counts as better*. At least a year is needed to tune on and a quarter to check on; a shorter history is refused with the numbers, and the fix (an earlier start date in the Backtest tab). Every try is scored on the tuning window only. The hold-out is looked at once, for the starting settings and for the winner, after the search has ended.

**The settings table** lists the starting values, the best found, the range tried and *Mattered*: the share of the differences between tries that goes with that setting, compared with the others you ticked. It is a rough number and needs a few dozen tries. ⚑ marks a best value at the end of its range.

**The charts.** *Growth of 1* shows the tuned and the starting settings with the divide between the two windows. Each setting has a chart of the score of every try against its value (a log axis where the range is logarithmic): a broad flat top is a setting that does not matter much, so the answer is safe; a sharp spike is fragile. *Every try, in order* shows whether the search was working: a drift upward after the first few random tries. *Where the new settings helped* is the return each ticker added on the untouched days with the tuned settings, minus with the starting ones. Every chart has a table view.

**How much to trust it** gives the number of tries, the chance the winner is more than luck (the deflated Sharpe ratio: 95% or more is convincing), and the probability of backtest overfitting (how often the best setting on one half of the tuning window ranks below the middle on the other half: under 50% is better). Both use the tuning window only; the hold-out comparison is a separate, plainer check. You will often see a verdict of ✓ with a deflated Sharpe ratio well under 95%: the settings really are better than the defaults, and the strategy is still not proven.

A real example, one run, to give a sense of the numbers and nothing more: the Kalman trend on the platform's 15 ETFs, 20 tries. The Sharpe ratio went from 0.06 to 0.57 on the tuning window (2008 to early 2021) and from 0.09 to 0.66 on the untouched days (to September 2026), a ✓. The page also said that holding all 15 ETFs equally scored 0.78 on those days, so tuning made the strategy better, not better than holding the tickers; that level noise and slope noise ended at the edges of their ranges; and that the chance the winner was more than luck was 71%. The strategy's volatility was about 1.2% a year: a high ratio on a very small position. Read the growth and volatility columns beside the ratio.

## What the search does

* **The first try is your starting settings.** Everything is compared with it, and "keep what you have" is a possible answer.
* **Scoring.** The score of a try is the *measure* on the tuning window. By default the window is cut into four equal parts and a try scores the mean of the measure over the parts less half of how much they differ, so a result that rests on one good stretch ranks below one that works throughout (switch it off under *4. What counts as better*). The tables show the plain measure.
* **The measure** is your choice: Sharpe ratio (return for the risk taken, the default), Sortino ratio (only losing days count as risk), Calmar ratio (growth against the worst fall) or yearly growth (no allowance for risk: it favours bold settings).
* **How it proposes tries.** A random start (a quarter of the tries, between 5 and 12), then a Tree-structured Parzen Estimator: it models where the good and the bad tries sat and proposes values that look like the good ones. It never runs the same combination twice, and when a small space (a few integers and a yes/no) is used up it stops and says so. The same seed gives the same search.
* **Every try is a trial.** Each is a backtest of these tickers, so it is counted in the trial counter in the top bar and in the deflated Sharpe ratio of the Backtest tab, under the same fingerprint a backtest of those settings has: the winner run there afterwards is not counted twice, and the dozens of looks it took to find it are not hidden. The search's own deflated Sharpe ratio counts earlier looks at the same tickers too.
* **A try that fails** (a combination the strategy refuses, a setting that leaves it almost no days) is counted, explained at the end and skipped. If almost every try fails the search stops and says why.
* **Position sizing** is the Backtest tab's automatic choice for the strategy (calibrated forecast, or independent sleeves where the strategy declares them). The allocation, regime, assets-under-management and cost settings of the Backtest tab are not part of the search; neither are settings that are lists or text.

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

`tuning.tune` is independent of how a strategy is run; the dashboard supplies the real pipeline (`webapp/tune_api.py`). `ops.hpo.tune_pipeline` is the lower-level search that this builds on, and `ops.hpo.Study.optimize` takes a `callback` (return `True` to stop) and `unique=True`.

The same tab is available over HTTP, with the dashboard's per-launch token in the `X-Token` header:

| Request | Does |
|---|---|
| `GET /api/tune/plan?strategy=kalman_trend` | The strategy's settings with range, scale, help text and whether each is ticked, and the offered efforts, measures and hold-out shares. |
| `POST /api/tune/run` | Starts a search as a job (`{"job": id}`, polled at `/api/job/<id>`). Body: the Backtest universe (`tickers`, `classes`, `start`, `source`), `strategy`, `tune` (`[{name, low, high, scale}]`), `values` (a setting's starting or fixed value), `allocator`, `objective`, `effort` (`quick`, `standard`, `thorough`, or `custom` with `trials` and `seconds`), `holdout`, `stable`, `seed`. |
| `POST /api/tune/stop` | `{"job": id}`: ends that search after the try it is on, keeping the tries so far. |

The result has the `verdict`, the two `windows`, `start` and `best` (settings, and the measures on both windows), `benchmark`, `settings` (one entry per tuned setting: its tries as points, importance, the range the best quarter sat in, an `edge` flag), `trials`, `history`, `curves`, `tickers`, `selection` (deflated Sharpe ratio, probability of overfitting), `search` (tries, failures, seconds, why it stopped), and `use`: the strategy with the settings that differ from the standard ones, as the Backtest tab would take them. It is kept with the other background runs, so a restarted app finds it again.

## The Machine learning tab

The Machine learning tab has its own tuning for its models (*Search*, then *Tune one model*): it varies a model's hyperparameters and the position rule on that tab's dataset, scores on the validation segment only, and runs the best on the test and forward segments once at the end. It works the same way: see [the machine-learning platform](ml_platform.md#tuning-a-model-for-beginners).
