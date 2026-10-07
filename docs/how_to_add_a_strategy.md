# How to add a strategy

There are three ways to test an idea, from least to most work. All three go through the same pipeline: calibration, combination, allocation, costs, the look-ahead check, benchmarks, the deflated Sharpe ratio and a tear sheet.

| You have | Use | Effort |
|---|---|---|
| A variation of something that exists (a different lookback, a different allocator) | A built-in strategy and its parameters in the dashboard | one click |
| An idea you can say in one line: "buy what rose over the last year but skip last month, among the calm ones" | A **formula** | one line, no code |
| An idea with several steps, its own parameters or extra data | A **Python file** | one small class |

Start the dashboard with `quant serve`. The **Strategy builder** tab has the formula editor and the Python template; the **Backtest** tab runs either.

## Route 1: a built-in strategy with different parameters

Pick it in the Strategy list and change the fields. Every field is a constructor argument of the strategy; the defaults are the ones used in the research reports. Remember that each variant you try counts as a trial: the deflated Sharpe ratio in the results is penalised for it.

## Route 2: write a formula

A formula gives every asset a score each day. Higher means you expect a higher return. Type it in **Strategy builder**, press *Check on my tickers* to see today's scores, then *Backtest this formula*.

```
mom(252, 21)                                  # 12-1 momentum: last year's return, skipping the last month
-ret(5)                                       # short-term reversal: fade last week's move
rank(mom(126, 21)) - rank(vol(63))            # momentum among the calm
where(close > sma(200), 1, -1)                # trend filter: long above the 200-day average, short below
close / hi(252) - 1                           # distance from the 52-week high
-tanh(zscore(20) / 2)                         # fade a stretch of more than a couple of standard deviations
```

The vocabulary is fixed, and every function looks backwards only, so a formula cannot peek at the future:

| Function | Meaning |
|---|---|
| `close` | adjusted closing prices |
| `ret(n)`, `mom(n, skip)` | return over n days; return from n days ago to `skip` days ago |
| `sma(n)`, `ema(n)`, `hi(n)`, `lo(n)` | moving averages, highest and lowest price of the last n days |
| `vol(n)`, `zscore(n)`, `rsi(n)` | volatility, stretch from the n-day mean, relative strength index |
| `lag(x, n)`, `rollmean(x, n)`, `rollstd(x, n)`, `rollmax(x, n)`, `rollmin(x, n)`, `ts_z(x, n)` | the same ideas applied to any expression x |
| `rank(x)`, `zs(x)`, `demean(x)` | compare assets with each other on each day |
| `sign`, `abs`, `log`, `tanh`, `clip(x, lo, hi)`, `where(cond, a, b)` | shaping and conditions; combine conditions with `&`, `|`, `~` |
| `macro("VIX")` | a macro series as it was known on each date |

Two modes: **rank assets against each other** (cross-sectional: long the high scores, short the low ones) and **each asset on its own** (time series: the sign and size of the score decide the position).

A formula is *parsed*, not executed. Names, attributes, indexing, lambdas and anything outside the table are rejected with a message that says why. The same formula works from the command line:

```bash
quant backtest --model expression --param "expr=rank(mom(126, 21)) - rank(vol(63))" --allocator score_stack --tearsheet
```

**Trading the score as written.** By default the framework calibrates a forecast: it learns from matured history how much a unit of your score has paid, and holds nothing if it has not paid. That is the honest default for a library strategy, but surprising for your own formula (a fade like `-ret(5)` would get no position if fading has not worked). In the dashboard, formulas use *Trade the signal as written* (`score_stack`): the score is ranked and scaled exactly as you wrote it. Switch Allocation back to *Automatic* to see the calibrated version.

## Route 3: write a Python file

```bash
quant new-strategy high_proximity
```

writes `user_strategies/high_proximity.py`. Every `.py` file in that folder is imported when the library loads (files starting with `_` are skipped), so the strategy shows up in `quant list models`, in `quant backtest --model`, and, after pressing *Reload my strategies*, in the dashboard under "My strategies" with its parameters as form fields.

### The example: proximity to the 52-week high

George and Hwang (2004) found that assets close to their 52-week high keep outperforming. The score is the ratio of price to its trailing 252-day maximum.

```python
# user_strategies/high_proximity.py
from src.framework.forecasting import ForecastModel
from src.framework.registry import register_model


@register_model("high_proximity", "momentum", "Proximity to the 52-week high: favour assets closest to their trailing 252-day maximum")
class HighProximity(ForecastModel):
    """Assets near their highs have no overhead supply of disappointed holders (anchoring)."""

    name, family = "high_proximity", "momentum"

    def __init__(self, window: int = 252):
        self.window = window

    def score(self, data):
        high = data.prices.rolling(self.window, min_periods=self.window // 2).max()
        return (data.prices / high - 1.0).where(data.investable)
```

Four rules the framework relies on:

1. **`score` returns a table of dates by assets**, using only information up to each date. The look-ahead check replaces everything after a cutoff with noise and fails you if earlier scores change.
2. **Mask with `data.investable`**, so assets that did not exist yet get no position.
3. **Higher score means "expect a higher return"**. Negate a mean-reversion score.
4. **State what it bets on** in the registration description (more than 30 characters) and in the docstring: it becomes the strategy's explanation.

What `data` gives you: `data.prices` (adjusted closes), `data.returns`, `data.high`, `data.low`, `data.volume` (when available), `data.macro` (macro series known on each date), `data.assets`, `data.asset_class`.

### Run it

```bash
quant backtest --model high_proximity --tearsheet --group my_ideas
```

or in Python:

```python
from src.framework import Pipeline, load_default_bundle, load_library
from src.utils.config import load_config

config = load_config(); load_library()            # imports user_strategies/ too
bundle = load_default_bundle(config)
result = Pipeline({"name": "high_proximity", "models": [{"name": "high_proximity"}]}, config, bundle).run()
print(result.metrics["sharpe"], result.validation["deflated_sharpe_probability"])
```

User files run as ordinary Python with your permissions: only put code there that you wrote or trust. The dashboard never accepts code from the browser; that is why formulas are parsed and Python strategies are files.

### Structured trades

If the strategy *is* a set of weights, as in a duration-neutral curve steepener, implement `weights(data)` instead of `score` and set `structured = True`; see `src/strategies/fixed_income.py`.

## Your own tickers

In the dashboard, add any Yahoo Finance symbol to the Tickers list: `NVDA`, `BTC-USD`, `^GSPC`, `EURUSD=X`. New tickers are downloaded as split- and dividend-adjusted prices and cached under `data/user/prices` for a day. Give each new ticker an asset class if you use macro strategies that tilt by class. From the command line, `quant backtest --prices my_prices.csv ...` takes a wide CSV of prices.

## What you get for free

- `result.validation["causality"]`: proof the score does not read the future.
- Paired bootstrap comparisons against equal weight and risk parity, and a list of red flags with their thresholds.
- The deflated Sharpe ratio, with `n_trials` = the number of distinct ideas you tried on the same tickers. Try ten variants and the bar rises each time.

## What you must still do yourself

- Think about cost: a score that flips sign every few days will be eaten by turnover.
- Think about what else it is: a momentum score is mostly an equity-beta timing rule on this universe. Run the factor attribution (`experiments/stage40_factors.py` shows how).
- Say how many variants you tried.

## When something looks wrong

| You see | Why |
|---|---|
| "This strategy held (almost) no positions" | The calibrated forecast holds nothing when a signal has not paid in the past. Use *Trade the signal as written*, or check the window is shorter than the history. |
| A flat line at the start | The book waits for enough investable assets or history. The page trims the wait; check the first date in the header. |
| "needs macro series" | The strategy reads a macro series the bundle does not have. The platform's macro panel is attached to every universe; a series outside it is not available. |
| The look-ahead check fails | Your score uses a future value: a negative shift, `center=True` in a rolling window, or a statistic computed over the whole sample. |
| Benchmarks missing | The portfolio builders need at least five assets. |
