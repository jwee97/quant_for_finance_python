# How to add a strategy

A strategy is a forecast model. You write one class with one method, register it, and the pipeline does the rest: calibration, combination, allocation, costs, the causality test, benchmarks, the deflated Sharpe ratio and a tear sheet.

## The example: proximity to the 52-week high

George and Hwang (2004) found that assets close to their 52-week high keep outperforming. The score is the ratio of price to its trailing 252-day maximum.

```python
# my_strategies.py
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

1. **`score` returns a table of dates by assets**, using only information up to each date. The causality test will catch you if it does not.
2. **Mask with `data.investable`**, so assets that did not exist yet get no position.
3. **Higher score means "expect a higher return"**. Negate a mean-reversion score.
4. **State what it bets on** in the registration description: it becomes the strategy card and the explanation.

## Run it

```bash
python - <<'PY'
import my_strategies                               # registers the model
from src.cli import main
main(["backtest", "--model", "high_proximity", "--tearsheet", "--group", "my_ideas"])
PY
```

Or put a `Pipeline` in a script:

```python
from src.framework import Pipeline, load_default_bundle, load_library
from src.utils.config import load_config

config = load_config(); load_library()
import my_strategies
bundle = load_default_bundle(config)
result = Pipeline({"name": "high_proximity", "models": [{"name": "high_proximity"}]}, config, bundle).run()
print(result.metrics["sharpe"], result.validation["deflated_sharpe_probability"])
```

## What you get for free

- `result.validation["causality"]`: proof the score does not read the future.
- `result.tables["benchmarks"]`: paired bootstrap comparisons against equal weight and risk parity.
- `result.validation["deflated_sharpe_probability"]` with `n_trials` = the number of distinct specifications in your group. Run ten variants under `--group my_ideas` and the bar rises each time.

## What you must still do yourself

- Think about cost: a score that flips sign every few days will be eaten by turnover.
- Think about what else it is: a momentum score is mostly an equity-beta timing rule on this universe. Run the factor attribution (`experiments/stage40_factors.py` shows how).
- Say how many variants you tried.

## Structured trades

If the strategy *is* a set of weights, as in a duration-neutral curve steepener, implement `weights(data)` instead of `score` and set `structured = True`; see `src/strategies/fixed_income.py`.
