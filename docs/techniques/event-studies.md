---
title: "Event studies: abnormal returns around a date"
slug: event-studies
difficulty: 2
chapter: Ch. 12
prerequisites: [regression-and-panel-statistics]
stages: []
files: [src/stats/event_study.py]
figures: []
tests: [tests/test_stats.py]
models: []
---

# Event studies: abnormal returns around a date

## In one sentence

An event study measures how an asset's return differs from what a model expected on and around a date (an earnings release, a rate decision, a market crash), and tests whether the average difference is more than noise.

## The idea

For each event `i` pick an **estimation window** well before the event, fit a model of normal returns, and apply it in the **event window** `[-k, +k]` around the date. The **abnormal return** is `AR_it = R_it - E[R_it]`. The model is the market model `R = a + b R_m` (the default), a constant mean, or a market-adjusted return. Summing over the window gives the cumulative abnormal return `CAR_i`, and averaging over events gives `CAAR`. `test_car(a, b)` reports the ordinary cross-sectional t-statistic, the Patell z, the Boehmer-Musumeci-Poulsen (BMP) standardised statistic, which stays valid when event-date volatility rises (as it does on almost every event day), and a sign test. `cluster_by_date=True` collapses events that share a date, because simultaneous events are not independent (a crash day is one observation, not fifty).

`buy_and_hold_abnormal_return` measures longer horizons by compounding, where cumulating arithmetic abnormal returns would be biased.

## Why it matters

Event studies are the standard way to ask whether anything changed at a date: an announcement, a policy shift, or "what do bonds do when equities fall 3%". Bad versions leak the future, by estimating the model on a window that overlaps the event, or by ignoring that events cluster in time.

## How this repo uses it

`event_study` takes a returns frame, an events frame (asset, date) and an optional market series, checks that the estimation window ends before the event window starts, drops events with too little history and reports them. The institutional findings script runs one on the platform's ETFs: what the other 14 do when SPY falls more than 3% in a day. Event-based signals for the strategy library can reuse the same event definition.

## What we found

The tests plant a known jump in a simulation and recover it, confirm that the estimation window never overlaps the event window, and confirm that changing data after the event does not change the model estimated before it. The ETF result in [the findings](../institutional_findings.md) is an exploratory observation, with the clustering caveat above.

## Pitfalls

- Overlapping estimation windows (events within a few weeks of each other) share data and understate variance; use `cluster_by_date`.
- Long-horizon abnormal returns have well-known size and power problems; prefer short windows.
- The market model assumes the event does not itself move the market, which fails for market-wide events like crashes (use a different benchmark there).
- Defining events by a return threshold selects on the outcome of interest.

## Try it

```python
import numpy as np, pandas as pd
from src.stats.event_study import event_study

rng = np.random.default_rng(1)
idx = pd.bdate_range("2020-01-01", periods=600)
mkt = pd.Series(rng.normal(0, 0.01, 600), index=idx, name="mkt")
rets = pd.DataFrame({k: 0.9 * mkt + rng.normal(0, 0.01, 600) for k in "ABCDE"}, index=idx)
events = pd.DataFrame({"asset": list("ABCDE"), "date": idx[[300, 320, 340, 360, 380]]})
for a, d in zip(events["asset"], events["date"]):
    rets.loc[d, a] += 0.03
res = event_study(rets, events, market=mkt)
print(res.summary[["AAR", "CAAR"]].loc[-1:2].round(4).to_dict("list"))
print({k: round(v, 3) for k, v in res.test_car(0, 0).items() if k in ("caar", "t_bmp", "p_bmp")})
```
