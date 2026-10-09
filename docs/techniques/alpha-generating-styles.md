---
title: "Alpha-generating styles: long-term, short-term, news, outlook and corporate actions"
slug: alpha-generating-styles
difficulty: 2
chapter: Ch. 20
prerequisites: [plugin-framework, event-studies, multiple-testing]
stages: []
files: [src/strategies/alpha_styles.py]
figures: []
tests: [tests/test_alpha_styles.py]
models: [jensen_alpha, adaptive_autocorrelation, squeeze_breakout, abnormal_volume_drift, event_study_drift, news_sentiment, panel_signal]
---

# Alpha-generating styles: long-term, short-term, news, outlook and corporate actions

## In one sentence

Seven strategies that try to earn a return no market exposure explains, over horizons from days to years, from prices and volume alone or from dated information you supply (headlines, analyst scores, corporate-action calendars), with the discipline that nothing is used before it was knowable.

## The idea

**Long term.** `jensen_alpha` ranks assets on the appraisal ratio of the last three years: the intercept of a regression on the market (Jensen's alpha) divided by the volatility of what the market does not explain, skipping the last month. Dividing by residual risk demotes a lucky streak in a noisy asset.

**Short term.** `adaptive_autocorrelation` continues yesterday's move where an asset's own first-order autocorrelation over the last year is significantly positive and fades it where it is significantly negative; with no significant autocorrelation it takes almost no position. `squeeze_breakout` waits for Bollinger bandwidth to be in the lowest fifth of its half-year range and then goes with the first close outside the band, holding until the close crosses the middle band.

**Company news.** Real news needs a feed. `abnormal_volume_drift` is the proxy made from prices: a move of more than two standard deviations on at least twice the usual volume continues for ten days (under-reaction), while the same move on ordinary volume is partly faded. `news_sentiment` reads headlines from a file you supply, scores each with a small finance word list (or takes your own score column), adds the scores per ticker with a half-life, and uses a headline from the trading day after its date.

**Company outlook.** `panel_signal` turns any dated score by company (analyst revisions, earnings surprises, guidance flags) into a cross-sectional signal with a publication lag and an expiry, optionally as a change, and standardised across companies.

**Corporate actions, index changes, announcements.** `event_study_drift` is a walk-forward event study. Every event has a date, an asset and a type. The strategy measures, for each type, the average cumulative abnormal return (the asset's return minus the market's) over the next days, using only events whose whole window has finished by the decision date, and holds the expected remaining drift of the events currently in their window when, and only when, that average is at least `tstat` standard errors from zero on at least `min_events` events. It learns whether a type continues or reverts and stays flat on noise. Without a file the events are price shocks split by direction and by heavy or ordinary volume; with `path=events.csv` (a file in `data/user/` with `date, ticker, type` and optionally `size`) they are yours: splits, buybacks, upgrades, index additions and deletions.

## Why it matters

Alpha strategies are where a backtest most easily lies. A signal built on a headline that arrived after the close, an analyst score that was revised later, or an index change announced after it was "effective" trades on the future. Each model here states the earliest day it may act on a piece of information and a test checks it: moving data after a cutoff must not change any earlier score.

## How this repo uses it

The seven models live in `src/strategies/alpha_styles.py` and run like any other strategy in the dashboard and the CLI. The ones that read a file say so and refuse to run without it (`news_sentiment needs the file data/user/headlines.csv ...`) instead of inventing a signal; files go in `data/user/`, which git ignores. Dates in a file are the first day the information could have been traded on. `event_study_drift` and `news_sentiment` use the trading day on or after the date, `news_sentiment` and `panel_signal` add a further `lag`, and a missing or malformed file is explained in the error. The tests plant the effect (a drift after one event type, a persistent alpha, autocorrelation of a chosen sign, a decaying headline score) and check that it is found and that a null type is left alone.

## What we found

The 15-ETF survey ([part two](../strategy_survey_2.md): 92 strategies counted as trials, 10 bps costs) is the wrong place for most of these, because they are written for single stocks and company information. On ETFs: `jensen_alpha` +0.13 net Sharpe against +0.84 for equal weight over the same dates (it needs a wide cross-section of assets with different alphas); `adaptive_autocorrelation` earns +0.09 gross but trades 116 times a year and loses 0.84 net, a clean illustration of a real but tiny edge eaten by costs; `squeeze_breakout` (-0.41) and `abnormal_volume_drift` (-0.55) do not pay on liquid funds; `event_study_drift` stays almost flat (0.3% volatility) because no price-shock type has a significant drift, which is what it should do when there is nothing to find. `news_sentiment` and `panel_signal` cannot run without a file. None of this says the ideas fail on the assets they were written for; it says what to expect from them on index funds, and the dashboard lets you run them on thirty or more stocks.

## Going deeper

```
jensen_alpha:             score = shift_21( alpha_hat / sigma_resid ),   r_i = alpha + beta r_m + e   over 756 days
adaptive_autocorrelation: score = tanh( sqrt(250) * rho_hat / 2 ) * clip( r_t / vol_t, -3, 3 ),   rho_hat = corr(r_t, r_{t-1}) over 250 days
event_study_drift:        score_t = tanh( sum over live events of  E[AR from t+1 to the end of the window] / ( vol * sqrt(days left) ) )
                          traded only for types with  n >= min_events  and  |mean CAR / (sd CAR / sqrt(n))| >= tstat, learned from finished events
news_sentiment:           level_t = sum of scores * 0.5^(age / half_life);  score = tanh(level / scale);  a headline dated D acts from D + lag trading days
```

## Pitfalls

- A file of events is only as good as its dates. If the "announcement date" is really the effective date, the strategy trades on the future.
- Headlines scored by a short word list are crude (Tetlock 2007 and Loughran and McDonald 2011 show why finance needs its own word lists, and that tone is a weak signal). The plumbing is the point; replace the scorer or supply a `score` column.
- A walk-forward event study with many event types is a multiple-testing machine: with forty types, two will look significant by chance. Count every type you try as a trial.
- Fast signals and liquid funds are a poor match at 10 bps of cost. Check turnover before the Sharpe ratio.

## Try it

```bash
quant backtest --model jensen_alpha --tearsheet
quant backtest --model squeeze_breakout --allocator sleeves
# with a file of events in data/user/events.csv (date,ticker,type[,size]); headlines.csv and signals.csv work the same way:
quant backtest --model event_study_drift --param path=events.csv --allocator sleeves
quant backtest --model news_sentiment --allocator sleeves
```
