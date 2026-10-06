---
title: "Crypto carry: funding, basis and stablecoin flows"
slug: crypto-carry
difficulty: 3
chapter: Ch. 22
prerequisites: [backtest-engine-and-costs]
stages: [32]
files: [src/data/crypto.py, src/strategies/crypto.py, src/framework/crypto_bundle.py, experiments/stage32_crypto.py]
figures: [65]
tests: [tests/test_crypto.py]
models: [funding_carry, basis_reversion, stablecoin_flow]
---

# Crypto carry: funding, basis and stablecoin flows

## In one sentence

Perpetual futures have no expiry, so they pay a funding rate to stay near spot; holding spot and shorting the perpetual collects it with little price risk.

## The idea

A perpetual swap trades near the spot price because longs pay shorts (or the reverse) a periodic funding rate. A cash-and-carry position (long spot, short perpetual) earns funding plus any basis change and
is nearly market neutral. Timing rules switch the carry on or off from trailing funding or from stablecoin supply growth.

## Why it matters

It is an example of a risk premium that is paid for providing a service, and also of how venue risk, margin and linearisation assumptions hide in an innocent-looking backtest.

## How this repo uses it

Data come from Deribit (funding, perpetual and index daily series at the 08:00 UTC settlement) and DefiLlama (stablecoin supply). Binance and Bybit were unreachable from the build environment, so the branch is one venue and four assets.
The carry asset's return uses a linear approximation for the inverse perpetual.

## What we found

None of the three timing rules beat the unconditioned alternatives after correction (EXP-083).

## Going deeper

```
carry position: long 1 unit of spot, short 1 unit of the perpetual
daily P&L ~ funding rate paid to the short (when positive) + (spot return - perpetual return) ~ funding + basis change
timing rules: on if trailing funding > 0 | on if perpetual premium to index has reverted | on if stablecoin supply growth > 0
```
The perpetual here is inverse (margined in coin), so the stage uses a linear approximation, disclosed in the stage notes.

## Pitfalls

- One exchange is one counterparty.
- Funding regimes change; the sample is a few years.
- Cross-exchange spreads and executable basis are not modelled.

## Try it

```bash
python -m experiments.stage32_crypto
quant backtest --model funding_carry
```
