# Systematic multi-asset research platform

A research platform for systematic investing across 15 ETFs, built to follow *Quantitative Finance with Case Studies in Python*, and a teaching tool whose reports mostly say *this did not work, and here is how we know*.

- **New here?** Read [Start here](START_HERE.md), or run `quant demo` (thirteen seconds on the cached data).
- **Want to understand a technique?** The [technique guides](techniques/index.md) explain 123 of them, from PCA to diffusion models and from HAC errors to option surfaces, with formulas, what this repository found, and how to run each one. `quant explain <term>` prints the same from the terminal.
- **Want to forecast a price?** The Forecast tab (or `quant forecast SPY`) forecasts a stock as a range of futures with Monte Carlo, Markov chains, Gaussian processes, a Kalman filter and more, then tests every method on the past: [Forecasting a stock's price, and how far to trust it](forecasting.md).
- **Want to study company accounts, or search them for alphas?** The Fundamentals tab (or `quant fundamentals`) keeps company statements as filed in a local database, tests factors against what happened next, searches formulas while counting every one it tried, and keeps the failures: [Company fundamentals: studying factors, searching for alphas, and counting honestly](fundamentals.md). It runs on a simulated demo world with no account; real data needs an OpenFilings key.
- **Want to price an option under a model, or cost a hedge?** The Options tab's Model fit, Model prices and Hedging cost views (and `quant models`) fit Merton, Kou, variance gamma, NIG and Heston to a chain, price a digital, barrier, average-price or American option under each, and cost the hedging of an option when trades cost money: [Option models](option_models.md).
- **Want to run a strategy?** 118 are registered; see the [strategy cards](strategies/index.md) and [how to add your own](how_to_add_a_strategy.md).
- **Want the platform's scope?** The [capability matrix](capability_matrix.md) says what is complete and what is not, the [architecture](architecture.md) shows how it fits together and the [research survey](research_survey.md) why.
- **Want the results?** The [findings digest](generated/findings.md) lists every hypothesis the project declared with its decision, and [what was built](roadmap_coverage.md) says honestly what was not.
- **Want to know how good the app itself is?** The [review and grades](review_2026_10.md) give each area a grade with the evidence, what was fixed and what still holds it back.
- **Want to explore?** `quant dashboard` writes a self-contained page; the notebooks in `docs/notebooks` run the main ideas live.

Browse offline with `mkdocs serve` (install with `pip install -e ".[docs]"`).
