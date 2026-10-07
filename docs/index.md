# Systematic multi-asset research platform

A research platform for systematic investing across 15 ETFs, built to follow *Quantitative Finance with Case Studies in Python*, and a teaching tool whose reports mostly say *this did not work, and here is how we know*.

- **New here?** Read [Start here](START_HERE.md), or run `quant demo` (thirteen seconds on the cached data).
- **Want to understand a technique?** The [technique guides](techniques/index.md) explain 73 of them, from PCA to diffusion models and from HAC errors to option surfaces, with formulas, what this repository found, and how to run each one. `quant explain <term>` prints the same from the terminal.
- **Want to run a strategy?** Ninety are registered; see the [strategy cards](strategies/index.md) and [how to add your own](how_to_add_a_strategy.md).
- **Want the platform's scope?** The [capability matrix](capability_matrix.md) says what is complete and what is not, the [architecture](architecture.md) shows how it fits together and the [research survey](research_survey.md) why.
- **Want the results?** The [findings digest](generated/findings.md) lists every hypothesis the project declared with its decision, and [what was built](roadmap_coverage.md) says honestly what was not.
- **Want to explore?** `quant dashboard` writes a self-contained page; the notebooks in `docs/notebooks` run the main ideas live.

Browse offline with `mkdocs serve` (install with `pip install -e ".[docs]"`).
