"""A machine-learning research platform that never trains a model in isolation.

A model here is one step of a pipeline: a leak-free dataset (``datasets``, ``features``, ``targets``) is split along time (``splitters``), a model (``models``) is fitted walk-forward (``training``), its
predictions are scored (``evaluation``), turned into positions (``signals``), run through the platform's own backtest engine with costs (``backtest``), stressed (``robustness``), explained
(``explainability``), recorded (``registry``, ``experiments``) and, once chosen, followed forward in time (``monitoring``). Searches over models and settings (``optimization``, ``strategy_search``) count
every trial and deflate the winner, and ``ensembles`` combines strategies. ``visualization`` shapes everything for the dashboard.

The entry point is ``experiments.run_experiment``; the command line is ``python -m src.cli ml-build-dataset | ml-train | ml-backtest | ml-report``.
"""
