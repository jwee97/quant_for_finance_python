"""Build and execute the tutorial notebooks: ``python docs/notebooks/build_notebooks.py``.

The notebooks are generated from this file so their code is reviewed as code, and executed so that the committed copies carry real outputs.
A test re-executes them from scratch (tests/test_notebooks.py).
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

SETUP = """import logging, sys
from pathlib import Path
ROOT = Path.cwd()
while not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from src.utils.config import load_config
from src.framework import Pipeline, load_default_bundle, load_library
logging.getLogger("src.backtest.engine").setLevel(logging.WARNING)
config = load_config(); load_library()
bundle = load_default_bundle(config)
print(f"{len(bundle.assets)} assets, {bundle.index[0].date()} to {bundle.index[-1].date()}")"""

NOTEBOOKS = {}

NOTEBOOKS["01_first_backtest.ipynb"] = [
    new_markdown_cell("# Your first honest backtest\n\nThis notebook runs one strategy through the whole pipeline and reads the verdict the way a sceptical reviewer would. Everything executes on the cached data; nothing is downloaded.\n\n"
                      "**What you will learn:** what a net-of-cost backtest is, why the benchmark comparison matters more than the Sharpe ratio, and what the causality test proves."),
    new_code_cell(SETUP),
    new_markdown_cell("## 1. Run a strategy\n\nDual momentum holds the four assets with the best 12-month return, but only those that beat cash. A strategy here is a *forecast model* named in a specification; the pipeline does the rest."),
    new_code_cell("spec = {'name': 'dual_momentum', 'models': [{'name': 'dual_momentum'}]}\nresult = Pipeline(spec, config, bundle).run()\nm = result.metrics\nprint(f\"net Sharpe {m['sharpe']:+.2f} (gross {m['gross_sharpe']:+.2f}), CAGR {m['cagr']:.1%}, vol {m['ann_vol']:.1%}, max drawdown {m['max_drawdown']:.1%}, turnover {m['ann_turnover']:.1f}x a year\")"),
    new_markdown_cell("## 2. Against the dumb alternatives\n\nA positive Sharpe is not the question. The question is whether the strategy beats *holding everything in equal weight*, which needs no forecasts."),
    new_code_cell("fig, ax = plt.subplots(figsize=(9, 4))\n(1 + result.window).cumprod().plot(ax=ax, label='dual momentum (net)')\nfor name, series in result.benchmarks.items():\n    (1 + series.reindex(result.window.index).fillna(0)).cumprod().plot(ax=ax, label=name, alpha=0.7)\nax.set_yscale('log'); ax.set_ylabel('growth of 1'); ax.legend(); plt.show()\nresult.tables['benchmarks'][['benchmark', 'difference', 'p_value']]"),
    new_markdown_cell("The paired bootstrap resamples both return series on the same dates and asks how often the *difference* in Sharpe ratios would be this large by chance. A large p-value means: no evidence this strategy beats the benchmark."),
    new_markdown_cell("## 3. The red-flag list"),
    new_code_cell("from src.framework.tearsheet import red_flags\npd.DataFrame(red_flags(result))[['rule', 'observed', 'flag']]"),
    new_markdown_cell("## 4. What the causality test proves\n\nEach model's forecasts are recomputed after replacing everything beyond a cutoff date with noise. If any forecast on or before the cutoff changes, the model has read the future."),
    new_code_cell("pd.DataFrame(result.validation['causality']).T"),
    new_markdown_cell("## 5. Your turn\n\nChange `lookback` to 63 and then to 315 and rerun. Notice how much the Sharpe ratio moves. *That spread is what a researcher who reports only the best parameter hides* (see the technique guide on multiple testing)."),
    new_code_cell("rows = []\nfor lookback in (63, 126, 252, 315):\n    r = Pipeline({'name': f'dm{lookback}', 'models': [{'name': 'dual_momentum', 'params': {'lookback': lookback}}], 'evaluation': {'causality': False}}, config, bundle).run(validate=False)\n    rows.append({'lookback': lookback, 'net Sharpe': round(r.metrics['sharpe'], 2)})\npd.DataFrame(rows)"),
]

NOTEBOOKS["02_regimes_and_adaptation.ipynb"] = [
    new_markdown_cell("# Market regimes: detect, then adapt\n\nRegime detection labels each day with the market state the detector believes it is in. This notebook shows the composite detector, what the regimes look like in the data, and the one test that matters: is the *timing* of the labels doing anything a placebo would not?"),
    new_code_cell(SETUP),
    new_code_cell("from src.framework import DETECTORS\nregimes = DETECTORS.create('composite').detect(bundle)\nregimes.share().round(3)"),
    new_markdown_cell("## The probabilities through time\n\nThe detector outputs a probability for each regime on each day, using only the information available that day (*filtered*, not *smoothed*)."),
    new_code_cell("p = regimes.probabilities.dropna().resample('ME').mean()\nfig, ax = plt.subplots(figsize=(10, 3.5))\nax.stackplot(p.index, [p[c] for c in p.columns], labels=list(p.columns), alpha=0.9)\nax.set_ylim(0, 1); ax.legend(ncol=5, fontsize=8, loc='upper left'); ax.set_ylabel('probability'); plt.show()"),
    new_markdown_cell("## Do regimes differ?\n\nThe annualised volatility and return of the equal-weight market proxy in each regime (hard labels). Volatility regimes are separated by construction, so large differences in volatility prove little; the test is whether *returns* differ."),
    new_code_cell("proxy = bundle.returns.mean(axis=1)\nlabels = regimes.hard_labels().reindex(proxy.index)\ntable = proxy.groupby(labels).agg(days='size', ann_return=lambda x: x.mean() * 252, ann_vol=lambda x: x.std() * np.sqrt(252))\ntable.round(3)"),
    new_markdown_cell("## A placebo for regime timing\n\nTake a simple timing rule: hold the equal-weight proxy, but go to cash the next day whenever the combined probability of HighVol and Crisis is above one half. Then shift the regime path by a random number of days and repeat. If the real path is not better than most shifted paths, the *timing* adds nothing."),
    new_code_cell("risky = (regimes.probabilities[['HighVol', 'Crisis']].sum(axis=1).shift(1) > 0.5)\nreal = proxy.where(~risky, 0.0).dropna()\nsharpe = lambda r: np.sqrt(252) * r.mean() / r.std()\nrng = np.random.default_rng(7)\nplacebo = []\nfor _ in range(200):\n    k = int(rng.integers(252, len(risky) - 252))\n    shifted = pd.Series(np.roll(risky.to_numpy(), k), index=risky.index)\n    placebo.append(sharpe(proxy.where(~shifted, 0.0).dropna()))\nprint(f'de-risk rule Sharpe {sharpe(real):.2f}; placebo mean {np.mean(placebo):.2f}; share of placebos at least as good {np.mean(np.array(placebo) >= sharpe(real)):.3f}')"),
    new_markdown_cell("Read the last number as a p-value. The platform's Stage 31 does this for the full adaptive allocator; see the technique guide on regime-adaptive allocation for what it found."),
]

NOTEBOOKS["03_how_many_ideas_did_you_try.ipynb"] = [
    new_markdown_cell("# How many ideas did you try?\n\nA simulation with no finance in it at all. We create strategies with **zero** true skill and see how good the best one looks, then apply the corrections the platform uses.\n\n"
                      "**What you will learn:** why the best of many random strategies looks impressive, what the deflated Sharpe ratio and the Reality Check do about it, and what false-discovery control buys."),
    new_code_cell("import numpy as np, pandas as pd, matplotlib.pyplot as plt, sys\nfrom pathlib import Path\nROOT = Path.cwd()\nwhile not (ROOT / 'pyproject.toml').exists():\n    ROOT = ROOT.parent\nsys.path.insert(0, str(ROOT))\nfrom src.backtest.metrics import deflated_sharpe_ratio, probabilistic_sharpe_ratio\nfrom src.validation.multiple_testing import white_reality_check"),
    new_markdown_cell("## 200 strategies of pure noise\n\nEach is a year-by-year coin toss with the volatility of a typical ETF book. Five years of daily returns."),
    new_code_cell("rng = np.random.default_rng(3)\nn_days, n_strats = 252 * 5, 200\nR = rng.normal(0, 0.01, size=(n_days, n_strats))\nsharpe = np.sqrt(252) * R.mean(axis=0) / R.std(axis=0, ddof=1)\nprint(f'best Sharpe {sharpe.max():.2f}; median {np.median(sharpe):.2f}; share above 0.5: {np.mean(sharpe > 0.5):.1%}')\nplt.hist(sharpe, bins=30); plt.axvline(sharpe.max(), color='red'); plt.xlabel('Sharpe of a zero-skill strategy'); plt.show()"),
    new_markdown_cell("The best of 200 coin-flip strategies has a Sharpe above 1, with *no skill at all*. Reporting it alone is how fake alpha is born."),
    new_markdown_cell("## The deflated Sharpe ratio\n\nIt asks: given that we tried 200 things, how likely is it that the best one beats what the best of 200 random strategies would show?"),
    new_code_cell("best = int(sharpe.argmax())\nr = pd.Series(R[:, best])\nfrom scipy.stats import skew, kurtosis\nalone = probabilistic_sharpe_ratio(sharpe[best], 0.0, n_days, skew(r), kurtosis(r, fisher=False))\ndeflated = deflated_sharpe_ratio(sharpe[best], n_strats, n_days, skew(r), kurtosis(r, fisher=False))\nprint(f'probability the true Sharpe is above zero if you pretend it was the only idea: {alone:.2f}; counting all {n_strats} tries: {deflated:.2f}')"),
    new_markdown_cell("## White's Reality Check\n\nA bootstrap test that the best strategy beats zero, with the search built into the null."),
    new_code_cell("out = white_reality_check(R, n_samples=500, mean_block=21.0, seed=7)\nout"),
    new_markdown_cell("## False discovery control\n\nTest every strategy's mean at 5%: about 10 of the 200 will 'work'. Benjamini-Hochberg keeps the share of false discoveries among the declared winners under control."),
    new_code_cell("from scipy import stats\nt = R.mean(axis=0) / (R.std(axis=0, ddof=1) / np.sqrt(n_days))\np = 2 * stats.norm.sf(np.abs(t))\nprint('naive 5% rejections:', int((p < 0.05).sum()))\nfrom src.validation.forecast_tests import benjamini_hochberg\nprint('after Benjamini-Hochberg at 10%:', int(benjamini_hochberg(p, 0.10).sum()))"),
    new_markdown_cell("## Your turn\n\nPlant a real edge (add a small positive drift to ten of the strategies) and see how many survive the corrections. That is power: Stage 39 of the platform does this properly."),
]


def build() -> list[Path]:
    paths = []
    for name, cells in NOTEBOOKS.items():
        nb = new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
        NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(HERE)}}).execute()
        path = HERE / name
        nbformat.write(nb, path)
        paths.append(path)
    return paths


if __name__ == "__main__":
    for p in build():
        print(p.relative_to(ROOT))
