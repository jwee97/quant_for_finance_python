# Data

## What is committed, and why

`data/raw/` **is committed.** It is the immutable record the entire project is
built on, it is only 8.4 MB, and committing it means the research is
reproducible offline and stays reproducible even when the provider's API,
adjustment methodology or history changes.

`data/processed/` and `data/features/` are **not** committed. They are rebuilt
from raw by `python -m experiments.stage01_data`, deterministically.

## Immutability

The downloader writes each raw file once and refuses to overwrite it without
an explicit `--force`. Every stage rebuilds from raw rather than editing in
place, so there is exactly one copy of the ground truth.

## Provenance

`data/metadata/manifest.json` records, per series: provider, requested window,
actual window, download timestamp, observation count, column list and a
sha256 of the file. The hash of those hashes is the dataset's `data_version`,
which every stage logs and which is stamped into every experiment-registry
entry.

## A caveat specific to adjusted prices

Yahoo computes adjusted prices **retroactively**: the adjusted history for any
past date changes when a future distribution occurs. The adjusted series in
this repository is therefore a snapshot as of its download date, not a
timeless fact. This is why the download date is part of the data version, and
why re-downloading produces a different `data_version` even when no new
trading days have elapsed.

## Layout

```
raw/         one immutable CSV per ticker: date, ticker, OHLC, adj_close, volume
processed/   cleaned wide panels (date x ticker), rebuilt
metadata/    manifest.json — provenance and the data version
features/    cached feature panels, rebuilt
```
