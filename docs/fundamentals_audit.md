# Fundamentals platform: audit, what was added, and what is still open

This audit was made before any change: the request was a Fundamentals Alpha Lab with a historical data foundation. Most of the research side already existed; the data foundation had real gaps. The rule
followed was to extend working components (the SQLite store, the point-in-time engine, the SEC and OpenFilings providers, Enter reports, the factor engine, the backtest bridge) and not to rebuild them.

## What already existed (reused, not rewritten)

| Layer | Where | What it does |
|---|---|---|
| Store | `src/fundamentals/store.py`, `schema.py` | One SQLite database (`data/user/fundamentals.db`); periods keep `filing_date`, `accepted_date`, `pit_quality` and a `vintage` (a restatement is a new row, never an overwrite); long canonical observations; raw payloads deduplicated by hash; usage and fetch logs |
| Point in time | `pit.py` | A figure is available on its filing date plus a lag, never on the period end; an unknown date is estimated conservatively and flagged; annual and trailing-twelve-month views |
| Sources | `providers/openfilings.py`, `providers/sec_edgar.py`, `providers/simulated.py`, `manual.py` | OpenFilings (annual only, metered), SEC company facts (free, annual), a simulated world, figures typed or pasted in Enter reports |
| Normalisation | `canonical.py` | One vocabulary of 46 lines; SEC element mapping; cash-outflow signs; derived lines marked `derived` |
| Quality | `quality.py` | `identity_error`, `negative_value`, `suspicious_jump`, `provider_disagreement`, `restated`, `derived`, `estimated` flags |
| Coverage planning | `coverage.py`, `service.py` | Which fiscal years are held, missing, empty, due; a plan with its cost before anything is fetched; resumable sync with a budget |
| Factors | `factors.py`, `transforms.py`, `expr.py`, `scores.py` | 104 factors in 13 families (value, quality, profitability, growth, investment, leverage, liquidity, fundamental momentum, earnings quality, efficiency, risk, price momentum, size); a formula language; winsorise, z-score, rank, percentile, sector/industry/country neutralisation |
| Diagnostics | `evaluate.py`, `calibration.py` | Spearman and Pearson rank IC with Newey-West t, rolling IC, IC decay over seven horizons, quantile returns and monotonicity, long-short, turnover, Fama-MacBeth premium with controls, breakdowns by country, sector, region and regime, power analysis, look-ahead audit, placebo calibration |
| Discovery | `discovery.py`, `library.py` | Staged formula search counting every try, locked-date test, deflated Sharpe and false-discovery control, the library and its graveyard, the monitor |
| Models | `combine.py` | Equal, IC, ICIR, decorrelated and clustered combinations; ridge, lasso, elastic net, random forest, extra trees, gradient boosting, LightGBM and XGBoost, fitted walk-forward with an embargo |
| Backtest | `bridge.py` | Long-only and long-short books, top-n or percentile, equal, signal or inverse-volatility weights, sector, industry, country, beta, turnover, liquidity and market-cap limits, costs and borrow, through the platform's own engine |
| Experiments | `workbench.py`, `experiments` table | Saved studies with the data version and code version |
| UI | `static/fundamentals*.js` | Overview, Company explorer, Enter reports, Factor lab, Alpha discovery, Alpha library, Alpha monitor, Data manager |

## The gaps found, and what was done

| Requirement | Finding | Done in this change |
|---|---|---|
| SEC 10-K **and 10-Q** | Only the 10-K facts were read ("quarterly figures are not read by this application yet"); so no trailing-twelve-month series from real filings | `providers/sec_quarters.py`: Q1 to Q3 from the 10-Q facts, year-to-date flows de-cumulated from the earlier filing, Q4 as the year less the nine months (marked derived, public on the 10-K's date), per-share and weighted-share lines never derived; `SecEdgarProvider.get_financials(period="quarter")` plugs into the existing sync, coverage and PIT code |
| Historical filings list | The submissions record was used only for lookups | `SecEdgarProvider.filings()`: 10-K and 10-Q filings with accession, filing date, period, document address and whether it carries XBRL, including the older pages of the submissions record |
| XBRL company disclosures, uploads of PDF, Excel, CSV, XBRL, ZIP, direct URLs, batches | Only typing and pasting a table | `src/fundamentals/importer/`: `xbrl.py` (XBRL and inline XBRL read exactly as tagged, segment contexts skipped, a 10-Q's year-to-date lines left out), `tabular.py` (CSV and Excel through the Enter-reports reader), `pdf.py` (words and positions to rows, page and line kept, always reviewed, scanned PDFs refused), `fetch.py` (public addresses only, retries, rate limit, redirects checked), `pipeline.py` (batches, ZIP with limits, content-hash duplicate detection, resumable processing) |
| Report Import page | Enter reports has a grid but no staging | Fundamentals mode **Report import**: add files, links or SEC filings; see what was read; correct figures, unit, period and filing date; conflicts with held figures need a decision; approve; history of batches |
| Validation and reconciliation | Per-period identities existed in `manual.check_entry` | `importer/reconcile.py`: cash flow against the change in cash, quarters against the year, year-to-date mistaken for a quarter, duplicates, restatement and source conflicts against held figures, units against the company's history, currency |
| Provenance | Observations kept only the source field and provider | `observation_provenance`: file or link, content hash, accession, page, taxonomy tag, the figure as reported before unit scaling, extraction method, whether a person edited it. The first reading of an import is kept apart from the corrected copy |
| Point-in-time integrity | Already enforced in one place | Unchanged. Imports store as the `manual` source with a real filing date; a restatement is a new vintage public from its own date; a missing date is estimated and flagged, never the period end |
| Coverage dashboard | A per-company calendar of fiscal years, annual only | Fundamentals mode **Data coverage**: companies by fiscal year and by quarter, held / derived / missing / not yet due, sources, awaiting review, conflicts, failed imports, assumed dates, a backfill button (SEC annual and quarterly) for one company or all with gaps |

## Requirements not done in this change, and why

* **DuckDB and Parquet as the store.** The store is SQLite and everything reads through it. Moving it would rewrite working code and the point-in-time engine's queries for no change in correctness. DuckDB is installed as an optional extra for analytic reads; no code depends on it yet. A Parquet export of the point-in-time observations is the right first step, and is not built.
* **Arelle.** Done as an opt-in cross-check (`importer/arelle_check.py`; `FUNDAMENTALS_ARELLE=1` or the `arelle` hint): given a full ESEF package it compares its own numeric facts with the importer's and reports disagreements and filer-side validation errors as review issues. Verified on a live package (208 of 208 facts agree). For US filings it needs the SEC taxonomy online, which this environment could not reach.
* **Structured-output AI extraction.** Not built. A PDF is read deterministically and always reviewed; an AI reading would be a second unverified source of numbers.
* **Company investor-relations pages.** Supported as links to a report and as uploads; there is no crawler for an investor-relations site.
* **Earnings revisions and analyst estimates.** No source of analyst estimates is connected, and none is invented. The factor is not offered.
* **Risk and performance attribution page, Optuna and SHAP in the alpha model builder, a side-by-side comparison of saved experiments, an Alphalens/vectorbt/CVXPY/Riskfolio backtest engine, Plotly and Streamlit.** The backtest and the models run on the platform's own engine and scikit-learn/LightGBM/XGBoost; the charts are ECharts through the existing chart API; the application is a vanilla JavaScript dashboard, not Streamlit, and was kept so. The attribution page, the explanation of tree models and the experiment comparison are the next work (Phases 3 and 4 of the request).

## Limits of the data, stated plainly

* SEC structured data starts with a company's first XBRL filing (2009 for the largest filers, 2011 for the rest). Earlier statements exist only as documents; they come in through PDF import and are reviewed by hand.
* The SEC has no restated-as-of series in the company facts; a restatement appears as a later filing's figure and is stored as a later vintage.
* A fourth quarter is derived, never filed. Quarterly cash-flow lines exist only as year-to-date in a 10-Q, so a quarter's cash flow depends on the earlier filing of the same year being present; when it is not, the line is left out.
* OpenFilings gives annual statements only.
* The point-in-time lag for an import without a filing date is 120 days for a year and 60 for a quarter, and is flagged estimated.
* PDF figures are as reliable as the layout of the PDF; scanned reports are refused (there is no OCR).
* The SEC refused automated requests to `www.sec.gov` from the environment this was built in (HTTP 403), so the filing list and the document fetch are tested against stand-ins of the documented shape, not against the live service.


## Global filings (second change)

Added: country connectors for EDINET, OpenDART, CVM, ESEF and link-only sources (HKEXnews, SGXNet, CNINFO, SSE, SZSE, SEDAR+) with each source's access rules enforced in code; the Japanese XBRL taxonomies; line names in eight languages; HTML table reading in the page's own encoding; OCR and Docling adapters (optional engines); AI-assisted extraction with a verbatim-evidence check; per-country coverage calendars; the Parquet/DuckDB point-in-time store with an as-of query; the Streamlit interface.

Not done, and why: the **live success answers of EDINET and OpenDART** (they need API keys; ESEF, CVM and the Japanese element names were checked live, see `docs/fundamentals.md`); **PaddleOCR, Surya and Docling** were not installed or run (heavy, optional, adapters tested with stand-ins); **dart-fss** is not used: OpenDART's API is called directly, which avoids a dependency and is the same interface; **SEDAR+, HKEXnews, SGXNet, CNINFO, SSE, SZSE** have no interface for programs and restrict automated collection, so they are links-and-files only; **Chinese XBRL** does not exist for Mainland companies (reports are PDFs); **Singapore's** and **Hong Kong's** financial statements are PDFs for most companies, so their history depends on the review of PDF readings: the PDF reader now handles their date, half-year, Notes-column and currency conventions but has been tested on generated PDFs, not on filed reports. A restatement in CVM's data is noted but only the latest version is read; EDINET corrections (type 130, 150, 170) are listed and read as separate filings.
