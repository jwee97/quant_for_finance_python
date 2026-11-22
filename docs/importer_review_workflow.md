# Fundamentals import: discover, confirm, store

**The scanner discovers financial information; the person confirms what is correct; the application stores and analyses only approved observations.**

This document covers the review workflow added on top of the extraction engine (`docs/importer_architecture.md`), the audit that motivated it, what was measured, and what is *not* done.
Everything measured here uses **synthetic look-alikes** of seven report structures. No original PDF was available or processed; no result below says anything about the real reports.

## 1. Audit: what the code did, and where

| # | Finding (confirmed in the code before the change) | Where |
|---|---|---|
| 1 | **A row whose label matched no taxonomy line was dropped before anyone could see it.** `if m is None: continue` (and `row not in s.lines`) in the candidate loop; segment revenue, deferred revenue, deliveries, capital ratios, adjusted measures never reached a draft. | `importer/matcher.py` `build()` |
| 2 | Draft edits could only name canonical lines: `if code not in manual.canonical.METRICS: raise FundamentalsError("… is not a line this application knows")`. A person could not add or keep an unfamiliar figure. | `pipeline.py` `review()`, `_apply_remap()` (still true for the *draft-level* legacy edit API; observations have no such limit) |
| 3 | **Scanner confidence counted as verification**: `verified = bool(s.get("verified") or s.get("edited") or s.get("confidence") is None or s["confidence"] >= LOW_CONFIDENCE)`. A missing confidence verified a figure; an edit verified it; 0.75+ verified it. | `pipeline.py` `_provenance()`; `_unverified()` |
| 4 | Approval stored every figure of the draft (flagging low-confidence ones `unverified`); unconfirmed figures were therefore *in* the research tables, only flagged. | `pipeline.py` `approve()`, `_entry()` |
| 5 | Provider data outranked a person's confirmed import: `PitSettings.provider_priority = ("sec_edgar", "openfilings", "simulated")` put `manual` last. | `pit.py` |
| 6 | A provider's figures in another currency could fill gaps in the leading provider's row (a KRW balance sheet under a USD income statement). | `pit.py` `_merge_providers()` |
| 7 | A fallback reading replaced the draft wholesale when it scored higher; observations only the first reading had found were lost. | `engine.py` `_targeted()` |
| 8 | Not defects, checked: statement/column/period logic, per-column currency/unit, targeted fallback and four-valued validation (earlier hardening, `docs/importer_hardening_report.md`); provider request history and rate limits (`usage.py`); PIT availability and `vintage == 0` as-first-filed (`pit.py`). | |

Things the brief assumed that are **not** true here: the Fundamentals UI is the vanilla-JS/ECharts web app (`src/webapp/static/fundamentals*.js`), not Streamlit/Plotly (`streamlit_app.py` is a separate, older front end); there is no `reconcile.py` problem of the kind described. I kept the existing architecture and charting stack.

## 2. The three layers

1. **Raw observations** (`raw_observations`, written at extraction): *every* figure of every table the scanner read, mapped or not: label, printed text, value, page, table, row, column, box, column heading and role (current, prior, change, percent change, forecast, note reference…), period (and whether it was inferred), currency, unit, scope, statement group (income statement, balance sheet, cash flow, equity, segment, notes, KPI, summary, other), the scanner's *guess* of the canonical line (`metric_guess`, a suggestion), extractor/mapping/taxonomy versions. Page numbers, note references, contact details and labels without words are not observations (`observations.is_financial`). A later reading adds what it finds and refreshes only unreviewed rows; a second reading **never removes** an observation the first one found (`observations.merge`, used by `engine._targeted`).
2. **Reviewed observations** (same rows, `status` and `user_*` columns, `history`): `UNCONFIRMED` (every observation starts here) → `CONFIRMED`, `CORRECTED_AND_CONFIRMED`, `EXCLUDED`, `UNRESOLVED`. Corrections (value, currency, unit incl. a custom multiplier, period start/end/type, fiscal period, scope, line or custom name, note) are stored beside the scanner's reading, with a history of every change. Editing a confirmed observation returns it to `UNCONFIRMED`.
3. **Research-ready** (`fundamental_observations` for canonical lines, `custom_observations` for everything else): only what a person confirmed and approved. A confirmed figure in another currency than the report's is stored under its own currency in `custom_observations`; it is never merged into the report's figures and never relabelled. `custom_observations` carries `filing_date`, `vintage` (a later different value is a new vintage; the earlier one stays) and is read through `store.custom_observations(ids, as_of)` which applies point-in-time availability.

Rules enforced in code (`importer/review.py`) and tested (`tests/test_fund_review_workflow.py`):

* scanner confidence never confirms; a missing confidence never confirms; an edit never confirms;
* an *inferred* period must be accepted or corrected by a person before its figure can be confirmed;
* two different confirmed figures for one line and period are a conflict that blocks approval until one is excluded;
* approval refuses when nothing is confirmed, runs the existing checks (reconciliations, restatement/correction decisions, FX conversion with rate provenance) on the draft rebuilt from confirmed observations only, then stores; the same file is not imported twice; an approved item can no longer be edited;
* warnings are visible and never discard data; hard blocks (`check_failed`, `unit_mismatch`, `period_unresolved`, `currency_unknown`) can be overridden only with a written reason.

## 3. The wizard (Fundamentals → Report import → open an item)

1. *Filing details*: company, ticker, exchange, filing type, accounting standard, fiscal year/period, period start/end, publication date, currency, units, scope (suggested, editable); "accept the inferred periods".
2. *Review statements*: every observation, grouped (statement groups plus **Unmapped**), searchable, sortable, filterable by decision; **the stored PDF page with the figure's box highlighted** beside it.
3. *Correct and classify*: edit any observation (value, currency, unit, period, scope, line, custom name), choose among competing candidates, add a figure the scanner missed, exclude, mark unresolved, bulk confirm/exclude, "confirm what the scanner chose for each line" (an explicit bulk decision).
4. *Warnings*: the validation findings, overrides with a written reason, restatement conflicts.
5. *Final approval*: counts (extracted, confirmed, corrected, excluded, unresolved, unconfirmed), the exact lines and custom figures that will be committed, what cannot be, and an acknowledgement box before the button is enabled.

API: `GET /api/fund/import/{observations,plan,source}`, `POST /api/fund/import/decide`, `GET /api/fund/custom`, `GET/POST /api/fund/sources/{compare,prefer}`, `GET /api/fund/lineage`, `GET /api/fund/alpha/inputs`.

## 4. Integration

* **Source priority**: figures a person confirmed (`manual`) now rank first; a provider's update does not overwrite them. `sources.compare` shows every provider's and import's figure side by side with currency, filing date, vintage, quality flag, confirmation and the difference; a person can choose a source per figure (`source_preferences`; a choice can never make a figure visible before its source made it public; nothing is deleted).
* **Currency**: figures in another currency never fill gaps in the leading provider's row.
* **Lineage**: `lineage.explain` re-evaluates a derived metric with a recording context: formula, the inputs the formula actually read (current/prior), each input's source (file, page, method, confirmed), currency, fiscal periods, availability date, version, calculation time, missing inputs. A metric with a missing input is *not calculated* (never filled).
* **Alpha Lab**: `lineage.alpha_inputs` lists, per factor field, securities/periods held, share from confirmed imports vs providers, flagged-out count, estimated-availability count and the number of unconfirmed scanner readings waiting (they are in no factor). Unconfirmed figures are not in `fundamental_observations`, so `PeriodTables`, ratios, factors, backtests and exports cannot see them (tested through the PIT panel).

## 5. Measured (synthetic; executed in this environment)

`python -m src.fundamentals.importer.benchmark --filings DIR --expected DIR/expected.json --output OUT` (writes `results.json`, `results.csv`, `failures.json`, `summary.html`; `--no-workflow` skips the workflow checks) on the seven synthetic look-alikes (`benchmarks/fundamentals_importer/results/review_workflow_synthetic.json`):

| | before this change | after |
|---|---|---|
| figures kept from the 7 documents | 137 (canonical lines only) | 228 observations archived (137 chosen for lines, 19 with no canonical line, the rest prior/change/percent columns and rejected candidates, with their roles) |
| raw recall of the expected reference values (anywhere in the archive) | not measured | 1.00 |
| canonical recall | 1.00 | 1.00 (unchanged: no regression) |
| nothing stored before confirmation | no (everything was stored, low-confidence flagged) | 7/7 |
| confirmation survives reading the file again | n/a | 7/7 |
| stored lines = confirmed lines (+ arithmetic derived from them) | no | 7/7 |
| research gate holds (no `unverified`/unconfirmed in research) | no | 7/7 |

These synthetic documents were built for the earlier hardening and are easy; this is a regression check of the workflow, not evidence of accuracy on real filings. **Real-PDF benchmark: not run (the originals are not available here).** Put the PDFs and an `expected.json` (see `benchmarks/fundamentals_importer/real_filings/expected.example.json`) in a folder and run the command above.

Performance (measured here, one run): bulk-confirming 330 observations took 1.1 s, building the final plan with 302 custom figures 0.02 s, adding one figure through the API about 20 ms each (each call re-runs the checks); extraction of the synthetic files took 0.07–3.2 s each. Not tested at thousands of observations in the browser.

Tests executed: `tests/test_fund_review_workflow.py` (24 tests: raw preservation incl. unmapped and change columns, fallback merge, confidence/edit/missing-confidence never confirming, corrected-and-confirmed history, inferred-period gate, exclude/unresolved, custom metrics with PIT availability and vintages, currency/unit handling, conflicts, duplicate import, reprocessing keeps decisions, end-to-end PDF → archive → confirm → plan → approve → store → PIT panel, source comparison and preference, lineage, alpha inputs, currency-mixing guard); the existing fundamentals suites were updated where they approved without confirming; the browser tests for the import page drive the five steps in Chromium.

## 6. Limitations (not done, or not verified)

* **No original PDF has been processed.** Real-document accuracy is unknown. Scanned reports need PaddleOCR (not installed here; adapter tested with stand-ins only).
* The PDF-beside-values viewer works for stored **PDFs** (a page image with the box); HTML/Excel items show no page. Highlight boxes come from the reader's coordinates.
* **Custom metrics are stored and PIT-readable but are not yet inputs of the factor DSL**: Alpha Lab can show them in `alpha_inputs`/`/api/fund/custom`; using one as a factor field needs a catalogue extension. Canonical lines are fully gated.
* Not done from the brief: valuation-model assumption editing and scenario comparison with per-input lineage; industry-specific ratio packs (banking, insurance, regulatory capital); a Plotly front end (the app uses ECharts); factor-correlation/IC-stability additions to Alpha Lab; provider health dashboard beyond the existing usage/rate-limit history; held-out cross-market generalisation beyond the earlier synthetic splits; a large-filing performance pass (the observation table is paged at 100 rows in the UI; extraction time on the synthetic files is under four seconds each).
* Approval writes canonical lines (`manual.save`), their provenance and custom figures in separate transactions: a crash in between can leave a period saved without its custom figures (re-approving is refused; use the import history to see it).
* The legacy draft-level review API (typing a value into the period table) still exists and is routed through observations (a typed value becomes an unconfirmed correction); its line names are limited to the canonical set.

## 7. Quarterly derivation from year-to-date figures (`quarters.py`)

Audit: the SEC quarterly provider (`providers/sec_quarters.py`) de-cumulates year-to-date facts but keeps only the derived quarter (no original cumulative figure, no derivation text); the importer rejected year-to-date columns outright, so a company reporting cumulatively could not be imported as quarters at all; `pit._build_ttm` assumes standalone quarters and guards against cumulative ones only by a revenue check against the annual figure.

Added: `ytd_observations` keeps cumulative figures as reported (never overwritten); `quarters.decumulate/derive` produce standalone quarters (100, 250, 390, 600 → 100, 150, 140, 210) with the derivation text, the inputs (period, end date, filing date, vintage), `available_from` (the latest input's filing date), the length in days (a 14-week quarter is 98 days and is not rejected; a 248-day "quarter" is queried), and flags for mixed vintages (restated inputs), missing filing dates and mixed currencies (not derived). A quarter with a missing input is not derived (no half-year: Q2/Q3 are not separable; the second half is). Derived quarters are written as provider `derived`, flag `derived`, and feed TTM from their availability date. Tested in `tests/test_fund_quarters.py`. **Not yet done:** the importer does not yet route year-to-date columns into `ytd_observations` (it still rejects them), and the SEC provider still stores only the derived quarter.

## 8. Source priority policy (changed)

`PitSettings.provider_priority` is now `("manual", "sec_edgar", …)`: a person's confirmed or typed figures lead, providers fill the lines they lack. Two older tests in `tests/test_fund_manual.py` that asserted the opposite policy were rewritten; the difference is still reported by `overlaps` and `/api/fund/sources/compare`.
