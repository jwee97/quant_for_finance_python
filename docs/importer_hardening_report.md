# Report importer hardening: what was wrong, what changed, what is verified

**Read this first.** The seven original reports were not processed. All results below are on **synthetic look-alikes** of their structures and on the project's synthetic benchmark. They show that the
failure *mechanisms* are fixed; they do **not** show that the original PDFs pass, and they do not support a claim about any market. `python -m src.fundamentals.importer.benchmark` (see the end) is how the
originals get tested locally.

**Verdict.** The importer is **not ready for unattended production use**, and it is not claimed to be. It is safe to use *with its review gate*, which this work made stricter: nothing reaches the database
without a person's approval, a figure that is low-confidence, unvalidated-and-failed or overridden is stored `unverified` and excluded from research until confirmed, and the findings that most often mean a
wrong number (a failed reconciliation, mixed units, an inferred period, no currency) block approval until fixed or overridden with a written reason.

## Before and after

| | Before | After |
|---|---|---|
| Seven synthetic look-alikes, 44 expected observations (`real_structures.py`) | **10 correct**, 7 wrong, 27 missing; two reports "no statement recognised" | **44 correct**, 0 wrong, 0 missing |
| Toyota FY2020 total assets | 743,487 (the change column) | 52,680,436 |
| Samsung revenue | 30,474,830 (a quarterly segment table) | 236,806,988; the four KRW/USD columns kept as distinct observations |
| Project benchmark, dev (14) / validation (8) | 100% / 100% | 100% / 100% (unchanged; a regression was found and fixed on the way) |
| Project benchmark, held-out (24, frozen) | recall 68%, 0 confident errors | recall 70.7%, 0 confident errors (**contaminated, see `importer_benchmark_report.md`**) |
| Automated tests | | `tests/test_fund_real_structures.py` (26 tests) and the earlier suites pass |

## Changes by module: what was wrong, what changed, which test verifies it

| Module | Was wrong | Now | Tests |
|---|---|---|---|
| `layout/pdf_layout.py` | header words belonged to one segment, so a merged heading ("March 31, 2019 March 31, 2020") or a group heading ("Group", "In millions of Korean won") sat in one column; a `%` cell was not a column; a long title was read as a header; nothing about geometry survived | headings are assigned to columns by geometry (a heading over several columns spans them, group headings take the columns nearer to them than to another heading, merged date/year/term headings are split); percentages are positioned cells; currency symbols join their figure; title rows are told from header rows; raw words, boxes and column extents are kept; a looser alternative grouping exists for the second reading | `test_samsung…`, `test_singtel…`, `test_toyota…` |
| `layout/pdfplumber_layout.py` (new) | no independent second reading | pdfplumber words with different tolerances, looser grouping, for chosen pages only | `test_a_failed_first_reading_is_re_read…` |
| `columns.py` (new), `statements.resolve_columns` | **positional** matching of dates to figure columns; one currency/unit per table | every column gets a role (current, prior, historical, change, percent, forecast, restated, note, per share, other currency), a period, currency, unit, scope, confidence and the reasons, from its header hierarchy and, failing that, its figures (difference of two columns, small whole numbers) | `test_toyota_change_and_percent_columns…`, `test_a_change_column_is_found_from_its_figures…` |
| `periods.py` (new), `model.FinancialPeriod` | a year anywhere could be a period; no instant/span distinction; calendar-year assumptions | end date from the column's own heading, then the heading's day/month, then the report's fiscal year end, else a flagged guess; instants vs spans; annual / quarter / half / YTD; forecast and restated flags; 52/53-week years; fiscal year and quarter from the fiscal year end | `test_period_model…`, `test_apple_fiscal_periods…` |
| `statements.py` | a continuation page was an unrelated table; summaries and non-consolidated results were not separated | `merge_continuations` (titleless or "continued" tables that align with the previous one, even if a note column is missing); `SUMMARY` class (highlights, five-year, quarterly, forecast); non-consolidated and Company scope | `test_singtel…[True/False]`, `test_byd…`, `test_toyota_forecasts…` |
| `matcher.py` | one pass: first/highest-scoring value won, rejected candidates were forgotten, currencies and scopes were merged | candidate extraction → normalisation (exact `Decimal`, raw kept) → ranking (authority, totals over components, column confidence) → validation → selection; **every candidate and its rejection reason are kept**; alternative currency → `presentations`; parent scope → `scoped`; accounting basis read from the report ("prepared in accordance with …") and a table of another basis is rejected; ties of equal authority are **conflicts**, not coin tosses | `test_asml…`, `test_samsung…`, `test_every_candidate_is_kept…` |
| `taxonomy.py`, `canonical.py`, `filings/labels.py` | `net_income` also meant parent-attributable profit; adjusted/non-IFRS lines could match; bilingual labels never matched; totals vs components | `net_income_attributable` is its own line; adjusted / non-IFRS / minority labels are never statutory lines; bilingual labels are matched per script and must agree; statement vocabulary added | `test_tencent…`, `test_byd…`, `test_toyota_consolidated_net_income…` |
| `checks.py` | a missing input produced *no result*; one identity for all; fixed tolerance | four outcomes (PASS / FAIL / INSUFFICIENT_DATA / NOT_APPLICABLE); mezzanine-equity identity; gross profit, operating bridge, cash cross-checks, cash roll-forward; scale-aware tolerance; duplicate conflicts; `validation()` verdict that needs structure, resolved periods, one currency/unit per statement, no conflicts and every critical line *confirmed* | `test_validation_has_four_outcomes…` |
| `engine.py` | fallback when fewer than 4 lines were found (and then to the context-free reader) | `assess()` judges statements, critical coverage, columns, currency/unit, periods, reconciliation, conflicts, provenance; fallback is **targeted at the affected pages**, each step is re-validated and accepted only if validation improves (never because it returned more numbers); cached by document hash and pages; AI only for missing critical lines, verbatim-verified, stored unverified | `test_a_reading_with_many_fields_is_not_reliable…`, `test_a_second_reading_is_accepted_only_if…` |
| `trace.py` (new) | no diagnostics | structured JSON events for classification, column resolution, candidates, rejection, reconciliation (off by default; `FUNDAMENTALS_IMPORT_TRACE` or hint `trace`) | `test_the_decision_trace_is_structured_json…` |
| `pipeline.py` | statuses mixed extraction, validation and approval; corrections could not be audited; no way to pick between candidates | lifecycle states UPLOADED…COMMITTED derived from status and validation; hard blocks (failed check, mixed units, inferred period, unknown currency) need a fix or a written override; `confirm_periods`; `select` a candidate; `remap` a row; every correction is recorded in the draft; figures in failed reconciliations are stored `unverified`; richer provenance | `test_a_guessed_period_blocks…`, `test_an_override_needs_a_reason…`, `test_a_person_can_choose…` |
| `schema.py` | provenance lacked currency, unit multiplier, exact raw/normalised text, table/row/column, role, validation, override | additive columns, added when a database is opened; schema version unchanged | `test_new_provenance_columns_are_added…` |
| `fundamentals_import.js` | no view of what was read | lifecycle chips, validation findings with override form, statements-and-columns table, candidates with reasons and *Use this*, other-currency presentations | browser tests still pass |
| `benchmark.py` (new) | no way to run the original PDFs | CLI described below | `test_the_benchmark_runner…` |

## Requirement checklist

| Acceptance criterion | Status |
|---|---|
| Critical values extracted or explicitly rejected | yes: a critical line that cannot be confirmed makes the draft REVIEW_REQUIRED |
| Incorrect periods cannot silently enter | inferred periods block approval; confirming or overriding is recorded |
| Currency/unit mismatches block automatic approval | `unit_mismatch`, `currency_unknown` are hard blocks; other-currency columns never become figures |
| Invalid candidate selections are detectable | all candidates and reasons kept; conflicts are failures |
| Reconciliation where appropriate | four-valued, mezzanine-aware; missing inputs are INSUFFICIENT_DATA, never a pass |
| Missing data visible | `missing_critical`, INSUFFICIENT_DATA counts, diagnostics |
| Provenance retained | page, table, row, column, box, role, authority, versions, raw and normalised text |
| Manual corrections preserved and auditable | `corrections` log, `edited`, `edit_note`, overrides with reason and time |
| Duplicates safe | content hash (unchanged) |
| Restatements versioned | unchanged vintage mechanism (`restate` / `correct` / `skip`) |
| Alpha Lab cannot consume unapproved data | nothing is stored before approval; `unverified` excluded by the point-in-time engine and not cleared by the quality pass |
| Existing functionality | all earlier Fundamentals suites pass |

**Not done or only partly done, said plainly**
* *Automatic* approval does not exist: every import still needs a person. The benchmark reports the precision of what *would* be auto-approvable (VALIDATED drafts), but there is no real-filing evidence for any threshold.
* The database commit (`manual.save`) and the provenance write are two transactions: a crash between them leaves stored figures without provenance (the diagnostics can show it; it is not yet repaired automatically).
* Point-in-time fields: filing date, publication date, restatement vintage and import time exist (existing schema); a missing filing date is flagged as estimated and the lag rule applies. No new timestamp columns were added.
* Docling is still used only by the older chain; the new fallback chain is pdfplumber → OCR (if installed) → AI (if switched on) → manual.
* Not handled: statements side by side on one page, rotated pages, a continuation whose columns do not align, tables whose cells are drawn rather than typeset, a heading that does not sit above its figures, banks/insurers' presentation (no taxonomy for their lines), year-to-date columns (rejected, not de-cumulated), three-decimal numbers with a dot.
* The held-out claim: 24 synthetic documents, 70.7% line recall (the rest are unseen label wordings), no confident errors; **0 of 15 market/format pairs pass the acceptance criteria**, and none was tested on real filings.

## Dependencies and migration

No new dependency (pdfplumber and PyMuPDF were already in the optional extras; the models are standard-library dataclasses). The database change is additive (new provenance columns, `import_extractions` from the earlier work);
an older build still opens the file. A canonical line was added (`net_income_attributable`); `net_income` keeps its meaning for the SEC path (the tag `NetIncomeLoss`), and is the consolidated total for documents read by the engine.

## Running the benchmark on the original filings

```
python -m src.fundamentals.importer.benchmark --filings ./test_filings --expected ./test_filings/expected.json --output ./benchmark_results [--trace]
```
Instructions, the reference file template (`benchmarks/fundamentals_importer/real_filings/expected.example.json`, **unverified** values copied from the task) and the held-out discipline are in
`benchmarks/fundamentals_importer/real_filings/README.md`. Outputs: `results.json`, `results.csv`, `failures.json` (the root-cause trail of every wrong or missing observation), `summary.html`. A filing that is
not processed is reported as `not_processed`. Tune on `development` filings only; keep `heldout` filings out of every rule.
