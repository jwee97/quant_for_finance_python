# Report importer: architecture, audit, limitations

This is the filing-agnostic extraction engine behind **Report import**. It sits *beside* the existing readers and feeds the same review, approval and storage code, so existing interfaces and databases keep working.

## 1. Audit of what existed, and what was reused

| Existing piece | Reused? | Why |
|---|---|---|
| `importer/xbrl.py` (XBRL/iXBRL facts, JP/US taxonomies, parent-only handling) | kept as is | already fact-based and context-aware: periods, units and scope come from the contexts, not from guessing. It does not go through the table IR (there is no table). Its figures carry no confidence field, so they are treated as exact. |
| `importer/pdf.py` (words to rows, headings, unit, dates) | reused as helpers and as the **fallback** | date/term/heading parsing (`_headings`, `context_from_lines`, `month_of`) is shared; the line-by-line reading itself is now only the fallback, with a confidence cap, because it has no statement context |
| `importer/html.py`, `tabular.py` | reused (charset decoding, unit and currency words, Enter-reports template) | CSV/Excel in the Enter-reports template still read first |
| `importer/ocr.py`, `docling_adapter.py`, `ai_extract.py` | reused | OCR output (words with boxes) is the input of the PDF layout reader; AI extraction keeps its verbatim-evidence rule and stays opt-in |
| `importer/reconcile.py`, `manual.check_entry` | reused | cross-period and against-the-database checks are unchanged; the engine adds *within-document* checks before them |
| `importer/currency.py` | reused | conversion happens after extraction |
| `manual.ALIASES`, `filings/labels.py` | reused as the base label list of every mapping | |
| `import_blobs`, `import_items`, `observation_provenance` | reused, extended additively | the original bytes are never replaced |
| the data-quality flags (`quality.py`, PIT `exclude_flags`) | reused for gating | a new flag, `unverified`, rides the existing mechanism |

## 2. Layers (each knows only the one before it)

```
bytes ──► layout/ (physical)      pdf_layout, html_layout, xlsx_layout  ──►  ir.Document (tables, cells, boxes, pages, titles, text)
          ir.Document ──► statements.py   which table is which statement; consolidated vs parent; period columns; unit; currency
          statements ──► taxonomy.py      what a row label means, per accounting standard (versioned mappings; custom taxonomies)
          mapped rows ──► matcher.py      statement-first matching, candidate choice, confidence
          draft ──► checks.py             reconciliations, cross-statement checks, duplicates, anomalies, confidence adjustment
          draft ──► engine.py             fallbacks (legacy reader → Docling → opt-in AI), every one recorded
```

* **IR** (`importer/ir.py`): `Document` → `Table` → `Cell` with text, row, column, page, bounding box, parsed number; plus `Section`, `Fact` (for structured sources), format, language, warnings. JSON round-trips.
* **Statement first**: a table is used only if it is recognised as an income statement, balance sheet or cash flow statement (title and lines vote independently), is consolidated (a parent-company statement is rejected, with the reason), and has period columns that could be read (headings, numbered terms such as 제 55(당) 기, dates in every script and several languages' month names, or, flagged as a guess, the report's latest date). A segment note, a five-year summary and a changes-in-equity table are not primary statements.
* **Context-aware matching**: a label that matches is accepted only in the statement it belongs to (revenue is not taken from a balance sheet; the cash flow statement's first "profit" line and its closing cash are kept as *cross-check values*, not as observations). Wrapped labels are joined before matching. Candidates compete by confidence; the same line twice with different values lowers confidence and is reported.
* **Standards**: `us-gaap`, `ifrs`, `sfrs-i`, `k-ifrs`, `j-gaap`, `cas`, plus `taxonomy.custom(name, {label: code})`. A mapping adds labels and *exclusions* (Japanese ordinary income is not net income). The standard is taken from the hint, the country, the document's words or script, else IFRS (and the draft says it was assumed).
* **Confidence** per figure = statement confidence × label tier × column certainty × scope × reader (text layer 0.95, OCR 0.7, HTML 0.95, workbook 1.0) × a 0.7 factor when the unit is not stated; reconciliations then add 0.08 (agree) or multiply by 0.6 (disagree).
* **Gating**: below **0.75** a figure is stored with the quality flag `unverified`, which the point-in-time engine excludes like the other serious flags, so *no Alpha Lab calculation sees it*. The data-quality pass never clears it. A person releases it by ticking *Confirmed* on the line in the review table, by typing the value, or afterwards with *Confirm the low-confidence figures*.

## 3. Originals, normalised values, versions

* The **original file** is stored once by hash (`import_blobs`) and never changed. Each figure's provenance records the value **as reported**, its unit, the original currency, page, box, statement, scope, label, mapping rule, confidence, and the **extractor and mapping versions** (`observation_provenance`, new columns).
* The **normalised** value (scaled to the currency's own units, converted if asked) is the observation; the explicit unit, fiscal period, filing date and currency are columns of the period.
* Every reading is kept in `import_extractions` with `extractor_version`, `mapping_version`, `taxonomy_version`, standard, method, line count and mean confidence. `reprocess(item)` reads the stored original again with the current code and returns the figures that changed, appeared or disappeared; a saved item's stored figures are **not** touched (store the new reading as a restatement or correction through the usual approval if wanted). Versions are in `importer/taxonomy.py` (`EXTRACTOR_VERSION`, `TAXONOMY_VERSION`, each mapping's `version`).
* **Migration**: additive only (new columns and one new table, added when a database is opened); the schema version number is unchanged, so an older build still opens the file. Tested (`test_an_old_database_is_migrated…`).

## 4. Interfaces

Unchanged: every existing route and service method. Added: `POST /api/fund/import/verify`, `POST /api/fund/import/reprocess`, `GET /api/fund/import/diagnostics`, `GET /api/fund/import/extractions`; hints `standard` and `engine: "legacy"` (forces the old readers); draft keys `statements`, `checks`, `confidence`, `rejections`, `standard`, `fallbacks`, `versions`; sources keys `confidence`, `statement`, `scope`, `bbox`, `label`, `mapping_rule`, `alternatives`.

## 5. Diagnostics

*Report import → Importer diagnostics* shows, from what was actually read here: extraction success (overall and by format), statement coverage, reconciliation outcomes by check, fallback usage, rejection reasons and failure reasons, the share of low-confidence figures, how many stored figures a person corrected (and the critical ones), the figures held out of research, and extraction versions in use; and, from the benchmark, line recall, critical-line accuracy, period / unit / currency accuracy and confident errors per split. True accuracy is only known for the benchmark: live numbers are proxies.

## 6. Benchmark and held-out discipline

`benchmarks/fundamentals_importer/`: `generate.py` (deterministic synthetic documents and truth), `harness.py` (scores), `report.py` (writes `docs/importer_benchmark_report.md`), `acceptance.json` (thresholds), `heldout.lock` (hash of the frozen held-out files; a test fails if they change). Run `python benchmarks/fundamentals_importer/harness.py --split heldout`. **Read `docs/importer_benchmark_report.md`: it states what was and was not measured.**

## 7. Limitations (what this does not do)

* **No real filing was benchmarked.** All 46 documents are synthetic; layouts and wording are my reconstruction of each market's conventions. A pass here is not support for a market. **No market or format is claimed as supported.**
* **OCR accuracy is unmeasured.** No OCR engine is installed; scanned cases use a simulator that damages digits. What is tested is that damaged digits do not become confident values.
* **Label coverage is the main generalisation limit**: unseen wording is simply not extracted (held-out line recall 68% against 100% on the tuned splits). Lists were extended from dev and validation reports only; the held-out wording was never added. Real reports will contain wording that is also missing.
* The IR assumes **one statement per table, upright pages, one column band**. Not handled: two statements side by side, a statement continued across a page break (read as two tables; the second has no title), rotated or multi-column page layouts, merged-cell headers in complex HTML, and spans in workbooks.
* `1.300` (three decimals with a dot) is read as one thousand three hundred, never 1.3; a per-share figure with three decimals would be misread.
* Parent-company detection is keyword-based (parent, company only, 個別, 별도, 母公司, …); an unlabelled separate statement is not recognised and would be read as consolidated (confidence is slightly lower when scope is unstated).
* The accounting standard defaults to IFRS when nothing says otherwise; mappings differ little between standards, so this mostly matters for the exclusion lists.
* The XBRL path is unchanged and is not routed through the IR (there is nothing physical to extract); the "universal" representation for structured sources is the draft itself plus `Fact`, which the XBRL reader does not yet populate.
* Confidence is a calibrated *ordering*, not a probability; the 0.75 threshold is a design choice, not fitted to real data.
* Quarterly statements: year-to-date columns are rejected (not de-cumulated) for documents read by the layout engine; the SEC 10-Q path still de-cumulates.
