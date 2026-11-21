# Running the importer on the original filings

None of the seven original reports was available when the importer was hardened. Everything under `tests/fixtures/real_like.py` is a **synthetic look-alike** of their structures. Passing those proves the failure
classes are fixed; it does **not** prove the original PDFs pass. Do not report them as passed until you have run this.

1. Put the seven PDFs in a folder, named as in `expected.example.json` (or edit the `file` fields).
2. Copy `expected.example.json` to `expected.json` in that folder. The values in it were copied from the task description and are **unverified**: check each against the report, fill in the empty ones (Singtel, ASML
   beyond revenue) and mark each filing `"split": "development"` (the ones you tune on) or `"split": "heldout"` (the ones you never tune on).
3. `python -m src.fundamentals.importer.benchmark --filings ./test_filings --expected ./test_filings/expected.json --output ./benchmark_results`
   (add `--trace` to keep the structured decision trace of every filing in `results.json`).
4. Read `benchmark_results/summary.html`, then `failures.json`: for every wrong or missing observation it lists every candidate that matched (table, page, column role, authority, the reason it lost) and the
   reconciliation and fallback record, which is the trail from the words on the page to the value that was chosen.
5. A filing that was not processed is reported as `not_processed`, never as passed.

For held-out evidence, run additional, *unseen* filings (banks, insurers, manufacturers, technology companies, conglomerates; other fiscal years, languages and layouts) listed with `"split": "heldout"`. Report
precision, recall, coverage and the review rate separately (the summary does).
