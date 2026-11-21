"""Report import: files, links and archives of financial reports, extracted into drafts that a person reviews before anything is stored.

The order of preference is the one the data allows: structured XBRL (a filing's own tagged numbers, read exactly), then tables (CSV, Excel), then PDF text, which is the least reliable and is always
sent for review. Nothing here writes a figure to the research database without an approval, and every figure stored carries where it came from (:mod:`pipeline`).
"""
