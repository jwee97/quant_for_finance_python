"""An independent second reading of selected PDF pages, for the targeted fallback.

The primary reader takes its words from PyMuPDF. This one takes them from pdfplumber (a different word splitter, with its own tolerances) and rebuilds the tables with a *looser* grouping (headings and
labels that sat a little too far apart for the primary reader are joined; rows a little too far apart in height are not split), and, when the page has ruled tables, with the table finder's cells. It exists
so that a page whose first reading failed validation can be read by a different method and the two compared: it is never preferred because it returned more numbers.
"""

from __future__ import annotations

import io

from ..ir import Table
from . import pdf_layout


def available() -> bool:
    try:
        import pdfplumber  # noqa: F401
        return True
    except ImportError:
        return False


def tables_for_pages(data: bytes, pages: list[int], start_id: int = 10_000) -> dict[int, list[Table]]:
    """``{page: [Table]}`` for the 1-based ``pages`` of a PDF."""
    import pdfplumber
    out: dict[int, list[Table]] = {}
    counter = [start_id]
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for n in pages:
            if not 1 <= n <= len(pdf.pages):
                continue
            page = pdf.pages[n - 1]
            words = [(w["x0"], (w["top"] + w["bottom"]) / 2, w["x1"], w["text"]) for w in page.extract_words(x_tolerance=2.0, y_tolerance=2.5, keep_blank_chars=False, use_text_flow=False)]
            if not words:
                continue
            tables, _ = pdf_layout._build_page(n, words, "pdfplumber", counter, gap_scale=1.5, row_tolerance=4.0)
            out[n] = tables
    return out


__all__ = ["available", "tables_for_pages"]
