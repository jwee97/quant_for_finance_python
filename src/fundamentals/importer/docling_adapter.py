"""Docling for complex tables, used only when the plain reading of a PDF finds too little.

Docling (IBM) analyses a page's layout and recovers tables whose rows and columns the word positions alone cannot. It is heavy (PyTorch and layout models), optional and slower than the deterministic reader,
so it is tried only after the deterministic reader found fewer than a handful of known lines. Its tables are turned into the same tab-separated text the spreadsheet reader takes, so the line names, the period
headings and the unit are recognised by the same rules, and the result is marked ``pdf-docling`` and reviewed. The adapter follows Docling's documented converter interface and is tested here with a stand-in.
"""

from __future__ import annotations

import importlib.util
import os
import tempfile

from ..errors import FundamentalsError


def available() -> bool:
    return importlib.util.find_spec("docling") is not None


def tables(data: bytes) -> list[str]:
    """Each table of the document as tab-separated text (``available()`` first)."""
    try:
        from docling.document_converter import DocumentConverter
    except ImportError:
        raise FundamentalsError("Docling is not installed (pip install docling)") from None
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "report.pdf")
        with open(path, "wb") as fh:
            fh.write(data)
        doc = DocumentConverter().convert(path).document
        out = []
        for t in doc.tables:
            frame = t.export_to_dataframe()
            rows = ["\t".join([""] + [str(c) for c in frame.columns])] + ["\t".join(str(v) for v in row) for row in frame.itertuples(index=False)]
            out.append("\n".join(rows))
        return out
