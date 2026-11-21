"""Physical document readers: bytes in, :class:`~src.fundamentals.importer.ir.Document` out. No accounting is known here."""

from . import html_layout, pdf_layout, xlsx_layout

__all__ = ["pdf_layout", "html_layout", "xlsx_layout"]
