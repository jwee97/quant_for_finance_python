"""Country connectors: where a company's filings come from, country by country, and what each source lets an application do.

A connector lists the filings of a company (or says it cannot), and fetches one as a *draft* (a source that already gives structured figures: the Korean regulator's account API, Brazil's open data, a
filing's tagged XBRL) or as a *file* for the importer to read (an XBRL instance, an inline-XBRL report, a PDF). What a source permits is part of the connector, not an afterthought:

* ``api``        an official, documented interface for programs (SEC EDGAR, EDINET, OpenDART). It needs a key where the source asks for one, and the connector refuses without it.
* ``open_data``  files the regulator publishes for reuse (Brazil's CVM, the ESEF filing index). No key; the files are public.
* ``links_only`` sources whose websites are for people (HKEXnews, SGXNet, CNINFO, SSE, SZSE, SEDAR+): they publish no interface for programs, and their terms restrict automated collection. The connector does
  not search or crawl them. It accepts a link or a file that a person has chosen, checks the link is on the source's own host, and says which kind of report it is.
"""

from .base import Access, Connector, FilingRef, registry

__all__ = ["Access", "Connector", "FilingRef", "registry"]

from . import cvm, dart, edinet, esef, linkonly, sec  # noqa: E402,F401  (importing registers each connector)
