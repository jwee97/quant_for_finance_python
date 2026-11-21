"""Sources whose websites are for people: HKEXnews, SGXNet, CNINFO, the Shanghai and Shenzhen exchanges, and SEDAR+.

None of them publishes an interface for programs, and their terms restrict automated collection of their pages and documents. These connectors therefore **do not search, list or crawl**. What they do:

* accept a link or a file that a person has chosen (the person is the one using the website), checking that the link is on the source's own host so the import page is not an open fetcher;
* say which kind of report the title names (annual, interim, quarterly) in the source's languages;
* fetch that one document, when a link is given, as any browser download would (public addresses only, polite pacing, no key and no login).

For a company on one of these markets, the way to a complete history is to download the reports from the source's site and upload them (or paste their links) in Report import. Where a company is also
listed in the US, Japan, Korea, Brazil or the EU, the connector of that country is the better route.
"""

from __future__ import annotations

from ..errors import FundamentalsError
from . import calendar as cal
from .base import Access, Connector, FilingRef, registry


class LinkOnly(Connector):
    access = Access("links_only", None, None, "No interface for programs: only links and files a person has chosen are accepted.")
    formats = ("pdf", "html", "xlsx", "csv")
    why = ""

    def list_filings(self, company: dict, since: str | None = None, until: str | None = None, kinds=None, limit: int = 100) -> dict:
        return {"filings": [], "complete": False, "next_since": None, "note": f"{self.label} has no interface for programs and restricts automated collection, so it is not searched. {self.why} "
                "Download the reports you want and upload them, or paste their links, in Report import."}

    def get(self, ref: FilingRef) -> dict:
        if not ref.url:
            raise FundamentalsError("give the link of the document")
        self.classify_link(ref.url, ref.title)                                                    # the link must be on the source's own host
        got = self._fetch(ref.url)
        return {"kind": "file", "name": ref.doc_id or "report", "data": got.data, "hint": {"filing_date": ref.filed_at}}


def _make(name, label, countries, hosts, words, why, languages):
    cls = type(name.title().replace("_", ""), (LinkOnly,), {"name": name, "label": label, "countries": countries, "hosts": hosts, "why": why, "languages": languages, "words": lambda self, w=words: w + cal.GENERIC_WORDS})
    return registry.register(cls)


HKEX = _make("hkex", "HKEXnews (Hong Kong)", ("HK",), ("hkexnews.hk",), cal.HK_WORDS, "Annual and interim reports are PDFs, with inline XBRL for some issuers.", ("en", "zh"))
SGX = _make("sgx", "SGXNet (Singapore)", ("SG",), ("sgx.com", "links.sgx.com"), cal.SG_WORDS, "Results announcements and annual reports are PDFs.", ("en",))
CNINFO = _make("cninfo", "CNINFO (Mainland China, Shenzhen and Shanghai)", ("CN",), ("cninfo.com.cn",), cal.CHINA_WORDS, "The statutory disclosure site of the CSRC; reports are PDFs.", ("zh",))
SSE = _make("sse", "Shanghai Stock Exchange", ("CN",), ("sse.com.cn",), cal.CHINA_WORDS, "Annual, half-year and quarterly reports are PDFs.", ("zh", "en"))
SZSE = _make("szse", "Shenzhen Stock Exchange", ("CN",), ("szse.cn",), cal.CHINA_WORDS, "Annual, half-year and quarterly reports are PDFs.", ("zh", "en"))
SEDAR = _make("sedar", "SEDAR+ (Canada)", ("CA",), ("sedarplus.ca", "sedarplus.com"), cal.GENERIC_WORDS, "Filings are PDFs and inline-XBRL documents behind a search form.", ("en", "fr"))
