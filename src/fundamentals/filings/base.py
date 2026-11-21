"""The shape every country connector has, so the import page, the API and the backfill treat a source in one way."""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from urllib import parse

from ..errors import FundamentalsError


@dataclass(frozen=True)
class Access:
    """What a source lets a program do. ``discovery``: ``api`` (an official interface for programs), ``open_data`` (files published for reuse) or ``links_only`` (a website for people: no search, no crawl)."""

    discovery: str
    key_env: str | None = None                  # the environment variable that holds the key the source asks for, if any
    terms_url: str | None = None
    note: str = ""


@dataclass
class FilingRef:
    """One filing: where it is, what it is and when it became public."""

    source: str
    country: str
    company_id: str
    doc_id: str
    doc_type: str = ""
    fiscal_period: str | None = None            # FY, Q1..Q4, H1, H2, or "Q" when the quarter is not known before the file is read
    period_end: str | None = None
    filed_at: str | None = None
    url: str | None = None
    format: str = "unknown"                      # xbrl | ixbrl | json | csv | html | pdf | unknown
    title: str = ""
    language: str | None = None
    version: int = 0                             # a correction or restatement of the same report has a higher version
    correction: bool = False
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


class Connector:
    name = "base"
    label = "Base"
    countries: tuple[str, ...] = ()
    hosts: tuple[str, ...] = ()                  # the source's own hosts: a link a person gives must be on one of them
    formats: tuple[str, ...] = ()
    access = Access("links_only")
    languages: tuple[str, ...] = ("en",)

    def __init__(self, fetcher=None, env=None):
        self.fetcher = fetcher
        self.env = os.environ if env is None else env

    # ------------------------------------------------------------------------------------------------------------ what it is
    def key(self) -> str:
        return str(self.env.get(self.access.key_env, "")).strip() if self.access.key_env else ""

    def available(self) -> tuple[bool, str]:
        if self.access.key_env and not self.key():
            return False, f"{self.label} asks for a key: set {self.access.key_env} (see {self.access.terms_url or 'its website'}); no request is made without one"
        return True, ""

    def describe(self) -> dict:
        ok, why = self.available()
        return {"name": self.name, "label": self.label, "countries": list(self.countries), "formats": list(self.formats), "discovery": self.access.discovery, "available": ok, "reason": why,
                "key_env": self.access.key_env, "terms_url": self.access.terms_url, "note": self.access.note, "languages": list(self.languages), "hosts": list(self.hosts)}

    def _need(self) -> None:
        ok, why = self.available()
        if not ok:
            raise FundamentalsError(why)

    def _fetch(self, url: str):
        if self.fetcher is None:
            from ..importer.fetch import Fetcher
            self.fetcher = Fetcher()
        return self.fetcher.fetch(url)

    def _json(self, url: str) -> dict:
        import json
        got = self._fetch(url)
        try:
            data = json.loads(got.data.decode("utf-8-sig"))
        except (ValueError, UnicodeDecodeError):
            raise FundamentalsError(f"{self.label} answered with something that is not JSON (a maintenance page, or a refused key?)") from None
        return data if isinstance(data, dict) else {"results": data}

    def on_host(self, url: str) -> bool:
        host = (parse.urlsplit(url).hostname or "").lower()
        return any(host == h or host.endswith("." + h) for h in self.hosts)

    # ------------------------------------------------------------------------------------------------------------ what it does
    def list_filings(self, company: dict, since: str | None = None, until: str | None = None, kinds=None, limit: int = 100) -> dict:
        """``{"filings": [FilingRef...], "complete": bool, "next_since": str | None, "note": str}``. A source with no interface for programs says so in ``note`` and lists nothing."""
        raise NotImplementedError

    def get(self, ref: FilingRef) -> dict:
        """``{"kind": "draft", "draft": {...}, "raw": bytes, "name": str}`` (figures already structured) or ``{"kind": "file", "name": str, "data": bytes}`` (for the importer to read)."""
        raise NotImplementedError

    def classify_link(self, url: str, title: str = "") -> FilingRef:
        """A link a person chose, on the source's own host, described (kind of report, format) but not fetched."""
        if not self.on_host(url):
            raise FundamentalsError(f"that link is not on {self.label}'s own site ({', '.join(self.hosts) or 'no host is allowed'}); only its own documents are accepted for this source")
        from . import calendar as cal
        kind = cal.classify_title(title or url, self.words())
        ext = re.search(r"\.(pdf|xhtml|xml|xbrl|htm|html|csv|xlsx|zip)(?:$|\?)", url.lower())
        fmt = {"pdf": "pdf", "xhtml": "ixbrl", "xml": "xbrl", "xbrl": "xbrl", "htm": "html", "html": "html", "csv": "csv", "xlsx": "xlsx", "zip": "zip"}.get(ext.group(1) if ext else "", "unknown")
        return FilingRef(self.name, self.countries[0] if self.countries else "", "", url.rsplit("/", 1)[-1][:80], "", kind, None, None, url, fmt, title)

    def words(self):
        from . import calendar as cal
        return cal.GENERIC_WORDS


class Registry:
    def __init__(self):
        self.classes: dict[str, type] = {}

    def register(self, cls: type) -> type:
        self.classes[cls.name] = cls
        return cls

    def make(self, name: str, fetcher=None, env=None) -> Connector:
        if name not in self.classes:
            raise FundamentalsError(f"no connector called {name!r}: choose one of {', '.join(sorted(self.classes))}")
        return self.classes[name](fetcher, env)

    def for_country(self, country: str) -> list[str]:
        c = (country or "").upper()
        return [n for n, k in self.classes.items() if c in k.countries]

    def describe(self, fetcher=None, env=None) -> list[dict]:
        return [self.make(n, fetcher, env).describe() for n in sorted(self.classes)]


registry = Registry()
