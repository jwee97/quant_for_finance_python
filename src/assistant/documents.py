"""The FOMC statement corpus, point-in-time stamped (Generation 4, Priority 16).

Statements are downloaded from the Federal Reserve Board's website, parsed to plain text and cached as one file
per statement with a manifest holding the source URL and the sha256 of the downloaded page. A statement dated
``D`` is treated as AVAILABLE from ``D + 1`` calendar day, which is conservative: it is released at 2:00 p.m.
Eastern, before the close.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.request
from pathlib import Path
from typing import Callable

import pandas as pd

BASE = "https://www.federalreserve.gov"
HISTORICAL = BASE + "/monetarypolicy/fomchistorical{year}.htm"
CURRENT = BASE + "/monetarypolicy/fomccalendars.htm"
LINK = re.compile(r'href="(/newsevents/(?:press/monetary/|pressreleases/monetary)(\d{8})a\.htm)"')
_CUT = ("Voting for the monetary policy action", "Voting against", "For media inquiries", "Implementation Note", "Statement on Longer-Run Goals")


def fetch(url: str, retries: int = 3, pause: float = 1.0) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research; contact via repository)"})
    last: Exception | None = None
    for attempt in range(retries):
        try:
            return urllib.request.urlopen(request, timeout=40).read().decode("utf-8", "ignore")
        except Exception as error:                                   # network errors are retried, then re-raised
            last = error
            time.sleep(pause * (2 ** attempt))
    raise RuntimeError(f"could not fetch {url}: {last}")


def statement_links(first_year: int = 2006, last_year: int = 2026, get: Callable[[str], str] = fetch) -> dict[str, str]:
    """``{yyyymmdd: path}`` for every press release linked from the FOMC calendar pages."""
    found: dict[str, str] = {}
    pages = [HISTORICAL.format(year=y) for y in range(first_year, min(last_year, 2020) + 1)]
    if last_year >= 2021:
        pages.append(CURRENT)
    for page in pages:
        for path, date in LINK.findall(get(page)):
            found.setdefault(date, path)
    return dict(sorted(found.items()))


def parse_statement(html: str) -> tuple[str, str]:
    """``(title, body text)``. The body drops page furniture and the list of voters."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    title = (soup.title.string or "").replace("Federal Reserve Board -", "").strip() if soup.title else ""
    node = soup.select_one("div#article") or soup.select_one("div#content")
    if node is None:
        return title, ""
    text = re.sub(r"\s+", " ", node.get_text(" ")).strip()
    head = re.search(r"\bShare\b", text)
    if head and head.start() < 200:
        text = text[head.end():].strip()
    for marker in _CUT:
        at = text.find(marker)
        if at > 0:
            text = text[:at].strip()
    return title, text


def is_statement(title: str, text: str) -> bool:
    return "FOMC statement" in title or text.startswith("The Federal Open Market Committee decided")


def ensure_fomc_corpus(directory: str | Path, first_year: int = 2006, last_year: int = 2026, get: Callable[[str], str] = fetch,
                       refresh: bool = False) -> pd.DataFrame:
    """Download what is missing and return the corpus (date, text, url) sorted by date."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() and not refresh else {}
    if refresh or not manifest:
        for date, path in statement_links(first_year, last_year, get).items():
            if date in manifest and (directory / f"{date}.txt").exists():
                continue
            url = BASE + path
            html = get(url)
            title, text = parse_statement(html)
            if not is_statement(title, text) or len(text.split()) < 40:
                manifest[date] = {"url": url, "title": title, "sha256": hashlib.sha256(html.encode()).hexdigest(), "statement": False}
                continue
            (directory / f"{date}.txt").write_text(text + "\n", encoding="utf-8")
            manifest[date] = {"url": url, "title": title, "sha256": hashlib.sha256(html.encode()).hexdigest(), "statement": True, "words": len(text.split())}
        manifest_path.write_text(json.dumps(dict(sorted(manifest.items())), indent=1))
    return load_corpus(directory)


def load_corpus(directory: str | Path) -> pd.DataFrame:
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    rows = [{"date": pd.Timestamp(d), "available_at": pd.Timestamp(d) + pd.Timedelta(days=1), "url": m["url"],
             "text": (directory / f"{d}.txt").read_text(encoding="utf-8").strip()}
            for d, m in sorted(manifest.items()) if m.get("statement") and (directory / f"{d}.txt").exists()]
    return pd.DataFrame(rows)


def corpus_version(directory: str | Path) -> str:
    manifest = json.loads((Path(directory) / "manifest.json").read_text())
    payload = json.dumps({d: m["sha256"] for d, m in manifest.items() if m.get("statement")}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]
