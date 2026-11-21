"""Fetching a report from a link: a public http(s) URL, with the care a server that fetches addresses for a person needs.

* **Only public addresses.** The scheme must be http or https, there must be no user name in the link, and every address the host resolves to (and every redirect's) must be public: a link to ``localhost``,
  a private network, a link-local address or the cloud metadata address is refused, so the import page cannot be used to read the machine's own network.
* **Bounded.** A body larger than the limit (40 MB) is refused as it arrives; a request times out; redirects are followed by hand (at most five) so each hop is checked.
* **Polite and patient.** A host is never hit more than once in ``min_interval`` seconds; a 429, a 5xx or a timeout is retried with a growing wait (honouring ``Retry-After``, at most a minute) before it
  is reported; the SEC hosts require a declared contact in the User-Agent and the fetch is refused, with that reason, when none is configured.
* **Cached by the pipeline**, not here: the bytes are kept once by content hash, and a link already fetched is not fetched again unless asked.
"""

from __future__ import annotations

import ipaddress
import os
import socket
import time
from dataclasses import dataclass
from urllib import error as urlerror
from urllib import parse, request

from ..errors import FundamentalsError

MAX_BYTES = 40_000_000
MAX_REDIRECTS = 5
RETRYABLE = {429, 500, 502, 503, 504}


@dataclass
class Fetched:
    data: bytes
    content_type: str
    url: str
    status: int


def check_url(url: str, resolver=socket.getaddrinfo) -> str:
    """The URL if it is a public http(s) address, else a :class:`FundamentalsError` that says why."""
    text = str(url or "").strip()
    if not text or len(text) > 2000:
        raise FundamentalsError("give the link of a report (an address of up to 2000 characters)")
    parts = parse.urlsplit(text)
    if parts.scheme not in ("http", "https"):
        raise FundamentalsError("only http and https links can be fetched")
    if parts.username or parts.password:
        raise FundamentalsError("a link with a user name or password in it cannot be fetched")
    host = parts.hostname
    if not host:
        raise FundamentalsError("the link has no host")
    try:
        infos = resolver(host, parts.port or (443 if parts.scheme == "https" else 80), type=socket.SOCK_STREAM)
    except OSError:
        raise FundamentalsError(f"the host {host} could not be found") from None
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
            raise FundamentalsError(f"{host} is not a public address, so it will not be fetched")
    return text


class _NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def _urllib_transport(url: str, headers: dict, timeout: float) -> tuple[int, dict, bytes]:
    opener = request.build_opener(_NoRedirect)
    req = request.Request(url, headers=headers)
    try:
        with opener.open(req, timeout=timeout) as response:
            body = response.read(MAX_BYTES + 1)
            return response.status, dict(response.headers.items()), body
    except urlerror.HTTPError as err:
        return err.code, dict(err.headers.items()), err.read(200_000)
    except (urlerror.URLError, TimeoutError, OSError) as err:
        raise TimeoutError(str(err)) from None


class Fetcher:
    def __init__(self, transport=None, resolver=socket.getaddrinfo, sleep=time.sleep, clock=time.monotonic, min_interval: float = 0.5, retries: int = 3, timeout: float = 30.0, user_agent: str | None = None):
        self.transport, self.resolver, self.sleep, self.clock = transport or _urllib_transport, resolver, sleep, clock
        self.min_interval, self.retries, self.timeout = min_interval, retries, timeout
        self.user_agent = " ".join(str(user_agent or os.environ.get("SEC_EDGAR_USER_AGENT", "")).split())
        self._last: dict[str, float] = {}

    def _pace(self, host: str) -> None:
        wait = self._last.get(host, -1e9) + (0.15 if host.endswith("sec.gov") else self.min_interval) - self.clock()
        if wait > 0:
            self.sleep(wait)
        self._last[host] = self.clock()

    def fetch(self, url: str) -> Fetched:
        current = check_url(url, self.resolver)
        for hop in range(MAX_REDIRECTS + 1):
            host = parse.urlsplit(current).hostname or ""
            if host.endswith("sec.gov") and not self.user_agent:
                raise FundamentalsError("the SEC requires every automated request to say who is asking: set SEC_EDGAR_USER_AGENT (a name and an e-mail address) before fetching from sec.gov")
            headers = {"User-Agent": self.user_agent or "quant-fundamentals-import/1.0", "Accept": "*/*", "Accept-Encoding": "identity"}
            status, got, body = self._get(current, headers, host)
            if status in (301, 302, 303, 307, 308):
                location = {k.lower(): v for k, v in got.items()}.get("location")
                if not location:
                    raise FundamentalsError(f"the server redirected without saying where (HTTP {status})")
                current = check_url(parse.urljoin(current, location), self.resolver)
                continue
            if status >= 400:
                raise FundamentalsError(f"the server answered HTTP {status} for this link" + (" (the SEC refuses a client that does not identify itself or goes too fast)" if status == 403 and host.endswith("sec.gov") else ""))
            if len(body) > MAX_BYTES:
                raise FundamentalsError(f"the report is larger than {MAX_BYTES // 1_000_000} MB: it will not be fetched")
            ctype = {k.lower(): v for k, v in got.items()}.get("content-type", "")
            return Fetched(body, ctype.split(";")[0].strip().lower(), current, status)
        raise FundamentalsError("the link redirects too many times")

    def _get(self, url: str, headers: dict, host: str) -> tuple[int, dict, bytes]:
        delay = 1.0
        for attempt in range(self.retries + 1):
            self._pace(host)
            try:
                status, got, body = self.transport(url, headers, self.timeout)
            except TimeoutError:
                status, got, body = 0, {}, b""
            if status in RETRYABLE or status == 0:
                if attempt == self.retries:
                    raise FundamentalsError("the server did not answer after several tries" if status == 0 else f"the server kept answering HTTP {status} after several tries")
                after = {k.lower(): v for k, v in got.items()}.get("retry-after", "")
                self.sleep(min(60.0, float(after) if after.isdigit() else delay))
                delay *= 2
                continue
            return status, got, body
        raise FundamentalsError("the server did not answer")
