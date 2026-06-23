"""Optional same-site web crawler with SSRF guards.

A compact, dependency-light breadth-first crawler that stays on the seed's
registrable domain, normalizes/dedupes URLs, and refuses to fetch private or
loopback hosts. Requires ``requests`` (install the ``web`` extra). Returns a
list of ``{"url", "title", "text", "depth"}`` dicts for :func:`ingest.from_pages`.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from typing import List, Optional
from urllib.parse import urljoin, urlparse

_WS_RE = re.compile(r"\s+")


def _host_is_public(host: str) -> bool:
    host = (host or "").strip().lower()
    if not host or host == "localhost" or host.endswith(".localhost"):
        return False
    try:
        for info in socket.getaddrinfo(host, None):
            addr = ipaddress.ip_address(info[4][0])
            if (
                addr.is_private or addr.is_loopback or addr.is_link_local
                or addr.is_multicast or addr.is_reserved or addr.is_unspecified
            ):
                return False
        return True
    except (socket.gaierror, ValueError):
        return False


def _registrable(netloc: str) -> str:
    host = (urlparse(f"//{netloc}").hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def _in_scope(netloc: str, allowed: str) -> bool:
    n, a = _registrable(netloc), allowed
    return bool(n and a and (n == a or n.endswith("." + a)))


def _normalize(url: str) -> str:
    p = urlparse(url)
    path = p.path or "/"
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    return f"{(p.scheme or 'https').lower()}://{p.netloc.lower()}{path}"


def crawl_site(
    start_url: str,
    *,
    max_pages: int = 100,
    max_depth: int = 6,
    user_agent: str = "LibrarianBot/1.0",
    timeout: float = 10.0,
) -> List[dict]:
    try:
        import requests  # type: ignore
        from bs4 import BeautifulSoup  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Web crawling requires the 'web' extra: pip install 'librarian-ai[web]'"
        ) from exc

    parsed = urlparse(start_url if "://" in start_url else f"https://{start_url}")
    base = f"{parsed.scheme}://{parsed.netloc}"
    allowed = _registrable(parsed.netloc)
    if not _host_is_public(parsed.hostname or ""):
        return []

    seen = set()
    queue: List[tuple] = [(_normalize(base), 0)]
    pages: List[dict] = []
    headers = {"User-Agent": user_agent}

    while queue and len(pages) < max_pages:
        url, depth = queue.pop(0)
        if url in seen or depth > max_depth:
            continue
        seen.add(url)
        host = urlparse(url).hostname or ""
        if not _host_is_public(host) or not _in_scope(urlparse(url).netloc, allowed):
            continue
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            if "html" not in resp.headers.get("Content-Type", "").lower():
                continue
            soup = BeautifulSoup(resp.content, "html.parser")
        except Exception:
            continue
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            tag.decompose()
        text = _WS_RE.sub(" ", soup.get_text(" ")).strip()
        title = soup.title.get_text().strip() if soup.title else url
        if text:
            pages.append({"url": url, "title": title, "text": text[:100_000], "depth": depth})
        for link in soup.find_all("a", href=True):
            href = link["href"].strip()
            if not href or href.startswith(("mailto:", "tel:", "javascript:", "#")):
                continue
            nxt = _normalize(urljoin(url, href))
            if nxt not in seen and _in_scope(urlparse(nxt).netloc, allowed):
                queue.append((nxt, depth + 1))
    return pages
