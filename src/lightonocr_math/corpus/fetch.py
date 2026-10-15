"""Paced, version-pinned, resumable e-print fetcher.

Usage:
  from lightonocr_math.corpus import fetch
  lim = fetch.Limiter(fetch.DEFAULT_INTERVAL_S)
  meta = fetch.fetch_one(catalog_row, Path("data/sources"), limiter=lim, policy_ack="<decision text>")
  order = fetch.fetch_order(ids, seed=0)

Rules:
  * one process, one request in flight; the limiter runs before every request, retries included
  * the interval is clamped to MIN_INTERVAL_S (3.1 s); the default is the 3.1 s minimum (user decision
    2026-10-14: export.arxiv.org per info.arxiv.org/help/bulk_data.html). robots.txt disallows crawlers
    there, so a caller must pass an explicit policy_ack naming the decision to proceed; it is stored in every fetch.json
  * each bundle is requested as /e-print/<id>vN with the catalog-pinned version
  * 403 stops the stage (FetchForbidden); 429/503 honor Retry-After (capped 600 s), up to 3 retries
  * a bundle is stored as src.bin (sha256 recorded in fetch.json); an intact pair is never refetched

Redirects: urllib's automatic redirect is disabled. arXiv answers /e-print/<id>vN with a 3xx to /src/<id>vN; each hop
is a separate request that goes through the limiter, and its target must be https on ALLOWED_HOST (else FetchForbidden).
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from lightonocr_math.corpus import catalog
from lightonocr_math.corpus._io import read_jsonl, sha256_file, write_json
from lightonocr_math.ids import base_id, safe_id

MIN_INTERVAL_S = 3.1
DEFAULT_INTERVAL_S = 3.1
MAX_BUNDLE_BYTES = 200 * 1024 * 1024
MAX_RETRIES = 3
MAX_CONSECUTIVE_FAILURES = 5
MAX_RETRY_AFTER_S = 600.0
FETCH_VERSION = "1"
ALLOWED_HOST = "export.arxiv.org"
EPRINT_BASE = f"https://{ALLOWED_HOST}/e-print/"
MAX_REDIRECTS = 3
_REDIRECT_CODES = (301, 302, 303, 307, 308)
_CHUNK = 1 << 16


class FetchForbidden(RuntimeError):
    """arXiv answered 403: further requests would count as abuse, so the stage stops."""


class FetchAborted(RuntimeError):
    """Too many consecutive failures."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):  # noqa: ARG002
        return None


_NO_REDIRECT_OPENER = urllib.request.build_opener(_NoRedirect)


def default_open(req, timeout=None):
    """urlopen without automatic redirects: a 3xx surfaces as HTTPError so the caller can pace and validate it."""
    return _NO_REDIRECT_OPENER.open(req, timeout=timeout)


def _check_redirect(base_url: str, err: urllib.error.HTTPError) -> str:
    loc = err.headers.get("Location") if err.headers else None
    if not loc:
        raise FetchForbidden(f"redirect {err.code} without Location from {base_url}") from err
    target = urllib.parse.urljoin(base_url, loc)
    parsed = urllib.parse.urlparse(target)
    if parsed.scheme != "https" or parsed.hostname != ALLOWED_HOST:
        raise FetchForbidden(f"redirect from {base_url} to {target} leaves https://{ALLOWED_HOST}")
    return target


class Limiter:
    def __init__(self, min_interval_s: float, clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep):
        self.min = max(float(min_interval_s), MIN_INTERVAL_S)
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None

    def wait(self) -> None:
        if self._last is not None:
            gap = self._clock() - self._last
            if gap < self.min:
                self._sleep(self.min - gap)
        self._last = self._clock()


def eprint_url(arxiv_id: str, version: str) -> str:
    return f"{EPRINT_BASE}{base_id(arxiv_id)}{version}"


def sniff_kind(head: bytes) -> str:
    if head[:2] == b"\x1f\x8b":
        try:
            inner = zlib.decompressobj(31).decompress(head, 512)
        except zlib.error:
            inner = b""
        return "tar.gz" if inner[257:262] == b"ustar" else "gz"
    if head[:4] == b"%PDF":
        return "pdf"
    stripped = head.lstrip()
    if stripped[:1] == b"<" or stripped[:9].lower() == b"<!doctype":
        return "html"
    if head[257:262] == b"ustar":
        return "tar"
    return "unknown"


def fetch_order(ids: list[str], seed: int) -> list[str]:
    bases = sorted({base_id(i) for i in ids})
    return sorted(bases, key=lambda b: hashlib.sha256(f"{seed}|{b}".encode()).hexdigest())


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _retry_after(err: urllib.error.HTTPError) -> float:
    try:
        v = float(err.headers.get("Retry-After", "")) if err.headers else 0.0
    except (TypeError, ValueError):
        v = 0.0
    return min(max(v, 0.0), MAX_RETRY_AFTER_S)


def _intact(dir_: Path) -> dict | None:
    import json

    meta_p, bin_p = dir_ / "fetch.json", dir_ / "src.bin"
    if not meta_p.exists():
        return None
    try:
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
    except ValueError:
        return None
    if meta.get("status") != "ok":
        # missing / too_large are terminal answers; no src.bin exists
        return meta if meta.get("status") in ("missing", "too_large") and not bin_p.exists() else None
    if bin_p.exists() and sha256_file(bin_p) == meta.get("sha256"):
        return meta
    return None


def fetch_one(
    row: dict,
    sources_dir: Path,
    *,
    limiter: Limiter,
    policy_ack: str,
    opener: Callable = default_open,
    now: Callable[[], str] = _utc_now,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    arxiv_id, version = row["arxiv_id"], row["version"]
    out = Path(sources_dir) / safe_id(arxiv_id)
    done = _intact(out)
    if done is not None:
        return done
    out.mkdir(parents=True, exist_ok=True)
    url = eprint_url(arxiv_id, version)
    meta = {
        "arxiv_id": base_id(arxiv_id), "version": version, "url": url, "url_final": url, "http_status": None,
        "content_type": "", "bytes": 0, "sha256": "", "kind": "none", "status": "missing", "fetched_at": "",
        "interval_s": limiter.min, "policy_ack": policy_ack, "license_uri": row.get("license_uri"),
        "license_spdx": row.get("license_spdx"), "license_checked_at": row.get("license_checked_at"),
        "fetch_version": FETCH_VERSION,
    }
    attempt = 0
    hops = 0
    while True:
        limiter.wait()
        req = urllib.request.Request(url, headers={"User-Agent": catalog.UA})
        try:
            resp = opener(req, timeout=120)
        except urllib.error.HTTPError as e:
            if e.code in _REDIRECT_CODES:
                if hops >= MAX_REDIRECTS:
                    raise FetchForbidden(f"more than {MAX_REDIRECTS} redirects from {meta['url']}") from e
                hops += 1
                url = _check_redirect(url, e)
                meta["url_final"] = url
                continue  # the next iteration paces the redirect hop through the limiter
            if e.code == 403:
                raise FetchForbidden(f"403 for {url}") from e
            if e.code == 404:
                meta.update(http_status=404, fetched_at=now())
                write_json(out / "fetch.json", meta)
                return meta
            if e.code in (429, 503) and attempt < MAX_RETRIES:
                attempt += 1
                sleep(_retry_after(e))
                continue
            raise
        break
    fd, tmp = tempfile.mkstemp(dir=out, prefix="src.bin.", suffix=".tmp")
    h = hashlib.sha256()
    total = 0
    head = b""
    too_large = False
    try:
        with os.fdopen(fd, "wb") as fh, resp:
            while True:
                chunk = resp.read(_CHUNK)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_BUNDLE_BYTES:
                    too_large = True
                    break
                if len(head) < 1024:
                    head += chunk[: 1024 - len(head)]
                h.update(chunk)
                fh.write(chunk)
        status_code = getattr(resp, "status", 200)
        final = resp.geturl() if hasattr(resp, "geturl") else url
        ctype = ""
        if getattr(resp, "headers", None) is not None:
            ctype = resp.headers.get("Content-Type", "") or ""
        if too_large:
            os.unlink(tmp)
            meta.update(http_status=status_code, url_final=final, content_type=ctype, kind="too_large", status="too_large", fetched_at=now())
            write_json(out / "fetch.json", meta)
            return meta
        os.replace(tmp, out / "src.bin")
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    meta.update(
        http_status=status_code, url_final=final, content_type=ctype, bytes=total, sha256=h.hexdigest(),
        kind=sniff_kind(head), status="ok", fetched_at=now(),
    )
    write_json(out / "fetch.json", meta)
    return meta


def load_catalog_rows(paths_: list[str | Path]) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for p in paths_:
        for r in read_jsonl(p):
            rows[base_id(r["arxiv_id"])] = r
    return rows
