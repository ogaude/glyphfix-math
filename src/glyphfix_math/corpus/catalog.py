"""Enriched, version-pinned paper catalog from OAI-PMH `arXivRaw` (metadata only, no e-print).

Usage:
  python scripts/corpus_catalog.py build --candidates data/candidates.jsonl ...      (CLI, see the script)
  from glyphfix_math.corpus.catalog import OaiClient, list_day, enrich, write_catalog, recheck

What it does: re-harvests the OAI days that built data/candidates.jsonl, this time in the `arXivRaw`
format that lists every version and the license of the latest one. Per paper the pinned version is the last
version listed and the license is mapped by exact URI to an SPDX id; anything else is excluded with reason
`license`. Authors and title come from the frozen candidate row (arXivRaw authors are kept raw).
Catalog rows of benchmark papers and of all other papers are written to different files, so no
benchmark id ever reaches a training-side file.

Math-category definition: the OAI `math` set, which on this data equals "any category starting with math."
(2,692 both ways), cross-lists included; primary_category is the first category listed.

Catalog row keys: arxiv_id (base id), version, n_versions, license_uri, license_spdx, license_checked_at,
authors, authors_raw, title, categories, primary_category, set, split, quarantined, catalog_version.

Rules: at most one request per 3.1 s (a smaller interval raises ValueError), 503 Retry-After honored,
error pages never cached, every OAI id passes ids.is_arxiv_id before use.

Known limit: the license is the record's current license, which arXiv applies to the latest version;
a later version posted after the catalog was built is only noticed by recheck().
"""

from __future__ import annotations

import datetime as dt
import math
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable
from pathlib import Path

from glyphfix_math import splits
from glyphfix_math.corpus._io import write_jsonl
from glyphfix_math.corpus.records import Exclusion, exclusion_row
from glyphfix_math.ids import base_id, is_arxiv_id, safe_id

BASE = "https://oaipmh.arxiv.org/oai"
UA = "glyphfix-math-research (contact@example.invalid)"
NS = {"o": "http://www.openarchives.org/OAI/2.0/", "r": "http://arxiv.org/OAI/arXivRaw/"}
OK_LICENSES = {
    "http://creativecommons.org/licenses/by/4.0/": "CC-BY-4.0",
    "http://creativecommons.org/licenses/by/3.0/": "CC-BY-3.0",
    "http://creativecommons.org/publicdomain/zero/1.0/": "CC0-1.0",
}
CATALOG_VERSION = "1"
MIN_INTERVAL_S = 3.1
ALLOWED_HOST = urllib.parse.urlparse(BASE).hostname
MAX_REDIRECTS = 3
_REDIRECT_CODES = (301, 302, 303, 307, 308)
_DAY_RE = re.compile(r"^math_(\d{4}-\d{2}-\d{2})_\d+\.xml$")


class OaiError(RuntimeError):
    """An OAI-PMH error response (other than noRecordsMatch) or an unusable body."""


def _check_page(data: bytes) -> ET.Element:
    try:
        root = ET.fromstring(data)
    except ET.ParseError as e:
        raise OaiError(f"unparseable OAI response: {e}") from e
    err = root.find("o:error", NS)
    if err is not None:
        code = err.get("code", "")
        if code != "noRecordsMatch":
            raise OaiError(f"OAI error {code}: {(err.text or '').strip()}")
    return root


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):  # noqa: ARG002
        return None


_NO_REDIRECT_OPENER = urllib.request.build_opener(_NoRedirect)


def default_open(req, timeout=None):
    """urlopen without automatic redirects: a 3xx surfaces as HTTPError so the client can pace and validate it."""
    return _NO_REDIRECT_OPENER.open(req, timeout=timeout)


def _check_redirect(base_url: str, err: urllib.error.HTTPError) -> str:
    """Target of a 3xx answer; it must be https on ALLOWED_HOST (the OAI host), else OaiError."""
    loc = err.headers.get("Location") if err.headers else None
    if not loc:
        raise OaiError(f"redirect {err.code} without Location from {base_url}") from err
    target = urllib.parse.urljoin(base_url, loc)
    parsed = urllib.parse.urlparse(target)
    if parsed.scheme != "https" or parsed.hostname != ALLOWED_HOST:
        raise OaiError(f"redirect from {base_url} to {target} leaves https://{ALLOWED_HOST}")
    return target


def retry_after_seconds(headers, *, default: float = 30.0, cap: float = 600.0) -> float:
    """Retry-After as seconds; accepts delta-seconds and an HTTP-date; anything else gives `default`; clamped to [0, cap]."""
    import email.utils

    raw = headers.get("Retry-After") if headers else None
    if raw is None:
        return min(max(float(default), 0.0), cap)
    try:
        v = float(raw)
        if not math.isfinite(v):
            v = float(default)
    except (TypeError, ValueError):
        try:
            when = email.utils.parsedate_to_datetime(str(raw))
            v = (when - dt.datetime.now(when.tzinfo or dt.UTC)).total_seconds()
        except (TypeError, ValueError, IndexError):
            v = float(default)
    return min(max(v, 0.0), cap)


class OaiClient:
    """Paced, cached OAI-PMH GET. One request in flight, never closer than min_interval_s."""

    def __init__(
        self,
        cache_dir: Path | str,
        min_interval_s: float = MIN_INTERVAL_S,
        opener: Callable = default_open,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if min_interval_s < MIN_INTERVAL_S:
            raise ValueError(f"min_interval_s must be at least {MIN_INTERVAL_S} s (arXiv politeness), got {min_interval_s}")
        self.cache_dir = Path(cache_dir)
        self.min_interval_s = min_interval_s
        self._opener = opener
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None
        self.requests = 0

    def _pace(self) -> None:
        if self._last is not None:
            wait = self.min_interval_s - (self._clock() - self._last)
            if wait > 0:
                self._sleep(wait)
        self._last = self._clock()

    def get(self, params: dict, key: str) -> bytes:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        f = self.cache_dir / f"{key}.xml"
        if f.exists():
            data = f.read_bytes()
            try:
                _check_page(data)
                return data
            except OaiError:
                f.unlink()
        url = BASE + "?" + urllib.parse.urlencode(params)
        hops = 0
        for _attempt in range(5 + MAX_REDIRECTS):
            self._pace()  # every request, redirect hops included, goes through the limiter
            self.requests += 1
            try:
                req = urllib.request.Request(url, headers={"User-Agent": UA})
                data = self._opener(req, timeout=90).read()
            except urllib.error.HTTPError as e:
                if e.code in _REDIRECT_CODES:
                    if hops >= MAX_REDIRECTS:
                        e.close()
                        raise OaiError(f"more than {MAX_REDIRECTS} redirects for {key}") from e
                    hops += 1
                    try:
                        url = _check_redirect(url, e)
                    finally:
                        e.close()
                    continue
                if e.code == 503:
                    self._sleep(retry_after_seconds(e.headers))
                    continue
                raise
            _check_page(data)
            tmp = f.with_suffix(".xml.tmp")
            tmp.write_bytes(data)
            os.replace(tmp, f)
            return data
        raise OaiError(f"gave up on {key} after repeated 503")


def parse_arxivraw(xml_bytes: bytes) -> list[dict]:
    root = _check_page(xml_bytes)
    out = []
    for rec in root.iterfind(".//o:record", NS):
        header = rec.find("o:header", NS)
        if header is not None and header.get("status") == "deleted":
            continue
        md = rec.find(".//r:arXivRaw", NS)
        if md is None:
            continue
        arxiv_id = (md.findtext("r:id", "", NS) or "").strip()
        versions = [v.get("version", "") for v in md.iterfind("r:version", NS)]
        if not is_arxiv_id(arxiv_id) or not versions or not all(re.fullmatch(r"v\d+", v) for v in versions):
            continue
        out.append(
            {
                "id": arxiv_id,
                "versions": versions,
                "license": (md.findtext("r:license", "", NS) or "").strip(),
                "title": " ".join((md.findtext("r:title", "", NS) or "").split()),
                "authors_raw": " ".join((md.findtext("r:authors", "", NS) or "").split()),
                "categories": (md.findtext("r:categories", "", NS) or "").split(),
            }
        )
    return out


def _resumption_token(root: ET.Element) -> str:
    tok = root.find(".//o:resumptionToken", NS)
    return (tok.text or "").strip() if tok is not None else ""


def list_day(client: OaiClient, day: dt.date) -> list[dict]:
    params = {"verb": "ListRecords", "metadataPrefix": "arXivRaw", "set": "math", "from": day.isoformat(), "until": day.isoformat()}
    out: list[dict] = []
    page = 0
    while True:
        data = client.get(params, f"raw_math_{day}_{page}")
        out += parse_arxivraw(data)
        tok = _resumption_token(_check_page(data))
        if not tok:
            return out
        page += 1
        params = {"verb": "ListRecords", "resumptionToken": tok}


def get_record(client: OaiClient, arxiv_id: str) -> dict | None:
    if not is_arxiv_id(arxiv_id):
        raise ValueError(f"not an arXiv id: {arxiv_id!r}")
    params = {"verb": "GetRecord", "identifier": f"oai:arXiv.org:{base_id(arxiv_id)}", "metadataPrefix": "arXivRaw"}
    try:
        data = client.get(params, f"raw_get_{safe_id(arxiv_id)}")
    except OaiError as e:
        if "idDoesNotExist" in str(e):
            return None
        raise
    recs = parse_arxivraw(data)
    return recs[0] if recs else None


def math_days(cache_dir: Path | str) -> list[dt.date]:
    days = set()
    for p in Path(cache_dir).glob("math_*.xml"):
        m = _DAY_RE.match(p.name)
        if m:
            days.add(dt.date.fromisoformat(m.group(1)))
    return sorted(days)


def split_raw_authors(s: str) -> list[str]:
    return [a.strip() for a in re.split(r",\s*|\s+and\s+", s) if a.strip()]


def enrich(
    candidate_rows: Iterable[dict],
    raw_by_id: dict[str, dict],
    *,
    checked_at: str,
    fs: splits.FrozenSplit | None = None,
) -> tuple[list[dict], list[dict]]:
    """Return (catalog rows, license-excluded rows). Candidates absent from raw_by_id are skipped (the caller reports them)."""
    fs = fs or splits.frozen()
    rows, excluded = [], []
    for cand in candidate_rows:
        aid = base_id(cand["arxiv_id"])
        raw = raw_by_id.get(aid)
        if raw is None:
            continue
        split = splits.split_of(aid, fs)
        spdx = OK_LICENSES.get(raw["license"])
        if spdx is None:
            excluded.append(exclusion_row(aid, Exclusion.LICENSE, license_uri=raw["license"], split=split))
            continue
        cats = raw["categories"] or list(cand.get("categories", []))
        rows.append(
            {
                "arxiv_id": aid,
                "version": raw["versions"][-1],
                "n_versions": len(raw["versions"]),
                "license_uri": raw["license"],
                "license_spdx": spdx,
                "license_checked_at": checked_at,
                "authors": list(cand["authors"]),
                "authors_raw": raw["authors_raw"],
                "title": cand["title"],
                "categories": cats,
                "primary_category": cats[0] if cats else "",
                "set": cand.get("set", "math"),
                "split": split,
                "quarantined": aid in fs.quarantine,
                "catalog_version": CATALOG_VERSION,
            }
        )
    rows.sort(key=lambda r: r["arxiv_id"])
    excluded.sort(key=lambda r: r["arxiv_id"])
    return rows, excluded


def write_catalog(rows: Iterable[dict], train_dev_path: Path | str, bench_path: Path | str) -> tuple[int, int]:
    """Benchmark rows only to bench_path, every other row only to train_dev_path; returns (n_train_dev, n_benchmark)."""
    rows = sorted(rows, key=lambda r: r["arxiv_id"])
    bench = [r for r in rows if r["split"] == "benchmark"]
    other = [r for r in rows if r["split"] != "benchmark"]
    write_jsonl(train_dev_path, other)
    write_jsonl(bench_path, bench)
    return len(other), len(bench)


def recheck(rows: Iterable[dict], client: OaiClient, days: Iterable[dt.date] | None = None) -> list[dict]:
    """Drift rows for catalog rows whose current license or last version no longer matches.

    With `days`, those OAI days are re-harvested first and GetRecord is used only for ids not found there.
    """
    rows = list(rows)
    current: dict[str, dict] = {}
    for day in days or []:
        for rec in list_day(client, day):
            current[base_id(rec["id"])] = rec
    out = []
    for row in rows:
        aid = row["arxiv_id"]
        rec = current.get(aid) or get_record(client, aid)
        if rec is None:
            out.append(exclusion_row(aid, Exclusion.VERSION_DRIFT, catalog_version=row["version"], current_version=None))
        elif rec["license"] != row["license_uri"]:
            out.append(exclusion_row(aid, Exclusion.LICENSE, catalog_license=row["license_uri"], current_license=rec["license"]))
        elif rec["versions"][-1] != row["version"]:
            out.append(exclusion_row(aid, Exclusion.VERSION_DRIFT, catalog_version=row["version"], current_version=rec["versions"][-1]))
    return out
