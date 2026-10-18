"""Step 1 of the 1,000-paper corpus: candidate arXiv papers whose license allows redistribution.

Harvests arXiv metadata over OAI-PMH (https://oaipmh.arxiv.org/oai, format `arXiv`, which carries the license),
keeps only CC BY 4.0/3.0 and CC0 / public-domain papers, drops the ids listed in
data/candidates_excluded_ids.txt (the 3 papers the frozen candidate population was built without) and, if a local
data/external/arxiv_src exists, the papers in it (legacy data whose licenses were never checked; the directory was
deleted on 2026-10-17), and writes data/candidates.jsonl. Nothing but metadata is downloaded.

Sampling: one day per month over the last months, for each set in SETS. OAI datestamp = last modification,
so a day mixes new and revised papers; that is fine, the license is the one of the latest version.
Requests are spaced 3.1 s apart (arXiv asks for at most one request every 3 s); responses are cached
in data/candidates_cache/ so a rerun costs nothing.

Usage (from repo root):  python3 scripts/build_candidates.py [--months 24] [--day 10] [--asof YYYY-MM-DD]
[--cache data/candidates_cache] [--out data/candidates.jsonl]
"""

from __future__ import annotations

import argparse
import calendar
import datetime as dt
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from lightonocr_math.corpus.catalog import retry_after_seconds

BASE = "https://oaipmh.arxiv.org/oai"
CONTACT_ENV = "LIGHTONOCR_MATH_CONTACT"
# arXiv asks for a contact address in the User-Agent: set LIGHTONOCR_MATH_CONTACT to your own before harvesting.
UA = f"lightonocr-math-research ({os.environ.get(CONTACT_ENV) or 'contact@example.invalid'})"
SETS = ["math", "stat", "physics", "cs", "eess", "q-bio"]
OK_LICENSES = {
    "http://creativecommons.org/licenses/by/4.0/": "CC-BY-4.0",
    "http://creativecommons.org/licenses/by/3.0/": "CC-BY-3.0",
    "http://creativecommons.org/publicdomain/zero/1.0/": "CC0-1.0",
}
NS = {"o": "http://www.openarchives.org/OAI/2.0/", "a": "http://arxiv.org/OAI/arXiv/"}
CACHE = Path("data/candidates_cache")
# ids the candidate population was frozen without (CC BY papers whose sources sat in the legacy data/external/arxiv_src)
EXCLUDED_IDS_FILE = Path(__file__).resolve().parents[1] / "data" / "candidates_excluded_ids.txt"
_last = [0.0]


class OaiError(RuntimeError):
    """An OAI-PMH error response (other than noRecordsMatch) or an unusable body."""


def _check_page(data: bytes) -> ET.Element:
    """Parse one OAI page. Only noRecordsMatch counts as a valid empty page; any other error raises."""
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


def fetch(params: dict, key: str) -> bytes:
    f = CACHE / f"{key}.xml"
    if f.exists():
        data = f.read_bytes()
        try:
            _check_page(data)
            return data
        except OaiError:
            f.unlink()  # a corrupt or error page in the cache is a miss
    for attempt in range(5):
        wait = 3.1 - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        try:
            req = urllib.request.Request(BASE + "?" + urllib.parse.urlencode(params), headers={"User-Agent": UA})
            data = urllib.request.urlopen(req, timeout=90).read()
        except urllib.error.HTTPError as e:
            if e.code == 503:  # arXiv asks to retry after a delay
                time.sleep(retry_after_seconds(e.headers))
                continue
            raise
        _check_page(data)  # never cache an error or truncated response
        tmp = f.with_suffix(".xml.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, f)
        return data
    raise RuntimeError(f"gave up on {key}")


def _resumption_token(root: ET.Element) -> str:
    tok = root.find(".//o:resumptionToken", NS)
    return (tok.text or "").strip() if tok is not None else ""


def _cached_chain_complete(set_: str, day: dt.date) -> bool:
    """False when the cached pages of a set/day end in a resumption token whose next page is not cached."""
    page = 0
    while True:
        f = CACHE / f"{set_}_{day}_{page}.xml"
        if not f.exists():
            return page == 0
        try:
            root = _check_page(f.read_bytes())
        except OaiError:
            return False
        if not _resumption_token(root):
            return True
        page += 1


def harvest(set_: str, day: dt.date):
    if not _cached_chain_complete(set_, day):
        for stale in CACHE.glob(f"{set_}_{day}_*.xml"):
            stale.unlink()
    params = {"verb": "ListRecords", "metadataPrefix": "arXiv", "set": set_, "from": day.isoformat(), "until": day.isoformat()}
    page = 0
    while True:
        root = _check_page(fetch(params, f"{set_}_{day}_{page}"))
        if root.find("o:error", NS) is not None:  # noRecordsMatch: genuinely empty
            return
        for rec in root.iterfind(".//o:record", NS):
            md = rec.find(".//a:arXiv", NS)
            if md is None:
                continue

            def g(t: str, md=md) -> str:
                return (md.findtext(f"a:{t}", default="", namespaces=NS) or "").strip()

            authors = [" ".join(filter(None, [a.findtext("a:forenames", "", NS), a.findtext("a:keyname", "", NS)])).strip()
                       for a in md.iterfind("a:authors/a:author", NS)]
            yield {"arxiv_id": g("id"), "title": " ".join(g("title").split()), "authors": authors, "categories": g("categories").split(),
                   "created": g("created"), "updated": g("updated"), "license_url": g("license")}
        tok = _resumption_token(root)
        if not tok:
            return
        page += 1
        params = {"verb": "ListRecords", "resumptionToken": tok}


def sample_days(asof: dt.date, months: int, day: int) -> list[dt.date]:
    """One day per month going back from ``asof``; the day is clamped to the month length, then moved to a weekday."""
    first = asof.replace(day=1)
    days = []
    for m in range(1, months + 1):
        y, mo = divmod(first.year * 12 + first.month - 1 - m, 12)
        d = dt.date(y, mo + 1, min(day, calendar.monthrange(y, mo + 1)[1]))
        while d.weekday() >= 5:
            d += dt.timedelta(days=1)
        days.append(d)
    return days


def main(argv: list[str] | None = None) -> None:
    global CACHE
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=24)
    ap.add_argument("--day", type=int, default=10, help="day of month sampled (clamped to the month, moved to the next weekday)")
    ap.add_argument("--asof", type=dt.date.fromisoformat, default=None, help="reference date, default today")
    ap.add_argument("--cache", default="data/candidates_cache")
    ap.add_argument("--out", default="data/candidates.jsonl")
    args = ap.parse_args(argv)
    CACHE = Path(args.cache)
    CACHE.mkdir(parents=True, exist_ok=True)
    legacy = Path("data/external/arxiv_src")
    known = {x.strip() for x in EXCLUDED_IDS_FILE.read_text().splitlines() if x.strip()}
    if legacy.is_dir():
        known |= {p.name.replace("_", "/", 1) if not p.name[0].isdigit() else p.name for p in legacy.iterdir()}
        known |= {p.name for p in legacy.iterdir()}
    days = sample_days(args.asof or dt.date.today(), args.months, args.day)
    seen: dict[str, dict] = {}
    total = Counter()
    for d in days:
        for s in SETS:
            n = 0
            for r in harvest(s, d):
                n += 1
                total["records"] += 1
                total["lic:" + (OK_LICENSES.get(r["license_url"]) or ("other" if r["license_url"] else "none"))] += 1
                lic = OK_LICENSES.get(r["license_url"])
                if not lic or r["arxiv_id"] in known:
                    continue
                r["license"] = lic
                r["set"] = s
                seen.setdefault(r["arxiv_id"], r)
            print(f"{d} {s:8} {n:5} records, candidates so far {len(seen)}", flush=True)
    out = Path(args.out)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w") as fh:
        for r in sorted(seen.values(), key=lambda r: r["arxiv_id"]):
            fh.write(json.dumps(r) + "\n")
    os.replace(tmp, out)
    cats = Counter(c.split(".")[0] for r in seen.values() for c in r["categories"][:1])
    print(json.dumps({"records_seen": total["records"], "license_breakdown": {k: v for k, v in total.items() if k.startswith("lic:")},
                      "candidates": len(seen), "by_primary_category": cats.most_common(12)}, indent=2))


if __name__ == "__main__":
    main()
