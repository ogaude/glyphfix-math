"""Step 1 of the 1,000-paper corpus: candidate arXiv papers whose license allows redistribution.

Harvests arXiv metadata over OAI-PMH (https://oaipmh.arxiv.org/oai, format `arXiv`, which carries the license),
keeps only CC BY 4.0/3.0 and CC0 / public-domain papers, drops papers already in data/external/arxiv_src
(their licenses were never checked), and writes data/candidates.jsonl. Nothing but metadata is downloaded.

Sampling: one day per month over the last months, for each set in SETS. OAI datestamp = last modification,
so a day mixes new and revised papers; that is fine, the license is the one of the latest version.
Requests are spaced 3.1 s apart (arXiv asks for at most one request every 3 s); responses are cached
in data/candidates_cache/ so a rerun costs nothing.

Usage (from repo root):  python3 scripts/build_candidates.py [--months 24] [--day 10]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

BASE = "https://oaipmh.arxiv.org/oai"
UA = "lightonocr-math-research (contact@example.invalid)"
SETS = ["math", "stat", "physics", "cs", "eess", "q-bio"]
OK_LICENSES = {
    "http://creativecommons.org/licenses/by/4.0/": "CC-BY-4.0",
    "http://creativecommons.org/licenses/by/3.0/": "CC-BY-3.0",
    "http://creativecommons.org/publicdomain/zero/1.0/": "CC0-1.0",
}
NS = {"o": "http://www.openarchives.org/OAI/2.0/", "a": "http://arxiv.org/OAI/arXiv/"}
CACHE = Path("data/candidates_cache")
_last = [0.0]


def fetch(params: dict, key: str) -> bytes:
    f = CACHE / f"{key}.xml"
    if f.exists():
        return f.read_bytes()
    for attempt in range(5):
        wait = 3.1 - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        try:
            req = urllib.request.Request(BASE + "?" + urllib.parse.urlencode(params), headers={"User-Agent": UA})
            data = urllib.request.urlopen(req, timeout=90).read()
            f.write_bytes(data)
            return data
        except urllib.error.HTTPError as e:
            if e.code == 503:  # arXiv asks to retry after a delay
                time.sleep(int(e.headers.get("Retry-After", "30")))
                continue
            raise
    raise RuntimeError(f"gave up on {key}")


def harvest(set_: str, day: dt.date):
    params = {"verb": "ListRecords", "metadataPrefix": "arXiv", "set": set_, "from": day.isoformat(), "until": day.isoformat()}
    page = 0
    while True:
        data = fetch(params, f"{set_}_{day}_{page}")
        root = ET.fromstring(data)
        if root.find("o:error", NS) is not None:  # noRecordsMatch
            return
        for rec in root.iterfind(".//o:record", NS):
            md = rec.find(".//a:arXiv", NS)
            if md is None:
                continue
            g = lambda t: (md.findtext(f"a:{t}", default="", namespaces=NS) or "").strip()
            authors = [" ".join(filter(None, [a.findtext("a:forenames", "", NS), a.findtext("a:keyname", "", NS)])).strip()
                       for a in md.iterfind("a:authors/a:author", NS)]
            yield {"arxiv_id": g("id"), "title": " ".join(g("title").split()), "authors": authors, "categories": g("categories").split(),
                   "created": g("created"), "updated": g("updated"), "license_url": g("license")}
        tok = root.find(".//o:resumptionToken", NS)
        if tok is None or not (tok.text or "").strip():
            return
        page += 1
        params = {"verb": "ListRecords", "resumptionToken": tok.text.strip()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=24)
    ap.add_argument("--day", type=int, default=10, help="day of month sampled (moved to the next weekday)")
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    known = {p.name.replace("_", "/", 1) if not p.name[0].isdigit() else p.name for p in Path("data/external/arxiv_src").iterdir()}
    known |= {p.name for p in Path("data/external/arxiv_src").iterdir()}
    today = dt.date.today().replace(day=1)
    days = []
    for m in range(1, args.months + 1):
        y, mo = divmod(today.year * 12 + today.month - 1 - m, 12)
        d = dt.date(y, mo + 1, args.day)
        while d.weekday() >= 5:
            d += dt.timedelta(days=1)
        days.append(d)
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
    out = Path("data/candidates.jsonl")
    with out.open("w") as fh:
        for r in sorted(seen.values(), key=lambda r: r["arxiv_id"]):
            fh.write(json.dumps(r) + "\n")
    cats = Counter(c.split(".")[0] for r in seen.values() for c in r["categories"][:1])
    print(json.dumps({"records_seen": total["records"], "license_breakdown": {k: v for k, v in total.items() if k.startswith("lic:")},
                      "candidates": len(seen), "by_primary_category": cats.most_common(12)}, indent=2))


if __name__ == "__main__":
    main()
