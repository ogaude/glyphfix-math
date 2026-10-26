"""Offline tests for the arXivRaw catalog: fixture XML and a fake opener, no network."""

import datetime as dt
import itertools
import urllib.error
import urllib.request
from email.message import Message
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from glyphfix_math import splits
from glyphfix_math.corpus import catalog as cat
from scripts import corpus_catalog as cli

FIX = Path(__file__).parent / "fixtures" / "oai"
DAY = dt.date(2025, 1, 10)


class _Resp:
    def __init__(self, data: bytes):
        self._d = data

    def read(self) -> bytes:
        return self._d


class FakeNet:
    def __init__(self, listrecords=None, getrecord=None, fail503=0):
        self.urls: list[str] = []
        self.list = listrecords or (FIX / "arxivraw_listrecords.xml").read_bytes()
        self.get = getrecord or (FIX / "arxivraw_getrecord.xml").read_bytes()
        self.fail503 = fail503

    def __call__(self, req, timeout=None):
        self.urls.append(req.full_url)
        if self.fail503 > 0:
            self.fail503 -= 1
            h = Message()
            h["Retry-After"] = "7"
            raise urllib.error.HTTPError(req.full_url, 503, "busy", h, None)
        q = parse_qs(urlparse(req.full_url).query)
        return _Resp(self.get if q.get("verb") == ["GetRecord"] else self.list)


class Clock:
    def __init__(self):
        self.t = 100.0
        self.sleeps: list[float] = []

    def now(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


def _client(tmp_path, net, clock=None):
    clock = clock or Clock()
    return cat.OaiClient(tmp_path / "cache", opener=net, clock=clock.now, sleep=clock.sleep), clock


def test_parse_arxivraw_skips_deleted_and_reads_versions():
    recs = cat.parse_arxivraw((FIX / "arxivraw_listrecords.xml").read_bytes())
    assert [r["id"] for r in recs] == ["9912.00001", "9912.00002"]
    r = recs[0]
    assert r["versions"] == ["v1", "v2", "v3"]
    assert r["license"] == "http://creativecommons.org/licenses/by/4.0/"
    assert r["title"] == "Fictitious paper one"
    assert r["authors_raw"] == "Jane Doe, John Roe and Max Poe"
    assert r["categories"] == ["math.CA", "math.PR"]
    assert cat.split_raw_authors(r["authors_raw"]) == ["Jane Doe", "John Roe", "Max Poe"]


def test_license_uris_exact():
    assert cat.OK_LICENSES == {
        "http://creativecommons.org/licenses/by/4.0/": "CC-BY-4.0",
        "http://creativecommons.org/licenses/by/3.0/": "CC-BY-3.0",
        "http://creativecommons.org/publicdomain/zero/1.0/": "CC0-1.0",
    }


def _cand(aid, **kw):
    return {"arxiv_id": aid, "title": f"T {aid}", "authors": ["Jane Doe"], "categories": ["math.CA"], "set": "math", "license": "CC-BY-4.0", **kw}


def test_enrich_pins_version_and_excludes_bad_license():
    raw = {r["id"]: r for r in cat.parse_arxivraw((FIX / "arxivraw_listrecords.xml").read_bytes())}
    cands = [_cand("9912.00001", license="CC0-1.0"), _cand("9912.00002"), _cand("9912.00009")]
    rows, excluded = cat.enrich(cands, raw, checked_at="2026-10-14T00:00:00Z")
    assert [r["arxiv_id"] for r in rows] == ["9912.00001"]
    r = rows[0]
    assert r["version"] == "v3" and r["n_versions"] == 3
    assert r["license_spdx"] == "CC-BY-4.0"  # arXivRaw wins over the candidate field
    assert r["authors"] == ["Jane Doe"] and r["title"] == "T 9912.00001"
    assert r["authors_raw"] == "Jane Doe, John Roe and Max Poe"
    assert r["split"] == splits.split_of("9912.00001")
    assert r["quarantined"] == ("9912.00001" in splits.frozen().quarantine)
    assert r["license_checked_at"] == "2026-10-14T00:00:00Z"
    assert [(e["arxiv_id"], e["reason"]) for e in excluded] == [("9912.00002", "license")]


def _row(aid, split, **kw):
    return {"arxiv_id": aid, "split": split, "version": "v1", "license_uri": "http://creativecommons.org/licenses/by/4.0/", **kw}


def test_write_catalog_routes_by_side_sorted(tmp_path):
    rows = [_row("2401.00003", "train"), _row("2401.00001", "benchmark"), _row("2401.00002", "dev"), _row("2401.00000", "benchmark")]
    td, b = tmp_path / "td.jsonl", tmp_path / "b" / "b.jsonl"
    assert cat.write_catalog(rows, td, b) == (2, 2)
    import json

    assert [json.loads(x)["arxiv_id"] for x in td.read_text().splitlines()] == ["2401.00002", "2401.00003"]
    assert [json.loads(x)["arxiv_id"] for x in b.read_text().splitlines()] == ["2401.00000", "2401.00001"]


def test_math_days(tmp_path):
    for n in ("math_2025-02-10_0.xml", "math_2025-01-10_0.xml", "math_2025-01-10_1.xml", "cs_2025-03-10_0.xml", "math_bad_0.xml"):
        (tmp_path / n).write_text("x")
    assert cat.math_days(tmp_path) == [dt.date(2025, 1, 10), dt.date(2025, 2, 10)]


def test_client_spacing_503_and_no_error_cache(tmp_path):
    clock = Clock()
    net = FakeNet(fail503=1)
    stamps: list[float] = []

    def stamped(req, timeout=None):
        stamps.append(clock.now())
        return net(req, timeout)

    client, _ = _client(tmp_path, stamped, clock)
    client.get({"verb": "ListRecords"}, "a")
    client.get({"verb": "ListRecords", "x": "1"}, "b")
    assert len(stamps) == 3
    assert 7 in clock.sleeps  # Retry-After honored
    assert all(b - a >= 3.1 - 1e-9 for a, b in itertools.pairwise(stamps))
    err = b'<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><error code="badArgument">no</error></OAI-PMH>'
    bad, _ = _client(tmp_path / "e", FakeNet(listrecords=err))
    with pytest.raises(cat.OaiError):
        bad.get({"verb": "ListRecords"}, "k")
    assert not (tmp_path / "e" / "cache" / "k.xml").exists()


def test_client_interval_floor(tmp_path):
    with pytest.raises(ValueError):
        cat.OaiClient(tmp_path, min_interval_s=3.0)


def test_list_day_and_get_record(tmp_path):
    net = FakeNet()
    client, _ = _client(tmp_path, net)
    assert [r["id"] for r in cat.list_day(client, DAY)] == ["9912.00001", "9912.00002"]
    rec = cat.get_record(client, "9912.00004")
    assert rec["versions"] == ["v1", "v2"]
    n = len(net.urls)
    cat.list_day(client, DAY)  # cached
    assert len(net.urls) == n


def test_get_record_unknown_id_returns_none(tmp_path):
    err = b'<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><error code="idDoesNotExist">no</error></OAI-PMH>'
    client, _ = _client(tmp_path, FakeNet(getrecord=err))
    assert cat.get_record(client, "9912.00004") is None


def test_recheck_reports_version_and_license_drift(tmp_path):
    client, _ = _client(tmp_path, FakeNet())
    ok = {"arxiv_id": "9912.00001", "version": "v3", "license_uri": "http://creativecommons.org/licenses/by/4.0/"}
    old_ver = {"arxiv_id": "9912.00004", "version": "v1", "license_uri": "http://creativecommons.org/licenses/by/3.0/"}
    old_lic = {"arxiv_id": "9912.00002", "version": "v1", "license_uri": "http://creativecommons.org/licenses/by/4.0/"}
    out = cat.recheck([ok, old_ver, old_lic], client, [DAY])
    assert {(d["arxiv_id"], d["reason"]) for d in out} == {("9912.00004", "version_drift"), ("9912.00002", "license")}


def test_cli_rejects_interval_below_floor(tmp_path):
    rc = cli.main(["build", "--candidates", str(tmp_path / "c.jsonl"), "--interval", "3.0"])
    assert rc == 2
    rc = cli.main(["recheck", "--catalog", str(tmp_path / "c.jsonl"), "--out", str(tmp_path / "o"), "--interval", "3.0"])
    assert rc == 2


def test_cli_build_end_to_end_offline(tmp_path, monkeypatch):
    import json

    cache_days = tmp_path / "days"
    cache_days.mkdir()
    (cache_days / "math_2025-01-10_0.xml").write_text("x")
    cands = tmp_path / "c.jsonl"
    cands.write_text("".join(json.dumps(c) + "\n" for c in [_cand("9912.00001"), _cand("9912.00002"), _cand("9912.00004"), _cand("9912.00099", set="cs")]))
    net = FakeNet()
    clock = Clock()
    real = cat.OaiClient
    monkeypatch.setattr(cat, "OaiClient", lambda c, min_interval_s: real(c, min_interval_s, opener=net, clock=clock.now, sleep=clock.sleep))
    args = [
        "build", "--candidates", str(cands), "--days-from", str(cache_days), "--cache", str(tmp_path / "cc"),
        "--out-train-dev", str(tmp_path / "td.jsonl"), "--out-benchmark", str(tmp_path / "b.jsonl"),
        "--out-benchmark-excluded", str(tmp_path / "bx.jsonl"), "--stats", str(tmp_path / "stats.json"),
    ]
    assert cli.main(args) == 0
    stats = json.loads((tmp_path / "stats.json").read_text())
    assert stats["n_math_candidates"] == 3
    assert stats["n_found_listrecords"] == 2 and stats["n_found_getrecord"] == 1
    assert stats["n_license_excluded"] == 1 and stats["n_missing"] == 0


def test_retry_after_seconds_forms():
    import datetime as dt
    import email.utils
    from email.message import Message

    from glyphfix_math.corpus import catalog as C

    def hdr(v=None):
        m = Message()
        if v is not None:
            m["Retry-After"] = v
        return m

    assert C.retry_after_seconds(hdr("7")) == 7.0
    assert C.retry_after_seconds(hdr()) == 30.0
    assert C.retry_after_seconds(hdr("99999")) == 600.0
    assert C.retry_after_seconds(hdr("garbage"), default=0.0) == 0.0
    assert C.retry_after_seconds(hdr("nan")) == 30.0
    assert C.retry_after_seconds(hdr("inf")) == 30.0
    future = email.utils.format_datetime(dt.datetime.now(dt.UTC) + dt.timedelta(seconds=120))
    assert 100 <= C.retry_after_seconds(hdr(future)) <= 120
    past = email.utils.format_datetime(dt.datetime.now(dt.UTC) - dt.timedelta(seconds=120))
    assert C.retry_after_seconds(hdr(past)) == 0.0


class RedirNet:
    """Answers each call from a script: ('redir', code, location) raises a 3xx, ('ok',) returns the list page."""

    def __init__(self, script):
        self.script = list(script)
        self.urls: list[str] = []

    def __call__(self, req, timeout=None):
        self.urls.append(req.full_url)
        step = self.script.pop(0)
        if step[0] == "redir":
            h = Message()
            if step[2] is not None:
                h["Location"] = step[2]
            raise urllib.error.HTTPError(req.full_url, step[1], "moved", h, None)
        return _Resp((FIX / "arxivraw_listrecords.xml").read_bytes())


def test_default_opener_does_not_follow_redirects():
    assert cat.OaiClient(Path("x"))._opener is cat.default_open
    h = Message()
    h["Location"] = "https://oaipmh.arxiv.org/elsewhere"
    req = urllib.request.Request("https://oaipmh.arxiv.org/oai")
    assert cat._NoRedirect().redirect_request(req, None, 302, "Found", h, "https://oaipmh.arxiv.org/elsewhere") is None


def test_redirect_hop_is_paced_and_validated(tmp_path):
    net = RedirNet([("redir", 301, "/oai2?verb=x"), ("ok",)])
    client, clock = _client(tmp_path, net)
    client.get({"verb": "ListRecords"}, "k")
    assert net.urls[1] == "https://oaipmh.arxiv.org/oai2?verb=x"
    assert client.requests == 2
    assert clock.sleeps and clock.sleeps[0] == pytest.approx(cat.MIN_INTERVAL_S)  # the hop waited for the limiter


@pytest.mark.parametrize("loc", ["http://oaipmh.arxiv.org/oai", "https://evil.example.com/oai", "https://oaipmh.arxiv.org.evil.com/", None])
def test_redirect_off_host_or_http_or_missing_location_refused(tmp_path, loc):
    net = RedirNet([("redir", 302, loc), ("ok",)])
    client, _ = _client(tmp_path, net)
    with pytest.raises(cat.OaiError):
        client.get({"verb": "ListRecords"}, "k")
    assert len(net.urls) == 1  # the bad target is never requested


def test_redirect_hops_capped(tmp_path):
    net = RedirNet([("redir", 302, "/oai")] * (cat.MAX_REDIRECTS + 1) + [("ok",)])
    client, _ = _client(tmp_path, net)
    with pytest.raises(cat.OaiError, match="redirects"):
        client.get({"verb": "ListRecords"}, "k")
    assert len(net.urls) == cat.MAX_REDIRECTS + 1
