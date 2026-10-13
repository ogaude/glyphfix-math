"""Offline tests for the OAI harvester: fixture XML, mocked urlopen, no network."""

import datetime as dt
import hashlib
import shutil
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from scripts import build_candidates as bc

FIX = Path(__file__).parent / "fixtures" / "oai"
ROOT = Path(__file__).resolve().parents[1]
DAY = dt.date(2025, 1, 10)


class _Resp:
    def __init__(self, data: bytes):
        self._d = data

    def read(self) -> bytes:
        return self._d


class FakeNet:
    """urlopen stand-in: routes by request URL, records every URL."""

    def __init__(self, first: str = "page0_token.xml", raw: bytes | None = None):
        self.urls: list[str] = []
        self.first = first
        self.raw = raw

    def __call__(self, req, timeout=None):
        self.urls.append(req.full_url)
        q = parse_qs(urlparse(req.full_url).query)
        if self.raw is not None:
            return _Resp(self.raw)
        if q.get("resumptionToken") == ["tok-1"]:
            return _Resp((FIX / "page1_last.xml").read_bytes())
        return _Resp((FIX / self.first).read_bytes())


@pytest.fixture
def env(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(bc, "CACHE", cache)
    monkeypatch.setattr(bc.time, "sleep", lambda s: None)
    return cache


def _install(monkeypatch, net):
    monkeypatch.setattr(bc.urllib.request, "urlopen", net)
    return net


def test_two_page_harvest_and_cached_rerun(env, monkeypatch):
    net = _install(monkeypatch, FakeNet())
    recs = list(bc.harvest("math", DAY))
    assert [r["arxiv_id"] for r in recs] == ["9912.00001", "9912.00002", "9912.00003"]
    first = recs[0]
    assert first["title"] == "Fictitious paper one"
    assert first["authors"] == ["Jane Doe"]
    assert first["categories"] == ["math.CA", "math.PR"]
    assert first["created"] == "2025-01-09" and first["updated"] == "2025-01-10"
    assert first["license_url"] == "http://creativecommons.org/licenses/by/4.0/"
    assert (env / "math_2025-01-10_0.xml").exists() and (env / "math_2025-01-10_1.xml").exists()
    assert len(net.urls) == 2
    assert "resumptionToken=tok-1" in net.urls[1]
    assert list(bc.harvest("math", DAY)) == recs
    assert len(net.urls) == 2  # rerun came from cache


def test_no_records_match_is_cached_empty_page(env, monkeypatch):
    net = _install(monkeypatch, FakeNet(first="no_records.xml"))
    assert list(bc.harvest("math", DAY)) == []
    assert (env / "math_2025-01-10_0.xml").exists()
    assert len(net.urls) == 1
    assert list(bc.harvest("math", DAY)) == []
    assert len(net.urls) == 1


@pytest.mark.parametrize(("fixture", "code"), [("bad_resumption_token.xml", "badResumptionToken"), ("bad_argument.xml", "badArgument")])
def test_other_oai_errors_raise_and_are_not_cached(env, monkeypatch, fixture, code):
    _install(monkeypatch, FakeNet(first=fixture))
    with pytest.raises(bc.OaiError, match=code):
        list(bc.harvest("math", DAY))
    assert list(env.iterdir()) == []


def test_unparseable_response_is_not_cached(env, monkeypatch):
    _install(monkeypatch, FakeNet(raw=b"<OAI-PMH><truncated"))
    with pytest.raises(bc.OaiError):
        bc.fetch({"verb": "ListRecords"}, "math_2025-01-10_0")
    assert list(env.iterdir()) == []


def test_corrupt_cached_page_is_refetched_atomically(env, monkeypatch):
    (env / "math_2025-01-10_0.xml").write_bytes(b"<OAI")
    net = _install(monkeypatch, FakeNet(first="page1_last.xml"))
    data = bc.fetch({"verb": "ListRecords"}, "math_2025-01-10_0")
    assert len(net.urls) == 1
    assert data == (FIX / "page1_last.xml").read_bytes()
    assert (env / "math_2025-01-10_0.xml").read_bytes() == data
    assert not list(env.glob("*.tmp"))


def test_incomplete_cached_chain_is_reharvested_from_page_zero(env, monkeypatch):
    shutil.copy(FIX / "page0_token.xml", env / "math_2025-01-10_0.xml")
    net = _install(monkeypatch, FakeNet())
    recs = list(bc.harvest("math", DAY))
    assert len(recs) == 3
    first = parse_qs(urlparse(net.urls[0]).query)
    assert first["verb"] == ["ListRecords"] and first["set"] == ["math"]
    assert "resumptionToken" not in first
    assert "resumptionToken=tok-1" in net.urls[1]


def test_complete_cached_chain_is_kept(env, monkeypatch):
    shutil.copy(FIX / "page0_token.xml", env / "math_2025-01-10_0.xml")
    shutil.copy(FIX / "page1_last.xml", env / "math_2025-01-10_1.xml")
    net = _install(monkeypatch, FakeNet())
    assert len(list(bc.harvest("math", DAY))) == 3
    assert net.urls == []


def test_sample_days_clamps_day_to_month_length():
    assert bc.sample_days(dt.date(2026, 3, 13), 2, 31) == [dt.date(2026, 3, 2), dt.date(2026, 2, 2)]
    for day in (29, 30, 31):
        bc.sample_days(dt.date(2026, 3, 13), 24, day)


def test_sample_days_default_matches_real_cache():
    cache = ROOT / "data" / "candidates_cache"
    if not cache.exists():
        pytest.skip("no real candidates cache")
    days = bc.sample_days(dt.date(2026, 10, 13), 24, 10)
    assert len(days) == 24 and days[0] == dt.date(2026, 9, 10) and days[-1] == dt.date(2024, 10, 10)
    for d in days:
        assert (cache / f"math_{d}_0.xml").exists()


def test_cache_only_rebuild_reproduces_candidates(tmp_path, monkeypatch):
    cache = ROOT / "data" / "candidates_cache"
    committed = ROOT / "data" / "candidates.jsonl"
    if not cache.exists() or not (ROOT / "data" / "external" / "arxiv_src").exists():
        pytest.skip("cache or arxiv_src missing")
    copy = tmp_path / "cache"
    shutil.copytree(cache, copy)

    def no_network(*a, **k):
        raise AssertionError("network")

    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(bc.urllib.request, "urlopen", no_network)
    monkeypatch.setattr(bc, "CACHE", copy)
    out = tmp_path / "c.jsonl"
    bc.main(["--asof", "2026-10-13", "--cache", str(copy), "--out", str(out)])
    assert hashlib.sha256(out.read_bytes()).hexdigest() == hashlib.sha256(committed.read_bytes()).hexdigest()
