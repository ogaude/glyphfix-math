"""Fetcher, pacing, bundle sniffing, order and CLI guards with a fake opener (no network)."""

import gzip
import hashlib
import io
import json
import tarfile
import urllib.error
from email.message import Message

import pytest

from lightonocr_math.corpus import fetch as F
from lightonocr_math.corpus._io import write_jsonl
from scripts import corpus_fetch as cli


class _Resp:
    status = 200

    def __init__(self, data: bytes, url: str = "u"):
        self._b = io.BytesIO(data)
        self._url = url
        self.headers = {"Content-Type": "application/x-eprint-tar"}

    def read(self, n=-1):
        return self._b.read(n)

    def geturl(self):
        return self._url

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _http_error(code, retry_after=None):
    h = Message()
    if retry_after is not None:
        h["Retry-After"] = str(retry_after)
    return urllib.error.HTTPError("u", code, "x", h, None)


def _tgz(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in files.items():
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


ROW = {"arxiv_id": "2301.01234", "version": "v2", "license_uri": "http://x", "license_spdx": "CC-BY-4.0", "license_checked_at": "2026-10-14"}


class _Opener:
    def __init__(self, *items):
        self.items = list(items)
        self.calls = []

    def __call__(self, req, timeout=None):
        self.calls.append(req)
        it = self.items.pop(0)
        if isinstance(it, Exception):
            raise it
        return it


def _lim():
    return F.Limiter(15, clock=lambda: 0.0, sleep=lambda s: None)


def test_limiter_clamp():
    assert F.Limiter(1.0).min == 3.1


def test_limiter_pacing():
    t = [0.0]
    sleeps = []

    def sleep(s):
        sleeps.append(s)
        t[0] += s

    lim = F.Limiter(15, clock=lambda: t[0], sleep=sleep)
    lim.wait()
    t[0] += 2
    lim.wait()
    lim.wait()
    assert sleeps == [13.0, 15.0]


def test_constants():
    assert F.DEFAULT_INTERVAL_S == 3.1 and F.MIN_INTERVAL_S == 3.1


def test_eprint_url():
    assert F.eprint_url("2301.01234", "v2") == "https://export.arxiv.org/e-print/2301.01234v2"
    assert F.eprint_url("math/0508005v1", "v1") == "https://export.arxiv.org/e-print/math/0508005v1"


def test_sniff_kind():
    assert F.sniff_kind(_tgz({"a.tex": b"x"})[:1024]) == "tar.gz"
    assert F.sniff_kind(gzip.compress(b"\\documentclass{article}")) == "gz"
    assert F.sniff_kind(b"%PDF-1.5 xx") == "pdf"
    assert F.sniff_kind(b"<!DOCTYPE html><html>") == "html"
    assert F.sniff_kind(b"  <html>") == "html"
    raw = bytearray(512)
    raw[257:262] = b"ustar"
    assert F.sniff_kind(bytes(raw)) == "tar"
    assert F.sniff_kind(b"hello") == "unknown"


def test_fetch_ok_and_resume(tmp_path):
    data = _tgz({"main.tex": b"\\documentclass{article}"})
    op = _Opener(_Resp(data, "https://arxiv.org/src/2301.01234v2"))
    meta = F.fetch_one(ROW, tmp_path, limiter=_lim(), policy_ack="ack", opener=op)
    assert meta["status"] == "ok" and meta["kind"] == "tar.gz" and meta["bytes"] == len(data)
    assert meta["sha256"] == hashlib.sha256(data).hexdigest()
    assert meta["url"].endswith("/e-print/2301.01234v2") and meta["url_final"].endswith("/src/2301.01234v2")
    assert meta["policy_ack"] == "ack" and meta["license_spdx"] == "CC-BY-4.0" and meta["http_status"] == 200
    d = tmp_path / "2301.01234"
    assert json.loads((d / "fetch.json").read_text())["sha256"] == meta["sha256"]
    assert op.calls[0].get_header("User-agent").startswith("lightonocr-math-research")
    # intact: zero requests
    op2 = _Opener()
    assert F.fetch_one(ROW, tmp_path, limiter=_lim(), policy_ack="ack", opener=op2)["sha256"] == meta["sha256"]
    assert op2.calls == []
    # corrupted: refetched
    (d / "src.bin").write_bytes(b"corrupt")
    op3 = _Opener(_Resp(data))
    F.fetch_one(ROW, tmp_path, limiter=_lim(), policy_ack="ack", opener=op3)
    assert len(op3.calls) == 1 and (d / "src.bin").read_bytes() == data


def test_fetch_404(tmp_path):
    op = _Opener(_http_error(404))
    meta = F.fetch_one(ROW, tmp_path, limiter=_lim(), policy_ack="a", opener=op)
    assert meta["status"] == "missing" and meta["kind"] == "none"
    assert not (tmp_path / "2301.01234" / "src.bin").exists()
    assert (tmp_path / "2301.01234" / "fetch.json").exists()


def test_fetch_403(tmp_path):
    with pytest.raises(F.FetchForbidden):
        F.fetch_one(ROW, tmp_path, limiter=_lim(), policy_ack="a", opener=_Opener(_http_error(403)))


def test_fetch_429_retry_after(tmp_path):
    slept = []
    op = _Opener(_http_error(429, 7), _http_error(503, 9999), _Resp(b"%PDF-1"))
    meta = F.fetch_one(ROW, tmp_path, limiter=_lim(), policy_ack="a", opener=op, sleep=slept.append)
    assert slept == [7.0, 600.0] and meta["kind"] == "pdf" and len(op.calls) == 3


def test_fetch_retries_exhausted(tmp_path):
    op = _Opener(*[_http_error(503, 1) for _ in range(4)])
    with pytest.raises(urllib.error.HTTPError):
        F.fetch_one(ROW, tmp_path, limiter=_lim(), policy_ack="a", opener=op, sleep=lambda s: None)
    assert len(op.calls) == 4


def test_fetch_too_large(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "MAX_BUNDLE_BYTES", 100)
    meta = F.fetch_one(ROW, tmp_path, limiter=_lim(), policy_ack="a", opener=_Opener(_Resp(b"x" * 500)))
    assert meta["status"] == "too_large" and meta["kind"] == "too_large"
    assert not (tmp_path / "2301.01234" / "src.bin").exists()
    assert [p.name for p in (tmp_path / "2301.01234").iterdir()] == ["fetch.json"]


def _redirect(loc, code=301):
    h = Message()
    h["Location"] = loc
    return urllib.error.HTTPError("u", code, "x", h, None)


def test_redirect_hop_is_paced_and_recorded(tmp_path):
    t = [0.0]
    sleeps = []

    def sleep(s):
        sleeps.append(s)
        t[0] += s

    lim = F.Limiter(3.1, clock=lambda: t[0], sleep=sleep)
    data = _tgz({"main.tex": b"x"})
    op = _Opener(_redirect("/src/2301.01234v2"), _Resp(data, "https://export.arxiv.org/src/2301.01234v2"))
    meta = F.fetch_one(ROW, tmp_path, limiter=lim, policy_ack="a", opener=op)
    assert len(op.calls) == 2 and sleeps == [3.1]
    assert op.calls[1].full_url == "https://export.arxiv.org/src/2301.01234v2"
    assert meta["status"] == "ok" and meta["url_final"].endswith("/src/2301.01234v2")


def test_redirect_off_host_refused(tmp_path):
    for loc in ("https://evil.example/x", "http://export.arxiv.org/src/x", "https://export.arxiv.org.evil.example/x"):
        op = _Opener(_redirect(loc))
        with pytest.raises(F.FetchForbidden):
            F.fetch_one(ROW, tmp_path, limiter=_lim(), policy_ack="a", opener=op)
        assert len(op.calls) == 1


def test_redirect_loop_capped(tmp_path):
    op = _Opener(*[_redirect("/src/2301.01234v2") for _ in range(10)])
    with pytest.raises(F.FetchForbidden):
        F.fetch_one(ROW, tmp_path, limiter=_lim(), policy_ack="a", opener=op)
    assert len(op.calls) == F.MAX_REDIRECTS + 1


def test_default_opener_does_not_follow_redirects():
    assert F.default_open is not None
    assert F._NoRedirect().redirect_request(None, None, 301, "x", {}, "http://a") is None


def test_fetch_order_deterministic():
    ids = ["2301.00001", "2301.00002v3", "math/0508005", "2302.12345"]
    a = F.fetch_order(ids, 0)
    assert a == F.fetch_order(list(reversed(ids)), 0)
    assert a == sorted(["2301.00001", "2301.00002", "math/0508005", "2302.12345"],
                       key=lambda b: hashlib.sha256(f"0|{b}".encode()).hexdigest())
    assert F.fetch_order(ids, 1) != a or len(ids) < 3


def _catalog(tmp_path, rows):
    p = tmp_path / "cat.jsonl"
    write_jsonl(p, rows)
    return p


def _row(i, **kw):
    return {"arxiv_id": i, "version": "v1", "license_spdx": "CC-BY-4.0", "quarantined": False, **kw}


def test_cli_order_and_refuse_overwrite(tmp_path):
    cat = _catalog(tmp_path, [_row("2301.00001"), _row("2301.00002"), _row("2301.00003", quarantined=True),
                              _row("2301.00004", license_spdx="CC-BY-NC-4.0")])
    out = tmp_path / "o.json"
    assert cli.main(["order", "--catalog", str(cat), "--out", str(out), "--seed", "0"]) == 0
    d = json.loads(out.read_text())
    assert sorted(d["ids"]) == ["2301.00001", "2301.00002"] and d["seed"] == 0 and d["catalog_sha256"]
    assert cli.main(["order", "--catalog", str(cat), "--out", str(out), "--seed", "0"]) == 0
    assert cli.main(["order", "--catalog", str(cat), "--out", str(out), "--seed", "1"]) in (0, 2)
    cat2 = _catalog(tmp_path, [_row("2301.00001")])
    assert cli.main(["order", "--catalog", str(cat2), "--out", str(out), "--seed", "0"]) == 2


def _fetch_args(tmp_path, cat, order, *extra):
    return ["fetch", "--order", str(order), "--catalog", str(cat), "--sources", str(tmp_path / "src"), *extra]


def test_cli_fetch_guards(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("request attempted")

    monkeypatch.setattr(F, "fetch_one", boom)
    cat = _catalog(tmp_path, [_row("2301.00001"), _row("2301.00002", quarantined=True)])
    order = tmp_path / "o.json"
    order.write_text(json.dumps({"ids": ["2301.00001"]}))
    assert cli.main(_fetch_args(tmp_path, cat, order)) == 2
    assert cli.main(_fetch_args(tmp_path, cat, order, "--policy-ack", "x", "--interval", "3.0")) == 2
    order.write_text(json.dumps({"ids": ["2301.00009"]}))
    assert cli.main(_fetch_args(tmp_path, cat, order, "--policy-ack", "x")) == 2
    order.write_text(json.dumps({"ids": ["2301.00002"]}))
    assert cli.main(_fetch_args(tmp_path, cat, order, "--policy-ack", "x")) == 2


def test_cli_fetch_runs_and_stops(tmp_path, monkeypatch):
    cat = _catalog(tmp_path, [_row("2301.00001"), _row("2301.00002")])
    order = tmp_path / "o.json"
    order.write_text(json.dumps({"ids": ["2301.00001", "2301.00002"]}))
    calls = []

    def ok(row, sources, **k):
        calls.append(row["arxiv_id"])
        return {"status": "ok", "kind": "tar.gz", "bytes": 1, "http_status": 200}

    monkeypatch.setattr(F, "fetch_one", ok)
    assert cli.main(_fetch_args(tmp_path, cat, order, "--policy-ack", "x")) == 0
    assert calls == ["2301.00001", "2301.00002"]
    assert len((tmp_path / "src" / "FETCH_LOG.jsonl").read_text().splitlines()) == 2

    def forbidden(row, sources, **k):
        raise F.FetchForbidden("403")

    monkeypatch.setattr(F, "fetch_one", forbidden)
    assert cli.main(_fetch_args(tmp_path, cat, order, "--policy-ack", "x")) == 3

    def fail(row, sources, **k):
        raise OSError("down")

    monkeypatch.setattr(F, "fetch_one", fail)
    ids = [f"2301.{n:05d}" for n in range(1, 8)]
    cat7 = _catalog(tmp_path, [_row(i) for i in ids])
    order.write_text(json.dumps({"ids": ids}))
    assert cli.main(_fetch_args(tmp_path, cat7, order, "--policy-ack", "x")) == 4
