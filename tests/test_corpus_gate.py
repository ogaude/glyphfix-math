import json
import random

import pytest
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from lightonocr_math import normalize, splits
from lightonocr_math.corpus import gate as G
from lightonocr_math.corpus.records import record_id

BASE = "a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 + a_{10} + a_{11} + a_{12}"


def _bench_id() -> str:
    return min(i for i in splits.frozen().benchmark if "/" not in i)


def _free_ids(n, start=1):
    out = []
    k = start
    while len(out) < n:
        i = f"2401.{k:05d}"
        k += 1
        if splits.split_of(i) != "benchmark" and i not in splits.frozen().quarantine:
            out.append(i)
    return out


def _rec(arxiv_id, latex, split="train"):
    return {"id": record_id(arxiv_id, "v1", latex), "arxiv_id": arxiv_id, "paper_version": "v1", "latex": latex, "split": split, "flags": ["plain"], "license_spdx": "CC-BY-4.0"}


def _cat(arxiv_id, title, authors, split="train"):
    return {"arxiv_id": arxiv_id, "title": title, "authors": authors, "authors_raw": ", ".join(authors), "split": split}


@pytest.fixture
def cand(tmp_path):
    p = tmp_path / "candidates.jsonl"
    p.write_text("")
    return p


def _setup(corpus_latex, *, title="Quiet title about zebras", authors=("Alice Zed", "Bob Yarn"), bench_latex=BASE):
    b = _bench_id()
    c = _free_ids(1)[0]
    bench = [_rec(b, bench_latex, "benchmark")]
    corpus = [_rec(c, corpus_latex)]
    cat = [_cat(b, "Benchmark paper on kernels", ["Carl Xu", "Dina Wu"], "benchmark"), _cat(c, title, list(authors))]
    return corpus, bench, cat


def test_exact_overlap_dropped_including_short(cand):
    corpus, bench, cat = _setup(BASE)
    res = G.apply_gate(corpus, bench, cat, candidates_path=cand)
    assert res.kept == [] and res.dropped[0]["check"] == "exact"
    assert res.report["checks"]["exact"]["count"] == 1
    corpus, bench, cat = _setup("x + y", bench_latex="x + y")
    res = G.apply_gate(corpus, bench, cat, candidates_path=cand)
    assert res.kept == [] and res.dropped[0]["check"] == "exact" and res.report["short_collisions"]


def test_missing_candidates_file_fails_closed(tmp_path):
    corpus, bench, cat = _setup("m + n + o")
    missing = tmp_path / "nope.jsonl"
    with pytest.raises(splits.LeakageError):
        G.apply_gate(corpus, bench, cat, candidates_path=missing)
    with pytest.raises(splits.LeakageError):
        G.assert_clean(corpus, bench, cat, candidates_path=missing)
    res = G.apply_gate(corpus, bench, cat, candidates_path=missing, allow_missing_candidates=True)
    assert res.report["candidates_used"] is False


def test_near_duplicate_threshold(cand):
    toks = normalize.normalize_tokens(BASE)
    close = BASE.replace("a_{12}", "b_{12}")
    sim = Levenshtein.normalized_similarity(toks, normalize.normalize_tokens(close))
    assert sim >= 0.9
    corpus, bench, cat = _setup(close)
    res = G.apply_gate(corpus, bench, cat, candidates_path=cand)
    assert res.dropped and res.dropped[0]["check"] == "near"
    far = BASE.replace("a_{12}", "b_{12}").replace("a_{11}", "c_{11}").replace("a_{10}", "d_{10}").replace("a_9", "e_9").replace("a_8", "f_8").replace("a_7", "g_7").replace("a_6", "h_6")
    sim = Levenshtein.normalized_similarity(toks, normalize.normalize_tokens(far))
    assert 0.5 < sim < 0.9
    corpus, bench, cat = _setup(far)
    res = G.apply_gate(corpus, bench, cat, candidates_path=cand)
    assert len(res.kept) == 1


def test_title_overlap_drops_whole_paper(cand):
    b = _bench_id()
    cs = _free_ids(1)
    corpus = [_rec(cs[0], "z + w + q"), _rec(cs[0], "z + w + r")]
    bench = [_rec(b, BASE, "benchmark")]
    cat = [_cat(b, "Benchmark paper on kernels", ["Carl Xu", "Dina Wu"], "benchmark"), _cat(cs[0], "Benchmark paper on kernels", ["Eve Pat", "Fay Lee"])]
    res = G.apply_gate(corpus, bench, cat, candidates_path=cand)
    assert res.kept == [] and {d["check"] for d in res.dropped} == {"paper_title"}


def test_shared_equations_drops_paper(cand):
    b = _bench_id()
    c = _free_ids(1)[0]
    eqs = [f"a_{i} + b_{i} + c_{i} + d_{i} + e_{i} + f_{i} + g_{i} + h_{i}" for i in range(1, 6)]
    bench = [_rec(b, e, "benchmark") for e in eqs]
    corpus = [_rec(c, e) for e in eqs[:5]] + [_rec(c, "u + v + w")]
    cat = [_cat(b, "Benchmark paper on kernels", ["Carl Xu", "Dina Wu"], "benchmark"), _cat(c, "Another quiet title", ["Eve Pat", "Fay Lee"])]
    res = G.apply_gate(corpus, bench, cat, candidates_path=cand)
    assert res.kept == []
    assert any(d["check"] == "paper_shared_eq" for d in res.dropped)


def test_same_author_dropped_then_assert_clean_passes(cand):
    corpus, bench, cat = _setup("u + v + w + x + y", authors=("Carl Xu", "Eve Pat"))
    with pytest.raises(splits.LeakageError):
        G.assert_clean(corpus, bench, cat, candidates_path=cand)
    res = G.apply_gate(corpus, bench, cat, candidates_path=cand)
    assert res.kept == [] and res.dropped[0]["check"] == "same_author"
    G.assert_clean(res.kept, bench, cat, candidates_path=cand)


def test_same_author_via_candidates_file(tmp_path):
    corpus, bench, cat = _setup("u + v + w + x + y", authors=("Zed Quark", "Eve Pat"))
    cand = tmp_path / "c.jsonl"
    cand.write_text(json.dumps({"arxiv_id": bench[0]["arxiv_id"], "authors": ["Zed Quark", "Other Person"]}) + "\n")
    with pytest.raises(splits.LeakageError):
        G.assert_clean(corpus, bench, cat, candidates_path=cand)


def test_benchmark_never_dropped_and_dropped_rows_leak_nothing(cand):
    corpus, bench, cat = _setup(BASE, title="Benchmark paper on kernels", authors=("Carl Xu", "Eve Pat"))
    before = json.dumps(bench)
    res = G.apply_gate(corpus, bench, cat, candidates_path=cand)
    assert json.dumps(bench) == before
    bid, btitle = bench[0]["arxiv_id"], "Benchmark paper on kernels"
    for d in res.dropped:
        assert set(d) == {"arxiv_id", "reason", "id", "check"}
        text = json.dumps(d)
        assert bid not in text and btitle not in text and bench[0]["id"] not in text
    assert any(o.get("bench_id") for o in res.report["checks"]["exact"]["offenders"])


def test_assert_clean_ids(cand):
    b = _bench_id()
    bench = [_rec(b, BASE, "benchmark")]
    with pytest.raises(splits.LeakageError):
        G.assert_clean([_rec(b, "p + q + r")], bench, [], candidates_path=cand)
    q = min(splits.frozen().quarantine)
    with pytest.raises(splits.LeakageError):
        G.assert_clean([_rec(q, "p + q + r")], bench, [], candidates_path=cand)


def test_assert_clean_near_and_exact_fail(cand):
    corpus, bench, cat = _setup(BASE)
    with pytest.raises(splits.LeakageError):
        G.assert_clean(corpus, bench, cat, candidates_path=cand)
    corpus, bench, cat = _setup("m + n + o")
    G.assert_clean(corpus, bench, cat, candidates_path=cand)


def test_side_scan_catches_ignored_style_subdir(tmp_path):
    root = tmp_path / "corpus"
    (root / "images" / _bench_id()).mkdir(parents=True)
    with pytest.raises(splits.LeakageError):
        G.assert_side_clean(root)
    ok = tmp_path / "ok"
    (ok / "shards").mkdir(parents=True)
    (ok / "shards" / "r.jsonl").write_text(json.dumps({"arxiv_id": _free_ids(1)[0]}) + "\n")
    G.assert_side_clean(ok)
    (ok / "shards" / "bad.jsonl").write_text(json.dumps({"arxiv_id": _bench_id()}) + "\n")
    with pytest.raises(splits.LeakageError):
        G.assert_side_clean(ok)


def test_prefilter_is_lossless():
    rng = random.Random(1)
    enc = G._Encoder()
    vocab = [f"t{i}" for i in range(6)]

    def make(n, p):
        out = []
        for i in range(n):
            out.append((f"{p}{i}", enc.encode([rng.choice(vocab) for _ in range(rng.randint(12, 24))])))
        return out

    bench, train = make(40, "b"), make(120, "t")
    thr = 0.45
    fast, _ = G._sweep(bench, train, thr, prefilter=True)
    slow, _ = G._sweep(bench, train, thr, prefilter=False)
    assert {(a, b) for a, b, _ in fast} == {(a, b) for a, b, _ in slow}
    ref = process.cdist([s for _, s in bench], [s for _, s in train], scorer=Levenshtein.normalized_similarity, score_cutoff=thr)
    assert len(slow) == int((ref >= thr).sum()) > 0


def test_regression_overlap():
    c = _free_ids(1)[0]
    tr = [_rec(c, "a + b + c"), _rec(c, "d + e + f")]
    reg = [{"id": "zzz", "latex": "a  +  b + c"}, {"id": tr[1]["id"], "latex": "q"}]
    assert G.regression_overlap(tr, reg) == {tr[0]["id"], tr[1]["id"]}


def test_timing_probe_small():
    r = G.timing_probe(30, 200, 0)
    assert {"seconds", "n_bench", "n_train", "pairs_compared"} <= set(r)
    assert r["pairs_compared"] <= 30 * 200


def _write_tree(tmp_path, corpus_latex):
    corpus, bench, cat = _setup(corpus_latex)
    (tmp_path / "corpus" / "train").mkdir(parents=True)
    (tmp_path / "corpus" / "dev").mkdir(parents=True)
    (tmp_path / "bench").mkdir()
    (tmp_path / "corpus" / "train" / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in corpus))
    (tmp_path / "bench" / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in bench))
    (tmp_path / "cat.jsonl").write_text("".join(json.dumps(r) + "\n" for r in cat))
    (tmp_path / "cand.jsonl").write_text("")
    return ["check", "--corpus", str(tmp_path / "corpus"), "--benchmark", str(tmp_path / "bench"),
            "--catalog", str(tmp_path / "cat.jsonl"), "--candidates", str(tmp_path / "cand.jsonl")]


def test_cli_exit_codes(tmp_path):
    assert G.main(_write_tree(tmp_path, "m + n + o")) == 0
    leaky = tmp_path / "leaky"
    leaky.mkdir()
    assert G.main(_write_tree(leaky, BASE)) == 1


def _shard(root, arxiv_id, recs, order_ids=None):
    (root / "shards" / "records").mkdir(parents=True, exist_ok=True)
    (root / "shards" / "papers").mkdir(parents=True, exist_ok=True)
    (root / "shards" / "records" / f"{arxiv_id}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    (root / "shards" / "papers" / f"{arxiv_id}.json").write_text(json.dumps({"arxiv_id": arxiv_id, "builder_version": "1", "canon_version": "1"}))


def test_corpus_select_script_end_to_end(tmp_path, monkeypatch):
    import importlib.util

    splits.frozen()
    monkeypatch.setenv("LIGHTONOCR_MATH_ROOT", str(tmp_path))
    spec = importlib.util.spec_from_file_location("corpus_select_script", "scripts/corpus_select.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    b = _bench_id()
    c1, c2, c3 = _free_ids(3)
    data = tmp_path / "data"
    bench_root, corpus_root = data / "benchmark", data / "corpus"
    _shard(bench_root, b, [_rec(b, BASE, "benchmark"), _rec(b, "q + r + s", "benchmark")])
    _shard(corpus_root, c1, [_rec(c1, "m + n + o"), _rec(c1, "m + n + p")])
    _shard(corpus_root, c2, [_rec(c2, BASE)])
    _shard(corpus_root, c3, [_rec(c3, "k + l + o", "dev")])
    (bench_root / "fetch_order.json").write_text(json.dumps({"ids": [b]}))
    (corpus_root / "fetch_order.json").write_text(json.dumps({"ids": [c1, c2, c3]}))
    (data / "catalog").mkdir()
    (data / "catalog" / "train_dev.jsonl").write_text("".join(json.dumps(_cat(c, f"Title {c} unique words {c}", [f"Name{c[-1]} Surname{c[-1]}"])) + "\n" for c in (c1, c2, c3)))
    (bench_root / "catalog.jsonl").write_text(json.dumps(_cat(b, "Benchmark paper on kernels", ["Carl Xu", "Dina Wu"], "benchmark")) + "\n")
    (data / "candidates.jsonl").write_text("")
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"seed": 0, "cap": 30, "quotas": {"benchmark": 5, "dev": 5, "train": 5}, "floors": {"corpus": {}, "benchmark": {}}}))

    assert mod.main(["run", "--config", str(cfg), "--skip-recheck"]) == 0
    train_ids = {r["arxiv_id"] for r in (json.loads(x) for x in (corpus_root / "train" / "records.jsonl").read_text().splitlines())}
    assert train_ids == {c1}
    excluded = [json.loads(x) for x in (corpus_root / "excluded_selection.jsonl").read_text().splitlines()]
    assert [e["check"] for e in excluded if e["reason"] == "leak"] == ["exact"]
    assert b not in (corpus_root / "excluded_selection.jsonl").read_text()
    assert b not in (corpus_root / "selection_manifest.json").read_text()
    assert (bench_root / "leak_report.json").exists()
    assert (corpus_root / "dev" / "records.jsonl").exists()


def test_corpus_select_requires_recheck_or_explicit_skip(tmp_path, monkeypatch, capsys):
    import importlib.util

    monkeypatch.setenv("LIGHTONOCR_MATH_ROOT", str(tmp_path))
    spec = importlib.util.spec_from_file_location("corpus_select_script2", "scripts/corpus_select.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"seed": 0}))
    assert mod.main(["run", "--config", str(cfg)]) == 2
    assert "--skip-recheck" in capsys.readouterr().err
    # with the explicit skip the next check (missing fetch_order) is reached instead
    assert mod.main(["run", "--config", str(cfg), "--skip-recheck"]) == 2
    assert "fetch_order" in capsys.readouterr().err


def test_corpus_select_rejects_invalid_shard_record(tmp_path, monkeypatch, capsys):
    import importlib.util

    splits.frozen()
    monkeypatch.setenv("LIGHTONOCR_MATH_ROOT", str(tmp_path))
    spec = importlib.util.spec_from_file_location("corpus_select_script3", "scripts/corpus_select.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    b = _bench_id()
    c = _free_ids(1)[0]
    data = tmp_path / "data"
    bad = _rec(c, "m + n + o")
    bad["license_spdx"] = "CC-BY-NC-4.0"
    _shard(data / "benchmark", b, [_rec(b, BASE, "benchmark")])
    _shard(data / "corpus", c, [bad])
    (data / "benchmark" / "fetch_order.json").write_text(json.dumps({"ids": [b]}))
    (data / "corpus" / "fetch_order.json").write_text(json.dumps({"ids": [c]}))
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"seed": 0}))
    assert mod.main(["run", "--config", str(cfg), "--skip-recheck"]) == 1
    assert "INVALID" in capsys.readouterr().err
    assert not (data / "corpus" / "train" / "records.jsonl").exists()
