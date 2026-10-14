"""Pilot report: metrics over built train/dev shards, engine go/no-go numbers, automated label audit, contact sheet.

Usage:
  from lightonocr_math.corpus import pilot
  m = pilot.pilot_metrics(Path("data/corpus/shards"), ids)
  summaries, records, excluded = pilot.load_shards(Path("data/corpus/shards"), ids)
  m["audit"] = pilot.automated_audit(records, excluded, summaries=summaries)
  pilot.render_contact_sheet(pilot.contact_sheet_sample(records), Path("pilot_contact_sheet.html"))
  Path("PILOT_REPORT.md").write_text(pilot.render_report(m))

Rules:
  * ids must be disjoint from the benchmark (splits.assert_disjoint); benchmark shards are never read here
  * under D-10 the crop is rendered from the label, so "render-match every label" would be tautological. The automated
    audit is instead: canon validation (the paper's verbatim starred environment vs the label through the scorer wrapper),
    re-render determinism of a seeded sample with a fresh cache, structural checks, and the layout-only token diff.
    The manual 30-pair contact-sheet check carries the fidelity judgement (D-09)
  * engine stop rule (D-07): stop for a user decision when the engine-flagged share of otherwise-good papers exceeds 10%
  * every text field written into the contact sheet passes html.escape; images are base64 data URIs, no scripts

Known limit: near-duplicate profiling is O(n^2) in the number of kept labels (rapidfuzz cdist), fine for a pilot of a few thousand.
"""

from __future__ import annotations

import base64
import html
import io
import json
import math
import random
import shutil
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from lightonocr_math import paths, render, splits
from lightonocr_math.corpus import build, canon
from lightonocr_math.corpus._io import read_jsonl
from lightonocr_math.corpus.flags import FLAGS
from lightonocr_math.ids import safe_id

ENGINE_STOP_SHARE = 0.10
SYSTEMATIC_SHARE = 0.02
SYSTEMATIC_PAPERS = 5
YIELD_CAP = 30
QUOTAS = {"benchmark": 150, "dev": 50, "train": 800}
_HIST_EDGES = (0, 1, 2, 6, 11, 21, 51)


def load_shards(shard_root: Path, ids: list[str]) -> tuple[list[dict], list[dict], list[dict]]:
    """(paper summaries, kept records, excluded rows) of the given papers; refuses benchmark ids."""
    splits.assert_disjoint(ids, context="pilot shards")
    root = Path(shard_root)
    summaries: list[dict] = []
    records: list[dict] = []
    excluded: list[dict] = []
    for i in ids:
        sid = safe_id(i)
        sp = root / "papers" / f"{sid}.json"
        if not sp.is_file():
            continue
        summaries.append(json.loads(sp.read_text(encoding="utf-8")))
        for sub, bucket in (("records", records), ("excluded", excluded)):
            p = root / sub / f"{sid}.jsonl"
            if p.is_file():
                bucket.extend(read_jsonl(p))
    splits.assert_disjoint({r["arxiv_id"] for r in records}, context="pilot shard records")
    return summaries, records, excluded


def _dist(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "median": None, "p10": None, "p90": None, "mean": None}
    a = np.asarray(values, dtype=float)
    return {
        "n": len(values), "median": float(np.median(a)), "p10": float(np.percentile(a, 10)),
        "p90": float(np.percentile(a, 90)), "mean": float(a.mean()),
    }


def _histogram(values: list[int]) -> dict[str, int]:
    labels = ["0", "1", "2-5", "6-10", "11-20", "21-50", "51+"]
    out = dict.fromkeys(labels, 0)
    for v in values:
        idx = max(j for j, e in enumerate(_HIST_EDGES) if v >= e)
        out[labels[idx]] += 1
    return out


def engine_numbers(summaries: list[dict], excluded: list[dict]) -> dict:
    """The three engine go/no-go numbers (RESEARCH Pitfall 5) and the stop decision."""
    good = [
        s for s in summaries
        if s.get("status") == "built"
        and s.get("n_extracted", 0) - s.get("exclusions", {}).get("macros", 0) - s.get("exclusions", {}).get("short_long", 0) > 0
    ]
    flagged = [s for s in good if s.get("engine_flagged")]
    share = len(flagged) / len(good) if good else 0.0
    flagged_all = [s for s in summaries if s.get("status") == "built" and s.get("engine_flagged")]
    return {
        "n_good_papers": len(good), "n_engine_flagged_good": len(flagged), "engine_flagged_share": share,
        "engine_flagged_zero_kept": sum(1 for s in flagged_all if s.get("n_kept", 0) == 0),
        "equations_lost_to_engine": sum(1 for r in excluded if r.get("reason") == "engine"),
        "threshold": ENGINE_STOP_SHARE, "stop": share > ENGINE_STOP_SHARE,
    }


def pilot_metrics(shard_root: Path, ids: list[str]) -> dict:
    from lightonocr_math.pixels import smart_resize_dims

    summaries, records, excluded = load_shards(shard_root, ids)
    built = [s for s in summaries if s.get("status") == "built"]
    by_status: dict[str, int] = {}
    for s in summaries:
        by_status[s["status"]] = by_status.get(s["status"], 0) + 1
    eq_reasons: dict[str, int] = {}
    paper_reasons: dict[str, int] = {}
    for r in excluded:
        bucket = paper_reasons if r.get("eq_index") is None else eq_reasons
        bucket[r["reason"]] = bucket.get(r["reason"], 0) + 1
    kept = [s.get("n_kept", 0) for s in built]
    flag_eq = dict.fromkeys(FLAGS, 0)
    flag_papers: dict[str, set] = {f: set() for f in FLAGS}
    for rec in records:
        for f in rec["flags"]:
            flag_eq[f] = flag_eq.get(f, 0) + 1
            flag_papers.setdefault(f, set()).add(rec["arxiv_id"])
    widths = [r["width"] for r in records]
    heights = [r["height"] for r in records]
    tokens, upscaled = [], 0
    for r in records:
        w, h = smart_resize_dims(r["width"], r["height"], 65536, 602112)
        tokens.append(w * h // 1024)
        upscaled += r["width"] * r["height"] < 65536
    usable_by_split: dict[str, dict] = {}
    for s in built:
        d = usable_by_split.setdefault(s.get("split") or "unknown", {"built": 0, "usable": 0, "kept": 0})
        d["built"] += 1
        d["usable"] += s.get("n_kept", 0) > 0
        d["kept"] += s.get("n_kept", 0)
    return {
        "n_ids": len(ids), "papers_by_status": dict(sorted(by_status.items())),
        "paper_level_exclusions": dict(sorted(paper_reasons.items())),
        "equation_exclusions": dict(sorted(eq_reasons.items())),
        "n_extracted": sum(s.get("n_extracted", 0) for s in built), "n_kept": len(records),
        "yield": {
            "per_paper": _dist([float(k) for k in kept]), "histogram": _histogram(kept),
            "capped": {"cap": YIELD_CAP, "per_paper": _dist([float(min(k, YIELD_CAP)) for k in kept]),
                       "total": sum(min(k, YIELD_CAP) for k in kept)},
        },
        "flags": {f: {"equations": flag_eq.get(f, 0), "papers": len(flag_papers.get(f, ()))} for f in FLAGS},
        "crop": {
            "width": _dist([float(x) for x in widths]), "height": _dist([float(x) for x in heights]),
            "image_tokens": _dist([float(x) for x in tokens]),
            "share_upscaled_to_floor": upscaled / len(records) if records else 0.0,
        },
        "seconds_per_paper": _dist([float(s.get("seconds", 0.0)) for s in built]),
        "engine": engine_numbers(summaries, excluded),
        "usable_by_split": usable_by_split,
        "versions": {
            "builder": build.BUILDER_VERSION, "canon": canon.CANON_RULES_VERSION, "pad_px": build.PAD_PX,
            "preamble_sha256": render.PREAMBLE_SHA256,
        },
    }


def automated_audit(
    records: list[dict], excluded: list[dict], *, summaries: list[dict] | None = None, sample: int = 50, seed: int = 0,
    cache: Path | None = None,
) -> dict:
    mism = [r for r in excluded if r.get("reason") == "canon_mismatch" and r["detail"].get("cause") != "structural"]
    n_validated = len(records) + len(mism)
    grouped: dict[str, int] = {}
    papers_by_cause: dict[str, set] = {}
    cause_totals: dict[str, int] = {}
    for r in mism:
        d = r["detail"]
        k = f"{d.get('cause')} | {r.get('env')} | multi_column={d.get('multi_column')}"
        grouped[k] = grouped.get(k, 0) + 1
        cause_totals[d.get("cause")] = cause_totals.get(d.get("cause"), 0) + 1
        papers_by_cause.setdefault(d.get("cause"), set()).add(r["arxiv_id"])
    systematic = sorted(
        c for c, n in cause_totals.items()
        if n_validated and (n / n_validated >= SYSTEMATIC_SHARE or len(papers_by_cause[c]) >= SYSTEMATIC_PAPERS)
    )
    rng = random.Random(seed)
    pool = sorted(records, key=lambda r: r["id"])
    picked = rng.sample(pool, min(sample, len(pool)))
    same = 0
    if picked:
        tmp = Path(tempfile.mkdtemp(prefix="pilot_determinism_"))
        try:
            res = render.render_many([r["latex"] for r in picked], cache=tmp)
            same = sum(1 for r, x in zip(picked, res, strict=True) if x.status == "rendered" and x.bitmap_sha256 == r["bitmap_sha256"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    structural = [{"id": r["id"], "problems": canon.structural_problems(r["latex"])} for r in records]
    structural = [x for x in structural if x["problems"]]
    layout = []
    for r in records:
        diff = canon.layout_only_diff(r["raw_body"], r["latex"])
        if diff:
            layout.append({"id": r["id"], "diff": diff[:10]})
    redefining = [
        {"arxiv_id": s["arxiv_id"], "commands": s["redefined_core"], "n_kept": s.get("n_kept", 0)}
        for s in (summaries or []) if s.get("redefined_core")
    ]
    return {
        "canon_validation": {
            "validated": n_validated, "passed": len(records), "mismatched": len(mism), "by_cause": dict(sorted(grouped.items())),
            "structural_excluded": sum(1 for r in excluded if r.get("reason") == "canon_mismatch" and r["detail"].get("cause") == "structural"),
            "tiers": {
                k: sum((s.get("validation_tiers") or {}).get(k, 0) for s in (summaries or [])) for k in ("identical", "glyph_equal")
            },
        },
        "systematic_causes": systematic,
        "systematic": bool(systematic),
        "determinism": {"sampled": len(picked), "identical": same, "seed": seed},
        "structural_failures": {"count": len(structural), "examples": structural[:5]},
        "layout_only_diff_failures": {"count": len(layout), "examples": layout[:5]},
        "redefined_core_papers": {
            "n_papers": len(redefining), "n_equations": sum(x["n_kept"] for x in redefining), "papers": redefining[:20],
        },
    }


def near_duplicate_profile(records: list[dict], *, threshold: float = 0.85, top: int = 20) -> dict:
    """Nearest cross-paper normalized token similarity per label (rapidfuzz cdist)."""
    from rapidfuzz import process
    from rapidfuzz.distance import Levenshtein

    from lightonocr_math.normalize import normalize_tokens

    recs = sorted(records, key=lambda r: r["id"])
    if len(recs) < 2:
        return {"n": len(recs), "threshold": threshold, "histogram": {}, "n_above": 0, "top_pairs": []}
    toks = [tuple(normalize_tokens(r["latex"])) for r in recs]
    sim = process.cdist(toks, toks, scorer=Levenshtein.normalized_similarity, workers=1)
    papers = np.array([r["arxiv_id"] for r in recs])
    sim = np.asarray(sim, dtype=float)
    sim[papers[:, None] == papers[None, :]] = -1.0
    best_j = sim.argmax(axis=1)
    best = sim[np.arange(len(recs)), best_j]
    edges = [0.0, 0.5, 0.7, 0.85, 0.95, 1.0001]
    hist = {f"{edges[k]:.2f}-{min(edges[k + 1], 1.0):.2f}": int(((best >= edges[k]) & (best < edges[k + 1])).sum()) for k in range(len(edges) - 1)}
    pairs = sorted(
        ({"a": recs[i]["id"], "b": recs[int(best_j[i])]["id"], "similarity": round(float(best[i]), 4)}
         for i in range(len(recs)) if best[i] >= threshold and recs[i]["id"] < recs[int(best_j[i])]["id"]),
        key=lambda x: (-x["similarity"], x["a"]),
    )
    return {"n": len(recs), "threshold": threshold, "histogram": hist, "n_above": int((best >= threshold).sum()), "top_pairs": pairs[:top]}


def contact_sheet_sample(records: list[dict], *, n: int = 30, seed: int = 0, per_paper: int = 2) -> list[dict]:
    """n kept records, seeded, about n/5 per flag (plain included), at most per_paper from one paper."""
    rng = random.Random(seed)
    pool = sorted(records, key=lambda r: r["id"])
    rng.shuffle(pool)
    per_flag = math.ceil(n / len(FLAGS))
    chosen: list[dict] = []
    ids: set[str] = set()
    count: dict[str, int] = {}

    def take(r: dict) -> bool:
        if r["id"] in ids or count.get(r["arxiv_id"], 0) >= per_paper or len(chosen) >= n:
            return False
        chosen.append(r)
        ids.add(r["id"])
        count[r["arxiv_id"]] = count.get(r["arxiv_id"], 0) + 1
        return True

    avail = {f: sum(1 for r in pool if f in r["flags"]) for f in FLAGS}
    for f in sorted(FLAGS, key=lambda f: (avail[f], FLAGS.index(f))):
        got = 0
        for r in pool:
            if got >= per_flag:
                break
            if f in r["flags"] and take(r):
                got += 1
    for r in pool:
        take(r)
    return chosen


def _png_b64(arr: np.ndarray) -> str:
    padded = np.pad(np.asarray(arr, dtype=np.uint8), build.PAD_PX, mode="constant", constant_values=255)
    buf = io.BytesIO()
    Image.fromarray(padded, mode="L").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def render_contact_sheet(sample: list[dict], out_path: Path, *, cache: Path | None = None, repo_root: Path | None = None) -> None:
    """One self-contained HTML file: per pair the crop, the verbatim-original render and the texts, all escaped."""
    root = Path(repo_root) if repo_root is not None else paths.repo_root()
    cache = Path(cache) if cache is not None else paths.cache_dir()
    originals = [canon.original_for_validation(r["raw_body"], r["env"]) for r in sample]
    results = render.render_many(originals, cache=cache, wrapper="env") if sample else []
    esc = html.escape
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Pilot contact sheet</title>",
        (
            "<style>body{font-family:sans-serif;margin:1em}.pair{border:1px solid #888;margin:1em 0;padding:.5em}"
            "img{background:#fff;border:1px solid #ccc;max-width:100%}pre{white-space:pre-wrap;background:#f4f4f4;padding:.3em}"
            ".cols{display:flex;gap:1em;flex-wrap:wrap}.cols>div{flex:1;min-width:20em}</style></head><body>"
        ),
        f"<h1>Pilot contact sheet: {len(sample)} pairs</h1>",
        (
            "<p>Left: the crop that will be stored (rendered from the label). Right: the paper's own environment rendered verbatim. "
            "They should look the same; judge whether the label is a faithful transcription.</p>"
        ),
    ]
    for k, (r, res) in enumerate(zip(sample, results, strict=True), 1):
        crop_path = root / r["image"]
        crop = f"<img alt='crop' src='data:image/png;base64,{base64.b64encode(crop_path.read_bytes()).decode('ascii')}'>" if crop_path.is_file() else "<em>crop file missing</em>"
        bm = render.load_bitmap(res, cache) if res.status == "rendered" else None
        orig = f"<img alt='original' src='data:image/png;base64,{_png_b64(bm)}'>" if bm is not None else f"<em>original not rendered: {esc(res.status)}</em>"
        parts.append(
            f"<div class='pair'><h3>{k}. {esc(r['id'])} &middot; {esc(r['arxiv_id'])} &middot; {esc(str(r['source_file']))}:{esc(str(r['source_line']))}"
            f" &middot; flags: {esc(', '.join(r['flags']))}</h3><div class='cols'><div><b>crop (label)</b><br>{crop}</div>"
            f"<div><b>original environment</b><br>{orig}</div></div>"
            f"<b>raw source</b><pre>{esc(r['raw_body'])}</pre><b>canonical label</b><pre>{esc(r['latex'])}</pre></div>"
        )
    parts.append("</body></html>\n")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("".join(parts), encoding="utf-8")


def _table(rows: list[tuple], header: tuple[str, ...]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def _fmt(x: object) -> str:
    return "n/a" if x is None else (f"{x:.1f}" if isinstance(x, float) else str(x))


def render_report(m: dict) -> str:
    L: list[str] = ["# Pilot report", ""]
    L += ["## Papers", ""] + _table([(k, v) for k, v in m["papers_by_status"].items()], ("status", "papers"))
    if m.get("fetch"):
        L += ["", "Fetch log for these papers: " + ", ".join(f"{k} {v}" for k, v in m["fetch"]["by_status"].items()) + "."]
    L += ["", f"Papers requested: {m['n_ids']}. Equations extracted from built papers: {m['n_extracted']}, kept: {m['n_kept']}.", ""]
    L += ["## Exclusions", "", "Equation-level (one reason each):", ""]
    L += _table(list(m["equation_exclusions"].items()), ("reason", "equations"))
    L += ["", "Paper-level:", ""] + _table(list(m["paper_level_exclusions"].items()), ("reason", "papers"))
    y = m["yield"]
    L += ["", "## Yield", "", "Kept equations per built paper:", ""]
    pp, cp = y["per_paper"], y["capped"]
    L += _table([("uncapped", pp["n"], _fmt(pp["median"]), _fmt(pp["p10"]), _fmt(pp["p90"]), m["n_kept"]),
                 (f"capped at {cp['cap']}", cp["per_paper"]["n"], _fmt(cp["per_paper"]["median"]), _fmt(cp["per_paper"]["p10"]),
                  _fmt(cp["per_paper"]["p90"]), cp["total"])], ("view", "papers", "median", "p10", "p90", "total"))
    L += ["", "Histogram (kept per paper):", ""] + _table(list(y["histogram"].items()), ("kept", "papers"))
    L += ["", "## Notation mix", ""]
    L += _table([(f, v["equations"], v["papers"]) for f, v in m["flags"].items()], ("flag", "equations", "papers"))
    e = m["engine"]
    L += ["", "## Engine go/no-go", "",
          f"1. Engine-flagged share of otherwise-good papers: {e['n_engine_flagged_good']}/{e['n_good_papers']} = {e['engine_flagged_share']:.1%}",
          f"2. Engine-flagged papers with zero kept equations: {e['engine_flagged_zero_kept']}",
          f"3. Equations lost to the engine reason: {e['equations_lost_to_engine']}", "",
          f"Rule: stop when number 1 exceeds {e['threshold']:.0%}. Result: "
          + ("**STOP for user decision** (engine-flagged share is over the threshold)" if e["stop"] else "continue (under the threshold)") + "."]
    L += ["", "## Label audit", ""]
    L += ["Manual audit: 30 pairs per D-09 (user choice); the roadmap text says 100-200. Open pilot_contact_sheet.html and judge each pair.", "",
          ("Render-matching a crop against its own label is tautological under D-10 (the crop is rendered from the label), so the "
           "automated audit is: canon validation against the verbatim original environment, re-render determinism with a fresh cache, "
           "structural checks and the layout-only token diff."), ""]
    a = m.get("audit")
    if a is None:
        L.append("Automated audit not computed.")
    else:
        cv = a["canon_validation"]
        L += [f"- Canon validation: {cv['passed']}/{cv['validated']} passed; {cv['mismatched']} mismatched; {cv['structural_excluded']} excluded structurally",
              f"- Systematic mismatch causes (>= 2% of validated or >= 5 papers): {', '.join(a['systematic_causes']) or 'none'}"]
        L += [f"  - {k}: {v}" for k, v in cv["by_cause"].items()]
        tiers = cv.get("tiers")
        if tiers:
            L.append(f"- Validation tiers of the passed labels: {tiers['identical']} bitmap-identical, {tiers['glyph_equal']} glyph-equal "
                     "(same ink components, positions snapped by at most 2 px)")
        d = a["determinism"]
        L += [f"- Determinism: {d['identical']}/{d['sampled']} identical bitmaps on re-render (seed {d['seed']})",
              f"- Structural failures in kept labels: {a['structural_failures']['count']}",
              f"- Layout-only diff failures: {a['layout_only_diff_failures']['count']}",
              f"- Papers redefining core commands: {a['redefined_core_papers']['n_papers']} ({a['redefined_core_papers']['n_equations']} kept equations)"]
    c = m["crop"]
    L += ["", "## Crop sizes and scale", ""]
    L += _table([(k, _fmt(c[k]["median"]), _fmt(c[k]["p10"]), _fmt(c[k]["p90"])) for k in ("width", "height", "image_tokens")],
                ("quantity", "median", "p10", "p90"))
    L += ["", f"Padding: {m['versions']['pad_px']} px. Share of crops below the 65,536 px floor (upscaled by the pixel policy): {c['share_upscaled_to_floor']:.1%}."]
    if m.get("crop_scale") is not None:
        L += ["", "Crop scale versus page scale:", "", "```json", json.dumps(m["crop_scale"], indent=2, sort_keys=True), "```"]
    L += ["", "## Near-duplicates", ""]
    nd = m.get("near_duplicates")
    if nd is None:
        L.append("Not computed.")
    else:
        L += [f"Labels: {nd['n']}; cross-paper nearest similarity >= {nd['threshold']}: {nd['n_above']}.", ""]
        L += _table(list(nd["histogram"].items()), ("similarity", "labels"))
        L += [""] + _table([(p["a"], p["b"], p["similarity"]) for p in nd["top_pairs"]], ("a", "b", "similarity"))
    L += ["", "## Projections", "",
          "Quotas of usable papers (at least one kept equation): " + ", ".join(f"{k} {v}" for k, v in QUOTAS.items()) + ".", ""]
    usable = m["usable_by_split"]
    tot_b = sum(v["built"] for v in usable.values())
    tot_u = sum(v["usable"] for v in usable.values())
    pooled = tot_u / tot_b if tot_b else 0.0
    rows = []
    for split, quota in QUOTAS.items():
        v = usable.get(split)
        if v and v["built"]:
            rate, src = v["usable"] / v["built"], f"{v['usable']}/{v['built']}"
        else:
            rate, src = pooled, f"pooled {tot_u}/{tot_b} (no pilot papers on this side)"
        rows.append((split, src, f"{rate:.1%}", math.ceil(quota / rate) if rate > 0 else "n/a"))
    L += _table(rows, ("split", "usable / built", "usable rate", "papers to fetch"))
    L.append("")
    return "\n".join(L)
