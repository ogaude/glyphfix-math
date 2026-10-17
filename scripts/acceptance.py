"""Acceptance CLI: the fixed command sequence that scores the tuned model on the held-out benchmark exactly once.

Subcommands:
  verify-env   check the Ollama engine, the stock model files and the server environment against the stored stock run
  precheck     dev pre-check on 100 fixed dev crops: determinism, stored-prediction equality and prompt token parity
  generate     the only path to benchmark predictions: lock-protected, resume-only generation of one arm
  score, report, compare-results, register-verdict   (added with the scoring and report steps)

Usage (from repo root, Ollama server running with no model loaded):
  uv run python -m scripts.acceptance verify-env
  uv run python -m scripts.acceptance precheck
  uv run python -m scripts.acceptance generate --arm tuned-crops
  uv run python -m scripts.acceptance generate --arm tuned-pages
  uv run python -m scripts.acceptance generate --arm tuned-crops --resume    # after an interrupted run only

Behaviour: generate refuses unless the constants are pinned, the two code files are committed and clean and unchanged
since the pre-check, the pre-check passed, and the Ollama server environment equals the one the pre-check ran in.
RUN_LOCK.json is written before the first request; a second fresh generation of an arm is refused and a completed arm
without errors cannot be resumed. The lock stores the git blob hashes of the two code files so the scoring step can
prove they did not change after the predictions were made.

Exit codes: 0 ok, 1 check failed, 2 bad input or refused, 3 render infrastructure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import urllib.error
from pathlib import Path

from lightonocr_math import acceptance as accmod
from lightonocr_math import guard
from lightonocr_math.paths import repo_root
from scripts.baseline_infer import model_blobs
from scripts.spikes.common import gpu_state, read_jsonl, write_json
from scripts.spikes.ollama_infer import DEFAULT_HOST, OllamaError, _get, _model_digest, _post
from scripts.spikes.parity_report import modal_template_offset

TUNED_TAG = "mathocr-lora-q8:p4-full-1"
STOCK_TAG = "lighton-ocr3-q8-16k"
ACC_DIR = Path("data/benchmark/acceptance")
EVIDENCE_DIR = Path(".planning/phases/05-acceptance-evaluation/evidence")
REPORT_PATH = Path(".planning/phases/05-acceptance-evaluation/reports/ACCEPTANCE.md")
LOCK_PATH = ACC_DIR / "RUN_LOCK.json"
REGISTRY_PATH = Path("data/models/REGISTRY.json")
FROZEN_CODE = ("src/lightonocr_math/acceptance.py", "scripts/acceptance.py")
STOCK_PARAMS = {"num_ctx": 32768, "num_predict": 24576}
ENV_PREFIXES = ("OLLAMA_", "HIP_", "ROCR_", "HSA_", "GGML_", "CUDA_", "GPU_")
# Same floors baseline_infer uses for its own preflight, with the guard watchdog added around the whole run.
GUARD_ARGS = ["--vram-gb", "6", "--ram-gb", "2", "--timeout", "1800"]
PRECHECK_N = 100
EXPECTED = {"crops": 2688, "pages": 50, "page_rows": 297}
PAGE_MATCH_VERSION = "2"
RECORDS = Path("data/benchmark/records.jsonl")
PAGES = Path("data/benchmark/pages/pages.jsonl")
PAGE_SHARDS = Path("data/benchmark/shards/records")
ARM_NAMES = ("tuned-crops", "tuned-pages", "stock-crops", "stock-pages")


class Refused(Exception):
    """Generation or a check refused to start; the message names the reason."""


# ------------------------------------------------------------------------------------------- helpers


def _git(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=repo_root(), check=False)


def git_blob(path: str | Path) -> str:
    """Git blob hash of the file as it is on disk."""
    return _git(["hash-object", str(path)]).stdout.strip()


def is_committed_clean(path: str | Path) -> bool:
    """True when the file is tracked and has no change against HEAD."""
    return _git(["ls-files", "--error-unmatch", str(path)]).returncode == 0 and _git(["diff", "--quiet", "HEAD", "--", str(path)]).returncode == 0


def file_commit(path: str | Path) -> tuple[str | None, str | None]:
    """(commit hash, committer date) of the last commit that touched the file."""
    parts = _git(["log", "-1", "--format=%H %cI", "--", str(path)]).stdout.split()
    return (parts[0], parts[1]) if len(parts) == 2 else (None, None)


def git_head() -> str | None:
    return _git(["rev-parse", "HEAD"]).stdout.strip() or None


def run_cmd(cmd: list[str]) -> int:
    """Run an argument list (never a shell string) and return its exit code."""
    return subprocess.run(cmd, check=False).returncode


def read_server_env() -> dict | None:
    """OLLAMA_/HIP_/ROCR_/HSA_/GGML_/CUDA_/GPU_ variables of the running `ollama serve` process, None if unreadable."""
    try:
        pids = subprocess.run(["pgrep", "-f", "ollama serve"], capture_output=True, text=True, check=False).stdout.split()
        if not pids:
            return None
        raw = Path(f"/proc/{pids[0]}/environ").read_bytes()
    except OSError:
        return None
    env = {}
    for item in raw.split(b"\0"):
        key, sep, val = item.decode("utf-8", "replace").partition("=")
        if sep and key.startswith(ENV_PREFIXES):
            env[key] = val
    return env


def env_sha(env: dict | None) -> str | None:
    if env is None:
        return None
    return hashlib.sha256(json.dumps(env, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _show(host: str, model: str) -> dict:
    return _post(host, "/api/show", {"model": model}, 60)


def parse_params(text: str) -> dict:
    """Parameter lines of /api/show ("num_ctx    32768") as a dict, numbers converted."""
    out: dict = {}
    for line in (text or "").splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            val: object = parts[1].strip()
            try:
                val = int(val)
            except ValueError:
                pass
            out[parts[0]] = val
    return out


def _safe(fn, *args):
    """Result of fn, or None when the server cannot be reached or answers badly."""
    try:
        return fn(*args)
    except (OllamaError, urllib.error.URLError, OSError, ValueError, KeyError):
        return None


def _load(path: Path) -> dict | None:
    p = Path(path)
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _count_lines(path: Path) -> int:
    with open(path, encoding="utf-8") as fh:
        return sum(1 for ln in fh if ln.strip())


def _ps(host: str) -> list[str] | None:
    got = _safe(_get, host, "/api/ps")
    return None if got is None else [m.get("name", "") for m in got.get("models", [])]


def registry_digest(tag: str = TUNED_TAG) -> str | None:
    reg = _load(REGISTRY_PATH) or {}
    for e in reg.get("entries", []):
        if e.get("tag") == tag:
            return e.get("ollama_digest")
    return None


def _check(expected, actual) -> dict:
    return {"expected": expected, "actual": actual, "ok": actual is not None and actual == expected}


def _frozen_state() -> dict:
    return {p: {"blob": git_blob(p), "clean": is_committed_clean(p)} for p in FROZEN_CODE}


# ------------------------------------------------------------------------------------------ verify-env


def cmd_verify_env(a: argparse.Namespace) -> int:
    stock = _load(a.stock_manifest)
    if stock is None:
        print(f"stock manifest not found: {a.stock_manifest}", file=sys.stderr)
        return 2
    blobs = _safe(model_blobs, a.host, STOCK_TAG) or {}
    params = parse_params((_safe(_show, a.host, STOCK_TAG) or {}).get("parameters", ""))
    stock_params_actual = {k: params.get(k) for k in STOCK_PARAMS} if params else None
    version = (_safe(_get, a.host, "/api/version") or {}).get("version")
    stock_blobs = stock.get("gguf_blobs", {})
    checks = {
        "ollama_version": _check(accmod.ACCEPTANCE["engine"]["ollama_version"], version),
        "stock_digest": _check(stock.get("model_digest"), _safe(_model_digest, a.host, STOCK_TAG)),
        "projector_blob": _check(stock_blobs.get("projector"), blobs.get("projector")),
        "text_blob": _check(stock_blobs.get("text"), blobs.get("text")),
        "template_sha256": _check(stock.get("template_sha256"), blobs.get("template_sha256")),
        "stock_params": _check(dict(STOCK_PARAMS), stock_params_actual),
        "tuned_digest": _check(registry_digest(), _safe(_model_digest, a.host, TUNED_TAG)),
    }
    env = read_server_env()
    module_path = FROZEN_CODE[0]
    commit, commit_time = file_commit(module_path)
    frozen = _frozen_state()
    result = {
        "checks": checks,
        "pass": all(c["ok"] for c in checks.values()) and env is not None,
        "readiness": {
            "server_env": env,
            "server_env_sha256": env_sha(env),
            "ollama_ps": _ps(a.host),
            "gpu": gpu_state(),
            "mem_available_bytes": guard.mem_available_bytes(),
            "pin": {
                "acceptance_sha256": accmod.ACCEPTANCE_SHA256,
                "digest_now": accmod.constants_digest(),
                "commit": commit,
                "commit_time": commit_time,
                "clean": frozen[module_path]["clean"],
            },
            "frozen_code": frozen,
        },
    }
    write_json(a.out, result)
    for name, c in checks.items():
        print(f"{'ok  ' if c['ok'] else 'FAIL'} {name}: expected {c['expected']!r} actual {c['actual']!r}")
    print(f"server env readable: {env is not None}; MemAvailable {result['readiness']['mem_available_bytes'] / 1024**3:.1f} GiB")
    failed = [n for n, c in checks.items() if not c["ok"]]
    stock_changed = [n for n in failed if n != "tuned_digest"]
    if stock_changed:
        print(f"CHANGED: {', '.join(stock_changed)}: the stock arm must be re-run (D-07): generate --arm stock-crops / stock-pages")
    if "tuned_digest" in failed:
        print("CHANGED: tuned_digest: the tuned tag is not the registered model; stop")
    print("PASS" if result["pass"] else "FAIL")
    return 0 if result["pass"] else 1


# --------------------------------------------------------------------------------------------- precheck


def raw_text_cmp(a_rows: dict, b_rows: dict, ids: list[str]) -> dict:
    """Compare raw text of two runs over `ids` (rows keyed by id)."""
    missing = [i for i in ids if i not in a_rows or i not in b_rows]
    both = [i for i in ids if i not in missing]
    differing = [i for i in both if a_rows[i].get("raw_text") != b_rows[i].get("raw_text")]
    prompt_mismatch = [i for i in both if a_rows[i].get("prompt_eval_count") != b_rows[i].get("prompt_eval_count")]
    return {
        "n": len(ids),
        "n_identical": len(both) - len(differing),
        "differing_ids": differing,
        "missing_ids": missing,
        "prompt_count_mismatch_ids": prompt_mismatch,
    }


def prompt_offset_check(ol_rows: dict, hf_rows: dict, ids: list[str]) -> dict:
    """Ollama prompt token counts against the HF counts: every id measured and equal (offset 0)."""
    ol = {i: ol_rows[i] for i in ids if i in ol_rows}
    hf = {i: hf_rows[i] for i in ids if i in hf_rows}
    measured = [i for i in ids if i in ol and i in hf and ol[i].get("prompt_eval_count") is not None and hf[i].get("n_prompt_tokens") is not None]
    zero = [i for i in measured if ol[i]["prompt_eval_count"] == hf[i]["n_prompt_tokens"]]
    modal = modal_template_offset({i: hf[i] for i in measured}, {i: ol[i] for i in measured})
    return {
        "measured": len(measured),
        "modal_offset": modal,
        "n_offset_zero": len(zero),
        "pass": len(measured) == len(ids) and len(zero) == len(ids) and modal == 0,
    }


def _by_id(path: Path) -> dict:
    return {r["id"]: r for r in read_jsonl(path)}


def _guarded_infer(*, suite: str, records: Path, model: str, out: Path, sidecar: Path, host: str, ids_file: Path | None = None, resume: bool = False) -> list[str]:
    cmd = [
        sys.executable, "-m", "lightonocr_math.guard", *GUARD_ARGS, "--",
        sys.executable, "-m", "scripts.baseline_infer", "--suite", suite, "--records", str(records), "--model", model,
        "--out", str(out), "--sidecar", str(sidecar), "--host", host,
    ]
    if ids_file is not None:
        cmd += ["--ids-file", str(ids_file)]
    if resume:
        cmd.append("--resume")
    return cmd


def cmd_precheck(a: argparse.Namespace) -> int:
    ids = [ln.strip() for ln in a.ids_source.read_text(encoding="utf-8").splitlines() if ln.strip()][:PRECHECK_N]
    hf, stored_t, stored_s = _by_id(a.hf_sidecar), _by_id(a.stored_tuned), _by_id(a.stored_stock)
    absent = [i for i in ids if i not in hf or i not in stored_t or i not in stored_s]
    if len(ids) < PRECHECK_N or absent:
        print(f"precheck ids unusable: {len(ids)} ids, {len(absent)} missing from the HF or stored sidecars", file=sys.stderr)
        return 2
    work = a.work
    work.mkdir(parents=True, exist_ok=True)
    ids_file = work / "precheck_ids.txt"
    ids_file.write_text("".join(i + "\n" for i in ids), encoding="utf-8")
    runs = (("tuned_a", TUNED_TAG), ("tuned_b", TUNED_TAG), ("stock_a", STOCK_TAG))
    for name, model in runs:
        loaded = _ps(a.host)
        if loaded is None or loaded:
            print(f"ollama must be reachable with no model loaded before {name}; loaded: {loaded}", file=sys.stderr)
            return 2
        out, side = work / f"{name}.pred.jsonl", work / f"{name}.sidecar.jsonl"
        for stale in (out, side, out.with_name(out.stem + ".manifest.json")):
            stale.unlink(missing_ok=True)
        rc = run_cmd(_guarded_infer(suite="dev", records=a.dev_records, model=model, out=out, sidecar=side, host=a.host, ids_file=ids_file))
        if rc != 0:
            print(f"{name}: inference exited {rc}", file=sys.stderr)
            return 1 if rc == 1 else 2
    ta, tb, sa = (_by_id(work / f"{n}.sidecar.jsonl") for n in ("tuned_a", "tuned_b", "stock_a"))

    def _cmp_check(left: dict, right: dict) -> dict:
        res = raw_text_cmp(left, right, ids)
        return {**res, "ok": res["n_identical"] == len(ids) and not res["missing_ids"]}

    offset = prompt_offset_check(ta, hf, ids)
    checks = {
        "tuned_ab": _cmp_check(ta, tb),
        "tuned_vs_stored": _cmp_check(ta, stored_t),
        "prompt_offset": {**offset, "ok": offset["pass"]},
        "stock_vs_stored": _cmp_check(sa, stored_s),
    }
    env = read_server_env()
    version = (_safe(_get, a.host, "/api/version") or {}).get("version")
    result = {
        "ids_file": str(ids_file),
        "server_env": env,
        "server_env_sha256": env_sha(env),
        "ollama_version": version,
        "digests": {"tuned": _safe(_model_digest, a.host, TUNED_TAG), "stock": _safe(_model_digest, a.host, STOCK_TAG)},
        "frozen_code": {p: git_blob(p) for p in FROZEN_CODE},
        "checks": checks,
        "pass": all(c["ok"] for c in checks.values()) and env is not None,
    }
    write_json(a.out, result)
    for name, c in checks.items():
        print(f"{'ok  ' if c['ok'] else 'FAIL'} {name}")
    print("PASS" if result["pass"] else "FAIL")
    return 0 if result["pass"] else 1


# --------------------------------------------------------------------------------------------- generate


def arm_spec(name: str) -> dict:
    """Suite, records, model and output paths of one generation arm."""
    tuned = name.startswith("tuned")
    pages = name.endswith("pages")
    stem = ("tuned_q8" if tuned else "stock_rerun_q8") + ("_pages" if pages else "")
    spec = {
        "suite": "pages" if pages else "benchmark",
        "records": PAGES if pages else RECORDS,
        "model": TUNED_TAG if tuned else STOCK_TAG,
        "out": ACC_DIR / f"{stem}.pred.jsonl",
        "sidecar": ACC_DIR / f"{stem}.sidecar.jsonl",
        "expected": EXPECTED["pages"] if pages else EXPECTED["crops"],
        "text_key": "pred_text" if pages else "pred_latex",
    }
    if pages:
        pstem = "tuned_pages" if tuned else "stock_rerun_pages"
        spec["rows"] = ACC_DIR / f"{pstem}_rows.jsonl"
        spec["stats"] = ACC_DIR / f"{pstem}_match_stats.json"
    return spec


def _preconditions(a: argparse.Namespace, spec: dict) -> tuple[dict, str, str]:
    """Raise Refused unless generation may start; returns (precheck, expected digest, server env sha)."""
    try:
        accmod.assert_pinned()
    except accmod.AcceptanceError as e:
        raise Refused(str(e)) from e
    for p in FROZEN_CODE:
        if not is_committed_clean(p):
            raise Refused(f"{p} is not committed and clean")
    pre = _load(EVIDENCE_DIR / "precheck.json")
    if pre is None or pre.get("pass") is not True:
        raise Refused("precheck.json is missing or did not pass; run the precheck first")
    for p in FROZEN_CODE:
        if (pre.get("frozen_code") or {}).get(p) != git_blob(p):
            raise Refused(f"{p} changed since the precheck; re-run the precheck")
    sha = env_sha(read_server_env())
    if sha is None or sha != pre.get("server_env_sha256"):
        raise Refused("the Ollama server environment differs from the precheck (or cannot be read)")
    digest = _safe(_model_digest, a.host, spec["model"])
    if digest is None:
        raise Refused(f"cannot read the digest of {spec['model']} from the server")
    if spec["model"] == TUNED_TAG and digest != registry_digest():
        raise Refused(f"tuned digest {digest} differs from the registry digest {registry_digest()}")
    if spec["model"] == STOCK_TAG:
        env_check = _load(EVIDENCE_DIR / "env_check.json")
        if env_check is None or env_check.get("pass") is not False:
            raise Refused("stock arms are only re-run when env_check.json shows a failed environment check (D-07)")
    return pre, digest, sha


def _write_lock(lock: dict) -> None:
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_json(LOCK_PATH, lock)


def _finish(arm: str, spec: dict, entry: dict, lock: dict, digest: str) -> bool:
    """Verify a finished generation and mark the arm completed; False when any check fails."""
    out = spec["out"]
    rows = read_jsonl(out) if out.exists() else []
    n_err = sum(1 for r in rows if spec["text_key"] not in r)
    n_ok = len(rows) - n_err
    man = _load(out.with_name(out.stem + ".manifest.json")) or {}
    problems = []
    if len(rows) != spec["expected"]:
        problems.append(f"{len(rows)} rows, expected {spec['expected']}")
    if man.get("model_digest") != digest:
        problems.append(f"manifest model_digest {man.get('model_digest')} != {digest}")
    if "rows" in spec and not problems:
        rc = run_cmd([
            sys.executable, "-m", "scripts.corpus_pages", "rows", "--pages", str(PAGES), "--records", str(PAGE_SHARDS), "--preds", str(out),
            "--out", str(spec["rows"]), "--stats", str(spec["stats"]),
        ])
        stats = _load(spec["stats"]) or {}
        if rc != 0:
            problems.append(f"corpus_pages rows exited {rc}")
        elif not Path(spec["rows"]).is_file() or _count_lines(spec["rows"]) != EXPECTED["page_rows"]:
            problems.append(f"page rows file does not have {EXPECTED['page_rows']} rows")
        elif stats.get("page_match_version") != PAGE_MATCH_VERSION:
            problems.append(f"page_match_version {stats.get('page_match_version')} != {PAGE_MATCH_VERSION}")
    entry["n_ok"], entry["n_error"] = n_ok, n_err
    if problems:
        entry["problems"] = problems
        _write_lock(lock)
        print(f"{arm}: NOT completed: {'; '.join(problems)}", file=sys.stderr)
        return False
    entry.pop("problems", None)
    entry["completed"] = n_err == 0
    entry["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write_lock(lock)
    if n_err:
        print(f"{arm}: {n_err} request error(s); resume retries them: {_resume_hint(arm)}", file=sys.stderr)
    return n_err == 0


def _resume_hint(arm: str) -> str:
    return f"uv run python -m scripts.acceptance generate --arm {arm} --resume"


def cmd_generate(a: argparse.Namespace) -> int:
    spec = arm_spec(a.arm)
    try:
        return _generate(a, spec)
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2


def _generate(a: argparse.Namespace, spec: dict) -> int:
    _pre, digest, sha = _preconditions(a, spec)
    lock = _load(LOCK_PATH) or {"arms": {}}
    arms = lock.setdefault("arms", {})
    blobs = {"acceptance_module_blob": git_blob(FROZEN_CODE[0]), "acceptance_cli_blob": git_blob(FROZEN_CODE[1])}
    if a.arm == "tuned-pages" and not (arms.get("tuned-crops") or {}).get("completed"):
        raise Refused("tuned-pages starts only after tuned-crops is completed")
    entry = arms.get(a.arm)
    if entry is not None:
        if entry.get("completed") and entry.get("n_error") == 0:
            raise Refused(f"{a.arm} is completed without errors; it is never generated again")
        if not a.resume:
            raise Refused(f"{a.arm} was already started; only `{_resume_hint(a.arm)}` is allowed")
        for key, now in (("model_digest", digest), ("out", str(spec["out"])), *blobs.items()):
            if entry.get(key) != now:
                raise Refused(f"cannot resume {a.arm}: {key} is {now}, the lock has {entry.get(key)}")
        entry.setdefault("resumed", []).append(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        entry["completed"] = False
    else:
        if a.resume:
            raise Refused(f"{a.arm} has no lock entry, nothing to resume")
        if spec["out"].exists():
            raise Refused(f"{spec['out']} exists without a lock entry; refusing to overwrite predictions")
        commit, _ = file_commit(FROZEN_CODE[0])
        entry = arms[a.arm] = {
            "model": spec["model"], "model_digest": digest, "out": str(spec["out"]),
            "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "git_commit": git_head(),
            "acceptance_sha256": accmod.ACCEPTANCE_SHA256, "acceptance_commit": commit, **blobs,
            "server_env_sha256": sha, "completed": False,
        }
    _write_lock(lock)

    spec["out"].parent.mkdir(parents=True, exist_ok=True)
    env_before = {"ps": _ps(a.host), "gpu": gpu_state(), "mem_available_bytes": guard.mem_available_bytes()}
    rc = run_cmd(_guarded_infer(
        suite=spec["suite"], records=spec["records"], model=spec["model"], out=spec["out"], sidecar=spec["sidecar"], host=a.host, resume=a.resume,
    ))
    write_json(
        spec["out"].with_name(spec["out"].stem + ".run_env.json"),
        {
            "server_env": read_server_env(), "before": env_before,
            "after": {"ps": _ps(a.host), "gpu": gpu_state(), "mem_available_bytes": guard.mem_available_bytes()}, "exit_code": rc,
        },
    )
    if rc != 0:
        rows = read_jsonl(spec["out"]) if spec["out"].exists() else []
        entry["n_error"] = sum(1 for r in rows if spec["text_key"] not in r)
        entry["n_ok"] = len(rows) - entry["n_error"]
        entry["last_exit_code"] = rc
        _write_lock(lock)
        print(f"{a.arm}: generation exited {rc}; continue with: {_resume_hint(a.arm)}", file=sys.stderr)
        return 1
    return 0 if _finish(a.arm, spec, entry, lock, digest) else 1


# ------------------------------------------------------------------------------------------------ main


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("verify-env", help="check engine and stock model against the stored stock run (D-07)")
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--stock-manifest", type=Path, default=Path("data/benchmark/baseline/stock_q8.pred.manifest.json"))
    p.add_argument("--out", type=Path, default=None)
    p.set_defaults(func=cmd_verify_env)

    p = sub.add_parser("precheck", help="dev pre-check on 100 fixed dev crops (D-08, D-22)")
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--ids-source", type=Path, default=Path("data/train/dev_eval_ids.txt"))
    p.add_argument("--hf-sidecar", type=Path, default=Path("data/models/p4-full-1/hf_merged.sidecar.jsonl"))
    p.add_argument("--dev-records", type=Path, default=Path("data/corpus/dev/records.jsonl"))
    p.add_argument("--stored-tuned", type=Path, default=Path("data/runs/p4-full-1/dev_cand.sidecar.jsonl"))
    p.add_argument("--stored-stock", type=Path, default=Path("data/runs/p4-full-1/dev_stock.sidecar.jsonl"))
    p.add_argument("--work", type=Path, default=Path("data/runs/p5-precheck"))
    p.add_argument("--out", type=Path, default=None)
    p.set_defaults(func=cmd_precheck)

    p = sub.add_parser("generate", help="lock-protected, resume-only benchmark generation of one arm (D-19)")
    p.add_argument("--arm", required=True, choices=ARM_NAMES)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--host", default=DEFAULT_HOST)
    p.set_defaults(func=cmd_generate)
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    if getattr(a, "out", 1) is None:
        a.out = EVIDENCE_DIR / ("env_check.json" if a.cmd == "verify-env" else "precheck.json")
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
