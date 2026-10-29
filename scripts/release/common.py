"""Constants, layout contract and helpers shared by every release step.

Every pinned digest, count, placeholder token, license rule, forbidden pattern and staging path lives here, so the
release check recomputes from one place instead of re-deciding. The tests cross-check each digest against its source
of truth (REGISTRY.json, TRAINSET_MANIFEST.json, BENCHMARK_FREEZE.json, render.IMAGE).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from scripts.release.freeze_amendment import NEW_EVIDENCE_DIR

# ---- staging layout (gitignored, local only) -------------------------------------------------------------------
STAGING = Path("release")
DATASET_DIR = STAGING / "dataset"
MODEL_DIR = STAGING / "model"
CODE_DIR = STAGING / "code"
OVERLAY_DIR = CODE_DIR / "overlay"
WORK_CLONE = CODE_DIR / "work"
PUBLIC_GIT = CODE_DIR / "public.git"
REHEARSAL_DIR = STAGING / "rehearsal"
PUBLISH_MD = STAGING / "PUBLISH.md"
NAMES_JSON = STAGING / "NAMES.json"

# relative to DATASET_DIR
TRAIN_PARQUET = "data/train-00000-of-00001.parquet"
BENCH_PARQUET = "data/benchmark-00000-of-00001.parquet"

GGUF_NAME = "text-q8_0.gguf"
MMPROJ_NAME = "mmproj-stock-q8_0.gguf"
ADAPTER_SUBDIR = ""  # adapter files sit at the repository root so the Hub detects the PEFT adapter

TRAIN_COLUMNS = (
    "id", "image", "latex", "target_text", "arxiv_id", "paper_version", "title", "authors", "license_spdx",
    "license_uri", "arxiv_url", "flags", "image_sha256", "bitmap_sha256", "width", "height", "group",
    "mining_class", "control_glyph", "tags",
)
BENCH_COLUMNS = (
    "id", "image", "latex", "arxiv_id", "paper_version", "title", "authors", "license_spdx", "license_uri",
    "arxiv_url", "flags", "image_sha256", "bitmap_sha256", "width", "height", "render_image_digest",
    "preamble_sha256", "canon_version", "builder_version", "env",
)
# Dropped from the Parquet; the frozen data/benchmark/records.jsonl keeps them because its bytes are pinned.
STRIPPED_BENCH_FIELDS = ("raw_body", "source_file", "source_line")

# ---- placeholders (D-07): the chosen names never enter code or history -----------------------------------------
PLACEHOLDERS = {
    "dataset_name": "@@DATASET_NAME@@",
    "model_name": "@@MODEL_NAME@@",
    "repo_name": "@@REPO_NAME@@",
    "hf_namespace": "@@HF_NAMESPACE@@",
    "gh_owner": "@@GH_OWNER@@",
    "contact_email": "@@CONTACT_EMAIL@@",
    "copyright_holder": "@@COPYRIGHT_HOLDER@@",
}

# ---- pinned digests (cross-checked in tests) --------------------------------------------------------------------
TRAINSET_SHA256 = "4b5e5b3b01e3eeaf42d49531545a791f81fdc6cc905ade9c3cc215c55584e0b0"
ADAPTER_SHA256 = "11277e81ecab00881328ad69b29d5d82a21013dd6c3ba754e5ddae15e674cdae"
GGUF_Q8_SHA256 = "724e739d505c31d8466b0f9ee88f4d406b8ccd4786f2baef9f4fa5a8b21772be"
PROJECTOR_SHA256 = "fd5acde1a7a7439b3f9a720d3cead67378beeaabbecc57fe4b30a7f9b301b102"
OLLAMA_DIGEST = "c06474f6789e896e8acbddf0cb91dad88b0a29c563ee58d76bd31d846351e737"
BASE_MODEL_ID = "lightonai/LightOnOCR-3-4B"
BASE_REVISION = "b71010f095dc3735de7ee6b9969bff8f8f50b39a"
LLAMA_CPP_COMMIT = "a17e94d229fc7fed1567a799baebd11b771fe5af"
TEXLIVE_IMAGE = "texlive/texlive@sha256:0f4884ca7d287f47ab466d2dd405d4c901c576585363f628863a7ca7eea21301"
REGISTRY_TAG = "glyphfix-math-lora-q8:p4-full-1"

EXPECTED = {
    "train_rows": 8000,
    "train_papers": 767,
    "train_licenses": {"CC-BY-4.0": 7560, "CC0-1.0": 440},
    "bench_rows": 2688,
    "bench_papers": 150,
    "bench_licenses": {"CC-BY-4.0": 2477, "CC0-1.0": 211},
}

# ---- licenses -------------------------------------------------------------------------------------------------
ALLOWED_SPDX = {"CC-BY-4.0", "CC-BY-3.0", "CC0-1.0"}
LICENSE_URI = {
    "CC-BY-4.0": "http://creativecommons.org/licenses/by/4.0/",
    "CC-BY-3.0": "http://creativecommons.org/licenses/by/3.0/",
    "CC0-1.0": "http://creativecommons.org/publicdomain/zero/1.0/",
}
LICENSE_DEED = {
    "CC-BY-4.0": "https://creativecommons.org/licenses/by/4.0/legalcode",
    "CC-BY-3.0": "https://creativecommons.org/licenses/by/3.0/legalcode",
    "CC0-1.0": "https://creativecommons.org/publicdomain/zero/1.0/legalcode",
}

DISCLAIMER = "This is an independent fine-tune. It is not affiliated with, endorsed by, or sponsored by LightOn."
DATASET_DISCLAIMER = (
    "This dataset is an independent project. It is not affiliated with, endorsed by, or sponsored by LightOn."
)
NAME_FORBIDDEN = re.compile(r"light[\s_-]*on", re.IGNORECASE)

# ---- forbidden content and file names ---------------------------------------------------------------------------
# The maintainer's personal address must not appear in any public blob, commit header or message; the work address (the
# arXiv User-Agent contact in the source) is scrubbed from the public history the same way. The literals are split
# so this file itself does not contain them.
PERSONAL_ADDRESS = re.compile(r"olivier\.gaude" + r"@gmail\.com", re.IGNORECASE)
WORK_ADDRESS = re.compile(r"olivier\.gaude" + r"@karbo" + r"tium\.com", re.IGNORECASE)
PUBLIC_PLACEHOLDER_EMAIL = "contact@example.invalid"
# Identity email written into every exported commit (author and committer); names and dates are kept.
PUBLIC_IDENTITY_EMAIL = "noreply@example.invalid"
# Blob rewrites applied to kept paths while exporting history: (pattern, replacement).
SCRUB_REPLACEMENTS = ((PERSONAL_ADDRESS, PUBLIC_PLACEHOLDER_EMAIL), (WORK_ADDRESS, PUBLIC_PLACEHOLDER_EMAIL))
# Rewrites applied to commit messages while exporting history.
MESSAGE_REPLACEMENTS = (
    (re.compile("Nexa" + "ris", re.IGNORECASE), "a private project"),
    (PERSONAL_ADDRESS, PUBLIC_PLACEHOLDER_EMAIL),
    (WORK_ADDRESS, PUBLIC_PLACEHOLDER_EMAIL),
)

FORBIDDEN_PATTERNS = (
    ("personal_email", PERSONAL_ADDRESS),
    ("work_email", WORK_ADDRESS),
    ("local_path_mnt", re.compile(r"/mnt" + r"/datas")),
    ("local_path_home", re.compile(r"/home" + r"/ogaude")),
    ("hf_token", re.compile(r"hf_[A-Za-z0-9]{30,}")),
    ("github_token", re.compile(r"ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}")),
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("origin_project", re.compile("Nexa" + "ris", re.IGNORECASE)),
    ("comparison_dataset_a", re.compile(r"im2" + r"latex")),
    ("comparison_dataset_b", re.compile(r"LaTeX" + r"_OCR")),
    ("unsloth_studio", re.compile(r"Unsloth" + r" Studio")),
)

# Applied with re.search to a POSIX path relative to the repo or staging root. The private-data rule is anchored to
# the gitignored data directories on purpose: the kept scripts/spikes/** code is imported by kept scripts.
FORBIDDEN_FILE_PATTERNS = (
    ("env", re.compile(r"(^|/)\.env$")),
    ("key", re.compile(r"\.key$")),
    ("pem", re.compile(r"\.pem$")),
    ("bf16_gguf", re.compile(r"bf16\.gguf$")),
    ("private_data", re.compile(r"^data/(probes|spikes|external)/")),
)

# Internal decision labels such as "(D-02)" or "(D-11..D-13)" never appear in the generated public documents.
INTERNAL_LABEL = re.compile(r"\(D-\d+")
_INTERNAL_LABEL_FULL = re.compile(r"[ \t]*\(D-\d+(?:[ \t]*(?:,|\.\.)[ \t]*D-\d+)*\)")


def strip_internal_labels(text: str) -> str:
    """Remove parenthesised decision labels (and the space before them) from generated public text."""
    return _INTERNAL_LABEL_FULL.sub("", text)


def strip_mining_labels(text: str) -> str:
    """Drop the internal decision labels from the mining report: "(D-06)" and the bare "under D-09"."""
    return strip_internal_labels(re.sub(r" under D-\d+", "", text))


def phase4_public_bytes(src: Path) -> bytes:
    """The bytes a Phase 4 report has in the public results/: byte copies, except MINING.md loses its labels."""
    if src.name == "MINING.md":
        return strip_mining_labels(src.read_text(encoding="utf-8")).encode("utf-8")
    return src.read_bytes()


# Keys of public JSON/JSONL rows that hold LaTeX or model output; a row with one must belong to an attributed paper.
# The set comes from a census of every key in every JSON/JSONL file of the public tree (06-08 summary).
CONTENT_KEYS = {
    "latex", "gt_latex", "pred_latex", "raw_body", "target_text", "stock_pred", "tuned_pred", "stock_pred_latex",
    "pred_norm", "gt_norm", "raw_text", "pred_text",
}

# ---- scorer evidence kept in the public tree --------------------------------------------------------------------
PHASE5_EVIDENCE = NEW_EVIDENCE_DIR
RESULTS_JSON = PHASE5_EVIDENCE / "results.json"
FLIPS_JSONL = PHASE5_EVIDENCE / "flips.jsonl"
ENV_CHECK_JSON = PHASE5_EVIDENCE / "env_check.json"
SENSITIVITY_JSON = PHASE5_EVIDENCE / "sensitivity.json"
ACCEPTANCE_MD = PHASE5_EVIDENCE / "ACCEPTANCE.md"
FREEZE_AMENDMENT_JSON = PHASE5_EVIDENCE / "FREEZE_AMENDMENT.json"
FREEZE_AMENDMENT_2_JSON = PHASE5_EVIDENCE / "FREEZE_AMENDMENT_2.json"

PUBLIC_EVIDENCE_FILES = (
    RESULTS_JSON.as_posix(),  # scripts/acceptance.py report/register-verdict, tests/test_acceptance.py
    FLIPS_JSONL.as_posix(),  # scripts/acceptance.py score output
    ENV_CHECK_JSON.as_posix(),  # scripts/acceptance.py score --env-check default
    SENSITIVITY_JSON.as_posix(),  # scripts/acceptance.py score, tests/test_acceptance.py
    ACCEPTANCE_MD.as_posix(),  # scripts/acceptance.py report
    FREEZE_AMENDMENT_JSON.as_posix(),  # freeze_amendment.verify_amendment
    FREEZE_AMENDMENT_2_JSON.as_posix(),  # rename_amendment.verify_rename
)

# Development-only sources that 06-06 byte-copies into results/ (D-04).
_P4 = Path(".planning/phases/04-error-mining-and-lora-training")
PHASE4_SOURCES = (
    _P4 / "reports/DEV_ANALYSIS.md",
    _P4 / "reports/MINING.md",
    _P4 / "reports/PARITY_DIAGNOSIS.md",
    _P4 / "evidence/dev_gain_by_class.json",
)

RESERVED_SOURCES = (Path("data/corpus/train/records.jsonl"), Path("data/corpus/dev/records.jsonl"))

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")


def _contact_forbidden() -> frozenset[str]:
    from glyphfix_math.corpus.catalog import UA

    return frozenset(_EMAIL_RE.findall(UA))


# The arXiv User-Agent address is never the release contact; later plans reject it.
CONTACT_FORBIDDEN = _contact_forbidden()


# ---- helpers -----------------------------------------------------------------------------------------------------
SHA256SUMS = "SHA256SUMS"


def sha256_stream(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def iter_files(root: Path | str) -> list[Path]:
    root = Path(root)
    return sorted(p for p in root.rglob("*") if p.is_file())


def write_sha256sums(root: Path | str) -> Path:
    root = Path(root)
    lines = [
        f"{sha256_stream(p)}  {p.relative_to(root).as_posix()}"
        for p in iter_files(root)
        if p.relative_to(root).as_posix() != SHA256SUMS
    ]
    lines.sort(key=lambda s: s.split("  ", 1)[1])
    out = root / SHA256SUMS
    out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return out


def verify_sha256sums(root: Path | str) -> list[str]:
    root = Path(root)
    problems: list[str] = []
    listed: dict[str, str] = {}
    sums = root / SHA256SUMS
    if not sums.is_file():
        return [f"missing {SHA256SUMS}"]
    for line in sums.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, _, rel = line.partition("  ")
        listed[rel] = digest
    for rel, digest in listed.items():
        p = root / rel
        if not p.is_file():
            problems.append(f"missing: {rel}")
        elif sha256_stream(p) != digest:
            problems.append(f"changed: {rel}")
    for p in iter_files(root):
        rel = p.relative_to(root).as_posix()
        if rel != SHA256SUMS and rel not in listed:
            problems.append(f"unlisted: {rel}")
    return problems


_SLOT_RE = re.compile(r"\{\{([a-z_][a-z0-9_]*)\}\}")


def render_template(text: str, values: dict[str, str]) -> str:
    """Replace {{slot}} slots; fail closed on a missing value. Other brace text (e.g. {{ .Prompt }}) is untouched."""

    def sub(m: re.Match) -> str:
        name = m.group(1)
        if name not in values:
            raise KeyError(f"no value for template slot {{{{{name}}}}}")
        return str(values[name])

    out = _SLOT_RE.sub(sub, text)
    left = _SLOT_RE.search(out)
    if left:
        raise ValueError(f"template slot left after rendering: {left.group(0)}")
    return out


def relative_summary(h: dict) -> dict[str, str]:
    """The headline in relative terms, from the same numbers as the points: errors removed, correct crops gained."""
    n = h["n"]
    right = round(h["stock_rate"] * n)
    r2w, w2r = h["flip"]["right_to_wrong"], h["flip"]["wrong_to_right"]
    wrong_before, wrong_after = n - right, n - right + r2w - w2r
    return {
        "net_fixed": str(w2r - r2w),
        "wrong_before": str(wrong_before),
        "wrong_after": str(wrong_after),
        "fewer_wrong_pct": f"{(wrong_before - wrong_after) / wrong_before:.0%}",
        "more_right_pct": f"{(w2r - r2w) / right:.0%}",
        "r2w_of_right_pct": f"{r2w / right:.1%}",
        "flip_ratio": f"{w2r / r2w:.1f}",
    }


def load_json(path: Path | str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: Path | str) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def caveats(results: dict) -> list[str]:
    """The D-05 caveat sentences, built only from values in results.json."""
    t = results["transfer"]
    weak = [c["class"] for c in results["per_class"] if c["claim"] == "not enough events to claim"]
    dev = results["dev_reference"]
    out = [
        (
            "Page-level transfer holds under the pre-committed rule but is not a proven gain: the paired "
            f"paper-level 95% CI is [{t['ci_low']:.4f}, {t['ci_high']:.4f}]."
        ),
    ]
    pm = results.get("page_match")
    if pm:
        out.append(
            f"On full pages this model matched {pm['tuned']['n_matched']} of {pm['tuned']['n_gt']} ground-truth "
            f"equations against {pm['stock']['n_matched']} for the base model. In the page outputs we analysed, almost "
            "all of the equations it lost sit on two pages where it left a `$$` fence unbalanced, so the extractor "
            "mispaired the rest of those pages (the equation text itself was present), and one more is a multi-line "
            "equation it wrote as three blocks; a few equations were gained. The metric depends on how each model "
            "fences and groups display equations."
        )
    if weak:
        out.append(
            f"The classes {', '.join(weak)} have too few base-model errors to claim a change."
            if len(weak) > 1
            else f"The class {weak[0]} has too few base-model errors to claim a change."
        )
    out.append(
        f"The dev gain (delta {dev['delta']:.4f}) is in-sample because the checkpoint was selected on the dev split."
    )
    out.append(f"Results are published exactly as scored by the frozen scorer {SCORER_VERSION}.")
    return out


SCORER_VERSION = "1.2.0"
