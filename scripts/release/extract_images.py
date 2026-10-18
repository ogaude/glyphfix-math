"""Write the benchmark crop PNGs from the published Parquet to the paths the frozen records.jsonl expects.

Usage (from the repo root):
  uv run python -m scripts.release.extract_images --parquet <benchmark parquet> [--records data/benchmark/records.jsonl]

Every image is hashed against the Parquet's image_sha256 and the record's image_sha256 before it is written. An
identical existing file is skipped, a different one is never overwritten, and a record path outside
data/benchmark/images/ is refused.

Exit codes: 0 done, 1 a mismatch, refusal or missing record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

IMAGES_ROOT = "data/benchmark/images"


class ExtractError(RuntimeError):
    pass


def _safe_target(rel: str, base: Path) -> Path:
    p = Path(rel)
    if p.is_absolute() or ".." in p.parts or not (rel.startswith(IMAGES_ROOT + "/")):
        raise ExtractError(f"refusing image path outside {IMAGES_ROOT}/: {rel}")
    return base / p


def extract(parquet: Path, records: Path, base: Path = Path(".")) -> dict:
    import pyarrow.parquet as pq

    recs = {}
    for line in Path(records).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            recs[r["id"]] = r
    written = skipped = 0
    seen: set[str] = set()
    pf = pq.ParquetFile(parquet)
    for batch in pf.iter_batches(columns=["id", "image", "image_sha256"], batch_size=256):
        for row in batch.to_pylist():
            rid = row["id"]
            rec = recs.get(rid)
            if rec is None:
                raise ExtractError(f"{rid}: no record in {records}")
            data = row["image"]["bytes"]
            digest = hashlib.sha256(data).hexdigest()
            if digest != row["image_sha256"] or digest != rec["image_sha256"]:
                raise ExtractError(f"{rid}: image bytes sha256 {digest} differs from the recorded image_sha256")
            target = _safe_target(rec["image"], base)
            if target.exists():
                if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                    raise ExtractError(f"{rid}: {rec['image']} exists with different content; not overwritten")
                skipped += 1
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                written += 1
            seen.add(rid)
    missing = sorted(set(recs) - seen)
    if missing:
        raise ExtractError(f"{len(missing)} records have no image in the Parquet, first: {missing[0]}")
    return {"written": written, "skipped": skipped, "records": len(recs)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="extract_images", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", type=Path, required=True)
    ap.add_argument("--records", type=Path, default=Path("data/benchmark/records.jsonl"))
    a = ap.parse_args(argv)
    try:
        info = extract(a.parquet, a.records)
    except (ExtractError, OSError, KeyError) as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
