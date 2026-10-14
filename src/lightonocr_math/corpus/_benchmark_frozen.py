"""Pin of the frozen held-out benchmark.

BENCHMARK_FREEZE_SHA256 is the sha256 hex digest of data/benchmark/BENCHMARK_FREEZE.json, which records the
digest of data/benchmark/records.jsonl. Changing it requires a new benchmark_version and a new stock baseline.
"""

BENCHMARK_FREEZE_SHA256: str = "b5c62fc248d36f1a26eea9563c0a000589777b71bac2c1d0ec792f4020b1e2b8"

# sha256 hex digest of data/benchmark/pages/PAGES_FREEZE.json, which pins the page-level eval set (pages.jsonl digest,
# page image digest, equation-matching threshold and match version).
PAGES_FREEZE_SHA256: str = "c83e3a44bdd245c64d9b7fa6cdca9adebec13a19d9ec4474a7da4ce802f4c8c2"
