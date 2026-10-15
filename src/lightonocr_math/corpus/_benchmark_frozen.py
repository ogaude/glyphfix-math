"""Pin of the frozen held-out benchmark.

BENCHMARK_FREEZE_SHA256 is the sha256 hex digest of data/benchmark/BENCHMARK_FREEZE.json, which records the
digest of data/benchmark/records.jsonl. Changing it requires a new benchmark_version and a new stock baseline.
"""

BENCHMARK_FREEZE_SHA256: str = "b5c62fc248d36f1a26eea9563c0a000589777b71bac2c1d0ec792f4020b1e2b8"

# sha256 hex digest of data/benchmark/pages/PAGES_FREEZE.json, which pins the page-level eval set (pages.jsonl digest,
# page image digest, equation-matching threshold and match version).
PAGES_FREEZE_SHA256: str = "c83e3a44bdd245c64d9b7fa6cdca9adebec13a19d9ec4474a7da4ce802f4c8c2"

# sha256 hex digests of the frozen records themselves (also recorded inside the freeze files above). Pinned here so a
# guard can recognise the frozen content by its bytes even when the freeze file or the path is missing or different.
BENCHMARK_RECORDS_SHA256: str = "95710217ca31a301e14998e6480da1e40263d1c0a15fdb146d35a29baebfeba2"
PAGES_JSONL_SHA256: str = "b00a72987d7edcd7f19ac7fb52ca4113d9aae3c69988ec5b25f153d1caf82aa9"
