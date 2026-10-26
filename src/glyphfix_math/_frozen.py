"""Pin of the committed split artefacts.

FROZEN_SPLIT_DIGEST is the sha256 hex digest of data/splits/SHA256SUMS. It stays None until
the split is frozen and approved; once set, importing glyphfix_math.splits verifies every
committed split file against it. Changing the pinned value means the held-out benchmark
changed, which requires a new SPLIT_RULE_VERSION.
"""

FROZEN_SPLIT_DIGEST: str | None = "b0afe4144600ba3e7d19a6d37d85344a080fae84364aeff562c11e475ac3ab5e"
