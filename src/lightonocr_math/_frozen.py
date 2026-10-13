"""Pin of the committed split artefacts.

FROZEN_SPLIT_DIGEST is the sha256 hex digest of data/splits/SHA256SUMS. It stays None until
the split is frozen and approved; once set, importing lightonocr_math.splits verifies every
committed split file against it. Changing the pinned value means the held-out benchmark
changed, which requires a new SPLIT_RULE_VERSION.
"""

FROZEN_SPLIT_DIGEST: str | None = None
