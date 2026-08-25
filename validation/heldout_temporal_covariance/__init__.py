"""Metadata-only held-out cohort contracts for issue #53."""

from .cohort import (
    CohortValidationError,
    build_manifest,
    canonical_digest,
    discover_candidates,
    validate_manifest,
)
from .scorer import (
    score_receipt,
    score_slope_difference,
)

__all__ = [
    "CohortValidationError",
    "build_manifest",
    "canonical_digest",
    "discover_candidates",
    "score_receipt",
    "score_slope_difference",
    "validate_manifest",
]
