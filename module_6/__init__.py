"""Module 6: Permanently Shadowed Region (PSR) and Doubly-Shadowed Crater Detection."""
from .psr_detection import (
    compute_psr_mask,
    identify_craters,
    identify_doubly_shadowed_craters,
    compute_horizon_mask,
    make_synthetic_psr_data,
    PSRResult,
)

__all__ = [
    "compute_psr_mask",
    "identify_craters",
    "identify_doubly_shadowed_craters",
    "compute_horizon_mask",
    "make_synthetic_psr_data",
    "PSRResult",
]
