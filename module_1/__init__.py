"""Module 1: Radar Polarimetric Decomposition (DFSAR)."""
from .dfsar_reader import DFSARProduct, load_dfsar_product, load_dfsar_product_synthetic
from .polarimetry import compute_cpr, compute_dop, build_ice_mask
from .crim import invert_ice_fraction, compute_ice_volume

__all__ = [
    "DFSARProduct",
    "load_dfsar_product",
    "load_dfsar_product_synthetic",
    "compute_cpr",
    "compute_dop",
    "build_ice_mask",
    "invert_ice_fraction",
    "compute_ice_volume",
]
