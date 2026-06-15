"""Core mathematical utilities for the Python EMA_MSA replication."""

from .beam_func import beam_func
from .gauss_w import gauss_w
from .hooke import hooke, hooke_from_args
from .poly_integ import poly_integ, poly_integ_4term
from .polyexp import polyexp

__all__ = [
    "beam_func",
    "gauss_w",
    "hooke",
    "hooke_from_args",
    "poly_integ",
    "poly_integ_4term",
    "polyexp",
]
