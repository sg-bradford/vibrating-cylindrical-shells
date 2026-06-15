"""Gauss-Legendre quadrature utilities translated from the MATLAB EMA_MSA codebase."""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss

_SUPPORTED_ORDERS = {2, 4, 6, 8, 10, 12, 14, 20, 24}
_DEFAULT_ORDER = 20


def _matlab_ordered_leggauss(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return Gauss-Legendre nodes ordered like the MATLAB lookup tables."""

    xi, wi = leggauss(n)
    positive_mask = xi > 0.0
    xi_pos = np.sort(xi[positive_mask])
    wi_pos = wi[positive_mask][np.argsort(xi[positive_mask])]
    xi_out = np.concatenate((xi_pos, -xi_pos)).astype(np.float64, copy=False)
    wi_out = np.concatenate((wi_pos, wi_pos)).astype(np.float64, copy=False)
    return xi_out, wi_out


def gauss_w(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return Gauss-Legendre abscissae and weights.

    Source: ``Ema_Msa_7/gauss_w.m`` lines 9-192.
    Unsupported orders follow the MATLAB fallback behavior and return the
    20-point rule.
    """

    order = int(n)
    if order not in _SUPPORTED_ORDERS:
        order = _DEFAULT_ORDER
    return _matlab_ordered_leggauss(order)


__all__ = ["gauss_w"]
