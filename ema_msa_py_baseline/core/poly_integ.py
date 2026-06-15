"""Closed-form polynomial integration translated from the MATLAB EMA_MSA codebase."""

from __future__ import annotations

import numpy as np


def _poly_integ_single(c: np.ndarray, p: np.ndarray, lim: tuple[float, float]) -> np.ndarray:
    """Evaluate one MATLAB-style ``poly_integ(c, p, lim)`` block."""

    coeff, powers = np.broadcast_arrays(
        np.asarray(c, dtype=np.float64),
        np.asarray(p, dtype=np.float64),
    )
    lower = float(lim[0])
    upper = float(lim[1])

    invx = powers == -1.0
    pow_new = powers + 1.0

    if np.any(invx):
        if lower <= 0.0 or upper <= 0.0:
            if np.all(coeff[invx] == 0.0):
                pow_new = pow_new.copy()
                pow_new[invx] = 1.0
            else:
                raise ValueError("Singularity in integrand.")
        else:
            pow_new = pow_new.copy()
            pow_new[invx] = 1.0

    coeff_new = coeff / pow_new
    result = coeff_new * (upper**pow_new - lower**pow_new)
    result = np.asarray(result, dtype=np.float64)
    result[coeff == 0.0] = 0.0

    if np.any(invx) and not np.all(coeff[invx] == 0.0):
        result[invx] = coeff[invx] * (np.log(upper) - np.log(lower))

    return result


def poly_integ(c: np.ndarray, p: np.ndarray, lim: tuple[float, float]) -> np.ndarray:
    """Integrate ``c * x**p`` in closed form.

    Source: ``Ema_Msa_7/poly_integ.m`` lines 7-37.
    """

    return _poly_integ_single(c, p, lim)


def poly_integ_4term(
    c1: np.ndarray,
    c2: np.ndarray,
    c3: np.ndarray,
    c4: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    p4: np.ndarray,
    lim: tuple[float, float],
) -> np.ndarray:
    """Sum the four-term MATLAB ``poly_integ`` form.

    Source: ``Ema_Msa_7/poly_integ.m`` lines 40-82.
    """

    return (
        _poly_integ_single(c1, p1, lim)
        + _poly_integ_single(c2, p2, lim)
        + _poly_integ_single(c3, p3, lim)
        + _poly_integ_single(c4, p4, lim)
    )


__all__ = ["poly_integ", "poly_integ_4term"]
