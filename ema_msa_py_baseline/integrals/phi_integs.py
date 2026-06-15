"""Circumferential integrals translated from the MATLAB EMA_MSA codebase."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
FloatMatrix = NDArray[np.float64]


def _sector_limits(phi_start: float, phi_end: float, q: int) -> tuple[FloatArray, FloatArray]:
    """Reproduce the sector replication in ``Ema_Msa_7/phi_integs.m`` lines 16-23."""

    if q < 0:
        raise ValueError("q must be non-negative.")

    start_value = float(phi_start)
    end_value = float(phi_end)
    if end_value < start_value:
        raise ValueError("phi_end must be greater than or equal to phi_start.")

    if q in {0, 1}:
        starts = np.array([start_value], dtype=np.float64)
        ends = np.array([end_value], dtype=np.float64)
        return starts, ends

    qi = np.arange(q, dtype=np.float64)
    offset = (2.0 * math.pi / float(q)) * qi
    return start_value + offset, end_value + offset


def _sin_sin_integral(n: int, x0: FloatArray, x1: FloatArray) -> FloatArray:
    """Evaluate ``∫ sin(nφ)^2 dφ`` over each sector."""

    if n == 0:
        return np.zeros_like(x0, dtype=np.float64)

    n_value = float(n)
    return (2.0 * n_value * (x1 - x0) + np.sin(2.0 * n_value * x0) - np.sin(2.0 * n_value * x1)) / (
        4.0 * n_value
    )


def _cos_cos_integral(n: int, x0: FloatArray, x1: FloatArray) -> FloatArray:
    """Evaluate ``∫ cos(nφ)^2 dφ`` over each sector."""

    if n == 0:
        return x1 - x0

    n_value = float(n)
    return (2.0 * n_value * (x1 - x0) + np.sin(2.0 * n_value * x1) - np.sin(2.0 * n_value * x0)) / (
        4.0 * n_value
    )


def _cross_integral(n: int, x0: FloatArray, x1: FloatArray) -> FloatArray:
    """Evaluate ``∫ cos(nφ) sin(nφ) dφ`` over each sector."""

    if n == 0:
        return np.zeros_like(x0, dtype=np.float64)

    n_value = float(n)
    return (np.cos(2.0 * n_value * x0) - np.cos(2.0 * n_value * x1)) / (4.0 * n_value)


def phi_integs(n: int, phi_start: float, phi_end: float, q: int = 1) -> dict[str, int | float | FloatArray]:
    """Compute circumferential sector integrals.

    Faithful translation of the closed-form formulas in ``Ema_Msa_7/phi_integs.m``
    lines 12-31, with the MATLAB struct-array output flattened into a Python
    dictionary for one circumferential order and one angular span.
    """

    n_value = int(n)
    q_value = int(q)
    sector_starts, sector_ends = _sector_limits(phi_start, phi_end, q_value)

    i_sin = _sin_sin_integral(n_value, sector_starts, sector_ends)
    i_cos = _cos_cos_integral(n_value, sector_starts, sector_ends)
    i_cross = _cross_integral(n_value, sector_starts, sector_ends)

    return {
        "n": n_value,
        "phi_start": float(phi_start),
        "phi_end": float(phi_end),
        "q": q_value,
        "sector_starts": sector_starts,
        "sector_ends": sector_ends,
        "I_sin": i_sin,
        "I_cos": i_cos,
        "I_sin_sin": i_sin,
        "I_cos_cos": i_cos,
        "I_sin_cos": i_cross,
        "I_cos_sin": i_cross,
        "I_sin_total": float(np.sum(i_sin, dtype=np.float64)),
        "I_cos_total": float(np.sum(i_cos, dtype=np.float64)),
        "I_cross_total": float(np.sum(i_cross, dtype=np.float64)),
    }


__all__ = ["phi_integs"]
