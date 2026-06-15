"""Radial integrals translated from the MATLAB EMA_MSA codebase."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..core.poly_integ import poly_integ
from ..core.polyexp import polyexp
from ..data_structures import PolyExpResult, PolyExpTerm


def _normalize_polyord(polyord: int | Sequence[int]) -> tuple[int, int, int, int, int, int]:
    """Normalize the simplified radial-order input to MATLAB's 6-entry ``mp.polyord``."""

    if isinstance(polyord, int):
        if polyord < 1:
            raise ValueError("polyord must be at least 1.")
        return (1, 1, 1, polyord, polyord, polyord)

    values = tuple(int(value) for value in polyord)
    if len(values) == 3:
        if any(value < 1 for value in values):
            raise ValueError("All polynomial orders must be at least 1.")
        return (1, 1, 1, values[0], values[1], values[2])
    if len(values) == 6:
        if any(value < 1 for value in values):
            raise ValueError("All polynomial orders must be at least 1.")
        return values
    raise ValueError("polyord must be an int or a sequence of length 3 or 6.")


def _poly_terms(expansion: PolyExpResult) -> tuple[PolyExpTerm, PolyExpTerm]:
    """Return the first two MATLAB expansion terms used by ``r_integs.m``."""

    return expansion.terms[0], expansion.terms[1]


def _radial_integral(
    expansion: PolyExpResult,
    left_factor: np.ndarray | float,
    right_factor: np.ndarray | float,
    power_shift: float,
    limits: tuple[float, float],
) -> np.ndarray:
    """Evaluate one radial polynomial block from ``Ema_Msa_7/r_integs.m`` lines 38-72."""

    left_term, right_term = _poly_terms(expansion)
    return np.asarray(
        poly_integ(
            left_term.c2 * right_term.c2 * left_factor * right_factor,
            left_term.p2 + right_term.p2 + power_shift,
            limits,
        ),
        dtype=np.float64,
    )


def r_integs(r_in: float, r_out: float, polyord: int | Sequence[int]) -> dict[str, object]:
    """Compute radial polynomial integration blocks through the cylinder wall.

    Faithful translation of ``Ema_Msa_7/r_integs.m`` lines 9-75. The simplified
    Python API accepts either one shared radial order or explicit order tuples; for
    the scalar case it reproduces the MATLAB fallback ``p(1:3)=1`` and
    ``p(4:6)=polyord`` before evaluating the closed-form radial integrals.
    """

    r_in_value = float(r_in)
    r_out_value = float(r_out)
    if not np.isfinite(r_in_value) or not np.isfinite(r_out_value) or r_out_value <= r_in_value:
        raise ValueError("r_out must be greater than r_in, and both must be finite.")

    p = _normalize_polyord(polyord)
    limits = (r_in_value, r_out_value)

    p11 = polyexp(p[0], p[0], p[3], p[3], 1, 1, 1.0, True)
    p12 = polyexp(p[0], p[1], p[3], p[4], 1, 0, 1.0, True)
    p13 = polyexp(p[0], p[2], p[3], p[5], 1, 0, 1.0, True)
    p22 = polyexp(p[1], p[1], p[4], p[4], 0, 0, 1.0, True)
    p23 = polyexp(p[1], p[2], p[4], p[5], 0, 0, 1.0, True)
    p33 = polyexp(p[2], p[2], p[5], p[5], 0, 0, 1.0, True)

    p11_l, p11_r = _poly_terms(p11)
    p12_l, p12_r = _poly_terms(p12)
    p13_l, p13_r = _poly_terms(p13)
    p22_l, p22_r = _poly_terms(p22)
    p23_l, p23_r = _poly_terms(p23)
    p33_l, p33_r = _poly_terms(p33)

    k12r1 = _radial_integral(p12, 1.0, 1.0, 0.0, limits)
    k22r1 = _radial_integral(p22, 1.0, 1.0, -1.0, limits)
    k23r1 = _radial_integral(p23, 1.0, 1.0, -1.0, limits)
    k33r1 = _radial_integral(p33, 1.0, 1.0, -1.0, limits)

    return {
        "limits": limits,
        "polyord": p,
        "k11r1": _radial_integral(p11, 1.0, 1.0, 1.0, limits),
        "k11r2": _radial_integral(p11, p11_l.p2, p11_r.p2, -1.0, limits),
        "k11r3": _radial_integral(p11, 1.0, 1.0, -1.0, limits),
        "k12r1": k12r1,
        "k12r2": k12r1.copy(),
        "k13r1": _radial_integral(p13, 1.0, 1.0, 0.0, limits),
        "k13r2": _radial_integral(p13, 1.0, p13_r.p2, 0.0, limits),
        "k13r3": _radial_integral(p13, p13_l.p2, 1.0, 0.0, limits),
        "k22r1": k22r1,
        "k22r2": _radial_integral(p22, p22_l.p2, p22_r.p2, -1.0, limits),
        "k22r3": _radial_integral(p22, p22_l.p2, 1.0, -1.0, limits),
        "k22r4": _radial_integral(p22, 1.0, p22_r.p2, -1.0, limits),
        "k22r5": k22r1.copy(),
        "k22r6": _radial_integral(p22, 1.0, 1.0, 1.0, limits),
        "k23r1": k23r1,
        "k23r2": _radial_integral(p23, 1.0, p23_r.p2, -1.0, limits),
        "k23r3": _radial_integral(p23, p23_l.p2, 1.0, -1.0, limits),
        "k23r4": k23r1.copy(),
        "k33r1": k33r1,
        "k33r2": _radial_integral(p33, 1.0, p33_r.p2, -1.0, limits),
        "k33r3": _radial_integral(p33, p33_l.p2, 1.0, -1.0, limits),
        "k33r4": _radial_integral(p33, p33_l.p2, p33_r.p2, -1.0, limits),
        "k33r5": k33r1.copy(),
        "k33r6": _radial_integral(p33, 1.0, 1.0, 1.0, limits),
    }


__all__ = ["r_integs"]
