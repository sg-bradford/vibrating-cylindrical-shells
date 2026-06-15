"""Axial integrals translated from the MATLAB EMA_MSA codebase."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from ..core.beam_func import beam_func
from ..core.gauss_w import gauss_w
from ..core.poly_integ import poly_integ
from ..core.polyexp import polyexp
from ..data_structures import PolyExpResult, PolyExpTerm


def _canonical_bc_type(bc_type: str) -> str:
    """Normalize boundary-condition labels used by MATLAB and Python callers."""

    normalized = bc_type.strip().lower().replace("_", "-")
    mapping = {
        "free-free": "Free-Free",
        "ff": "Free-Free",
        "pinned-free": "Pinned-Free",
        "pinnedfree": "Pinned-Free",
        "pf": "Pinned-Free",
        "free-pinned": "Pinned-Free",
        "fp": "Pinned-Free",
        "pinned-pinned": "Pinned-Pinned",
        "pinnedpinned": "Pinned-Pinned",
        "pp": "Pinned-Pinned",
        "clamped-clamped": "Clamped-Clamped",
        "clampedclamped": "Clamped-Clamped",
        "cc": "Clamped-Clamped",
        "clamped-free": "Clamped-Free",
        "clampedfree": "Clamped-Free",
        "cf": "Clamped-Free",
        "free-clamped": "Clamped-Free",
        "fc": "Clamped-Free",
        "clamped-pinned": "Clamped-Pinned",
        "clampedpinned": "Clamped-Pinned",
        "cp": "Clamped-Pinned",
        "pinned-clamped": "Clamped-Pinned",
        "pc": "Clamped-Pinned",
    }
    try:
        return mapping[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported boundary condition {bc_type!r}.") from exc


def _beam_boundary_codes(bc_type: str) -> tuple[str, str, str]:
    """Map the MATLAB ``mp.zbound`` value to ``beam_func`` arguments."""

    mapping = {
        "Free-Free": ("l", "sff", "ff"),
        "Pinned-Free": ("l", "sfs", "fs"),
        "Pinned-Pinned": ("l", "sss", "ss"),
        "Clamped-Clamped": ("l", "scc", "cc"),
        "Clamped-Free": ("l", "scf", "cf"),
        "Clamped-Pinned": ("l", "scs", "cs"),
    }
    return mapping[bc_type]


def _validate_length(length: float) -> float:
    """Validate the axial length."""

    length_value = float(length)
    if not np.isfinite(length_value) or length_value <= 0.0:
        raise ValueError("length must be positive and finite.")
    return length_value


def _validate_limits(limits: tuple[float, float] | None, default: tuple[float, float]) -> tuple[float, float]:
    if limits is None:
        return default

    lower = float(limits[0])
    upper = float(limits[1])
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise ValueError("limits must be a finite interval with upper > lower.")
    return (lower, upper)


def _beam_bilinear_integral(weights: np.ndarray, values_1: np.ndarray, values_2: np.ndarray, jac: float) -> float:
    """Evaluate one Gauss-quadrature bilinear form."""

    return float(jac * np.sum(weights * values_1 * values_2, dtype=np.float64))


def _poly_terms(expansion: PolyExpResult) -> tuple[PolyExpTerm, PolyExpTerm, PolyExpTerm, PolyExpTerm]:
    """Return the four fixed MATLAB ``polyexp`` terms."""

    return tuple(expansion.terms)  # type: ignore[return-value]


def _poly_pair_sum(
    expansion: PolyExpResult,
    left_factor: Callable[[PolyExpTerm], np.ndarray],
    right_factor: Callable[[PolyExpTerm], np.ndarray],
    power_shift: float,
    limits: tuple[float, float],
) -> np.ndarray:
    """Reproduce the repeated 4-term ``poly_integ`` sums in ``z_integs.m``."""

    t1, t2, t3, t4 = _poly_terms(expansion)
    pairs = ((t1, t2), (t1, t4), (t2, t3), (t3, t4))

    result: np.ndarray | None = None
    for left_term, right_term in pairs:
        value = poly_integ(
            left_term.c1 * right_term.c1 * left_factor(left_term) * right_factor(right_term),
            left_term.p1 + right_term.p1 + power_shift,
            limits,
        )
        result = value if result is None else result + value

    if result is None:
        raise RuntimeError("polyexp() did not yield any integration terms.")
    return np.asarray(result, dtype=np.float64)


def _normalize_poly_orders(m1: int | Sequence[int], m2: int) -> tuple[int, int, int, int, int, int]:
    if isinstance(m1, Sequence) and not isinstance(m1, (str, bytes, bytearray)):
        values = tuple(int(value) for value in m1)
        if len(values) != 6 or any(value < 1 for value in values):
            raise ValueError("Poly-mode order sequences must contain exactly six positive integers.")
        return values

    u_order = int(m1)
    vw_order = int(m2)
    if u_order < 1 or vw_order < 1:
        raise ValueError("Poly-mode shorthand orders must be positive integers.")
    return (u_order, vw_order, vw_order, 1, 1, 1)


def _poly_expansions(
    poly_orders: tuple[int, int, int, int, int, int],
    bc_type: str,
    length: float,
    symmetric: bool,
) -> dict[str, PolyExpResult]:
    """Build the MATLAB polynomial expansions for the supported Poly cases."""

    p = poly_orders
    if bc_type == "Pinned-Pinned":
        return {
            "P11": polyexp(p[0], p[0], p[3], p[3], 1, 1, length, symmetric),
            "P12": polyexp(p[0], p[1], p[3], p[4], 1, 0, length, symmetric),
            "P13": polyexp(p[0], p[2], p[3], p[5], 1, 0, length, symmetric),
            "P22": polyexp(p[1], p[1], p[4], p[4], 0, 0, length, symmetric),
            "P23": polyexp(p[1], p[2], p[4], p[5], 0, 0, length, symmetric),
            "P33": polyexp(p[2], p[2], p[5], p[5], 0, 0, length, symmetric),
        }

    if bc_type == "Free-Free":
        return {
            "P11": polyexp(p[0], p[0], p[3], p[3], 1, 1, length, symmetric),
            "P12": polyexp(p[0], p[1], p[3], p[4], 1, 1, length, symmetric),
            "P13": polyexp(p[0], p[2], p[3], p[5], 1, 1, length, symmetric),
            "P22": polyexp(p[1], p[1], p[4], p[4], 1, 1, length, symmetric),
            "P23": polyexp(p[1], p[2], p[4], p[5], 1, 1, length, symmetric),
            "P33": polyexp(p[2], p[2], p[5], p[5], 1, 1, length, symmetric),
        }

    raise ValueError(f"Poly mode is only implemented for 'Free-Free' and 'Pinned-Pinned', not {bc_type!r}.")


def _z_integs_beam(
    m1: int,
    m2: int,
    bc_type: str,
    length: float,
    n_gauss: int,
    limits: tuple[float, float] | None,
) -> dict[str, object]:
    """Beam-mode translation of ``Ema_Msa_7/z_integs.m`` lines 16-95."""

    s_ux1, s_ux2, s_vx = _beam_boundary_codes(bc_type)
    xi, wx = gauss_w(int(n_gauss))

    interval = _validate_limits(limits, (0.0, length))
    lower, upper = interval

    x = 0.5 * (upper - lower) * xi + 0.5 * (upper + lower)
    jac = 0.5 * (upper - lower)

    u1 = beam_func(s_ux1, s_ux2, 0, m1, x, length)
    u1_d = beam_func(s_ux1, s_ux2, 1, m1, x, length)
    v1 = beam_func("b", s_vx, 0, m1, x, length)
    v1_d = beam_func("b", s_vx, 1, m1, x, length)
    w1 = v1
    w1_d = v1_d

    u2 = beam_func(s_ux1, s_ux2, 0, m2, x, length)
    u2_d = beam_func(s_ux1, s_ux2, 1, m2, x, length)
    v2 = beam_func("b", s_vx, 0, m2, x, length)
    v2_d = beam_func("b", s_vx, 1, m2, x, length)
    w2 = v2
    w2_d = v2_d

    k11z2 = _beam_bilinear_integral(wx, u1, u2, jac)
    k13z1 = _beam_bilinear_integral(wx, u1_d, w2, jac)
    k22z1 = _beam_bilinear_integral(wx, v1, v2, jac)
    k33z2 = _beam_bilinear_integral(wx, w1_d, w2_d, jac)

    return {
        "basis": "beam",
        "bc_type": bc_type,
        "interval": interval,
        "m1": int(m1),
        "m2": int(m2),
        "n_gauss": int(n_gauss),
        "k11z1": _beam_bilinear_integral(wx, u1_d, u2_d, jac),
        "k11z2": k11z2,
        "k11z3": k11z2,
        "k12z1": _beam_bilinear_integral(wx, u1_d, v2, jac),
        "k12z2": _beam_bilinear_integral(wx, u1, v2_d, jac),
        "k13z1": k13z1,
        "k13z2": k13z1,
        "k13z3": _beam_bilinear_integral(wx, u1, w2_d, jac),
        "k22z1": k22z1,
        "k22z2": _beam_bilinear_integral(wx, v1_d, v2_d, jac),
        "k23z1": _beam_bilinear_integral(wx, v1, w2, jac),
        "k33z1": _beam_bilinear_integral(wx, w1, w2, jac),
        "k33z2": k33z2,
    }


def _z_integs_poly(
    m1: int | Sequence[int],
    m2: int,
    bc_type: str,
    length: float,
    limits: tuple[float, float] | None,
    symmetric: bool | None,
) -> dict[str, object]:
    """Poly-mode translation of ``Ema_Msa_7/z_integs.m`` lines 99-254."""

    poly_orders = _normalize_poly_orders(m1, m2)
    symmetric_value = bool(symmetric) if symmetric is not None else True
    default_limits = (-0.5 * length, 0.5 * length) if symmetric_value else (0.0, length)
    interval = _validate_limits(limits, default_limits)
    expansions = _poly_expansions(poly_orders, bc_type, length, symmetric_value)

    p11 = expansions["P11"]
    p12 = expansions["P12"]
    p13 = expansions["P13"]
    p22 = expansions["P22"]
    p23 = expansions["P23"]
    p33 = expansions["P33"]

    k11z2 = _poly_pair_sum(p11, lambda term: 1.0, lambda term: 1.0, 0.0, interval)
    k13z1 = _poly_pair_sum(p13, lambda term: term.p1, lambda term: 1.0, -1.0, interval)
    k22z1 = _poly_pair_sum(p22, lambda term: 1.0, lambda term: 1.0, 0.0, interval)
    k33z1 = _poly_pair_sum(p33, lambda term: 1.0, lambda term: 1.0, 0.0, interval)

    return {
        "basis": "poly",
        "bc_type": bc_type,
        "interval": interval,
        "m1": poly_orders[0],
        "m2": poly_orders[1],
        "polyord": poly_orders,
        "symmetric": symmetric_value,
        "k11z1": _poly_pair_sum(p11, lambda term: term.p1, lambda term: term.p1, -2.0, interval),
        "k11z2": k11z2,
        "k11z3": k11z2.copy(),
        "k12z1": _poly_pair_sum(p12, lambda term: term.p1, lambda term: 1.0, -1.0, interval),
        "k12z2": _poly_pair_sum(p12, lambda term: 1.0, lambda term: term.p1, -1.0, interval),
        "k13z1": k13z1,
        "k13z2": k13z1.copy(),
        "k13z3": _poly_pair_sum(p13, lambda term: 1.0, lambda term: term.p1, -1.0, interval),
        "k22z1": k22z1,
        "k22z2": _poly_pair_sum(p22, lambda term: term.p1, lambda term: term.p1, -2.0, interval),
        "k23z1": _poly_pair_sum(p23, lambda term: 1.0, lambda term: 1.0, 0.0, interval),
        "k33z1": k33z1,
        "k33z2": _poly_pair_sum(p33, lambda term: term.p1, lambda term: term.p1, -2.0, interval),
    }


def z_integs(
    m1: int | Sequence[int],
    m2: int,
    bc_type: str,
    length: float,
    poly_mode: bool = False,
    n_gauss: int = 20,
    limits: tuple[float, float] | None = None,
    symmetric: bool | None = None,
) -> dict[str, object]:
    """Compute axial integration blocks for one mode pair or one polynomial basis set.

    MATLAB source: ``Ema_Msa_7/z_integs.m`` lines 14-258.

    Beam mode mirrors the MATLAB Gauss-quadrature path, generalized from one axial
    mode to a bilinear form between ``m1`` and ``m2``. Poly mode mirrors the MATLAB
    closed-form path using ``polyexp()`` and ``poly_integ()``; with the simplified
    Python API, ``m1`` is interpreted as the axial polynomial count for the axial
    displacement family and ``m2`` for the tangential/radial families.
    """

    bc_value = _canonical_bc_type(bc_type)
    length_value = _validate_length(length)

    if poly_mode:
        return _z_integs_poly(m1, int(m2), bc_value, length_value, limits, symmetric)
    if isinstance(m1, Sequence) and not isinstance(m1, (str, bytes, bytearray)):
        raise TypeError("Beam-mode z_integs() expects integer m1 and m2 values.")
    return _z_integs_beam(int(m1), int(m2), bc_value, length_value, int(n_gauss), limits)


__all__ = ["z_integs"]
