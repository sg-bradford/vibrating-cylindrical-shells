"""Polynomial expansion helpers translated from the MATLAB EMA_MSA codebase."""

from __future__ import annotations

import numpy as np

from ..data_structures import PolyExpResult, PolyExpTerm


def _index_pairs(start_a: int, count_a: int, start_b: int, count_b: int) -> np.ndarray:
    """Reproduce MATLAB ``meshgrid`` + ``(:)`` index-pair generation."""

    ia, ib = np.meshgrid(
        np.arange(start_a, start_a + count_a, dtype=np.float64),
        np.arange(start_b, start_b + count_b, dtype=np.float64),
        indexing="xy",
    )
    return np.column_stack((ia.ravel(order="F"), ib.ravel(order="F")))


def _repeat_column(values: np.ndarray, width: int) -> np.ndarray:
    """Match MATLAB ``repmat(column, 1, width)`` for column vectors."""

    return np.repeat(values[:, None], width, axis=1)


def _repeat_row(values: np.ndarray, height: int) -> np.ndarray:
    """Match MATLAB ``(repmat(column, 1, height))'`` for column vectors."""

    return np.repeat(values[None, :], height, axis=0)


def _build_terms(
    i1: np.ndarray,
    i2: np.ndarray,
    c1_main_1: float,
    c1_main_2: float,
    c1_corr_1: float,
    c1_corr_2: float,
    p1_shift_1: float,
    p1_shift_2: float,
) -> PolyExpResult:
    """Build the four MATLAB ``polyexp`` struct terms."""

    n1 = i1.shape[0]
    n2 = i2.shape[0]

    p1_main = _repeat_column(i1[:, 0], n2)
    p2_main = _repeat_column(i1[:, 1], n2)
    p1_other = _repeat_row(i2[:, 0], n1)
    p2_other = _repeat_row(i2[:, 1], n1)

    ones_matrix = np.ones((n1, n2), dtype=np.float64)

    term1 = PolyExpTerm(
        p1=p1_main,
        p2=p2_main,
        c1=np.full((n1, n2), c1_main_1, dtype=np.float64),
        c2=ones_matrix.copy(),
    )
    term2 = PolyExpTerm(
        p1=p1_other,
        p2=p2_other,
        c1=np.full((n1, n2), c1_main_2, dtype=np.float64),
        c2=ones_matrix.copy(),
    )
    term3 = PolyExpTerm(
        p1=term1.p1 + p1_shift_1,
        p2=term1.p2.copy(),
        c1=np.full((n1, n2), c1_corr_1, dtype=np.float64),
        c2=term1.c2.copy(),
    )
    term4 = PolyExpTerm(
        p1=term2.p1 + p1_shift_2,
        p2=term2.p2.copy(),
        c1=np.full((n1, n2), c1_corr_2, dtype=np.float64),
        c2=term2.c2.copy(),
    )

    return PolyExpResult([term1, term2, term3, term4])


def polyexp(
    p1: int,
    p2: int,
    p3: int,
    p4: int,
    cond1: int,
    cond2: int,
    L: float,
    sym: bool,
) -> PolyExpResult:
    """Generate the four-term polynomial expansion used by the MATLAB solver.

    Source: ``Ema_Msa_7/polyexp.m`` lines 10-212.
    """

    p1_i = int(p1)
    p2_i = int(p2)
    p3_i = int(p3)
    p4_i = int(p4)
    cond1_i = int(cond1)
    cond2_i = int(cond2)
    L_value = float(L)

    if cond1_i == 1 and cond2_i == 1:
        i1 = _index_pairs(0, p1_i, 0, p3_i)
        i2 = _index_pairs(0, p2_i, 0, p4_i)
        return _build_terms(i1, i2, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0)

    if cond1_i == 0 and cond2_i == 0:
        if sym:
            a_coeff = -0.25 * (L_value**2)
            i1 = _index_pairs(0, p1_i, 0, p3_i)
            i2 = _index_pairs(0, p2_i, 0, p4_i)
            return _build_terms(i1, i2, a_coeff, a_coeff, 1.0, 1.0, 2.0, 2.0)

        a_coeff = -L_value
        i1 = _index_pairs(1, p1_i, 0, p3_i)
        i2 = _index_pairs(1, p2_i, 0, p4_i)
        return _build_terms(i1, i2, a_coeff, a_coeff, 1.0, 1.0, 1.0, 1.0)

    if cond1_i == 1 and cond2_i == 0:
        if sym:
            a_coeff = -0.25 * (L_value**2)
            i1 = _index_pairs(0, p1_i, 0, p3_i)
            i2 = _index_pairs(0, p2_i, 0, p4_i)
            return _build_terms(i1, i2, 1.0, a_coeff, 0.0, 1.0, 0.0, 2.0)

        a_coeff = -L_value
        i1 = _index_pairs(0, p1_i, 0, p3_i)
        i2 = _index_pairs(1, p2_i, 0, p4_i)
        return _build_terms(i1, i2, 1.0, a_coeff, 0.0, 1.0, 0.0, 1.0)

    if cond1_i == 0 and cond2_i == 1:
        if sym:
            a_coeff = -0.25 * (L_value**2)
            i1 = _index_pairs(0, p1_i, 0, p3_i)
            i2 = _index_pairs(0, p2_i, 0, p4_i)
            return _build_terms(i1, i2, a_coeff, 1.0, 1.0, 0.0, 2.0, 0.0)

        a_coeff = -L_value
        i1 = _index_pairs(1, p1_i, 0, p3_i)
        i2 = _index_pairs(0, p2_i, 0, p4_i)
        return _build_terms(i1, i2, a_coeff, 1.0, 1.0, 0.0, 1.0, 0.0)

    raise ValueError("cond1 and cond2 must each be 0 or 1.")


__all__ = ["polyexp"]
