"""Generalized eigenvalue assembly translated from the MATLAB EMA_MSA solver."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from scipy.linalg import eig

from ..data_structures import CylinderComponent, EigenResult, ModeParams

_DEFAULT_FREQ_CUTOFF_HZ = 8000.0
_EPS = np.finfo(np.float64).eps


def _validate_lengths(
    components: Sequence[CylinderComponent],
    phi_integrals: Sequence[Mapping[str, object]],
    z_integrals: Sequence[object],
    r_integrals: Sequence[Mapping[str, object]],
    c_matrices: Sequence[np.ndarray],
) -> int:
    component_count = len(components)
    expected = {
        "phi_integrals": len(phi_integrals),
        "z_integrals": len(z_integrals),
        "r_integrals": len(r_integrals),
        "C_matrices": len(c_matrices),
    }
    for name, count in expected.items():
        if count != component_count:
            raise ValueError(
                f"{name} must contain exactly {component_count} entries, received {count}."
            )
    return component_count


def _dof_sizes(mode_params: ModeParams) -> tuple[int, int, int]:
    polyord = tuple(int(value) for value in mode_params.polyord)
    if len(polyord) != 6:
        raise ValueError("mode_params.polyord must contain exactly six integers.")
    if any(value < 1 for value in polyord):
        raise ValueError("All mode_params.polyord entries must be at least 1.")

    if mode_params.zdef == "Poly":
        return (
            polyord[0] * polyord[3],
            polyord[1] * polyord[4],
            polyord[2] * polyord[5],
        )
    if mode_params.zdef == "Beam":
        return polyord[3], polyord[4], polyord[5]
    raise ValueError(f"Unsupported mode_params.zdef value {mode_params.zdef!r}.")


def _phi_total(phi_data: Mapping[str, object], key: str) -> float:
    total_key = f"{key}_total"
    if total_key in phi_data:
        return float(phi_data[total_key])
    if key not in phi_data:
        raise KeyError(f"Missing circumferential integral {key!r}.")
    values = np.asarray(phi_data[key], dtype=np.float64)
    return float(np.sum(values, dtype=np.float64))


def _resolve_z_integral(component_z: object, m_index: int, m_value: int) -> Mapping[str, object]:
    if isinstance(component_z, Mapping) and "k11z1" in component_z:
        return component_z

    if isinstance(component_z, Sequence) and not isinstance(component_z, (str, bytes, bytearray)):
        if m_index < 0 or m_index >= len(component_z):
            raise IndexError(
                f"Missing axial integral entry for m_range index {m_index}."
            )
        candidate = component_z[m_index]
        if isinstance(candidate, Mapping) and "k11z1" in candidate:
            return candidate

    if isinstance(component_z, Mapping):
        for key in ((m_value, m_value), m_value, str(m_value), m_index):
            candidate = component_z.get(key)
            if isinstance(candidate, Mapping) and "k11z1" in candidate:
                return candidate

    raise ValueError(
        f"Unable to resolve z-integral data for axial wave number m={m_value}."
    )


def _as_block(value: object, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        if array.size != 1:
            raise ValueError(f"Block {name!r} must be scalar or 2D, got shape {array.shape}.")
        array = array.reshape(1, 1)
    elif array.ndim != 2:
        raise ValueError(f"Block {name!r} must be scalar or 2D, got shape {array.shape}.")
    return np.asarray(array, dtype=np.complex128)


def _as_square_matrix(value: object, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.complex128)
    if array.ndim != 2:
        raise ValueError(f"Matrix {name!r} must be 2D, got shape {array.shape}.")
    if array.shape[0] != array.shape[1]:
        raise ValueError(f"Matrix {name!r} must be square, got shape {array.shape}.")
    return array


def _validate_constitutive_matrix(matrix: np.ndarray, component_index: int) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.complex128)
    if array.shape != (6, 6):
        raise ValueError(
            f"C_matrices[{component_index}] must have shape (6, 6), got {array.shape}."
        )
    return array


def _assemble_component_blocks(
    n_value: int,
    rho_value: float,
    phi_data: Mapping[str, object],
    z_data: Mapping[str, object],
    r_data: Mapping[str, object],
    constitutive: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    i_cos = _phi_total(phi_data, "I_cos")
    i_sin = _phi_total(phi_data, "I_sin")
    n_sq = float(n_value * n_value)

    k11z1 = _as_block(z_data["k11z1"], "k11z1")
    k11z2 = _as_block(z_data["k11z2"], "k11z2")
    k11z3 = _as_block(z_data["k11z3"], "k11z3")
    k12z1 = _as_block(z_data["k12z1"], "k12z1")
    k12z2 = _as_block(z_data["k12z2"], "k12z2")
    k13z1 = _as_block(z_data["k13z1"], "k13z1")
    k13z2 = _as_block(z_data["k13z2"], "k13z2")
    k13z3 = _as_block(z_data["k13z3"], "k13z3")
    k22z1 = _as_block(z_data["k22z1"], "k22z1")
    k22z2 = _as_block(z_data["k22z2"], "k22z2")
    k23z1 = _as_block(z_data["k23z1"], "k23z1")
    k33z1 = _as_block(z_data["k33z1"], "k33z1")
    k33z2 = _as_block(z_data["k33z2"], "k33z2")

    k11r1 = _as_block(r_data["k11r1"], "k11r1")
    k11r2 = _as_block(r_data["k11r2"], "k11r2")
    k11r3 = _as_block(r_data["k11r3"], "k11r3")
    k12r1 = _as_block(r_data["k12r1"], "k12r1")
    k13r1 = _as_block(r_data["k13r1"], "k13r1")
    k13r2 = _as_block(r_data["k13r2"], "k13r2")
    k13r3 = _as_block(r_data["k13r3"], "k13r3")
    k22r1 = _as_block(r_data["k22r1"], "k22r1")
    k22r2 = _as_block(r_data["k22r2"], "k22r2")
    k22r3 = _as_block(r_data["k22r3"], "k22r3")
    k22r4 = _as_block(r_data["k22r4"], "k22r4")
    k22r5 = _as_block(r_data["k22r5"], "k22r5")
    k22r6 = _as_block(r_data["k22r6"], "k22r6")
    k23r1 = _as_block(r_data["k23r1"], "k23r1")
    k23r2 = _as_block(r_data["k23r2"], "k23r2")
    k23r3 = _as_block(r_data["k23r3"], "k23r3")
    k23r4 = _as_block(r_data["k23r4"], "k23r4")
    k33r1 = _as_block(r_data["k33r1"], "k33r1")
    k33r2 = _as_block(r_data["k33r2"], "k33r2")
    k33r3 = _as_block(r_data["k33r3"], "k33r3")
    k33r4 = _as_block(r_data["k33r4"], "k33r4")
    k33r5 = _as_block(r_data["k33r5"], "k33r5")
    k33r6 = _as_block(r_data["k33r6"], "k33r6")

    k11 = (
        constitutive[0, 0] * i_cos * k11z1 * k11r1
        + constitutive[4, 4] * i_cos * k11z2 * k11r2
        + n_sq * constitutive[5, 5] * i_sin * k11z3 * k11r3
    )
    m11 = float(rho_value) * i_cos * k11z2 * k11r1

    k12 = (
        float(n_value) * constitutive[0, 1] * i_cos * k12z1 * k12r1
        - float(n_value) * constitutive[5, 5] * i_sin * k12z2 * k12r1
    )
    m12 = np.zeros_like(k12, dtype=np.complex128)

    k13 = (
        constitutive[0, 1] * i_cos * k13z1 * k13r1
        + constitutive[0, 2] * i_cos * k13z2 * k13r2
        + constitutive[4, 4] * i_cos * k13z3 * k13r3
    )
    m13 = np.zeros_like(k13, dtype=np.complex128)

    k22 = (
        k22z1
        * (
            constitutive[1, 1] * n_sq * i_cos * k22r1
            + constitutive[3, 3] * i_sin * (k22r2 - k22r3 - k22r4 + k22r5)
        )
        + constitutive[5, 5] * i_sin * k22z2 * k22r6
    )
    m22 = float(rho_value) * i_sin * k22z1 * k22r6

    k23 = float(n_value) * k23z1 * (
        i_cos * (constitutive[1, 1] * k23r1 + constitutive[1, 2] * k23r2)
        + i_sin * constitutive[3, 3] * (-k23r3 + k23r4)
    )
    m23 = np.zeros_like(k23, dtype=np.complex128)

    k33 = (
        k33z1
        * (
            i_cos
            * (
                constitutive[1, 1] * k33r1
                + constitutive[1, 2] * k33r2
                + constitutive[1, 2] * k33r3
                + constitutive[2, 2] * k33r4
            )
            + constitutive[3, 3] * n_sq * i_sin * k33r5
        )
        + constitutive[4, 4] * k33z2 * i_cos * k33r6
    )
    m33 = float(rho_value) * i_cos * k33z1 * k33r6

    return k11, m11, k12, m12, k13, m13, k22, m22, k23, m23, k33, m33


def _assemble_component_blocks_beam(
    rho_value: float,
    z_data: Mapping[str, object],
    r_data: Mapping[str, object],
    constitutive: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    k11z1 = _as_block(z_data["k11z1"], "k11z1")
    k11z2 = _as_block(z_data["k11z2"], "k11z2")
    k11z3 = _as_block(z_data["k11z3"], "k11z3")
    k12z1 = _as_block(z_data["k12z1"], "k12z1")
    k12z2 = _as_block(z_data["k12z2"], "k12z2")
    k13z1 = _as_block(z_data["k13z1"], "k13z1")
    k13z2 = _as_block(z_data["k13z2"], "k13z2")
    k13z3 = _as_block(z_data["k13z3"], "k13z3")
    k22z1 = _as_block(z_data["k22z1"], "k22z1")
    k22z2 = _as_block(z_data["k22z2"], "k22z2")
    k23z1 = _as_block(z_data["k23z1"], "k23z1")
    k33z1 = _as_block(z_data["k33z1"], "k33z1")
    k33z2 = _as_block(z_data["k33z2"], "k33z2")

    k11r1 = _as_block(r_data["k11r1"], "k11r1")
    k11r2 = _as_block(r_data["k11r2"], "k11r2")
    k11r3 = _as_block(r_data["k11r3"], "k11r3")
    k12r1 = _as_block(r_data["k12r1"], "k12r1")
    k13r1 = _as_block(r_data["k13r1"], "k13r1")
    k13r2 = _as_block(r_data["k13r2"], "k13r2")
    k13r3 = _as_block(r_data["k13r3"], "k13r3")
    k22r1 = _as_block(r_data["k22r1"], "k22r1")
    k22r2 = _as_block(r_data["k22r2"], "k22r2")
    k22r3 = _as_block(r_data["k22r3"], "k22r3")
    k22r4 = _as_block(r_data["k22r4"], "k22r4")
    k22r5 = _as_block(r_data["k22r5"], "k22r5")
    k22r6 = _as_block(r_data["k22r6"], "k22r6")
    k23r1 = _as_block(r_data["k23r1"], "k23r1")
    k23r2 = _as_block(r_data["k23r2"], "k23r2")
    k23r3 = _as_block(r_data["k23r3"], "k23r3")
    k23r4 = _as_block(r_data["k23r4"], "k23r4")
    k33r1 = _as_block(r_data["k33r1"], "k33r1")
    k33r2 = _as_block(r_data["k33r2"], "k33r2")
    k33r3 = _as_block(r_data["k33r3"], "k33r3")
    k33r4 = _as_block(r_data["k33r4"], "k33r4")
    k33r5 = _as_block(r_data["k33r5"], "k33r5")
    k33r6 = _as_block(r_data["k33r6"], "k33r6")

    k11_00 = constitutive[0, 0] * k11z1 * k11r1 + constitutive[4, 4] * k11z2 * k11r2
    k11_11 = constitutive[5, 5] * k11z3 * k11r3
    m11_00 = float(rho_value) * k11z2 * k11r1

    k12_01 = constitutive[0, 1] * k12z1 * k12r1 - constitutive[5, 5] * k12z2 * k12r1
    k13_00 = (
        constitutive[0, 1] * k13z1 * k13r1
        + constitutive[0, 2] * k13z2 * k13r2
        + constitutive[4, 4] * k13z3 * k13r3
    )

    k22_00 = (
        k22z1 * constitutive[3, 3] * (k22r2 - k22r3 - k22r4 + k22r5)
        + constitutive[5, 5] * k22z2 * k22r6
    )
    k22_11 = k22z1 * constitutive[1, 1] * k22r1
    m22_00 = float(rho_value) * k22z1 * k22r6

    k23_01 = k23z1 * (
        -(constitutive[1, 1] * k23r1 + constitutive[1, 2] * k23r2)
        + constitutive[3, 3] * (-k23r3 + k23r4)
    )

    k33_00 = k33z1 * (
        constitutive[1, 1] * k33r1
        + constitutive[1, 2] * k33r2
        + constitutive[1, 2] * k33r3
        + constitutive[2, 2] * k33r4
    ) + constitutive[4, 4] * k33z2 * k33r6
    k33_11 = constitutive[3, 3] * k33z1 * k33r5
    m33_00 = float(rho_value) * k33z1 * k33r6

    du = k11_00.shape[0]
    dv = k22_00.shape[0]
    dw = k33_00.shape[0]

    zero_uu = np.zeros((du, du), dtype=np.complex128)
    zero_uv = np.zeros((du, dv), dtype=np.complex128)
    zero_uw = np.zeros((du, dw), dtype=np.complex128)
    zero_vv = np.zeros((dv, dv), dtype=np.complex128)
    zero_vw = np.zeros((dv, dw), dtype=np.complex128)
    zero_ww = np.zeros((dw, dw), dtype=np.complex128)

    k_00 = np.block(
        [[k11_00, zero_uv, k13_00], [zero_uv.T, k22_00, zero_vw], [k13_00.T, zero_vw.T, k33_00]]
    )
    k_11 = np.block(
        [[k11_11, zero_uv, zero_uw], [zero_uv.T, k22_11, zero_vw], [zero_uw.T, zero_vw.T, k33_11]]
    )
    k_01 = np.block(
        [[zero_uu, k12_01, zero_uw], [-k12_01.T, zero_vv, k23_01], [zero_uw.T, -k23_01.T, zero_ww]]
    )
    m_00 = np.block(
        [[m11_00, zero_uv, zero_uw], [zero_uv.T, m22_00, zero_vw], [zero_uw.T, zero_vw.T, m33_00]]
    )

    return k_00, k_11, k_01, m_00


def _select_reduced_system(
    *,
    zdef: str,
    n_value: int,
    m_value: int,
    du: int,
    dv: int,
    dw: int,
    k11: np.ndarray,
    m11: np.ndarray,
    k12: np.ndarray,
    m12: np.ndarray,
    k13: np.ndarray,
    m13: np.ndarray,
    k22: np.ndarray,
    m22: np.ndarray,
    k23: np.ndarray,
    m23: np.ndarray,
    k33: np.ndarray,
    m33: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    if zdef == "Poly":
        if n_value == 0:
            return (
                np.block([[k11, k13], [k13.T, k33]]),
                np.block([[m11, m13], [m13.T, m33]]),
                1,
            )
        return (
            np.block([[k11, k12, k13], [k12.T, k22, k23], [k13.T, k23.T, k33]]),
            np.block([[m11, m12, m13], [m12.T, m22, m23], [m13.T, m23.T, m33]]),
            0,
        )

    if zdef != "Beam":
        raise ValueError(f"Unsupported mode_params.zdef value {zdef!r}.")

    if n_value == 0 and m_value > 0:
        return (
            np.block([[k11, k13], [k13.T, k33]]),
            np.block([[m11, m13], [m13.T, m33]]),
            1,
        )
    if n_value == 0 and m_value == 0:
        return k33.copy(), m33.copy(), 2
    if m_value == 0:
        return (
            np.block([[k22, k23], [k23.T, k33]]),
            np.block([[m22, m23], [m23.T, m33]]),
            3,
        )
    return (
        np.block([[k11, k12, k13], [k12.T, k22, k23], [k13.T, k23.T, k33]]),
        np.block([[m11, m12, m13], [m12.T, m22, m23], [m13.T, m23.T, m33]]),
        0,
    )


def _pad_removed_dofs(shapes: np.ndarray, case_code: int, du: int, dv: int, dw: int) -> np.ndarray:
    mode_count = shapes.shape[1]
    if case_code == 0:
        expected_rows = du + dv + dw
        if shapes.shape[0] != expected_rows:
            raise ValueError(
                f"Expected {expected_rows} rows in full shape matrix, got {shapes.shape[0]}."
            )
        return shapes

    if case_code == 1:
        expected_rows = du + dw
        if shapes.shape[0] != expected_rows:
            raise ValueError(
                f"Expected {expected_rows} reduced rows for n=0 case, got {shapes.shape[0]}."
            )
        return np.vstack(
            (
                shapes[:du, :],
                np.zeros((dv, mode_count), dtype=np.complex128),
                shapes[du:, :],
            )
        )

    if case_code == 2:
        if shapes.shape[0] != dw:
            raise ValueError(
                f"Expected {dw} reduced rows for n=0, m=0 beam case, got {shapes.shape[0]}."
            )
        return np.vstack(
            (
                np.zeros((du, mode_count), dtype=np.complex128),
                np.zeros((dv, mode_count), dtype=np.complex128),
                shapes,
            )
        )

    if case_code == 3:
        expected_rows = dv + dw
        if shapes.shape[0] != expected_rows:
            raise ValueError(
                f"Expected {expected_rows} reduced rows for m=0 beam case, got {shapes.shape[0]}."
            )
        return np.vstack(
            (
                np.zeros((du, mode_count), dtype=np.complex128),
                shapes[:dv, :],
                shapes[dv:, :],
            )
        )

    raise ValueError(f"Unsupported DOF-elimination case code {case_code}.")


def _mass_normalize(modes: np.ndarray, mass: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if modes.size == 0:
        empty_real = np.empty((0,), dtype=np.float64)
        empty_complex = np.empty((0,), dtype=np.complex128)
        return modes, empty_complex, empty_real

    c_norm_complex = np.diag(modes.T @ mass @ modes)
    factors = np.zeros_like(c_norm_complex, dtype=np.complex128)
    valid = np.abs(c_norm_complex) > _EPS
    factors[valid] = np.sqrt(1.0 / c_norm_complex[valid])
    normalized = modes * factors[np.newaxis, :]

    maybe_real = np.real_if_close(c_norm_complex, tol=1000)
    if np.iscomplexobj(maybe_real):
        c_norm_store = np.abs(c_norm_complex).astype(np.float64)
    else:
        c_norm_store = np.asarray(maybe_real, dtype=np.float64)

    return normalized, c_norm_complex, c_norm_store


def calc_eigs(
    n: int,
    m_range: list[int],
    components: list[CylinderComponent],
    mode_params: ModeParams,
    phi_integrals: list[dict[str, object]],
    z_integrals: list[list[dict[str, object]]],
    r_integrals: list[dict[str, object]],
    C_matrices: list[np.ndarray],
    freq_cutoff_hz: float = _DEFAULT_FREQ_CUTOFF_HZ,
) -> list[EigenResult]:
    """Assemble K/M blocks and solve the MATLAB-style generalized eigenproblems.

    The return value is one [`EigenResult`](ema_msa_py/data_structures.py:181) per
    axial wave number in ``m_range``.
    """

    if not m_range:
        return []

    freq_cutoff_value = float(freq_cutoff_hz)
    if not np.isfinite(freq_cutoff_value) or freq_cutoff_value < 0.0:
        raise ValueError("freq_cutoff_hz must be finite and non-negative.")

    component_count = _validate_lengths(components, phi_integrals, z_integrals, r_integrals, C_matrices)
    du, dv, dw = _dof_sizes(mode_params)
    n_value = int(n)

    results: list[EigenResult] = []

    for m_index, m_value_raw in enumerate(m_range):
        m_value = int(m_value_raw)
        assembled_k: np.ndarray | None = None
        assembled_m: np.ndarray | None = None
        case_code: int | None = None

        for component_index in range(component_count):
            phi_data = phi_integrals[component_index]
            z_data = _resolve_z_integral(z_integrals[component_index], m_index, m_value)
            r_data = r_integrals[component_index]
            constitutive = _validate_constitutive_matrix(C_matrices[component_index], component_index)
            rho_value = float(components[component_index].Rho)

            blocks = _assemble_component_blocks(
                n_value=n_value,
                rho_value=rho_value,
                phi_data=phi_data,
                z_data=z_data,
                r_data=r_data,
                constitutive=constitutive,
            )
            reduced_k, reduced_m, current_case = _select_reduced_system(
                zdef=mode_params.zdef,
                n_value=n_value,
                m_value=m_value,
                du=du,
                dv=dv,
                dw=dw,
                k11=blocks[0],
                m11=blocks[1],
                k12=blocks[2],
                m12=blocks[3],
                k13=blocks[4],
                m13=blocks[5],
                k22=blocks[6],
                m22=blocks[7],
                k23=blocks[8],
                m23=blocks[9],
                k33=blocks[10],
                m33=blocks[11],
            )

            if assembled_k is None:
                assembled_k = np.asarray(reduced_k, dtype=np.complex128)
                assembled_m = np.asarray(reduced_m, dtype=np.complex128)
                case_code = current_case
            else:
                if reduced_k.shape != assembled_k.shape or reduced_m.shape != assembled_m.shape:
                    raise ValueError(
                        "All component contributions for a given (n, m) pair must share the same reduced matrix shape."
                    )
                assembled_k = assembled_k + reduced_k
                assembled_m = assembled_m + reduced_m
                if case_code != current_case:
                    raise ValueError("Inconsistent DOF-elimination case across components.")

        if assembled_k is None or assembled_m is None or case_code is None:
            raise ValueError("At least one component is required to assemble the eigenproblem.")

        assembled_k = 0.5 * (assembled_k + assembled_k.T)
        assembled_m = 0.5 * (assembled_m + assembled_m.T)

        eigenvalues, eigenvectors = eig(assembled_k, assembled_m, check_finite=False)
        poles = np.asarray(eigenvalues, dtype=np.complex128)

        freq = np.sqrt(np.abs(np.real(poles))) / (2.0 * np.pi)
        damp = np.imag(poles) / (np.real(poles) + _EPS)

        order = np.argsort(freq, kind="stable")
        poles = poles[order]
        freq = np.asarray(freq[order], dtype=np.float64)
        damp = np.asarray(damp[order], dtype=np.float64)
        eigenvectors = np.asarray(eigenvectors[:, order], dtype=np.complex128)

        normalized_shapes, _, c_norm_store = _mass_normalize(eigenvectors, assembled_m)

        keep = np.isfinite(freq) & np.isfinite(damp) & (freq < freq_cutoff_value)
        poles = poles[keep]
        freq = freq[keep]
        damp = damp[keep]
        normalized_shapes = normalized_shapes[:, keep]
        c_norm_store = c_norm_store[keep]

        padded_shapes = _pad_removed_dofs(normalized_shapes, case_code, du, dv, dw)

        results.append(EigenResult(n=n_value, m=m_value, Poles=poles, Freqs=freq, Damps=damp, Shapes=padded_shapes, C_Norms=c_norm_store))

    return results


__all__ = ["calc_eigs"]
