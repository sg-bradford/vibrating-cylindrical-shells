"""Admissible axial beam functions translated from the MATLAB EMA_MSA codebase."""

from __future__ import annotations

from math import pi
from typing import Literal, overload

import numpy as np
from numpy.typing import ArrayLike


def _as_vector(z: ArrayLike) -> np.ndarray:
    """Convert an input coordinate array to a 1-D ``float64`` NumPy vector."""

    return np.atleast_1d(np.asarray(z, dtype=np.float64)).reshape(-1)


def _filled_like(z: np.ndarray, value: float) -> np.ndarray:
    """Return a constant vector matching the shape of ``z``."""

    return np.full(z.shape, float(value), dtype=np.float64)


def _require_supported_derivative(deriv: int, supported: tuple[int, ...]) -> None:
    """Validate derivative order for the selected beam-function branch."""

    if deriv not in supported:
        supported_str = ", ".join(str(value) for value in supported)
        raise ValueError(f"Unsupported derivative order {deriv}; expected one of {supported_str}.")


def _beam_func_full(
    wave_type: Literal["l", "t", "s", "b"],
    boundary: str,
    deriv: int,
    m: int,
    z: ArrayLike,
    L: float,
) -> np.ndarray:
    """Evaluate the full MATLAB ``beam_func(type, bound, deriv, m, z, L)`` form.

    Source: ``Ema_Msa_7/beam_func.m`` lines 40-258.
    """

    z_vec = _as_vector(z)
    if m < 0:
        raise ValueError("Mode number m must be non-negative.")

    L_value = float(L)
    if not np.isfinite(L_value) or L_value <= 0.0:
        raise ValueError("Beam length L must be positive and finite.")

    z_norm = z_vec / L_value
    bound = boundary.lower()

    if wave_type in {"l", "t", "s"}:
        _require_supported_derivative(deriv, (0, 1, 2))
        if bound == "ff":
            k = m * pi
            kz = z_norm * k
            if deriv == 0:
                return np.cos(kz)
            if deriv == 1:
                return -(k / L_value) * np.sin(kz)
            return -((k**2) / (L_value**2)) * np.cos(kz)

        if bound == "cc":
            k = m * pi
            kz = z_norm * k
            if deriv == 0:
                return np.sin(kz)
            if deriv == 1:
                return (k / L_value) * np.cos(kz)
            return -((k**2) / (L_value**2)) * np.sin(kz)

        if bound == "cf":
            k = 0.5 * (2 * m - 1) * pi
            kz = z_norm * k
            if deriv == 0:
                return np.sin(kz)
            if deriv == 1:
                return (k / L_value) * np.cos(kz)
            return -((k**2) / (L_value**2)) * np.sin(kz)

        if bound == "sff":
            if m == 0:
                return _filled_like(z_vec, 0.0)
            if m == 1:
                if deriv == 0:
                    return _filled_like(z_vec, 1.0 / L_value)
                return _filled_like(z_vec, 0.0)
            k = (m - 0.5) * pi
            kz = z_norm * k
            ratio = (np.cosh(k) - np.cos(k)) / (np.sinh(k) - np.sin(k))
            if deriv == 0:
                return np.sinh(kz) - np.sin(kz) - (np.cosh(kz) + np.cos(kz)) * ratio
            if deriv == 1:
                return (k / L_value) * (
                    np.cosh(kz) - np.cos(kz) - (np.sinh(kz) - np.sin(kz)) * ratio
                )
            return ((k**2) / (L_value**2)) * (
                np.sinh(kz) + np.sin(kz) - (np.cosh(kz) - np.cos(kz)) * ratio
            )

        if bound == "scc":
            k = (m + 0.5) * pi
            kz = z_norm * k
            ratio = (np.cosh(k) - np.cos(k)) / (np.sinh(k) - np.sin(k))
            if deriv == 0:
                return np.sinh(kz) + np.sin(kz) - (np.cosh(kz) - np.cos(kz)) * ratio
            if deriv == 1:
                return (k / L_value) * (
                    np.cosh(kz) + np.cos(kz) - (np.sinh(kz) + np.sin(kz)) * ratio
                )
            return ((k**2) / (L_value**2)) * (
                np.sinh(kz) - np.sin(kz) - (np.cosh(kz) + np.cos(kz)) * ratio
            )

        if bound == "sss":
            k = m * pi
            kz = z_norm * k
            if deriv == 0:
                return np.cos(kz)
            if deriv == 1:
                return -(k / L_value) * np.sin(kz)
            return ((k**2) / (L_value**2)) * np.cos(kz)

        if bound == "scs":
            k = (m + 0.25) * pi
            kz = z_norm * k
            ratio = (np.cosh(k) - np.cos(k)) / (np.sinh(k) - np.sin(k))
            if deriv == 0:
                return np.sinh(kz) + np.sin(kz) - (np.cosh(kz) - np.cos(kz)) * ratio
            if deriv == 1:
                return (k / L_value) * (
                    np.cosh(kz) + np.cos(kz) - (np.sinh(kz) + np.sin(kz)) * ratio
                )
            return ((k**2) / (L_value**2)) * (
                np.sinh(kz) - np.sin(kz) - (np.cosh(kz) + np.cos(kz)) * ratio
            )

        if bound == "sfs":
            k = (m + 0.25) * pi
            kz = z_norm * k
            ratio = (np.cosh(k) - np.cos(k)) / (np.sinh(k) - np.sin(k))
            if deriv == 0:
                return np.sinh(kz) - np.sin(kz) - (np.cosh(kz) + np.cos(kz)) * ratio
            if deriv == 1:
                return (k / L_value) * (
                    np.cosh(kz) - np.cos(kz) - (np.sinh(kz) - np.sin(kz)) * ratio
                )
            return ((k**2) / (L_value**2)) * (
                np.sinh(kz) + np.sin(kz) - (np.sinh(kz) - np.cos(kz)) * ratio
            )

        if bound == "scf":
            k = (m - 0.5) * pi
            kz = z_norm * k
            ratio = (np.sinh(k) - np.sin(k)) / (np.cosh(k) + np.cos(k))
            if deriv == 0:
                return np.sinh(kz) + np.sin(kz) - (np.cosh(kz) - np.cos(kz)) * ratio
            if deriv == 1:
                return (k / L_value) * (
                    np.cosh(kz) + np.cos(kz) - (np.sinh(kz) + np.sin(kz)) * ratio
                )
            return ((k**2) / (L_value**2)) * (
                np.sinh(kz) - np.sin(kz) - (np.cosh(kz) + np.cos(kz)) * ratio
            )

        raise ValueError("Not a proper boundary condition for the selected wavetype.")

    if wave_type == "b":
        _require_supported_derivative(deriv, (-1, 0, 1, 2))
        if bound == "ff":
            if m == 0:
                if deriv == 0:
                    return _filled_like(z_vec, 1.0)
                if deriv in {1, 2}:
                    return _filled_like(z_vec, 0.0)
                return _filled_like(z_vec, L_value)

            if m == 1:
                if deriv == 0:
                    return z_norm - 0.5
                if deriv == 1:
                    return _filled_like(z_vec, 1.0 / L_value)
                if deriv == 2:
                    return _filled_like(z_vec, 0.0)
                return _filled_like(z_vec, 0.0)

            k = (m - 0.5) * pi
            kz = z_norm * k
            ratio = (np.cosh(k) - np.cos(k)) / (np.sinh(k) - np.sin(k))
            if deriv == 0:
                return np.cosh(kz) + np.cos(kz) - (np.sinh(kz) + np.sin(kz)) * ratio
            if deriv == 1:
                return (k / L_value) * (
                    np.sinh(kz) - np.sin(kz) - (np.cosh(kz) + np.cos(kz)) * ratio
                )
            if deriv == 2:
                return ((k**2) / (L_value**2)) * (
                    np.cosh(kz) - np.cos(kz) - (np.sinh(kz) - np.sin(kz)) * ratio
                )
            scalar = (L_value / k) * (
                np.sinh(k) + np.sin(k) + (-np.cosh(k) + np.cos(k)) * ratio
            )
            return _filled_like(z_vec, scalar)

        if bound == "cc":
            k = (m + 0.5) * pi
            kz = z_norm * k
            ratio = (np.cosh(k) - np.cos(k)) / (np.sinh(k) - np.sin(k))
            if deriv == 0:
                return np.cosh(kz) - np.cos(kz) - (np.sinh(kz) - np.sin(kz)) * ratio
            if deriv == 1:
                return (k / L_value) * (
                    np.sinh(kz) + np.sin(kz) - (np.cosh(kz) - np.cos(kz)) * ratio
                )
            if deriv == 2:
                return ((k**2) / (L_value**2)) * (
                    np.cosh(kz) + np.cos(kz) - (np.sinh(kz) + np.sin(kz)) * ratio
                )
            raise ValueError("Derivative -1 is not defined for this boundary condition.")

        if bound == "cf":
            k = (m - 0.5) * pi
            kz = z_norm * k
            ratio = (np.sinh(k) - np.sin(k)) / (np.cosh(k) + np.cos(k))
            if deriv == 0:
                return np.cosh(kz) - np.cos(kz) - (np.sinh(kz) - np.sin(kz)) * ratio
            if deriv == 1:
                return (k / L_value) * (
                    np.sinh(kz) + np.sin(kz) - (np.cosh(kz) - np.cos(kz)) * ratio
                )
            if deriv == 2:
                return ((k**2) / (L_value**2)) * (
                    np.cosh(kz) + np.cos(kz) - (np.sinh(kz) + np.sin(kz)) * ratio
                )
            raise ValueError("Derivative -1 is not defined for this boundary condition.")

        if bound == "ss":
            k = m * pi
            kz = z_norm * k
            if deriv == 0:
                return np.sin(kz)
            if deriv == 1:
                return (k / L_value) * np.cos(kz)
            if deriv == 2:
                return -((k**2) / (L_value**2)) * np.sin(kz)
            raise ValueError("Derivative -1 is not defined for this boundary condition.")

        if bound == "fs":
            k = (m + 0.25) * pi
            kz = z_norm * k
            ratio = (np.cosh(k) - np.cos(k)) / (np.sinh(k) - np.sin(k))
            if deriv == 0:
                return np.cosh(kz) + np.cos(kz) - (np.sinh(kz) + np.sin(kz)) * ratio
            if deriv == 1:
                return (k / L_value) * (
                    np.sinh(kz) - np.sin(kz) - (np.cosh(kz) + np.cos(kz)) * ratio
                )
            if deriv == 2:
                return ((k**2) / (L_value**2)) * (
                    np.cosh(kz) - np.cos(kz) - (np.cosh(kz) - np.sin(kz)) * ratio
                )
            raise ValueError("Derivative -1 is not defined for this boundary condition.")

        if bound == "cs":
            k = (m + 0.25) * pi
            kz = z_norm * k
            ratio = (np.cosh(k) - np.cos(k)) / (np.sinh(k) - np.sin(k))
            if deriv == 0:
                return np.cosh(kz) - np.cos(kz) - (np.sinh(kz) - np.sin(kz)) * ratio
            if deriv == 1:
                return (k / L_value) * (
                    np.sinh(kz) + np.sin(kz) - (np.cosh(kz) - np.cos(kz)) * ratio
                )
            if deriv == 2:
                return ((k**2) / (L_value**2)) * (
                    np.cosh(kz) + np.cos(kz) - (np.sinh(kz) + np.sin(kz)) * ratio
                )
            raise ValueError("Derivative -1 is not defined for this boundary condition.")

        raise ValueError("Not a proper boundary condition for the selected wavetype.")

    raise ValueError("Wrong type designation.")


_SHORTHAND_BOUNDARY_MAP: dict[str, str] = {
    "FF": "ff",
    "PF": "fs",
    "PP": "ss",
    "CC": "cc",
    "CF": "cf",
    "CP": "cs",
}


@overload
def beam_func(z_norm: ArrayLike, m: int, bc_type: str, deriv: int = 0) -> np.ndarray: ...


@overload
def beam_func(
    wave_type: Literal["l", "t", "s", "b"],
    boundary: str,
    deriv: int,
    m: int,
    z: ArrayLike,
    L: float,
) -> np.ndarray: ...


def beam_func(*args: object, deriv: int = 0) -> np.ndarray:
    """Evaluate admissible axial functions in shorthand or MATLAB-compatible form.

    Source: ``Ema_Msa_7/beam_func.m`` lines 1-258. The 4-argument shorthand
    requested for this phase maps ``FF/PF/PP/CC/CF/CP`` to the MATLAB bending
    cases ``ff/fs/ss/cc/cf/cs`` on the normalized interval ``[0, 1]``.
    """

    if len(args) == 3:
        z_norm, m_value, bc_type = args
        return _beam_func_full(
            "b",
            _SHORTHAND_BOUNDARY_MAP[str(bc_type).upper()],
            deriv,
            int(m_value),
            z_norm,
            1.0,
        )

    if len(args) == 4:
        z_norm, m_value, bc_type, deriv_value = args
        return _beam_func_full(
            "b",
            _SHORTHAND_BOUNDARY_MAP[str(bc_type).upper()],
            int(deriv_value),
            int(m_value),
            z_norm,
            1.0,
        )

    if len(args) == 6:
        wave_type, boundary, deriv_value, m_value, z, L = args
        return _beam_func_full(
            str(wave_type).lower(),
            str(boundary),
            int(deriv_value),
            int(m_value),
            z,
            float(L),
        )

    raise TypeError("beam_func() expects either 3/4 shorthand args or 6 MATLAB-style args.")


__all__ = ["beam_func"]
