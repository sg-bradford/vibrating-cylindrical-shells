"""Constitutive matrix utilities translated from the MATLAB EMA_MSA codebase."""

from __future__ import annotations

from typing import overload

import numpy as np

from ..data_structures import CylinderComponent

_DEFAULT_NU = 0.3
_DEFAULT_ETA = 0.1


def _complex_modulus(value: float, loss_factor_percent: float) -> complex:
    """Apply MATLAB-style hysteretic damping using ``complex(1, 0.01*eta)``."""

    return complex(value) * complex(1.0, 0.01 * loss_factor_percent)


def _validate_real_scalar(name: str, value: float) -> float:
    """Validate that a real-valued elastic input is finite."""

    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite.")
    return scalar


def _build_from_full_args(
    E_x: float,
    E_phi: float,
    E_z: float,
    nu_xphi: float,
    nu_xz: float,
    nu_phiz: float,
    G_phiz: float,
    G_xz: float,
    G_xphi: float,
    eta_x: float,
    eta_phi: float,
    eta_z: float,
    eta_phiz: float,
    eta_xz: float,
    eta_xphi: float,
) -> np.ndarray:
    """Build the orthotropic constitutive matrix from the full MATLAB argument set."""

    E_x_r = _validate_real_scalar("E_x", E_x)
    E_phi_r = _validate_real_scalar("E_phi", E_phi)
    E_z_r = _validate_real_scalar("E_z", E_z)
    nu_xphi_r = _validate_real_scalar("nu_xphi", nu_xphi)
    nu_xz_r = _validate_real_scalar("nu_xz", nu_xz)
    nu_phiz_r = _validate_real_scalar("nu_phiz", nu_phiz)
    G_phiz_r = _validate_real_scalar("G_phiz", G_phiz)
    G_xz_r = _validate_real_scalar("G_xz", G_xz)
    G_xphi_r = _validate_real_scalar("G_xphi", G_xphi)

    if E_x_r == 0.0 or E_phi_r == 0.0 or E_z_r == 0.0:
        raise ValueError("Elastic moduli must be non-zero.")
    if G_phiz_r == 0.0 or G_xz_r == 0.0 or G_xphi_r == 0.0:
        raise ValueError("Shear moduli must be non-zero.")

    E_x_c = _complex_modulus(E_x_r, _validate_real_scalar("eta_x", eta_x))
    E_phi_c = _complex_modulus(E_phi_r, _validate_real_scalar("eta_phi", eta_phi))
    E_z_c = _complex_modulus(E_z_r, _validate_real_scalar("eta_z", eta_z))
    G_phiz_c = _complex_modulus(G_phiz_r, _validate_real_scalar("eta_phiz", eta_phiz))
    G_xz_c = _complex_modulus(G_xz_r, _validate_real_scalar("eta_xz", eta_xz))
    G_xphi_c = _complex_modulus(G_xphi_r, _validate_real_scalar("eta_xphi", eta_xphi))

    nu_phix = nu_xphi_r * (E_phi_r / E_x_r)
    nu_zx = nu_xz_r * (E_z_r / E_x_r)
    nu_zphi = nu_phiz_r * (E_z_r / E_phi_r)

    compliance = np.zeros((6, 6), dtype=np.complex128)
    compliance[0, 0] = 1.0 / E_x_c
    compliance[1, 1] = 1.0 / E_phi_c
    compliance[2, 2] = 1.0 / E_z_c
    compliance[0, 1] = -nu_xphi_r / E_x_c
    compliance[1, 0] = -nu_phix / E_phi_c
    compliance[0, 2] = -nu_xz_r / E_x_c
    compliance[2, 0] = -nu_zx / E_z_c
    compliance[1, 2] = -nu_phiz_r / E_phi_c
    compliance[2, 1] = -nu_zphi / E_z_c
    compliance[3, 3] = 1.0 / G_phiz_c
    compliance[4, 4] = 1.0 / G_xz_c
    compliance[5, 5] = 1.0 / G_xphi_c

    try:
        constitutive = np.linalg.inv(compliance)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Compliance matrix is singular and cannot be inverted.") from exc

    return ((constitutive + constitutive.T) * 0.5).astype(np.complex128, copy=False)


def hooke_from_args(*args: float) -> np.ndarray:
    """Translate MATLAB ``hooke(varargin)`` exactly.

    Source: ``Ema_Msa_7/hooke.m`` lines 32-122.
    """

    if len(args) == 1:
        E = float(args[0])
        G = E / (2.0 * (1.0 + _DEFAULT_NU))
        return _build_from_full_args(
            E,
            E,
            E,
            _DEFAULT_NU,
            _DEFAULT_NU,
            _DEFAULT_NU,
            G,
            G,
            G,
            _DEFAULT_ETA,
            _DEFAULT_ETA,
            _DEFAULT_ETA,
            _DEFAULT_ETA,
            _DEFAULT_ETA,
            _DEFAULT_ETA,
        )

    if len(args) == 2:
        E = float(args[0])
        nu = float(args[1])
        G = E / (2.0 * (1.0 + nu))
        return _build_from_full_args(
            E,
            E,
            E,
            nu,
            nu,
            nu,
            G,
            G,
            G,
            _DEFAULT_ETA,
            _DEFAULT_ETA,
            _DEFAULT_ETA,
            _DEFAULT_ETA,
            _DEFAULT_ETA,
            _DEFAULT_ETA,
        )

    if len(args) == 3:
        E = float(args[0])
        nu = float(args[1])
        eta = float(args[2])
        G = E / (2.0 * (1.0 + nu))
        return _build_from_full_args(
            E,
            E,
            E,
            nu,
            nu,
            nu,
            G,
            G,
            G,
            eta,
            eta,
            eta,
            eta,
            eta,
            eta,
        )

    if len(args) == 10:
        return _build_from_full_args(
            float(args[0]),
            float(args[1]),
            float(args[1]),
            float(args[2]),
            float(args[2]),
            float(args[3]),
            float(args[4]),
            float(args[5]),
            float(args[5]),
            float(args[6]),
            float(args[7]),
            float(args[7]),
            float(args[8]),
            float(args[9]),
            float(args[9]),
        )

    if len(args) == 15:
        return _build_from_full_args(*(float(value) for value in args))

    raise ValueError("Not correct arguments.")


@overload
def hooke(material: CylinderComponent) -> np.ndarray: ...


def hooke(material: CylinderComponent) -> np.ndarray:
    """Build the 6×6 complex constitutive matrix for a cylinder component.

    Source: ``Ema_Msa_7/hooke.m`` lines 32-122; cylindrical field-to-argument
    mapping follows ``Ema_Msa_7/ema_msa_calculate_modes.m`` lines 33-36.
    """

    if not isinstance(material, CylinderComponent):
        raise TypeError("hooke() expects a CylinderComponent instance.")

    return hooke_from_args(
        material.Ez,
        material.Ephi,
        material.Er,
        material.mu_zphi,
        material.mu_zr,
        material.mu_phir,
        material.Gphir,
        material.Gzr,
        material.Gzphi,
        material.eta_z,
        material.eta_phi,
        material.eta_r,
        material.eta_phir,
        material.eta_zr,
        material.eta_zphi,
    )


__all__ = ["hooke", "hooke_from_args"]
