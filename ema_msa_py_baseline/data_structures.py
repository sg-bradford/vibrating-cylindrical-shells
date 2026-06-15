"""Shared dataclasses for the Python EMA_MSA replication."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
ComplexArray: TypeAlias = NDArray[np.complex128]
IntArray: TypeAlias = NDArray[np.int32]


def _empty_float_array() -> FloatArray:
    return np.array([], dtype=np.float64)


def _empty_complex_array() -> ComplexArray:
    return np.array([], dtype=np.complex128)


def _empty_vertices() -> FloatArray:
    return np.empty((0, 3), dtype=np.float64)


def _empty_faces() -> IntArray:
    return np.empty((0, 4), dtype=np.int32)


def _zero_complex_matrix() -> ComplexArray:
    return np.zeros((6, 6), dtype=np.complex128)


@dataclass(slots=True)
class ModeParams:
    """Mode calculation parameters — replaces MATLAB ``mp``."""

    n: list[int] = field(default_factory=list)
    m: list[int] = field(default_factory=list)
    L: tuple[float, float] = (0.0, 0.0)
    zdef: Literal["Beam", "Poly"] = "Beam"
    zbound: Literal[
        "Free-Free",
        "Pinned-Free",
        "Pinned-Pinned",
        "Clamped-Clamped",
        "Clamped-Free",
        "Clamped-Pinned",
    ] = "Free-Free"
    polyord: tuple[int, int, int, int, int, int] = (1, 1, 1, 1, 1, 1)
    P11: PolyExpResult | None = None
    P12: PolyExpResult | None = None
    P13: PolyExpResult | None = None
    P22: PolyExpResult | None = None
    P23: PolyExpResult | None = None
    P33: PolyExpResult | None = None


@dataclass(slots=True)
class CylinderComponent:
    """Material and geometry for one cylindrical layer."""

    name: str = ""
    R_lim: tuple[float, float] = (0.0, 0.0)
    Phi_lim: tuple[float, float] = (0.0, 0.0)
    Z_lim: tuple[float, float] = (0.0, 0.0)
    nr: int = 0
    nphi: int = 0
    nz: int = 0
    Rho: float = 0.0
    Er: float = 0.0
    Ephi: float = 0.0
    Ez: float = 0.0
    mu_phir: float = 0.0
    mu_zr: float = 0.0
    mu_zphi: float = 0.0
    Gphir: float = 0.0
    Gzr: float = 0.0
    Gzphi: float = 0.0
    eta_r: float = 0.0
    eta_phi: float = 0.0
    eta_z: float = 0.0
    eta_phir: float = 0.0
    eta_zr: float = 0.0
    eta_zphi: float = 0.0


@dataclass(slots=True)
class HxrCylindricalComponent(CylinderComponent):
    """Cylindrical HXR component."""


@dataclass(slots=True)
class HxrToothComponent(CylinderComponent):
    """Tooth component for the HXR machine model."""

    Qst: int = 0


@dataclass(slots=True)
class HxrRibComponent(CylinderComponent):
    """Rib component for the HXR machine model."""

    Qsr: int = 0
    Qfr: int = 0


@dataclass(slots=True)
class HxrWindingComponent(CylinderComponent):
    """Winding component for the HXR machine model."""

    Exist: int = 1


@dataclass(slots=True)
class HxrComponents:
    """Component collection for the HXR machine model."""

    tooth: HxrToothComponent = field(default_factory=HxrToothComponent)
    sback: HxrCylindricalComponent = field(default_factory=HxrCylindricalComponent)
    srib: HxrRibComponent = field(default_factory=HxrRibComponent)
    wind: HxrWindingComponent = field(default_factory=HxrWindingComponent)
    frame: HxrCylindricalComponent = field(default_factory=HxrCylindricalComponent)
    frib: HxrRibComponent = field(default_factory=HxrRibComponent)


@dataclass(slots=True)
class ComponentIntegrals:
    """Pre-computed integrals for one component."""

    component: str = ""
    rlim: tuple[float, float] = (0.0, 0.0)
    philim: tuple[float, float] = (0.0, 0.0)
    zlim: tuple[float, float] = (0.0, 0.0)
    q: int = 0
    C: ComplexArray = field(default_factory=_zero_complex_matrix)
    rho: float = 0.0
    I_sin: FloatArray = field(default_factory=_empty_float_array)
    I_cos: FloatArray = field(default_factory=_empty_float_array)
    k11z1: FloatArray = field(default_factory=_empty_float_array)
    k11z2: FloatArray = field(default_factory=_empty_float_array)
    k11z3: FloatArray = field(default_factory=_empty_float_array)
    k12z1: FloatArray = field(default_factory=_empty_float_array)
    k12z2: FloatArray = field(default_factory=_empty_float_array)
    k13z1: FloatArray = field(default_factory=_empty_float_array)
    k13z2: FloatArray = field(default_factory=_empty_float_array)
    k13z3: FloatArray = field(default_factory=_empty_float_array)
    k22z1: FloatArray = field(default_factory=_empty_float_array)
    k22z2: FloatArray = field(default_factory=_empty_float_array)
    k23z1: FloatArray = field(default_factory=_empty_float_array)
    k33z1: FloatArray = field(default_factory=_empty_float_array)
    k33z2: FloatArray = field(default_factory=_empty_float_array)
    k11r1: FloatArray = field(default_factory=_empty_float_array)
    k11r2: FloatArray = field(default_factory=_empty_float_array)
    k11r3: FloatArray = field(default_factory=_empty_float_array)
    k12r1: FloatArray = field(default_factory=_empty_float_array)
    k12r2: FloatArray = field(default_factory=_empty_float_array)
    k13r1: FloatArray = field(default_factory=_empty_float_array)
    k13r2: FloatArray = field(default_factory=_empty_float_array)
    k13r3: FloatArray = field(default_factory=_empty_float_array)
    k22r1: FloatArray = field(default_factory=_empty_float_array)
    k22r2: FloatArray = field(default_factory=_empty_float_array)
    k22r3: FloatArray = field(default_factory=_empty_float_array)
    k22r4: FloatArray = field(default_factory=_empty_float_array)
    k22r5: FloatArray = field(default_factory=_empty_float_array)
    k22r6: FloatArray = field(default_factory=_empty_float_array)
    k23r1: FloatArray = field(default_factory=_empty_float_array)
    k23r2: FloatArray = field(default_factory=_empty_float_array)
    k23r3: FloatArray = field(default_factory=_empty_float_array)
    k23r4: FloatArray = field(default_factory=_empty_float_array)
    k33r1: FloatArray = field(default_factory=_empty_float_array)
    k33r2: FloatArray = field(default_factory=_empty_float_array)
    k33r3: FloatArray = field(default_factory=_empty_float_array)
    k33r4: FloatArray = field(default_factory=_empty_float_array)
    k33r5: FloatArray = field(default_factory=_empty_float_array)
    k33r6: FloatArray = field(default_factory=_empty_float_array)


@dataclass(slots=True)
class EigenResult:
    """Eigen-solution results for one ``(n, m)`` pair."""

    n: int = 0
    m: int = 0
    Poles: ComplexArray = field(default_factory=_empty_complex_array)
    Freqs: FloatArray = field(default_factory=_empty_float_array)
    Damps: FloatArray = field(default_factory=_empty_float_array)
    Shapes: ComplexArray = field(default_factory=_empty_complex_array)
    C_Norms: FloatArray = field(default_factory=_empty_float_array)


ModeResult = EigenResult


@dataclass(slots=True)
class MSHeader:
    """Header metadata for mode shape results."""

    zdef: str = ""
    zbound: str = ""
    polyord: tuple[int, ...] = ()
    tot_length: tuple[float, float] = (0.0, 0.0)


@dataclass(slots=True)
class MSComponent:
    """Component information stored in reconstructed mode results."""

    component: str = ""
    C: ComplexArray = field(default_factory=_zero_complex_matrix)
    rlim: tuple[float, float] = (0.0, 0.0)
    philim: tuple[float, float] = (0.0, 0.0)
    zlim: tuple[float, float] = (0.0, 0.0)
    q: int = 0
    rho: float = 0.0


@dataclass(slots=True)
class ModeShapeResult:
    """Top-level results container — replaces MATLAB ``MS``."""

    header: MSHeader = field(default_factory=MSHeader)
    components: list[MSComponent] = field(default_factory=list)
    modes: list[EigenResult] = field(default_factory=list)


@dataclass(slots=True)
class NMFD:
    """Mode frequency/damping lookup table."""

    data: FloatArray = field(default_factory=lambda: np.empty((0, 6), dtype=np.float64))


@dataclass(slots=True)
class PolyExpTerm:
    """One term of a polynomial expansion."""

    p1: FloatArray = field(default_factory=_empty_float_array)
    p2: FloatArray = field(default_factory=_empty_float_array)
    c1: FloatArray = field(default_factory=_empty_float_array)
    c2: FloatArray = field(default_factory=_empty_float_array)


@dataclass(slots=True)
class PolyExpResult:
    """Result of ``polyexp`` as a fixed four-term expansion container."""

    terms: list[PolyExpTerm] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.terms) != 4:
            raise ValueError("PolyExpResult must contain exactly four PolyExpTerm items.")


@dataclass(slots=True)
class CylMeshSpec:
    """Specification for one cylindrical mesh block."""

    xlim: tuple[float, float] = (0.0, 0.0)
    nx: int = 0
    ylim: tuple[float, float] = (0.0, 0.0)
    ny: int = 0
    zlim: tuple[float, float] = (0.0, 0.0)
    nz: int = 0


@dataclass(slots=True)
class PatchMesh:
    """Surface mesh with cylindrical-coordinate vertices and quad faces."""

    vertices: FloatArray = field(default_factory=_empty_vertices)
    faces: IntArray = field(default_factory=_empty_faces)


__all__ = [
    "ComplexArray",
    "ComponentIntegrals",
    "CylinderComponent",
    "CylMeshSpec",
    "EigenResult",
    "FloatArray",
    "HxrComponents",
    "HxrCylindricalComponent",
    "HxrRibComponent",
    "HxrToothComponent",
    "HxrWindingComponent",
    "IntArray",
    "ModeParams",
    "ModeResult",
    "ModeShapeResult",
    "MSComponent",
    "MSHeader",
    "NMFD",
    "PatchMesh",
    "PolyExpResult",
    "PolyExpTerm",
]
