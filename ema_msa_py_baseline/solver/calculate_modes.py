"""Top-level modal-analysis orchestrator translated from the MATLAB EMA_MSA solver."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Literal

import numpy as np

from ..core.hooke import hooke
from ..data_structures import (
    CylinderComponent,
    HxrComponents,
    ModeParams,
    ModeShapeResult,
    MSComponent,
    MSHeader,
    NMFD,
)
from ..integrals.phi_integs import phi_integs
from ..integrals.r_integs import r_integs
from ..integrals.z_integs import z_integs

from .calc_eigs import calc_eigs

_EPS = np.finfo(np.float64).eps


@dataclass(slots=True)
class _AssemblyEntry:
    """One effective component ready for integral evaluation and eigensolution assembly."""

    component: CylinderComponent
    q: int
    constitutive: np.ndarray
    include_in_length: bool = True


def _positive_span(limits: tuple[float, float]) -> bool:
    lower = float(limits[0])
    upper = float(limits[1])
    return np.isfinite(lower) and np.isfinite(upper) and (upper - lower) > _EPS


def _has_nonzero_volume(component: CylinderComponent) -> bool:
    return _positive_span(component.R_lim) and _positive_span(component.Phi_lim) and _positive_span(component.Z_lim)


def _named_component(
    component: CylinderComponent,
    *,
    name: str,
    r_lim: tuple[float, float] | None = None,
    phi_lim: tuple[float, float] | None = None,
    z_lim: tuple[float, float] | None = None,
) -> CylinderComponent:
    return replace(
        component,
        name=str(name),
        R_lim=component.R_lim if r_lim is None else r_lim,
        Phi_lim=component.Phi_lim if phi_lim is None else phi_lim,
        Z_lim=component.Z_lim if z_lim is None else z_lim,
    )


def _build_entry(component: CylinderComponent, q: int, *, include_in_length: bool = True) -> _AssemblyEntry:
    q_value = int(q)
    return _AssemblyEntry(
        component=component,
        q=q_value,
        constitutive=hooke(component),
        include_in_length=include_in_length,
    )


def _assemble_cylinder_components(components: dict[str, CylinderComponent]) -> list[_AssemblyEntry]:
    entries: list[_AssemblyEntry] = []

    for name, component in components.items():
        if not isinstance(component, CylinderComponent):
            raise TypeError(f"Cylinder component {name!r} must be a CylinderComponent instance.")

        effective = _named_component(component, name=name)
        if not _has_nonzero_volume(effective):
            continue
        entries.append(_build_entry(effective, 0, include_in_length=True))

    return entries


def _assemble_hxr_components(components: HxrComponents) -> list[_AssemblyEntry]:
    entries: list[_AssemblyEntry] = []
    wind_geometry: tuple[tuple[float, float], tuple[float, float], tuple[float, float], int] | None = None

    for component_field in fields(HxrComponents):
        name = component_field.name
        component = getattr(components, name)
        if not isinstance(component, CylinderComponent):
            raise TypeError(f"HXR component {name!r} must be a CylinderComponent instance.")

        if name == "wind":
            if getattr(component, "Exist", 1) == 0 or wind_geometry is None:
                continue

            wind_r_lim, wind_phi_lim, wind_z_lim, wind_q = wind_geometry
            effective = _named_component(
                component,
                name=name,
                r_lim=wind_r_lim,
                phi_lim=wind_phi_lim,
                z_lim=wind_z_lim,
            )
            if not _has_nonzero_volume(effective):
                continue
            entries.append(_build_entry(effective, wind_q, include_in_length=False))
            continue

        effective = _named_component(component, name=name)
        if not _has_nonzero_volume(effective):
            continue

        q_value = 0
        if name == "tooth":
            q_value = int(getattr(component, "Qst", 0))
            if q_value > 0:
                wind_geometry = (
                    effective.R_lim,
                    (float(effective.Phi_lim[1]), float(effective.Phi_lim[0]) + (2.0 * np.pi / float(q_value))),
                    effective.Z_lim,
                    q_value,
                )
            else:
                wind_geometry = None
        elif name == "srib":
            q_value = int(getattr(component, "Qsr", 0))
        elif name == "frib":
            q_value = int(getattr(component, "Qfr", 0))

        entries.append(_build_entry(effective, q_value, include_in_length=True))

    return entries


def _assemble_entries(
    mod: int,
    components: dict[str, CylinderComponent] | HxrComponents,
) -> list[_AssemblyEntry]:
    mod_value = int(mod)

    if mod_value == 1:
        if not isinstance(components, HxrComponents):
            raise TypeError("mod=1 requires an HxrComponents instance.")
        return _assemble_hxr_components(components)

    if mod_value == 2:
        if not isinstance(components, dict):
            raise TypeError("mod=2 requires a dict[str, CylinderComponent].")
        return _assemble_cylinder_components(components)

    raise ValueError(f"Unsupported model type {mod_value}. Only mod=1 and mod=2 are supported.")


def _overall_length(entries: list[_AssemblyEntry]) -> tuple[float, float]:
    z_limits = [entry.component.Z_lim for entry in entries if entry.include_in_length]
    if not z_limits:
        raise ValueError("At least one non-degenerate component is required to derive the overall axial limits.")

    z_min = min(float(z_lim[0]) for z_lim in z_limits)
    z_max = max(float(z_lim[1]) for z_lim in z_limits)
    if not np.isfinite(z_min) or not np.isfinite(z_max) or (z_max - z_min) <= _EPS:
        raise ValueError("Overall axial limits must define a positive finite length.")
    return (z_min, z_max)


def _normalize_mode_params(mp: ModeParams, length_limits: tuple[float, float]) -> ModeParams:
    if not isinstance(mp, ModeParams):
        raise TypeError("mp must be a ModeParams instance.")

    zdef = str(mp.zdef).strip().capitalize()
    if zdef not in {"Beam", "Poly"}:
        raise ValueError(f"Unsupported deformation type {mp.zdef!r}.")

    n_values = [int(value) for value in mp.n]
    if not n_values:
        raise ValueError("mp.n must contain at least one circumferential wave number.")

    if zdef == "Poly":
        m_values = [1]
    else:
        m_values = [int(value) for value in mp.m]
        if not m_values:
            raise ValueError("mp.m must contain at least one axial wave number for Beam mode.")

    polyord = tuple(int(value) for value in mp.polyord)
    if len(polyord) != 6 or any(value < 1 for value in polyord):
        raise ValueError("mp.polyord must contain exactly six positive integers.")

    return replace(mp, n=n_values, m=m_values, L=length_limits, zdef=zdef, polyord=polyord)


def _phi_integrals_for_n(n_value: int, entries: list[_AssemblyEntry]) -> list[dict[str, object]]:
    return [
        phi_integs(
            n_value,
            float(entry.component.Phi_lim[0]),
            float(entry.component.Phi_lim[1]),
            entry.q,
        )
        for entry in entries
    ]


def _z_integrals_for_entries(mode_params: ModeParams, entries: list[_AssemblyEntry]) -> list[list[dict[str, object]]]:
    length_value = float(mode_params.L[1] - mode_params.L[0])
    symmetric = abs(float(mode_params.L[0] + mode_params.L[1])) <= 10.0 * _EPS

    if mode_params.zdef == "Beam":
        return [
            [
                z_integs(
                    int(m_value),
                    int(m_value),
                    mode_params.zbound,
                    length_value,
                    poly_mode=False,
                    n_gauss=24,
                    limits=entry.component.Z_lim,
                )
                for m_value in mode_params.m
            ]
            for entry in entries
        ]

    return [
        [
            z_integs(
                mode_params.polyord,
                0,
                mode_params.zbound,
                length_value,
                poly_mode=True,
                limits=entry.component.Z_lim,
                symmetric=symmetric,
            )
        ]
        for entry in entries
    ]


def _r_integrals_for_entries(mode_params: ModeParams, entries: list[_AssemblyEntry]) -> list[dict[str, object]]:
    return [
        r_integs(
            float(entry.component.R_lim[0]),
            float(entry.component.R_lim[1]),
            mode_params.polyord,
        )
        for entry in entries
    ]


def _to_ms_component(entry: _AssemblyEntry) -> MSComponent:
    component = entry.component
    return MSComponent(
        component=component.name,
        C=np.asarray(entry.constitutive, dtype=np.complex128),
        rlim=component.R_lim,
        philim=component.Phi_lim,
        zlim=component.Z_lim,
        q=entry.q,
        rho=float(component.Rho),
    )


def _build_nmfd(modes: list) -> NMFD:
    rows: list[np.ndarray] = []

    for group_index, mode in enumerate(modes, start=1):
        freq = np.asarray(mode.Freqs, dtype=np.float64)
        damp = np.asarray(mode.Damps, dtype=np.float64)
        if freq.size == 0:
            continue

        rows.append(
            np.column_stack(
                (
                    np.full(freq.size, float(mode.n), dtype=np.float64),
                    np.full(freq.size, float(mode.m), dtype=np.float64),
                    freq,
                    damp,
                    np.full(freq.size, float(group_index), dtype=np.float64),
                    np.arange(1, freq.size + 1, dtype=np.float64),
                )
            )
        )

    if not rows:
        return NMFD()
    return NMFD(data=np.vstack(rows).astype(np.float64, copy=False))


def calculate_modes(
    mod: Literal[1, 2],
    mp: ModeParams,
    components: dict[str, CylinderComponent] | HxrComponents,
    freq_cutoff: float = 8000.0,
) -> tuple[ModeShapeResult, NMFD]:
    """Assemble component integrals and solve all requested modal groups.

    MATLAB source: ``Ema_Msa_7/ema_msa_calculate_modes.m`` lines 1-177.
    """

    freq_cutoff_value = float(freq_cutoff)
    if not np.isfinite(freq_cutoff_value) or freq_cutoff_value < 0.0:
        raise ValueError("freq_cutoff must be finite and non-negative.")

    entries = _assemble_entries(int(mod), components)
    if not entries:
        raise ValueError("No non-degenerate components were found for modal analysis.")

    mode_params = _normalize_mode_params(mp, _overall_length(entries))

    z_integrals = _z_integrals_for_entries(mode_params, entries)
    r_integrals = _r_integrals_for_entries(mode_params, entries)

    effective_components = [entry.component for entry in entries]
    constitutive_matrices = [entry.constitutive for entry in entries]

    modes = []
    for n_value in mode_params.n:
        modes.extend(
            calc_eigs(
                int(n_value),
                list(mode_params.m),
                effective_components,
                mode_params,
                _phi_integrals_for_n(int(n_value), entries),
                z_integrals,
                r_integrals,
                constitutive_matrices,
                freq_cutoff_hz=freq_cutoff_value,
            )
        )

    ms = ModeShapeResult(
        header=MSHeader(
            zdef=mode_params.zdef,
            zbound=mode_params.zbound,
            polyord=tuple(mode_params.polyord),
            tot_length=mode_params.L,
        ),
        components=[_to_ms_component(entry) for entry in entries],
        modes=modes,
    )
    return ms, _build_nmfd(modes)


__all__ = ["calculate_modes"]
