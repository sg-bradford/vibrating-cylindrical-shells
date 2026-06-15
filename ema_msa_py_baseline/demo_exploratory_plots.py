from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from . import CylinderComponent, ModeParams, calculate_modes
from .core.beam_func import beam_func
from .data_structures import EigenResult, ModeShapeResult

_EPS = float(np.finfo(np.float64).eps)
_DEFAULT_MIN_FREQUENCY_HZ = 1.0
_DEFAULT_POLYORD = (1, 1, 1, 4, 4, 2)


@dataclass(frozen=True, slots=True)
class SelectedMode:
    n: int
    m: int
    label: str


SELECTED_MODES: tuple[SelectedMode, ...] = (
    SelectedMode(0, 0, "breathing"),
    SelectedMode(2, 0, "ovalization"),
    SelectedMode(2, 1, "ovalization + 1 axial half-wave"),
)


def _configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def _build_component() -> CylinderComponent:
    length = 0.510
    radius = 0.317
    thickness = 0.045
    inner_radius = radius - 0.5 * thickness
    outer_radius = radius + 0.5 * thickness
    density = 557.0 / (math.pi * (outer_radius**2 - inner_radius**2) * length)

    return CylinderComponent(
        name="yoke",
        R_lim=(inner_radius, outer_radius),
        Phi_lim=(0.0, 2.0 * math.pi),
        Z_lim=(0.0, length),
        nr=1,
        nphi=96,
        nz=64,
        Rho=density,
        Er=207.0e9,
        Ephi=207.0e9,
        Ez=1.0e9,
        mu_phir=0.3,
        mu_zr=0.0015,
        mu_zphi=0.0015,
        Gphir=79.6e9,
        Gzr=0.5e9,
        Gzphi=0.5e9,
        eta_r=0.0,
        eta_phi=0.0,
        eta_z=0.0,
        eta_phir=0.0,
        eta_zr=0.0,
        eta_zphi=0.0,
    )


def _build_mode_params() -> ModeParams:
    return ModeParams(
        n=[0, 2, 3, 4],
        m=[0, 1],
        zdef="Beam",
        zbound="Free-Free",
        polyord=_DEFAULT_POLYORD,
    )


def _phase_align(shape_matrix: np.ndarray, branch_index: int) -> np.ndarray:
    shapes = np.asarray(shape_matrix, dtype=np.complex128)
    if shapes.ndim != 2:
        raise ValueError("EigenResult.Shapes must be 2-D.")
    if branch_index < 0 or branch_index >= shapes.shape[1]:
        raise IndexError("Requested branch index is out of range.")

    column = np.asarray(shapes[:, branch_index], dtype=np.complex128)
    pivot_index = int(np.argmax(np.abs(column)))
    pivot_magnitude = abs(column[pivot_index])
    if pivot_magnitude <= _EPS:
        return column
    return column * np.exp(-1j * np.angle(column[pivot_index]))


def _radial_polynomial(coefficients: np.ndarray, radius: float) -> complex:
    coeffs = np.asarray(coefficients, dtype=np.complex128)
    if coeffs.ndim != 1:
        raise ValueError("Radial coefficient vector must be 1-D.")
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("Radius must be finite and positive.")
    powers = np.power(float(radius), np.arange(coeffs.size, dtype=np.float64))
    return complex(np.dot(coeffs, powers))


def _bending_boundary(zbound: str) -> str:
    mapping = {
        "Free-Free": "ff",
        "Pinned-Free": "fs",
        "Pinned-Pinned": "ss",
        "Clamped-Clamped": "cc",
        "Clamped-Free": "cf",
        "Clamped-Pinned": "cs",
    }
    try:
        return mapping[str(zbound)]
    except KeyError as exc:
        raise ValueError(f"Unsupported zbound {zbound!r}.") from exc


def _choose_branch(result: EigenResult, min_frequency_hz: float) -> tuple[int, float]:
    if not math.isfinite(min_frequency_hz) or min_frequency_hz < 0.0:
        raise ValueError("min_frequency_hz must be finite and non-negative.")

    freqs = np.asarray(result.Freqs, dtype=np.float64)
    valid = np.flatnonzero(np.isfinite(freqs) & (freqs >= min_frequency_hz))
    if valid.size == 0:
        raise ValueError(f"No valid branch found for n={result.n}, m={result.m}.")
    index = int(valid[0])
    return index, float(freqs[index])


def _reconstruct_outer_surface_w(
    ms: ModeShapeResult,
    result: EigenResult,
    branch_index: int,
    radius: float,
    phi_coordinates: np.ndarray,
    z_coordinates: np.ndarray,
) -> np.ndarray:
    du, dv, dw = _DEFAULT_POLYORD[3:6]
    shape = _phase_align(np.asarray(result.Shapes, dtype=np.complex128), branch_index)
    if shape.size < du + dv + dw:
        raise ValueError("Mode shape vector is smaller than the expected beam DOF count.")

    w_coefficients = shape[du + dv : du + dv + dw]
    total_length = float(ms.header.tot_length[1] - ms.header.tot_length[0])
    if not math.isfinite(total_length) or total_length <= 0.0:
        raise ValueError("ModeShapeResult.header.tot_length must define a positive length.")

    wz = np.asarray(
        beam_func("b", _bending_boundary(ms.header.zbound), 0, int(result.m), z_coordinates, total_length),
        dtype=np.float64,
    )
    phi_term = np.cos(float(result.n) * phi_coordinates)[:, np.newaxis]
    field = _radial_polynomial(w_coefficients, radius) * phi_term * wz[np.newaxis, :]
    return np.asarray(np.real(np.real_if_close(field, tol=1000)), dtype=np.float64)


def _plot_frequency_overview(ms: ModeShapeResult, output_path: Path, min_frequency_hz: float) -> None:
    figure, axis = plt.subplots(figsize=(10.0, 4.8), constrained_layout=True)
    legend_entries: dict[str, object] = {}
    x_offset = 0

    for result in ms.modes:
        freqs = np.asarray(result.Freqs, dtype=np.float64)
        freqs = freqs[np.isfinite(freqs) & (freqs >= min_frequency_hz)]
        if freqs.size == 0:
            continue

        x_values = np.arange(x_offset, x_offset + freqs.size, dtype=np.int32)
        color = f"C{int(result.n) % 10}"
        axis.vlines(x_values, 0.0, freqs, color=color, alpha=0.55, linewidth=1.3)
        scatter = axis.scatter(x_values, freqs, color=color, s=36, label=f"n={int(result.n)}, m={int(result.m)}")
        legend_entries.setdefault(f"n={int(result.n)}, m={int(result.m)}", scatter)
        x_offset += freqs.size + 1

    axis.set_title("Baseline EMA/MSA exploratory frequency overview")
    axis.set_xlabel("Retained modal branches")
    axis.set_ylabel("Natural frequency [Hz]")
    axis.grid(True, alpha=0.3)
    if legend_entries:
        axis.legend(legend_entries.values(), legend_entries.keys(), fontsize=8, ncols=2)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _plot_mode_heatmaps(
    ms: ModeShapeResult,
    component: CylinderComponent,
    output_path: Path,
    min_frequency_hz: float,
) -> list[tuple[str, int, int, int, float]]:
    groups = {(int(result.n), int(result.m)): result for result in ms.modes}
    phi_coordinates = np.linspace(0.0, 2.0 * math.pi, 181)
    z_coordinates = np.linspace(float(component.Z_lim[0]), float(component.Z_lim[1]), 180)

    figure, axes = plt.subplots(1, len(SELECTED_MODES), figsize=(15.0, 4.5), constrained_layout=True)
    axes_array = np.atleast_1d(axes)

    summaries: list[tuple[str, int, int, int, float]] = []
    image = None

    for axis, selected in zip(axes_array, SELECTED_MODES, strict=True):
        key = (selected.n, selected.m)
        if key not in groups:
            raise ValueError(f"Missing solved modal group for requested mode {key}.")

        result = groups[key]
        branch_index, frequency_hz = _choose_branch(result, min_frequency_hz)
        field = _reconstruct_outer_surface_w(
            ms=ms,
            result=result,
            branch_index=branch_index,
            radius=float(component.R_lim[1]),
            phi_coordinates=phi_coordinates,
            z_coordinates=z_coordinates,
        )
        scale = float(np.max(np.abs(field)))
        normalized = field if scale <= _EPS else field / scale

        image = axis.imshow(
            normalized,
            extent=(float(z_coordinates[0]), float(z_coordinates[-1]), 0.0, 360.0),
            origin="lower",
            aspect="auto",
            cmap="RdBu_r",
            vmin=-1.0,
            vmax=1.0,
        )
        axis.set_title(f"{selected.label}\nn={selected.n}, m={selected.m}, f={frequency_hz:.1f} Hz")
        axis.set_xlabel("z [m]")
        axis.set_ylabel("phi [deg]")
        summaries.append((selected.label, selected.n, selected.m, branch_index + 1, frequency_hz))

    if image is not None:
        figure.colorbar(image, ax=list(axes_array), label="Normalized radial displacement w")

    figure.suptitle("Baseline EMA/MSA outer-surface radial mode-shape patterns")
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return summaries


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate exploratory frequency and radial mode-shape plots from ema_msa_py_baseline."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory where the output PNG files will be written.",
    )
    parser.add_argument(
        "--min-frequency-hz",
        type=float,
        default=_DEFAULT_MIN_FREQUENCY_HZ,
        help="Minimum frequency used when selecting plotted branches.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    _configure_stdout()
    args = parse_args(argv)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    min_frequency_hz = float(args.min_frequency_hz)
    component = _build_component()
    mode_params = _build_mode_params()
    ms, nmfd = calculate_modes(mod=2, mp=mode_params, components={component.name: component})

    freqs_path = output_dir / "ema_msa_py_baseline_exploratory_freqs.png"
    modes_path = output_dir / "ema_msa_py_baseline_exploratory_modes.png"

    _plot_frequency_overview(ms, freqs_path, min_frequency_hz)
    summaries = _plot_mode_heatmaps(ms, component, modes_path, min_frequency_hz)

    print("Baseline exploratory plots generated")
    print(f"Solved groups: {len(ms.modes)}")
    print(f"Retained NMFD rows: {int(nmfd.data.shape[0])}")
    for label, n_value, m_value, branch, frequency_hz in summaries:
        print(f"{label}: n={n_value}, m={m_value}, branch={branch}, f={frequency_hz:.3f} Hz")
    print(f"Saved: {freqs_path}")
    print(f"Saved: {modes_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
