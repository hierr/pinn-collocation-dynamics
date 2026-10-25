"""Compare two saved Burgers resolutions without running either solver."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import dct
from scipy.interpolate import BarycentricInterpolator

from scripts.dataset_generation.burgers_spectral import (
    chebyshev_differentiation_matrix,
)

ROOT = Path(__file__).resolve().parents[2]
VISCOSITY = 0.001
DEGREES = (1024, 2048)
FINAL_TIME = 0.99  # Match the time interval used for PINN evaluation.
UNIFORM_POINTS = 8193
PROFILE_TIMES = (0.35, 0.505, 0.99)
FRONT_COORDINATES = np.linspace(-0.025, 0.025, 2001)


def load_references(paths):
    """Load matching Burgers cases, allowing only their spatial degrees to differ."""
    references = []
    previous_settings = None
    previous_times = None
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            x, t, u = data["x"], data["t"], data["U"]
            config = json.loads(data["config_json"].item())
            if float(data["nu"]) != VISCOSITY or config["nu"] != VISCOSITY:
                raise ValueError("Reference viscosity does not match the comparison.")
        expected_x = np.cos(np.pi * np.arange(config["N"] + 1) / config["N"])
        if not np.array_equal(x, expected_x) or u.shape != (len(t), len(x)):
            raise ValueError("Expected time-by-space data on Chebyshev-Lobatto nodes.")
        if not np.isfinite(u).all():
            raise ValueError("Reference contains non-finite solution values.")
        settings = {
            k: v for k, v in config.items() if k not in ("N", "number_of_spatial_nodes")
        }
        if previous_times is not None and (
            settings != previous_settings or not np.array_equal(t, previous_times)
        ):
            raise ValueError(
                "Reference times and non-spatial solver settings must match."
            )
        selected = t <= FINAL_TIME
        references.append({"x": x, "t": t[selected], "u": u[selected]})
        previous_settings, previous_times = settings, t
    if not np.array_equal(references[0]["x"], references[1]["x"][::2]):
        raise ValueError("This comparison requires nested grids with doubled degree.")
    return references


def sample_reference(x, u, coordinates):
    """Evaluate the Chebyshev solution and its spatial derivative at common points."""
    degree = len(x) - 1
    weights = (-1.0) ** np.arange(degree + 1)
    weights[[0, -1]] *= 0.5
    interpolator = BarycentricInterpolator(x, u, axis=1, wi=weights)
    values = interpolator(coordinates)

    # Differentiate the saved spectral polynomial, then evaluate on the same grid.
    _, differentiation = chebyshev_differentiation_matrix(degree)
    derivative = u @ differentiation.T
    interpolator.set_yi(derivative, axis=1)
    return values, interpolator(coordinates)


def main():
    """Save solution/gradient disagreement, time series, and front-profile figures."""
    tag = f"{VISCOSITY:.8g}".replace(".", "p")
    paths = [ROOT / f"data/burgers_dataset/burgers_nu_{tag}_N{n}.npz" for n in DEGREES]
    output = (
        ROOT
        / f"output/burgers_reference_validation/nu_{tag}_n{DEGREES[0]}_n{DEGREES[1]}"
    )
    coarse, fine = references = load_references(paths)
    times = coarse["t"]
    grid = np.linspace(-1, 1, UNIFORM_POINTS)
    sampled = [sample_reference(ref["x"], ref["u"], grid) for ref in references]
    u0, ux0 = sampled[0]
    u1, ux1 = sampled[1]

    # Compare shared nodes directly; use a uniform grid to check between those nodes.
    direct_difference = coarse["u"] - fine["u"][:, ::2]
    difference = u0 - u1
    relative = np.linalg.norm(difference, axis=1) / np.linalg.norm(u1, axis=1)
    peaks = np.array([np.max(np.abs(ux), axis=1) for _, ux in sampled])
    peak_change = np.abs(peaks[0] - peaks[1]) / peaks[1]
    maximum_index = np.unravel_index(np.argmax(np.abs(difference)), difference.shape)
    summary = {
        "viscosity": VISCOSITY,
        "degrees": list(DEGREES),
        "inputs": [str(path.relative_to(ROOT)) for path in paths],
        "time_interval": [float(times[0]), float(times[-1])],
        "snapshots": len(times),
        "uniform_points": UNIFORM_POINTS,
        "shared_nodes_relative_l2": float(
            np.linalg.norm(direct_difference) / np.linalg.norm(fine["u"][:, ::2])
        ),
        "shared_nodes_max_abs_difference": float(np.max(np.abs(direct_difference))),
        "uniform_relative_l2": float(np.linalg.norm(difference) / np.linalg.norm(u1)),
        "uniform_max_abs_difference": float(np.max(np.abs(difference))),
        "uniform_max_difference_location": {
            "t": float(times[maximum_index[0]]),
            "x": float(grid[maximum_index[1]]),
        },
        "uniform_gradient_relative_l2": float(
            np.linalg.norm(ux0 - ux1) / np.linalg.norm(ux1)
        ),
        "maximum_peak_gradient_relative_change": float(np.max(peak_change)),
        "references": [],
    }
    for degree, reference, peak in zip(DEGREES, references, peaks):
        # DCT-I gives Chebyshev coefficients after degree and endpoint normalization.
        coefficients = dct(reference["u"], type=1, axis=1) / degree
        coefficients[:, [0, -1]] *= 0.5
        tail_ratio = np.max(
            np.abs(coefficients[:, int(np.ceil(0.9 * degree)) :]), axis=1
        ) / np.max(np.abs(coefficients), axis=1)
        summary["references"].append(
            {
                "degree": degree,
                "peak_spatial_gradient": float(np.max(peak)),
                "peak_gradient_time": float(times[np.argmax(peak)]),
                "highest_10pct_modes_relative_max": float(np.max(tail_ratio)),
            }
        )

    # Keep numerical records separate from the interpretation in the campaign README.
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output / "time_series.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "t",
                "uniform_relative_l2",
                "uniform_max_abs_difference",
                "coarse_peak_gradient",
                "fine_peak_gradient",
                "peak_gradient_relative_change",
            ]
        )
        writer.writerows(
            zip(
                times,
                relative,
                np.max(np.abs(difference), axis=1),
                peaks[0],
                peaks[1],
                peak_change,
            )
        )

    # Show the central front before, near, and after its maximum steepening.
    indices = [int(np.argmin(np.abs(times - time))) for time in PROFILE_TIMES]
    fig, axes = plt.subplots(
        2,
        len(indices),
        figsize=(12, 6),
        sharex=True,
        sharey="row",
        layout="constrained",
    )
    for degree, reference, color, style in zip(
        DEGREES, references, ("#0072B2", "#D55E00"), ("-", "--")
    ):
        values, gradients = sample_reference(
            reference["x"], reference["u"][indices], FRONT_COORDINATES
        )
        for column, index in enumerate(indices):
            axes[0, column].plot(
                FRONT_COORDINATES,
                values[column],
                color=color,
                linestyle=style,
                label=f"N{degree}",
            )
            axes[1, column].plot(
                FRONT_COORDINATES, gradients[column], color=color, linestyle=style
            )
            axes[0, column].set_title(f"t = {times[index]:.3f}")
    for ax in axes.flat:
        ax.grid(alpha=0.25, linewidth=0.6)
    for ax in axes[1]:
        ax.set_xlabel("Position x")
    axes[0, 0].set_ylabel("Solution u")
    axes[1, 0].set_ylabel(r"Spatial gradient $u_x$")
    axes[0, -1].legend()
    fig.suptitle(r"Burgers front resolution at $\nu = 0.001$", weight="bold")
    fig.savefig(output / "front_profiles.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    axes[0].semilogy(times, 100 * np.maximum(relative, 1e-14), color="#0072B2")
    axes[0].set_ylabel("Solution relative L2 disagreement (%)")
    for degree, peak, color in zip(DEGREES, peaks, ("#0072B2", "#D55E00")):
        axes[1].plot(times, peak, label=f"N{degree}", color=color)
    axes[1].set_ylabel(r"Maximum $|u_x|$")
    axes[1].legend()
    for ax in axes:
        ax.set_xlabel("Physical time t")
        ax.grid(alpha=0.25, linewidth=0.6)
    fig.suptitle(r"Effect of spatial refinement at $\nu = 0.001$", weight="bold")
    fig.savefig(output / "resolution_over_time.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
