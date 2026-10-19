"""Describe spatial steepening in the three selected Burgers references."""

from pathlib import Path

import matplotlib
import numpy as np
import yaml
from numpy.polynomial.chebyshev import chebder
from scipy.fft import dct

# Render the documentation figure without an interactive display.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATHS = [
    ROOT / "config/burgers_nu_0p1_10seed.yaml",
    ROOT / "config/burgers_nu_0p01_10seed.yaml",
    ROOT / "config/burgers_nu_0p0031830989_10seed.yaml",
]
OUTPUT_PATH = ROOT / "docs/burgers_reference_regimes.png"
SLICES_OUTPUT_PATH = ROOT / "docs/burgers_reference_slices.png"
SLICE_TIMES = (0.25, 0.50, 0.75, 0.99)


def spatial_derivative(x, solution):
    """Return each snapshot's spatial derivative at descending Lobatto nodes."""
    if solution.ndim != 2 or solution.shape[1] != len(x):
        raise ValueError("Solution columns must match spatial nodes")
    degree = len(x) - 1
    width = x[0] - x[-1]
    expected = (x[0] + x[-1]) / 2 + width / 2 * np.cos(
        np.pi * np.arange(degree + 1) / degree
    )
    if width <= 0 or not np.allclose(x, expected, rtol=0, atol=1e-12):
        raise ValueError("Spatial nodes must be descending Chebyshev-Lobatto nodes")

    # Convert native values to Chebyshev coefficients and differentiate in x.
    coefficients = dct(solution, type=1, axis=1) / degree
    coefficients[:, [0, -1]] *= 0.5
    derivative = chebder(coefficients, axis=1) * (2 / width)

    # Evaluate the derivative polynomial on the same nodes using DCT-I scaling.
    reconstruction = np.pad(derivative, ((0, 0), (0, 1)))
    reconstruction[:, 1:-1] *= 0.5
    return dct(reconstruction, type=1, axis=1)


def characterize_reference(problem):
    """Return native coordinates, solution values, maximum gradients, and amplitudes."""
    with np.load(ROOT / problem["reference_path"], allow_pickle=False) as data:
        x, times, solution = data["x"].ravel(), data["t"].ravel(), data["U"]
        viscosity = float(data["nu"])
    if not np.isclose(viscosity, problem["viscosity"], rtol=1e-12, atol=0):
        raise ValueError("Reference viscosity differs from the configured PDE")
    if solution.shape != (len(times), len(x)) or not all(
        np.isfinite(array).all() for array in (x, times, solution)
    ):
        raise ValueError("Reference shape or values do not match its coordinates")

    # Order spatial nodes and retain exact snapshots in the campaign time window.
    order = np.argsort(x)[::-1]
    x, solution = x[order], solution[:, order]
    if not np.allclose(x[[-1, 0]], problem["x_bounds"], rtol=0, atol=1e-12):
        raise ValueError("Reference spatial endpoints differ from the PDE domain")
    tmin, tmax = problem["t_bounds"]
    selected = (times >= tmin) & (times <= tmax)
    times, solution = times[selected], solution[selected]
    if (
        len(times) < 2
        or not np.allclose(times[[0, -1]], [tmin, tmax], rtol=0, atol=1e-12)
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError("Reference times must span the configured time window")

    # Evaluate maximum spatial gradients and amplitudes at each saved time.
    gradients = np.max(np.abs(spatial_derivative(x, solution)), axis=1)
    amplitudes = np.max(np.abs(solution), axis=1)
    return x, times, solution, gradients, amplitudes


def main():
    """Print reference descriptors and save gradient and solution-slice comparisons."""
    configs = [yaml.safe_load(path.read_text()) for path in CONFIG_PATHS]
    common = configs[0]["problem"]
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    slices_fig, slices_axes = plt.subplots(
        1,
        len(configs),
        figsize=(4 * len(configs), 4),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    colors = ["#0072b2", "#009e73", "#d55e00"]

    # Describe each regime on the same domain, independently of training results.
    print("nu,peak_gradient,time_at_peak,amplitude_at_peak")
    for config, color, slice_ax in zip(configs, colors, slices_axes.flat):
        problem = config["problem"]
        if any(problem[key] != common[key] for key in ("x_bounds", "t_bounds")):
            raise ValueError("Reference characterization requires a common domain")
        x, times, solution, gradients, amplitudes = characterize_reference(problem)
        peak = int(np.argmax(gradients))
        nu = problem["viscosity"]
        print(
            f"{nu:.10g},{gradients[peak]:.6f},{times[peak]:.3f},{amplitudes[peak]:.6f}"
        )
        ax.plot(times, gradients, color=color, linewidth=1.8, label=f"ν = {nu:.6g}")
        ax.scatter(times[peak], gradients[peak], color=color, s=24, zorder=3)

        # Compare the same saved reference times in every viscosity panel.
        for time in SLICE_TIMES:
            index = int(np.argmin(np.abs(times - time)))
            slice_ax.plot(
                x, solution[index], linewidth=1.6, label=f"t = {times[index]:.2f}"
            )
        slice_ax.set(
            xlabel="x",
            title=f"ν = {nu:.6g}",
            xlim=problem["x_bounds"],
            xticks=[-1, -0.5, 0, 0.5, 1],
            ylim=(-1.05, 1.05),
        )
        slice_ax.grid(True, linewidth=0.4, alpha=0.35)

    # The shared initial condition -sin(pi*x) has maximum gradient pi.
    ax.axhline(
        np.pi, color="black", linestyle="--", linewidth=1, label="Initial gradient = π"
    )
    ax.set(
        xlabel="Time t",
        ylabel=r"Maximum spatial gradient $g_\nu(t)$",
        title="Spatial steepening in the Burgers reference solutions",
        xlim=common["t_bounds"],
        yscale="log",
    )
    ax.grid(True, which="major", linewidth=0.4, alpha=0.35)
    ax.legend(frameon=False)
    fig.savefig(OUTPUT_PATH, dpi=200)
    plt.close(fig)
    print(f"Saved {OUTPUT_PATH.relative_to(ROOT)}")

    # Share the slice colors, legend, and axis scales across the three regimes.
    slices_axes[0, 0].set_ylabel("u(x, t)")
    slices_fig.suptitle("Burgers reference solutions")
    slices_fig.legend(
        *slices_axes[0, 0].get_legend_handles_labels(),
        loc="lower center",
        ncol=len(SLICE_TIMES),
        frameon=False,
    )
    slices_fig.subplots_adjust(
        bottom=0.23, top=0.81, left=0.065, right=0.99, wspace=0.12
    )
    slices_fig.savefig(SLICES_OUTPUT_PATH, dpi=200)
    plt.close(slices_fig)
    print(f"Saved {SLICES_OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
