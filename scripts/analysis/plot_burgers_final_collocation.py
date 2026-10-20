"""Compare five saved final point populations over the same Burgers reference."""

from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "output/burgers/burgers_nu_0p0031830989_10seed"
# Use the first planned evaluation seed rather than selecting by final error.
SEED = 4000
OUTPUT_PATH = ROOT / "docs/figures/burgers_final_collocation.png"
METHODS = [
    ("fixed_hammersley", "Fixed Hammersley"),
    ("random_resampling", "Random resampling"),
    ("rad", "RAD"),
    ("pacmann_adam", "PACMANN"),
    ("pacmann_diffusion", "PACMANN + diffusion"),
]


def main():
    """Plot final interior points from matched runs without training or inference."""
    first_run = CAMPAIGN / METHODS[0][0] / f"seed_{SEED}"
    config = yaml.safe_load((first_run / "config.yaml").read_text())
    problem = config["problem"]
    xmin, xmax = problem["x_bounds"]
    tmin, tmax = problem["t_bounds"]
    with np.load(first_run / "points/initial.npz", allow_pickle=False) as saved:
        shared_initial = saved["interior"]

    # Reconstruct the common reference field in its archived coordinate order.
    with np.load(first_run / "predictions.npz", allow_pickle=False) as saved:
        coordinates, reference = saved["coordinates"], saved["reference"]
    x, times = np.unique(coordinates[:, 0]), np.unique(coordinates[:, 1])
    xx, tt = np.meshgrid(x, times)
    if not np.array_equal(coordinates, np.column_stack((xx.ravel(), tt.ravel()))):
        raise ValueError("Reference coordinates must form a time-major grid")
    reference = reference.reshape(len(times), len(x))

    # Center the two gradient-adaptive panels beneath the three sampling baselines.
    fig = plt.figure(figsize=(14, 7))
    grid = fig.add_gridspec(
        2,
        6,
        left=0.055,
        right=0.91,
        bottom=0.08,
        top=0.92,
        wspace=0.65,
        hspace=0.40,
    )
    axes = [
        fig.add_subplot(grid[0, 0:2]),
        fig.add_subplot(grid[0, 2:4]),
        fig.add_subplot(grid[0, 4:6]),
        fig.add_subplot(grid[1, 1:3]),
        fig.add_subplot(grid[1, 3:5]),
    ]
    for ax, (method, label) in zip(axes, METHODS):
        run = CAMPAIGN / method / f"seed_{SEED}"
        config = yaml.safe_load((run / "config.yaml").read_text())
        if (
            config["problem"] != problem
            or config["selected_method"] != method
            or config["selected_seed"] != SEED
        ):
            raise ValueError(f"Matched problem, method, or seed differs: {run}")
        with np.load(run / "points/initial.npz", allow_pickle=False) as saved:
            initial = saved["interior"]
        with np.load(run / "points/final.npz", allow_pickle=False) as saved:
            final = saved["interior"]
        if (
            initial.shape != (problem["interior_points"], 2)
            or not np.array_equal(initial, shared_initial)
            or final.shape != initial.shape
            or not np.isfinite(final).all()
            or np.any(final < [xmin, tmin])
            or np.any(final > [xmax, tmax])
        ):
            raise ValueError(f"Point budget, initial design, or domain differs: {run}")
        if method == "fixed_hammersley" and not np.array_equal(final, initial):
            raise ValueError("Fixed Hammersley must retain the initial points")

        # Hold the solution background and color scale fixed across all five panels.
        solution = ax.pcolormesh(
            x,
            times,
            reference,
            shading="auto",
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
        )
        ax.scatter(final[:, 0], final[:, 1], s=2.5, color="black", alpha=0.75, lw=0)
        ax.set(
            xlabel="x",
            ylabel="t",
            xlim=(xmin, xmax),
            ylim=(tmin, tmax),
            xticks=np.linspace(xmin, xmax, 5),
        )
        ax.set_title(label, fontsize=11, weight="semibold")
        ax.set_axisbelow("line")
        ax.grid(True, linewidth=0.4, color="0.5", alpha=0.35)

    colorbar_axis = fig.add_axes((0.935, 0.20, 0.013, 0.60))
    fig.colorbar(solution, cax=colorbar_axis, label="Reference u(x, t)")
    fig.suptitle("Final interior collocation points", fontsize=12, weight="semibold")
    fig.savefig(OUTPUT_PATH, dpi=200)
    plt.close(fig)
    print(f"Saved {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
