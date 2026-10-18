"""Plot the saved Burgers reference beneath the initial interior collocation points."""

from pathlib import Path

import matplotlib
import numpy as np

# Render the documentation figure without an interactive display.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = (
    ROOT / "output/burgers/nu_0p0031830989/sigma_0/pacmann_adam/seed_3000"
)
OUTPUT_PATH = ROOT / "docs/burgers_initial_collocation.png"


def main():
    """Save the reference solution with the selected run's initial Hammersley points."""
    # Read the common reference grid and the initial interior population.
    with np.load(RUN_DIR / "predictions.npz", allow_pickle=False) as data:
        x = np.unique(data["coordinates"][:, 0])
        times = np.unique(data["coordinates"][:, 1])
        reference = data["reference"].reshape(len(times), len(x))
    with np.load(RUN_DIR / "points/initial.npz", allow_pickle=False) as data:
        points = data["interior"]

    # Overlay the initial points on the space-time reference solution.
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    solution = ax.pcolormesh(
        x, times, reference, shading="auto", cmap="RdBu_r", vmin=-1, vmax=1, zorder=0
    )
    ax.scatter(points[:, 0], points[:, 1], s=5, color="black", zorder=3)
    fig.colorbar(solution, ax=ax, label="Reference u(x, t)")
    ax.set(
        xlabel="x", ylabel="t", title="Burgers solution and initial Hammersley points"
    )
    ax.set_axisbelow("line")
    ax.grid(True, linewidth=0.4, color="0.5", alpha=0.35)
    fig.savefig(OUTPUT_PATH, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
