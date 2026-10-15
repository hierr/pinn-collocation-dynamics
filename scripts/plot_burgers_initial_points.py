"""Plot the configured Burgers reference beneath initial Hammersley points."""

from pathlib import Path

import matplotlib
import numpy as np
import yaml

from src.burgers import build_model, load_reference

# Import DeepXDE after src selects the PyTorch backend.
from deepxde import config as dde_config  # isort: skip

# Render the documentation figure without an interactive display.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/burgers_template.yaml"
SEED = 3000
OUTPUT_PATH = ROOT / "docs/burgers_initial_collocation.png"


def main():
    """Save the reference solution and initial points without training a model."""
    # Load the common reference grid from the general experiment template.
    config = yaml.safe_load(CONFIG_PATH.read_text())
    coordinates, values, _ = load_reference(
        ROOT, config["problem"], config["evaluation"]["solution_nx"]
    )
    x = np.unique(coordinates[:, 0])
    times = np.unique(coordinates[:, 1])
    reference = values.reshape(len(times), len(x))

    # Construct the same initial physics population used by the trainer.
    dde_config.set_random_seed(SEED)
    dde_config.set_default_float("float64")
    model, _ = build_model(config["problem"], config["network"])
    points = model.data.train_x_all

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
