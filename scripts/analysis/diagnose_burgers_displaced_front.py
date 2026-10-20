"""Plot displaced fronts in two Burgers runs at viscosity 0.01/pi."""

from pathlib import Path

import matplotlib
import numpy as np
import torch
import yaml

from scripts.post_processing.analyze_burgers import (
    predict_batches,
    reconstruct_grid,
    solution_errors,
)
from src.burgers import build_model

# Import DeepXDE after src selects the PyTorch backend.
from deepxde import config as dde_config  # isort: skip

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "output/burgers/burgers_nu_0p0031830989_10seed"
CASES = [
    ("pacmann_adam", 4002, "PACMANN"),
    ("pacmann_diffusion", 4004, "PACMANN + diffusion"),
]
STAGES = [
    ("phase_02_l-bfgs", "Phase 2: L-BFGS"),
    ("phase_03_adam", "Phase 3: Adam"),
    ("phase_03_l-bfgs", "Phase 3: L-BFGS"),
]


def main():
    """Restore six checkpoints and save residual maps and solution slices."""
    # Evaluate saved weights on CPU without training.
    torch.set_num_threads(1)
    dde_config.set_default_float("float64")
    torch.set_default_device("cpu")
    plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.5})
    output = CAMPAIGN / "diagnostics"
    output.mkdir(exist_ok=True)

    for method, seed, title in CASES:
        run = CAMPAIGN / method / f"seed_{seed}"
        config = yaml.safe_load((run / "config.yaml").read_text())
        problem, settings = config["problem"], config["evaluation"]
        model, _ = build_model(problem, config["network"])
        # Initialize DeepXDE prediction functions; no optimizer steps are taken.
        model.compile("adam", lr=config["training"]["learning_rate"])
        with np.load(run / "predictions.npz", allow_pickle=False) as saved:
            coordinates, reference = saved["coordinates"], saved["reference"]
            x, times, reference_grid, _ = reconstruct_grid(saved)

        # Reuse the campaign's independent uniform residual midpoint grid.
        xmin, xmax = problem["x_bounds"]
        tmin, tmax = problem["t_bounds"]
        nx, nt = settings["residual_nx"], settings["residual_nt"]
        rx = xmin + (np.arange(nx) + 0.5) * (xmax - xmin) / nx
        rt = tmin + (np.arange(nt) + 0.5) * (tmax - tmin) / nt
        xx, tt = np.meshgrid(rx, rt)
        residual_coordinates = np.column_stack((xx.ravel(), tt.ravel()))

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for column, (stage, caption) in enumerate(STAGES):
            model.net.load_state_dict(
                torch.load(
                    run / "models" / f"{stage}.pt",
                    map_location="cpu",
                    weights_only=True,
                )
            )
            prediction = predict_batches(model, coordinates, settings["batch_size"])
            residual = predict_batches(
                model,
                residual_coordinates,
                settings["batch_size"],
                operator=model.data.pde,
            )
            l2 = solution_errors(reference, prediction)["final_relative_l2"]
            residual_rms = np.sqrt(np.mean(residual**2))
            with np.load(run / "points" / f"{stage}.npz", allow_pickle=False) as saved:
                points = saved["pde"]

            # Overlay the checkpoint's PDE population on its absolute residual field.
            ax = axes[0, column]
            image = ax.pcolormesh(
                rx,
                rt,
                np.abs(residual).reshape(nt, nx),
                shading="nearest",
                norm=LogNorm(1e-4, 100),
                cmap="magma",
                rasterized=True,
            )
            ax.scatter(points[:, 0], points[:, 1], c="black", s=2, alpha=0.5, lw=0)
            ax.set(
                xlabel="x",
                ylabel="t",
                title=caption,
                xlim=(xmin, xmax),
                ylim=(tmin, tmax),
            )
            ax.grid(alpha=0.3, color="white")
            ax.text(
                0.03,
                0.97,
                f"L2: {100 * l2:.3g}%\nGrid residual: {residual_rms:.3g}",
                transform=ax.transAxes,
                va="top",
                color="white",
                bbox={"facecolor": "black", "alpha": 0.65, "edgecolor": "none"},
            )

            # Draw predictions above references to expose differences at the front.
            prediction_grid = prediction.reshape(reference_grid.shape)
            for target, color in zip(
                [0.5, 0.75, 0.99], ["#0072b2", "#009e73", "#d55e00"]
            ):
                index = int(np.argmin(abs(times - target)))
                axes[1, column].plot(
                    x,
                    reference_grid[index],
                    color=color,
                    lw=1.4,
                    label=f"Reference t={times[index]:.3g}",
                )
                axes[1, column].plot(
                    x,
                    prediction_grid[index],
                    color=color,
                    lw=1.7,
                    ls="--",
                    zorder=5,
                    label=f"Prediction t={times[index]:.3g}",
                )
            axes[1, column].set(
                xlabel="x",
                ylabel="u(x,t)",
                xlim=(-0.5, 0.5),
                title="Solution slices (central region)",
            )

        fig.colorbar(
            image,
            cax=fig.add_axes([0.94, 0.55, 0.012, 0.34]),
            label="Absolute PDE residual; black dots = saved PDE points",
        )
        fig.legend(
            *axes[1, 0].get_legend_handles_labels(),
            loc="lower center",
            ncol=3,
            frameon=False,
        )
        fig.suptitle(f"{title}, seed {seed}: good phase 2 solution is lost in phase 3")
        fig.subplots_adjust(
            left=0.06, right=0.92, bottom=0.14, top=0.91, wspace=0.25, hspace=0.28
        )
        fig.savefig(output / f"{method}_{seed}_transition.png", dpi=170)
        plt.close(fig)


if __name__ == "__main__":
    main()
