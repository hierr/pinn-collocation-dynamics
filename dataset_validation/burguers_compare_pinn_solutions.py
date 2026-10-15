"""Compare saved PACMANN predictions against three references on the original grid."""

import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.interpolate import BarycentricInterpolator

# Select the backend before importing DeepXDE.
os.environ["DDE_BACKEND"] = "pytorch"
from deepxde.nn.pytorch import FNN

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/burgers_dataset"
OUTPUT = DATA / "reference_comparison/pacmann_3seed"
RUNS = OUTPUT / "source_models"
SEEDS = [2000, 2001, 2002]


def main():
    """Restore saved models and write common-grid relative L2 errors and one plot."""
    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float64)

    # Keep the original reference grid and values unchanged.
    with np.load(DATA / "PACMANN_Burgers.npz", allow_pickle=False) as archive:
        x, times = archive["x"].ravel(), archive["t"].ravel()
        original = archive["usol"].T
    references = {"original": original}

    # Interpolate generated Chebyshev solutions onto that grid at exact shared times.
    for resolution in (512, 1024):
        path = DATA / f"burgers_nu_0p0031830989_N{resolution}.npz"
        with np.load(path, allow_pickle=False) as archive:
            nodes, snapshots = archive["x"].ravel(), archive["t"].ravel()
            solution, viscosity = archive["U"], float(archive["nu"])
        indices = np.argmin(np.abs(snapshots[:, None] - times[None, :]), axis=0)
        if not np.array_equal(snapshots[indices], times):
            raise ValueError("Generated reference lacks exact original snapshot times")
        if solution.shape != (len(snapshots), len(nodes)):
            raise ValueError("Generated solution shape does not match its coordinates")
        if not np.isclose(viscosity, 0.01 / np.pi, rtol=1e-12, atol=0):
            raise ValueError("Generated reference viscosity differs from the benchmark")
        expected_nodes = np.cos(np.pi * np.arange(resolution + 1) / resolution)
        if not np.allclose(nodes, expected_nodes, rtol=0, atol=1e-14):
            raise ValueError("Interpolation requires Chebyshev nodes")
        weights = (-1.0) ** np.arange(len(nodes))
        weights[[0, -1]] *= 0.5
        references[f"N{resolution}"] = BarycentricInterpolator(
            nodes, solution[indices], axis=1, wi=weights
        )(x)

    # Reload each fixed model and verify inference against its archived predictions.
    xx, tt = np.meshgrid(x, times)
    coordinates = np.column_stack((xx.ravel(), tt.ravel()))
    rows = []
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for seed in SEEDS:
        run = RUNS / f"seed_{seed}"
        config = yaml.safe_load((run / "config.yaml").read_text())
        if not np.isclose(
            config["problem"]["viscosity"], 0.01 / np.pi, rtol=1e-12, atol=0
        ):
            raise ValueError("Saved model viscosity differs from the reference")
        network = FNN(config["network"]["layers"], "tanh", "Glorot normal").double()
        network.load_state_dict(
            torch.load(run / "models/final.pt", map_location="cpu", weights_only=True)
        )
        network.eval()
        with torch.no_grad():
            prediction = (
                network(torch.from_numpy(coordinates)).numpy().reshape(original.shape)
            )
        with np.load(run / "predictions.npz", allow_pickle=False) as saved:
            if not np.array_equal(
                saved["coordinates"], coordinates
            ) or not np.array_equal(
                saved["reference"].reshape(original.shape), original
            ):
                raise ValueError("Archived predictions use a different original grid")
            if not np.allclose(
                prediction,
                saved["prediction"].reshape(original.shape),
                rtol=1e-10,
                atol=1e-12,
            ):
                raise ValueError("Restored network differs from archived predictions")

        # Compare the same prediction array against each reference on identical points.
        errors = []
        for name, reference in references.items():
            relative_l2 = float(
                np.linalg.norm(prediction - reference) / np.linalg.norm(reference)
            )
            rows.append({"seed": seed, "dataset": name, "relative_l2": relative_l2})
            errors.append(100 * relative_l2)
        ax.plot(list(references), errors, "o-", label=f"Seed {seed}")
    ax.set(
        xlabel="Reference dataset",
        ylabel="Final relative L2 (%)",
        title="Fixed PACMANN models evaluated on the original grid",
    )
    ax.legend()

    # Regenerate only the derived model-error table and figure.
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "model_errors.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    fig.savefig(OUTPUT / "comparison.png", dpi=160)
    plt.close(fig)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
