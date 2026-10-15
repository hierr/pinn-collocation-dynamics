"""Compare Burgers references using three existing PACMANN models, without training."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.interpolate import BarycentricInterpolator

from src.burgers import FNN

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/burgers_dataset"
RUNS = ROOT / "output/burgers_sigma_tuning_01_baseline/burgers/pacmann_adam"
OUTPUT = DATA / "comparison/pacmann_3seed"
SEEDS = [2000, 2001, 2002]


def errors(prediction, reference):
    """Return global relative L2, MSE, and maximum absolute error."""
    difference = prediction - reference
    return {
        "relative_l2": float(np.linalg.norm(difference) / np.linalg.norm(reference)),
        "mse": float(np.mean(difference**2)),
        "maximum_absolute_error": float(np.max(np.abs(difference))),
    }


def main():
    """Evaluate the saved references and models and write two tables and a figure."""
    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float64)
    print("Evaluation device: CPU; precision: float64; no training.")

    # Load the original reference, whose spatial grid and times define the comparison.
    original_path = DATA / "PACMANN_Burgers.npz"
    with np.load(original_path, allow_pickle=False) as archive:
        x = archive["x"].ravel()
        times = archive["t"].ravel()
        original = archive["usol"].T
    references = {"original": {"x": x, "native": original, "common": original}}
    print(f"Original SHA256: {hashlib.sha256(original_path.read_bytes()).hexdigest()}")

    # Select exact shared snapshot times and interpolate Chebyshev solutions in space.
    for resolution in (512, 1024):
        path = DATA / f"burgers_nu_0p0031830989_N{resolution}.npz"
        with np.load(path, allow_pickle=False) as archive:
            nodes, snapshots = archive["x"].ravel(), archive["t"].ravel()
            solution = archive["U"]
            settings = json.loads(str(archive["config_json"]))
        indices = np.argmin(np.abs(snapshots[:, None] - times[None, :]), axis=0)
        if not np.array_equal(snapshots[indices], times):
            raise ValueError("Generated reference lacks exact original snapshot times")
        if solution.shape != (len(snapshots), len(nodes)):
            raise ValueError("Generated solution shape does not match its coordinates")
        if not np.isclose(settings["nu"], 0.01 / np.pi, rtol=1e-12, atol=0):
            raise ValueError("Generated reference viscosity differs from the benchmark")
        expected_nodes = np.cos(np.pi * np.arange(resolution + 1) / resolution)
        if not np.allclose(nodes, expected_nodes, rtol=0, atol=1e-14):
            raise ValueError("Interpolation requires the recorded Chebyshev nodes")
        native = solution[indices]
        weights = (-1.0) ** np.arange(len(nodes))
        weights[[0, -1]] *= 0.5
        common = BarycentricInterpolator(nodes, native, axis=1, wi=weights)(x)
        references[f"N{resolution}"] = {"x": nodes, "native": native, "common": common}
        print(f"N{resolution} SHA256: {hashlib.sha256(path.read_bytes()).hexdigest()}")

    # Compare reference solutions on identical coordinates, retaining per-time errors.
    reference_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for candidate, baseline in (
        ("N512", "original"),
        ("N1024", "original"),
        ("N512", "N1024"),
    ):
        values = references[candidate]["common"]
        truth = references[baseline]["common"]
        reference_rows.append(
            {"candidate": candidate, "reference": baseline, **errors(values, truth)}
        )
        per_time = np.linalg.norm(values - truth, axis=1) / np.linalg.norm(
            truth, axis=1
        )
        axes[0].semilogy(
            times, 100 * np.maximum(per_time, 1e-14), label=f"{candidate} vs {baseline}"
        )
    axes[0].set(
        xlabel="Time",
        ylabel="Relative L2 (%)",
        title="Reference differences on original grid",
    )
    axes[0].legend()

    # Restore each archived network; first verify its original-grid predictions.
    model_rows = []
    for seed in SEEDS:
        run = RUNS / f"seed_{seed}"
        config = yaml.safe_load((run / "config.yaml").read_text())
        if not np.isclose(
            config["problem"]["viscosity"], 0.01 / np.pi, rtol=1e-12, atol=0
        ):
            raise ValueError("Saved model viscosity differs from the reference")
        network = FNN(config["network"]["layers"], "tanh", "Glorot normal").double()
        weights_path = run / "models/final.pt"
        network.load_state_dict(
            torch.load(weights_path, map_location="cpu", weights_only=True)
        )
        network.eval()
        print(
            f"Seed {seed} weights SHA256: {hashlib.sha256(weights_path.read_bytes()).hexdigest()}"
        )
        with np.load(run / "predictions.npz", allow_pickle=False) as saved:
            xx, tt = np.meshgrid(x, times)
            coordinates = np.column_stack((xx.ravel(), tt.ravel()))
            if not np.array_equal(
                saved["coordinates"], coordinates
            ) or not np.array_equal(
                saved["reference"].reshape(original.shape), original
            ):
                raise ValueError(
                    "Archived predictions use a different original reference grid"
                )
            with torch.no_grad():
                prediction = (
                    network(torch.from_numpy(coordinates))
                    .numpy()
                    .reshape(original.shape)
                )
            archived = saved["prediction"].reshape(original.shape)
            if not np.allclose(prediction, archived, rtol=1e-10, atol=1e-12):
                raise ValueError("Restored network differs from archived predictions")
            print(
                f"Seed {seed} reload maximum difference: {np.max(np.abs(prediction - archived)):.3e}"
            )

        # Evaluate native generated grids directly and common-grid references separately.
        for name, reference in references.items():
            if name == "original":
                model_rows.append(
                    {
                        "seed": seed,
                        "dataset": name,
                        "grid": "original",
                        **errors(prediction, original),
                    }
                )
                continue
            xx, tt = np.meshgrid(reference["x"], times)
            coordinates = np.column_stack((xx.ravel(), tt.ravel()))
            with torch.no_grad():
                native_prediction = (
                    network(torch.from_numpy(coordinates))
                    .numpy()
                    .reshape(reference["native"].shape)
                )
            model_rows.append(
                {
                    "seed": seed,
                    "dataset": name,
                    "grid": "native",
                    **errors(native_prediction, reference["native"]),
                }
            )
            model_rows.append(
                {
                    "seed": seed,
                    "dataset": name,
                    "grid": "original",
                    **errors(prediction, reference["common"]),
                }
            )

    # Summarize only the shared-grid model errors; native-grid scores remain distinct.
    names = list(references)
    for seed in SEEDS:
        values = [
            100 * row["relative_l2"]
            for row in model_rows
            if row["seed"] == seed and row["grid"] == "original"
        ]
        axes[1].plot(names, values, "o-", label=f"Seed {seed}")
    axes[1].set(
        ylabel="Final relative L2 (%)",
        title="Same PACMANN predictions, different references",
    )
    axes[1].legend()

    # Save a compact review package in a new directory, leaving source runs untouched.
    OUTPUT.mkdir(parents=True, exist_ok=False)
    for filename, rows in (
        ("reference_errors.csv", reference_rows),
        ("model_errors.csv", model_rows),
    ):
        with (OUTPUT / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    fig.savefig(OUTPUT / "comparison.png", dpi=160)
    plt.close(fig)
    print(f"Review artifacts: {OUTPUT}")
    for row in reference_rows + model_rows:
        print(row)


if __name__ == "__main__":
    main()
