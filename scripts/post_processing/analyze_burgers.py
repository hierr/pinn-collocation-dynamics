"""Evaluate the final Burgers model and plot one saved run."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch
import yaml

from src.artifacts import write_json
from src.burgers import build_model, load_reference

# Import DeepXDE after src selects the PyTorch backend.
from deepxde import config as dde_config  # isort: skip

# Render plots without requiring an interactive display.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]


def predict_batches(model, coordinates, batch_size: int, operator=None):
    """Return predictions or PDE derivatives in bounded batches."""
    values = np.concatenate(
        [
            model.predict(coordinates[start : start + batch_size], operator=operator)
            for start in range(0, len(coordinates), batch_size)
        ]
    )
    if not np.isfinite(values).all():
        raise FloatingPointError("Non-finite final evaluation")
    return values


def solution_errors(reference, prediction):
    """Return ordinary solution MSE and relative L2 on one common grid."""
    reference_norm = np.sum(reference**2)
    if reference_norm <= 0:
        raise ValueError("Relative L2 is undefined for a zero reference norm")
    squared_error = np.sum((prediction - reference) ** 2)
    return {
        "final_mse": float(squared_error / reference.size),
        "final_relative_l2": float(np.sqrt(squared_error / reference_norm)),
    }


def residual_rms(model, problem, nx: int, nt: int, batch_size: int):
    """Return PDE-residual RMS on a fixed uniform interior midpoint grid."""
    xmin, xmax = problem["x_bounds"]
    tmin, tmax = problem["t_bounds"]
    x = xmin + (np.arange(nx) + 0.5) * (xmax - xmin) / nx
    times = tmin + (np.arange(nt) + 0.5) * (tmax - tmin) / nt
    xx, tt = np.meshgrid(x, times)
    coordinates = np.column_stack((xx.ravel(), tt.ravel()))
    values = predict_batches(model, coordinates, batch_size, operator=model.data.pde)
    return float(np.sqrt(np.mean(values**2)))


def evaluate_final(model, coordinates, reference, problem, settings):
    """Return final predictions and solution, PDE, boundary, and initial errors."""
    batch_size = settings["batch_size"]
    prediction = predict_batches(model, coordinates, batch_size)
    metrics = solution_errors(reference, prediction)

    # Evaluate the PDE on points independent of the training and reference grids.
    metrics["final_residual_rms"] = residual_rms(
        model, problem, settings["residual_nx"], settings["residual_nt"], batch_size
    )

    # Evaluate boundary and initial constraints on the same uniform solution grid.
    x = np.unique(coordinates[:, 0])
    times = np.unique(coordinates[:, 1])
    bx, bt = np.meshgrid(problem["x_bounds"], times)
    boundary = np.column_stack((bx.ravel(), bt.ravel()))
    boundary_values = predict_batches(model, boundary, batch_size).reshape(-1, 2)
    metrics["boundary_rmse"] = float(np.sqrt(np.mean(boundary_values**2)))
    initial = np.column_stack((x, np.full(len(x), problem["t_bounds"][0])))
    initial_values = predict_batches(model, initial, batch_size).ravel()
    metrics["initial_rmse"] = float(
        np.sqrt(np.mean((initial_values + np.sin(np.pi * x)) ** 2))
    )
    return prediction, metrics


def load_arrays(path: Path):
    """Load finite numeric arrays from a saved NumPy artifact."""
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    for name, values in arrays.items():
        if not np.isfinite(values).all():
            raise ValueError(f"{path}: {name} contains non-finite values")
    return arrays


def load_artifacts(run: Path):
    """Return saved predictions, history, and final interior points."""
    predictions = load_arrays(run / "predictions.npz")
    history = load_arrays(run / "history.npz")
    points = load_arrays(run / "points/final.npz")["interior"]
    # Check shapes needed for the convergence and point plots.
    steps = history["steps"]
    for name in ("loss_train", "loss_test"):
        values = history[name]
        if values.ndim != 2 or len(values) != len(steps):
            raise ValueError(f"History {name} must have one row per recorded step")
    if steps.ndim != 1:
        raise ValueError("History steps must be one-dimensional")
    if "metrics_test" in history and (
        history["metrics_test"].ndim != 2
        or history["metrics_test"].shape != (len(steps), 2)
    ):
        raise ValueError("Historical metrics must contain MSE/relative-L2 columns")
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Interior points must contain [x, t] coordinate pairs")
    return predictions, history, points


def reconstruct_grid(predictions):
    """Return x, times, reference, and prediction grids in saved coordinate order."""
    coordinates = predictions["coordinates"]
    reference = predictions["reference"]
    predicted = predictions["prediction"]
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or len(coordinates) == 0:
        raise ValueError("Evaluation coordinates must be a nonempty [x, t] array")
    if reference.shape != (len(coordinates), 1) or predicted.shape != reference.shape:
        raise ValueError("Reference and prediction must have one value per coordinate")

    # Infer dimensions and verify the time-major Cartesian ordering before reshaping.
    x = np.unique(coordinates[:, 0])
    times = np.unique(coordinates[:, 1])
    xx, tt = np.meshgrid(x, times)
    expected = np.column_stack((xx.ravel(), tt.ravel()))
    if not np.array_equal(coordinates, expected):
        raise ValueError("Evaluation coordinates must form a complete time-major grid")
    shape = (len(times), len(x))
    return x, times, reference.reshape(shape), predicted.reshape(shape)


def calculate_diagnostics(x, times, reference, predicted):
    """Return maximum absolute error and ordinary relative L2 at each saved time."""
    norms = np.linalg.norm(reference, axis=1)
    if np.any(norms == 0):
        raise ValueError("Per-time relative L2 is undefined for a zero reference norm")
    error = np.abs(predicted - reference)
    time_index, space_index = np.unravel_index(np.argmax(error), error.shape)
    return {
        "maximum_absolute_error": float(error[time_index, space_index]),
        "maximum_error_coordinate": [float(x[space_index]), float(times[time_index])],
        "times": times.tolist(),
        "per_time_relative_l2": (np.linalg.norm(error, axis=1) / norms).tolist(),
    }


def plot_diagnostics(path, x, times, reference, predicted, history, points, config):
    """Save convergence, final errors, interior points, and solution slices."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    # Show optimizer loss curves; old runs can also contain reference-error history.
    axes[0, 0].semilogy(
        history["steps"], np.sum(history["loss_train"], axis=1), label="Training loss"
    )
    if "metrics_test" in history:
        axes[0, 0].semilogy(
            history["steps"], history["metrics_test"][:, 1], label="Relative L2"
        )
    else:
        axes[0, 0].semilogy(
            history["steps"], np.sum(history["loss_test"], axis=1), label="Test loss"
        )
    axes[0, 0].set(xlabel="Global optimizer step", title="Recorded convergence")
    axes[0, 0].legend()

    # Locate final prediction errors using the saved evaluation coordinates.
    image = axes[0, 1].pcolormesh(
        x, times, np.abs(predicted - reference), shading="auto"
    )
    axes[0, 1].set(xlabel="x", ylabel="t", title="Final absolute error")
    fig.colorbar(image, ax=axes[0, 1])

    # Display the final interior population without boundary/initial constraints.
    axes[1, 0].scatter(points[:, 0], points[:, 1], s=2)
    axes[1, 0].set(xlabel="x", ylabel="t", title="Final interior collocation points")

    # Compare representative Burgers slices at the nearest saved evaluation times.
    for time in (0.25, 0.50, 0.75, 0.99):
        index = int(np.argmin(np.abs(times - time)))
        (line,) = axes[1, 1].plot(
            x, reference[index], label=f"t={times[index]:.2f} reference"
        )
        axes[1, 1].plot(x, predicted[index], "--", color=line.get_color())
    axes[1, 1].set(xlabel="x", ylabel="u", title="Final solution (prediction dashed)")
    axes[1, 1].legend(fontsize=7)
    fig.suptitle(
        f"{config['selected_method']}: Burgers, seed {config['selected_seed']}"
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def analyze_run(run: Path):
    """Evaluate missing final artifacts and regenerate diagnostics from a saved run."""
    config = yaml.safe_load((run / "config.yaml").read_text())
    predictions_path = run / "predictions.npz"
    metrics_path = run / "metrics.json"

    # Evaluate controlled runs from saved weights; retain completed evaluation artifacts.
    if "evaluation" in config and (
        not predictions_path.exists() or not metrics_path.exists()
    ):
        settings = config["evaluation"]
        coordinates, reference, fingerprint = load_reference(
            ROOT, config["problem"], settings["solution_nx"]
        )
        metadata = json.loads((run / "metadata.json").read_text())
        if fingerprint != metadata["dataset_sha256"]:
            raise ValueError("Reference dataset differs from the training fingerprint")
        dde_config.set_default_float("float64")
        model, _ = build_model(config["problem"], config["network"])
        model.compile("adam", lr=config["training"]["learning_rate"])
        model.net.load_state_dict(
            torch.load(
                run / "models/final.pt",
                map_location=torch.get_default_device(),
                weights_only=True,
            )
        )
        prediction, metrics = evaluate_final(
            model, coordinates, reference, config["problem"], settings
        )
        stages = json.loads((run / "training.json").read_text())["stages"]
        metrics["training_seconds"] = sum(stage["seconds"] for stage in stages)
        metrics["movement_events"] = sum(stage["movement_events"] for stage in stages)
        if not predictions_path.exists():
            np.savez_compressed(
                predictions_path,
                coordinates=coordinates,
                reference=reference,
                prediction=prediction,
            )
        if not metrics_path.exists():
            write_json(metrics_path, metrics)

    predictions, history, points = load_artifacts(run)
    x, times, reference, predicted = reconstruct_grid(predictions)
    diagnostics = calculate_diagnostics(x, times, reference, predicted)
    # Save derived diagnostics separately from the final metric summary.
    (run / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, allow_nan=False) + "\n"
    )
    plot_diagnostics(
        run / "diagnostics.png", x, times, reference, predicted, history, points, config
    )
    return diagnostics


def main():
    """Analyze the run directory selected on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    diagnostics = analyze_run(args.run_dir)
    print(f"Diagnostics saved in {args.run_dir}")
    print(f"Maximum absolute error: {diagnostics['maximum_absolute_error']:.8g}")


if __name__ == "__main__":
    main()
