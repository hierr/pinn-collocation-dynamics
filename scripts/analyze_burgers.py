"""Calculate error diagnostics and plot one saved Burgers run."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import yaml

# Render plots without requiring an interactive display.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_arrays(path: Path):
    """Load finite numeric arrays from a saved NumPy artifact."""
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    for name, values in arrays.items():
        if not np.isfinite(values).all():
            raise ValueError(f"{path}: {name} contains non-finite values")
    return arrays


def load_artifacts(run: Path):
    """Return saved predictions, history, final interior points, and config."""
    predictions = load_arrays(run / "predictions.npz")
    history = load_arrays(run / "history.npz")
    points = load_arrays(run / "points/final.npz")["interior"]
    config = yaml.safe_load((run / "config.yaml").read_text())
    # Check shapes needed for the convergence and point plots.
    steps = history["steps"]
    for name in ("loss_train", "metrics_test"):
        values = history[name]
        if values.ndim != 2 or len(values) != len(steps):
            raise ValueError(f"History {name} must have one row per recorded step")
    if steps.ndim != 1 or history["metrics_test"].shape[1] < 2:
        raise ValueError("History must contain steps and MSE/relative-L2 columns")
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Interior points must contain [x, t] coordinate pairs")
    return predictions, history, points, config


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
    """Return maximum absolute error and relative L2 error at each saved time."""
    # Normalize each time slice by its own reference solution norm.
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
    # Show the recorded training objective alongside reference prediction error.
    axes[0, 0].semilogy(
        history["steps"], np.sum(history["loss_train"], axis=1), label="Training loss"
    )
    axes[0, 0].semilogy(
        history["steps"], history["metrics_test"][:, 1], label="Relative L2"
    )
    axes[0, 0].set(xlabel="Global optimizer step", title="Recorded convergence")
    axes[0, 0].legend()

    # Locate final prediction errors on the original reference grid.
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
    """Regenerate derived diagnostics without changing saved training artifacts."""
    predictions, history, points, config = load_artifacts(run)
    x, times, reference, predicted = reconstruct_grid(predictions)
    diagnostics = calculate_diagnostics(x, times, reference, predicted)
    # Save additional error analysis separately from the trainer's metric summary.
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
