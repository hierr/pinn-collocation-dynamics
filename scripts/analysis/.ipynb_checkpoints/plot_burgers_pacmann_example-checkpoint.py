"""Plot one successful PACMANN run using its saved final predictions."""

import json
from pathlib import Path

import matplotlib
import numpy as np

from scripts.post_processing.analyze_burgers import (
    load_arrays,
    reconstruct_grid,
    solution_errors,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
# The first planned seed is a successful illustration, not a method aggregate.
RUN = ROOT / "output/burgers/burgers_nu_0p0031830989_10seed/pacmann_adam/seed_4000"
OUTPUT_PATH = ROOT / "docs/figures/burgers_pacmann_example.png"


def main():
    """Save reference, learned solution, absolute error, and solution profiles."""
    # Reuse saved evaluation values without restoring or retraining the model.
    predictions = load_arrays(RUN / "predictions.npz")
    x, times, reference, learned = reconstruct_grid(predictions)
    metrics = json.loads((RUN / "metrics.json").read_text())
    relative_l2 = solution_errors(reference, learned)["final_relative_l2"]
    if not np.isclose(relative_l2, metrics["final_relative_l2"], rtol=1e-12, atol=0):
        raise ValueError("Saved predictions disagree with the archived solution score")

    plt.rcParams.update(
        {"font.size": 12, "axes.titlesize": 13, "axes.titleweight": "bold"}
    )
    fig, axes = plt.subplots(2, 2, figsize=(14, 7), constrained_layout=True)

    # Compare space-time fields using identical solution color scales.
    for ax, values, title in (
        (axes[0, 0], reference, "Reference solution"),
        (axes[0, 1], learned, "Learned solution"),
    ):
        field = ax.pcolormesh(
            x, times, values, shading="auto", cmap="RdBu_r", vmin=-1, vmax=1
        )
        ax.set(xlabel="x", ylabel="t", title=title, ylim=(times[0], times[-1]))
        fig.colorbar(field, ax=ax, label="u(x, t)")

    # Locate the final prediction errors on the same saved evaluation grid.
    error = axes[1, 0].pcolormesh(
        x, times, np.abs(learned - reference), shading="auto", cmap="viridis"
    )
    axes[1, 0].set(
        xlabel="x",
        ylabel="t",
        title="Final absolute error",
        ylim=(times[0], times[-1]),
    )
    fig.colorbar(error, ax=axes[1, 0])

    # Draw the diagnostic's four solution profiles with predictions on top.
    for time in (0.25, 0.50, 0.75, 0.99):
        index = int(np.argmin(np.abs(times - time)))
        (line,) = axes[1, 1].plot(
            x,
            reference[index],
            label=f"t = {times[index]:.2f}",
            alpha=0.55,
            linewidth=1.8,
            zorder=2,
        )
        axes[1, 1].plot(
            x, learned[index], "--", color=line.get_color(), linewidth=1.4, zorder=3
        )
    axes[1, 1].set(
        xlabel="x",
        ylabel="u",
        title="Final solution (prediction dashed)",
        ylim=(-1.05, 1.05),
    )
    axes[1, 1].legend(fontsize=10, ncol=2)

    # Match the thin diagnostic grids without adding collocation overlays.
    for ax in axes.flat:
        ax.set_xlim(x[0], x[-1])
        ax.set_xticks(np.linspace(x[0], x[-1], 5))
        ax.set_axisbelow("line")
        ax.grid(True, linewidth=0.4, color="0.5", alpha=0.35)
    fig.savefig(OUTPUT_PATH, dpi=200)
    plt.close(fig)
    print(
        f"Saved {OUTPUT_PATH.relative_to(ROOT)}; solution L2 = {100 * relative_l2:.4f}%"
    )


if __name__ == "__main__":
    main()
