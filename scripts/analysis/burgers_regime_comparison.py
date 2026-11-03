"""Reproduce the accepted final-error and physical-time comparisons for Burgers."""

import csv
import json
import subprocess
from io import BytesIO
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/burgers/regime_comparison"
FIGURES = ROOT / "docs/figures/burgers_regime_comparison"
PRESENTATION_FIGURES = ROOT / "docs/presentation/figures/burgers_regime_comparison"
# All five completed campaigns are retained in this fixed results checkpoint.
SOURCE_REVISION = "456207722c67ce4b11856efc3f4ace6f22f0c3e5"
SEEDS = range(4000, 4010)
# Peak gradients and times use the same native-grid reference characterization.
CASES = [
    ("output/burgers/burgers_nu_0p1_10seed", "0.1", 0.1, 4.211433, 0.240),
    (
        "output/burgers/burgers_nu_0p031415927_10seed",
        "π/100",
        np.pi / 100,
        11.872671,
        0.515,
    ),
    ("output/burgers/burgers_nu_0p01_10seed", "0.01", 0.01, 44.565582, 0.540),
    (
        "output/burgers/burgers_nu_0p0031830989_10seed",
        "0.01/π",
        0.01 / np.pi,
        151.932819,
        0.510,
    ),
    (
        "output/burgers/nu_0p001/burgers_nu_0p001_10seed",
        "0.001",
        0.001,
        494.985806,
        0.505,
    ),
]
# Keep pi/100 in the complete analysis; its inclusion in the main slides is deferred.
PRESENTATION_CASES = (0, 2, 3, 4)
METHODS = [
    "fixed_hammersley",
    "random_resampling",
    "rad",
    "pacmann_adam",
    "pacmann_diffusion",
]
LABELS = ["Hammersley", "Random", "RAD", "PACMANN", "PACMANN\n+ diffusion"]
# Separate method colors from the reference figures' time and viscosity palettes.
COLORS = ["#363b41", "#8a8f94", "#008b8b", "#b18b00", "#b5166b"]
METRICS = [
    ("solution_l2_percent", "Solution relative L2 (%)"),
    ("residual_rms", "PDE residual RMS"),
]


def read_snapshot(path):
    """Read an artifact from the accepted Git snapshot without changing checkout files."""
    return subprocess.run(
        ["git", "show", f"{SOURCE_REVISION}:{path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout


def prediction_errors(arrays):
    """Return times, per-time relative L2 and RMSE, and complete-grid relative L2."""
    coordinates, reference, prediction = (
        arrays[key] for key in ("coordinates", "reference", "prediction")
    )
    x, times = np.unique(coordinates[:, 0]), np.unique(coordinates[:, 1])
    xx, tt = np.meshgrid(x, times)
    if not np.array_equal(coordinates, np.column_stack((xx.ravel(), tt.ravel()))):
        raise ValueError("Predictions must use a complete, time-major Cartesian grid")
    if reference.shape != (xx.size, 1) or prediction.shape != reference.shape:
        raise ValueError(
            "Prediction and reference shapes must match the evaluation grid"
        )
    if not np.isfinite(reference).all() or not np.isfinite(prediction).all():
        raise ValueError("Predictions and references must be finite")
    reference = reference.reshape(xx.shape)
    error = prediction.reshape(xx.shape) - reference
    norms = np.linalg.norm(reference, axis=1)
    if np.any(norms == 0):
        raise ValueError("Per-time relative L2 requires nonzero reference norms")
    return (
        times,
        np.linalg.norm(error, axis=1) / norms,
        np.sqrt(np.mean(error**2, axis=1)),
        np.linalg.norm(error) / np.linalg.norm(reference),
    )


def paired_statistics(first, second):
    """Summarize same-seed error differences and ratios without discarding outcomes."""
    first, second = np.asarray(first), np.asarray(second)
    if first.shape != second.shape or first.ndim != 1 or first.size < 2:
        raise ValueError("Paired comparisons require matching seed vectors")
    if not all(
        np.isfinite(values).all() and np.all(values > 0) for values in (first, second)
    ):
        raise ValueError("Error ratios require finite, strictly positive errors")
    difference = first - second
    return {
        "mean_difference": float(np.mean(difference)),
        "sample_sd_difference": float(np.std(difference, ddof=1)),
        "improved": int(np.sum(difference < 0)),
        "tied": int(np.sum(difference == 0)),
        "worse": int(np.sum(difference > 0)),
        "median_ratio": float(np.median(first / second)),
    }


def write_csv(name, rows):
    """Save one derived table with explicit column names."""
    with (OUT / name).open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    """Read all 250 completed runs and save full and presentation comparisons."""
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    errors, temporal, sigmas = {}, {}, []
    final_rows, temporal_rows, summary_rows = [], [], []

    # Recompute solution errors from final predictions; retain archived residual scores.
    for case_index, (campaign, _, nu, gradient, _) in enumerate(CASES):
        first_run = f"{campaign}/{METHODS[0]}/seed_{SEEDS[0]}"
        common = yaml.safe_load(read_snapshot(f"{first_run}/config.yaml"))
        sigmas.append(common["methods"]["pacmann_diffusion"]["sigma"])
        shared_grid = None
        for method in METHODS:
            solution_errors, residual_errors, profiles = [], [], []
            for seed in SEEDS:
                run = f"{campaign}/{method}/seed_{seed}"
                config = yaml.safe_load(read_snapshot(f"{run}/config.yaml"))
                if (
                    any(
                        config[key] != common[key]
                        for key in (
                            "problem",
                            "evaluation",
                            "network",
                            "training",
                            "methods",
                        )
                    )
                    or not np.isclose(
                        config["problem"]["viscosity"], nu, rtol=1e-12, atol=0
                    )
                    or config["selected_method"] != method
                    or config["selected_seed"] != seed
                ):
                    raise ValueError(f"Scientific configuration mismatch: {run}")
                if (
                    json.loads(read_snapshot(f"{run}/status.json"))["status"]
                    != "complete"
                ):
                    raise ValueError(f"Incomplete planned run: {run}")
                metrics = json.loads(read_snapshot(f"{run}/metrics.json"))
                with np.load(
                    BytesIO(read_snapshot(f"{run}/predictions.npz")), allow_pickle=False
                ) as arrays:
                    times, relative, rmse, overall = prediction_errors(arrays)
                    if shared_grid is None:
                        shared_grid = (
                            arrays["coordinates"].copy(),
                            arrays["reference"].copy(),
                        )
                    elif not (
                        np.array_equal(shared_grid[0], arrays["coordinates"])
                        and np.array_equal(shared_grid[1], arrays["reference"])
                    ):
                        raise ValueError(f"Evaluation grid or reference differs: {run}")
                if not np.isclose(
                    overall, metrics["final_relative_l2"], rtol=1e-10, atol=1e-12
                ):
                    raise ValueError(
                        f"Archived solution score differs from predictions: {run}"
                    )
                solution_errors.append(100 * overall)
                residual_errors.append(metrics["final_residual_rms"])
                profiles.append(100 * relative)
                final_rows.append(
                    {
                        "nu": nu,
                        "peak_reference_gradient": gradient,
                        "selected_diffusion_sigma": sigmas[-1],
                        "method": method,
                        "seed": seed,
                        "solution_l2_percent": 100 * overall,
                        "residual_rms": metrics["final_residual_rms"],
                    }
                )
                temporal_rows.extend(
                    {
                        "nu": nu,
                        "method": method,
                        "seed": seed,
                        "time": time,
                        "solution_l2_percent": value,
                        "solution_rmse": absolute,
                    }
                    for time, value, absolute in zip(times, 100 * relative, rmse)
                )
            temporal[case_index, method] = (times, np.array(profiles))
            for (metric, _), values in zip(METRICS, (solution_errors, residual_errors)):
                values = np.array(values)
                if not np.isfinite(values).all() or np.any(values <= 0):
                    raise ValueError(
                        f"Ratios and log plots require positive finite {metric}"
                    )
                errors[case_index, method, metric] = values
                summary_rows.append(
                    {
                        "nu": nu,
                        "method": method,
                        "metric": metric,
                        "mean": np.mean(values),
                        "sample_sd": np.std(values, ddof=1),
                        "median": np.median(values),
                        "minimum": np.min(values),
                        "maximum": np.max(values),
                    }
                )
        print(f"Verified all 50 final solution scores at ν={nu:.10g}", flush=True)

    # Preserve seed pairing within each viscosity; do not pool different regimes.
    contrasts = [
        (method, baseline) for baseline in METHODS[:2] for method in METHODS[2:]
    ]
    contrasts += [(METHODS[1], METHODS[0]), (METHODS[4], METHODS[3])]
    paired_rows, ratio_rows = [], []
    for case_index, (_, _, nu, gradient, _) in enumerate(CASES):
        for method, baseline in contrasts:
            for metric, _ in METRICS:
                first, second = (
                    errors[case_index, method, metric],
                    errors[case_index, baseline, metric],
                )
                paired_rows.append(
                    dict(
                        nu=nu,
                        method=method,
                        baseline=baseline,
                        metric=metric,
                        **paired_statistics(first, second),
                    )
                )
                ratio_rows.extend(
                    {
                        "nu": nu,
                        "peak_reference_gradient": gradient,
                        "method": method,
                        "baseline": baseline,
                        "metric": metric,
                        "seed": seed,
                        "ratio": ratio,
                    }
                    for seed, ratio in zip(SEEDS, first / second)
                )
    write_csv("final_metrics.csv", final_rows)
    write_csv("final_summary.csv", summary_rows)
    write_csv("paired_summary.csv", paired_rows)
    write_csv("paired_seed_ratios.csv", ratio_rows)
    write_csv("physical_time_errors.csv", temporal_rows)

    plot_comparisons(errors, temporal, sigmas, range(len(CASES)), FIGURES)
    plot_comparisons(errors, temporal, sigmas, PRESENTATION_CASES, PRESENTATION_FIGURES)
    print(f"Saved five CSV tables in {OUT.relative_to(ROOT)}")


def plot_comparisons(errors, temporal, sigmas, case_indices, destination):
    """Save the four comparison figures for an explicit selection of viscosity cases."""
    destination.mkdir(parents=True, exist_ok=True)
    # Match figure widths and header spacing so titles keep the same displayed size.
    plt.rcParams.update(
        {
            "font.size": 10,
            "figure.titlesize": 12,
            "figure.titleweight": "semibold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    jitter = np.linspace(-0.14, 0.14, len(SEEDS))
    method_handles = [
        Line2D([], [], color=color, label=label.replace("\n", " "))
        for color, label in zip(COLORS, LABELS)
    ]
    selected_cases = [CASES[index] for index in case_indices]
    case_labels = [
        f"ν = {label}\nG = {gradient:.2f}"
        for _, label, _, gradient, _ in selected_cases
    ]

    # Show every final outcome, including tails, with full-seed mean and median.
    fig, axes = plt.subplots(
        2, len(case_indices), figsize=(14, 8), sharey="row", squeeze=False
    )
    for row, (metric, ylabel) in enumerate(METRICS):
        for column, case_index in enumerate(case_indices):
            ax = axes[row, column]
            for index, (method, color) in enumerate(zip(METHODS, COLORS)):
                values = errors[case_index, method, metric]
                ax.scatter(index + jitter, values, color=color, s=19, alpha=0.85)
                ax.scatter(
                    index, np.mean(values), color="black", marker="D", s=28, zorder=4
                )
                ax.hlines(
                    np.median(values),
                    index - 0.23,
                    index + 0.23,
                    color="black",
                    linewidth=1.8,
                )
            ax.set(yscale="log", xticks=range(5), xticklabels=LABELS)
            ax.tick_params(axis="x", labelsize=9, labelrotation=35)
            plt.setp(ax.get_xticklabels(), ha="right")
            ax.grid(axis="y", alpha=0.2)
            if row == 0:
                ax.set_title(case_labels[column])
            if column == 0:
                ax.set_ylabel(ylabel)
    fig.suptitle(
        "Final-model accuracy",
        y=1 - 0.12 / fig.get_figheight(),
    )
    fig.text(
        0.5,
        1 - 0.34 / fig.get_figheight(),
        "All ten seeds at each viscosity",
        ha="center",
        va="top",
        fontsize=9,
        weight="normal",
        color="0.35",
    )
    fig.legend(
        handles=[
            Line2D(
                [],
                [],
                color="black",
                marker="D",
                linestyle="none",
                label="Arithmetic mean (all seeds)",
            ),
            Line2D([], [], color="black", linewidth=1.8, label="Median (all seeds)"),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.savefig(destination / "final_errors.png", dpi=200)
    plt.close(fig)

    # Compare relative benefits to both sampling baselines using the same seeds.
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey="row")
    for row, (metric, _) in enumerate(METRICS):
        for column, baseline in enumerate(METHODS[:2]):
            ax = axes[row, column]
            for index, method in enumerate(METHODS[1:], start=1):
                if method == baseline:
                    continue
                medians = []
                for position_index, case_index in enumerate(case_indices):
                    ratio = (
                        errors[case_index, method, metric]
                        / errors[case_index, baseline, metric]
                    )
                    position = position_index + (index - 2.5) * 0.13
                    ax.scatter(
                        position + jitter * 0.18,
                        ratio,
                        color=COLORS[index],
                        s=16,
                        alpha=0.65,
                    )
                    medians.append(np.median(ratio))
                ax.plot(
                    np.arange(len(case_indices)) + (index - 2.5) * 0.13,
                    medians,
                    color=COLORS[index],
                    marker="o",
                    linewidth=1.5,
                )
            ax.axhline(1, color="black", linestyle="--", linewidth=1)
            ax.set(
                yscale="log", xticks=range(len(case_indices)), xticklabels=case_labels
            )
            ax.grid(axis="y", alpha=0.2)
            if row == 0:
                ax.set_title(f"Relative to {LABELS[column]}")
            if column == 0:
                ax.set_ylabel(
                    "Solution-error ratio" if row == 0 else "Residual-RMS ratio"
                )
    fig.suptitle(
        "Paired final-error ratios across reference regimes",
        y=1 - 0.12 / fig.get_figheight(),
    )
    fig.text(
        0.5,
        1 - 0.34 / fig.get_figheight(),
        "Below 1 favors the method; lines connect paired-ratio medians.",
        ha="center",
        va="top",
        fontsize=9,
        weight="normal",
        color="0.35",
    )
    fig.legend(handles=method_handles[1:], loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 0.92))
    fig.savefig(destination / "baseline_ratios.png", dpi=200)
    plt.close(fig)

    # Isolate diffusion's incremental effect under the frozen per-viscosity tuning policy.
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    for ax, (metric, ylabel) in zip(axes, METRICS):
        for position_index, case_index in enumerate(case_indices):
            ratio = (
                errors[case_index, METHODS[4], metric]
                / errors[case_index, METHODS[3], metric]
            )
            ax.scatter(position_index + jitter, ratio, color=COLORS[4], s=25)
            ax.hlines(
                np.median(ratio),
                position_index - 0.23,
                position_index + 0.23,
                color="black",
                linewidth=1.8,
            )
        ax.axhline(1, color="black", linestyle="--", linewidth=1)
        ax.set(
            yscale="log",
            title=ylabel,
            ylabel="Diffusion / PACMANN error ratio",
            xticks=range(len(case_indices)),
            xticklabels=[
                f"ν = {CASES[index][1]}\nσ = {sigmas[index]:g}"
                for index in case_indices
            ],
        )
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Paired diffusion effect", y=1 - 0.12 / fig.get_figheight())
    fig.text(
        0.5,
        1 - 0.34 / fig.get_figheight(),
        "Below 1 favors diffusion; black bars show paired-ratio medians.",
        ha="center",
        va="top",
        fontsize=9,
        weight="normal",
        color="0.35",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(destination / "diffusion_ratios.png", dpi=200)
    plt.close(fig)

    # Physical time refers to the solution coordinate, not optimizer or elapsed time.
    fig, axes = plt.subplots(
        2,
        len(case_indices),
        figsize=(14, 7.8),
        sharex=True,
        sharey="row",
        squeeze=False,
    )
    for column, case_index in enumerate(case_indices):
        case = CASES[case_index]
        for method, color in zip(METHODS, COLORS):
            times, profiles = temporal[case_index, method]
            axes[0, column].plot(times, np.mean(profiles, axis=0), color=color)
            axes[1, column].plot(times, np.median(profiles, axis=0), color=color)
        for row in range(2):
            axes[row, column].axvline(
                case[4], color="black", linestyle=":", linewidth=1
            )
            axes[row, column].set(yscale="log", xlim=(0, 0.99))
            axes[row, column].grid(alpha=0.2)
        axes[0, column].set_title(case_labels[column])
        axes[1, column].set_xlabel("Physical time t")
    axes[0, 0].set_ylabel("Mean solution relative L2 (%)")
    axes[1, 0].set_ylabel("Median solution relative L2 (%)")
    fig.suptitle(
        "Final-model solution error over physical time",
        y=1 - 0.12 / fig.get_figheight(),
    )
    fig.legend(
        handles=method_handles
        + [
            Line2D(
                [],
                [],
                color="black",
                linestyle=":",
                label="Peak reference gradient time",
            )
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.94))
    fig.savefig(destination / "solution_error_over_time.png", dpi=200)
    plt.close(fig)
    print(f"Saved four figures in {destination.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
