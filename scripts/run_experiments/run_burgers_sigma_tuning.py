"""Run the Burgers diffusion tuning sweep or summarize its saved artifacts."""

import json
import math
import statistics

import yaml

from scripts.run_experiments import run_burgers_campaign as campaign
from src.train import ROOT

# Select execution here. Summarization never launches training or writes artifacts.
ACTION = "run"
CONFIG_PATHS = [
    ROOT / "config/burgers_nu_0p01_sigma_0.yaml",
    ROOT / "config/burgers_nu_0p01_sigma_1e-5.yaml",
    ROOT / "config/burgers_nu_0p01_sigma_1e-4.yaml",
    ROOT / "config/burgers_nu_0p01_sigma_1e-3.yaml",
    ROOT / "config/burgers_nu_0p01_sigma_1e-2.yaml",
]


def load_configurations():
    """Return the full YAML configurations selected above, in execution order."""
    return [yaml.safe_load(path.read_text()) for path in CONFIG_PATHS]


def check_tuning_configs(configs):
    """Reject overlapping destinations or unmatched controls in the paired sweep."""
    if not configs or list(configs[0]["methods"]) != ["pacmann_adam"]:
        raise ValueError("The first tuning configuration must be ordinary PACMANN")
    baseline = configs[0]
    seeds = baseline["seeds"]
    if (
        not seeds
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("Tuning seeds must be distinct nonnegative integers")
    campaigns, sigmas = set(), [0.0]
    for index, config in enumerate(configs):
        if config["campaign"] in campaigns:
            raise ValueError("Tuning configurations must have distinct campaign names")
        campaigns.add(config["campaign"])
        for field in ("seeds", "problem", "network", "training", "evaluation"):
            if config.get(field) != baseline.get(field):
                raise ValueError(f"Unmatched tuning control: {field}")
        method = "pacmann_adam" if index == 0 else "pacmann_diffusion"
        if list(config["methods"]) != [method]:
            raise ValueError("Each tuning configuration must select its one method")
        sampling = dict(config["methods"][method])
        if index > 0:
            sigma = sampling.pop("sigma")
            if not math.isfinite(sigma) or sigma <= sigmas[-1]:
                raise ValueError(
                    "Diffusion sigmas must be finite, positive and increasing"
                )
            sigmas.append(sigma)
        if sampling != baseline["methods"]["pacmann_adam"]:
            raise ValueError("Coordinate-Adam settings must match the baseline")


def run_tuning(configs):
    """Check the entire sweep, then train and analyze configurations sequentially."""
    check_tuning_configs(configs)
    for config in configs:
        method = next(iter(config["methods"]))
        campaign.check_campaign(config, method)

    # Execute each matched method batch in its configured order.
    for config in configs:
        method = next(iter(config["methods"]))
        campaign.run_campaign(config, method)


def summarize_tuning(configs):
    """Return a Markdown comparison using archived configurations and final errors."""
    check_tuning_configs(configs)
    rows: list[dict] = []
    provenance = None
    for index, config in enumerate(configs):
        method = next(iter(config["methods"]))
        sigma = 0.0 if index == 0 else config["methods"][method]["sigma"]
        errors, residuals, seconds, failed, pending = {}, {}, [], 0, 0

        # Classify every planned seed, retaining failures and unfinished analysis.
        for seed in config["seeds"]:
            run = (
                ROOT
                / "output"
                / "burgers"
                / config["campaign"]
                / method
                / f"seed_{seed}"
            )
            status_path = run / "status.json"
            status = (
                json.loads(status_path.read_text())["status"]
                if status_path.exists()
                else "pending"
            )
            if status == "failed":
                failed += 1
                continue
            required = (
                "config.yaml",
                "metrics.json",
                "metadata.json",
                "diagnostics.json",
                "diagnostics.png",
            )
            if status != "complete" or not all(
                (run / name).is_file() for name in required
            ):
                pending += 1
                continue

            # Read executed settings and reject stale or incomparable summaries.
            saved = yaml.safe_load((run / "config.yaml").read_text())
            if saved != dict(config, selected_method=method, selected_seed=seed):
                raise ValueError(f"Archived configuration differs from sweep: {run}")
            sigma = (
                0.0 if method == "pacmann_adam" else saved["methods"][method]["sigma"]
            )
            metadata = json.loads((run / "metadata.json").read_text())
            identity = {
                key: metadata[key]
                for key in ("revision", "versions", "dataset_sha256", "device")
            }
            if provenance is None:
                provenance = identity
            elif identity != provenance:
                raise ValueError(
                    f"Tuning runs have different source, environment, data or device: {run}"
                )
            metrics = json.loads((run / "metrics.json").read_text())
            error = metrics["final_relative_l2"]
            residual = metrics["final_residual_rms"]
            duration = metrics["training_seconds"]
            if not all(
                math.isfinite(value) and value >= 0
                for value in (error, residual, duration)
            ):
                raise ValueError(f"Invalid final metric or training time: {run}")
            errors[seed] = error
            residuals[seed] = residual
            seconds.append(duration)

        rows.append(
            {
                "sigma": sigma,
                "errors": errors,
                "residuals": residuals,
                "seconds": seconds,
                "failed": failed,
                "pending": pending,
            }
        )

    # Report individual outcomes; aggregate only fully completed configurations.
    lines = [
        "| Sigma | Planned | Completed | Failed | Pending | Seed: final L2 (%) | Mean L2 (%) | Sample SD (%) | Mean paired change (percentage points) | Improved / paired | Completed training seconds |",
        "| ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    baseline_errors = rows[0]["errors"]
    planned = len(configs[0]["seeds"])
    for row in rows:
        errors = row["errors"]
        values = list(errors.values())
        complete = len(values) == planned
        mean = f"{100 * statistics.mean(values):.6g}" if complete else "—"
        sd = (
            f"{100 * statistics.stdev(values):.6g}"
            if complete and len(values) > 1
            else "—"
        )
        paired = [
            errors[seed] - baseline_errors[seed]
            for seed in errors
            if seed in baseline_errors
        ]
        change = (
            f"{100 * statistics.mean(paired):+.6g}" if len(paired) == planned else "—"
        )
        improved = (
            f"{sum(value < 0 for value in paired)}/{len(paired)}" if paired else "—"
        )
        individual = "; ".join(
            f"{seed}: {100 * errors[seed]:.6g}"
            if seed in errors
            else f"{seed}: unavailable"
            for seed in configs[0]["seeds"]
        )
        elapsed = f"{sum(row['seconds']):.1f}" if values else "—"
        lines.append(
            f"| {row['sigma']:g} | {planned} | {len(values)} | {row['failed']} | {row['pending']} | {individual} | {mean} | {sd} | {change} | {improved} | {elapsed} |"
        )

    # Report residual outcomes separately; they do not select sigma.
    lines.extend(
        [
            "",
            "Final PDE residual RMS (reported at the solution-error-selected setting):",
            "",
            "| Sigma | Planned | Completed | Failed | Pending | Seed: residual RMS | Mean RMS | Sample SD |",
            "| ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
        ]
    )
    for row in rows:
        values = list(row["residuals"].values())
        complete = len(values) == planned
        mean = f"{statistics.mean(values):.6g}" if complete else "—"
        sd = (
            f"{statistics.stdev(values):.6g}"
            if complete and len(values) > 1
            else "—"
        )
        individual = "; ".join(
            f"{seed}: {row['residuals'][seed]:.6g}"
            if seed in row["residuals"]
            else f"{seed}: unavailable"
            for seed in configs[0]["seeds"]
        )
        lines.append(
            f"| {row['sigma']:g} | {planned} | {len(values)} | {row['failed']} | {row['pending']} | {individual} | {mean} | {sd} |"
        )

    # Compare positive diffusion settings separately from ordinary PACMANN.
    if all(len(row["errors"]) == planned for row in rows):
        positive_rows = rows[1:]
        winner = min(
            positive_rows,
            key=lambda row: (statistics.mean(row["errors"].values()), row["sigma"]),
        )
        lines.append(
            "\nOrdinary PACMANN (sigma=0) is the comparison baseline."
        )
        lines.append(
            f"Selected positive sigma by mean final relative L2: {winner['sigma']:g}."
        )
        if winner is positive_rows[0]:
            lower = (winner["sigma"] / 100, winner["sigma"] / 10)
            lines.append(
                f"The winner is the lower tested boundary. The next approved lower expansion is {lower[0]:g}, {lower[1]:g}; after two expansion rounds, freeze the best tested positive sigma and report the boundary limitation."
            )
        elif winner is positive_rows[-1]:
            upper = (winner["sigma"] * 10, winner["sigma"] * 100)
            lines.append(
                f"The winner is the upper tested boundary. The next approved upper expansion is {upper[0]:g}, {upper[1]:g}; after two expansion rounds, freeze the best tested positive sigma and report the boundary limitation."
            )
    else:
        lines.append(
            "\nNo sigma selected: all planned training runs and diagnostics must complete successfully. Preserve failures for review."
        )
    lines.append(
        "\nNegative paired changes indicate improvement. SD is sample standard deviation across completed seeds; timings cover compilation and training only."
    )
    return "\n".join(lines)


def main():
    """Execute the selected action and print the saved-run comparison."""
    if ACTION not in ("run", "summarize"):
        raise ValueError("ACTION must be 'run' or 'summarize'")
    configs = load_configurations()
    if ACTION == "run":
        run_tuning(configs)
    print(summarize_tuning(configs))


if __name__ == "__main__":
    main()
