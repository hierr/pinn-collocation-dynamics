"""Run the paired sigma sweep or summarize its saved artifacts without training."""

import json
import math
import statistics

import yaml

from scripts import run_burgers_campaign as campaign
from src.train import ROOT

# Select execution here. Summarization never launches training or writes artifacts.
ACTION = "summarize"
CONFIG_PATHS = [
    ROOT / "config/burgers_sigma_baseline.yaml",
    ROOT / "config/burgers_sigma_1e-5.yaml",
    ROOT / "config/burgers_sigma_1e-4.yaml",
    ROOT / "config/burgers_sigma_1e-3.yaml",
    ROOT / "config/burgers_sigma_1e-2.yaml",
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
        for field in ("seeds", "problem", "network", "training"):
            if config[field] != baseline[field]:
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
        campaign.check_campaign(config)

    # Create short protocol links only when launching an actual campaign.
    for config in configs:
        directory = ROOT / "output" / config["campaign"]
        directory.mkdir(parents=True, exist_ok=True)
        readme = directory / "README.md"
        if not readme.exists():
            readme.write_text(
                f"# Burgers sigma tuning: {config['campaign']}\n\n"
                "- [Protocol and interpretation](../../docs/experimental_notes.md#burgers-sigma-tuning-01)\n"
                "- [Results](../../docs/results.md)\n"
                "- [Output dictionary](../../docs/output_format.md)\n\n"
                "Executed with `venv/bin/python -m scripts.run_burgers_sigma_tuning` "
                'and `ACTION = "run"`. Each seed directory archives its configuration '
                "and source revision.\n"
            )
        campaign.run_campaign(config)


def summarize_tuning(configs):
    """Return a Markdown comparison using archived configurations and final errors."""
    check_tuning_configs(configs)
    rows: list[dict] = []
    provenance = None
    for index, config in enumerate(configs):
        method = next(iter(config["methods"]))
        sigma = 0.0 if index == 0 else config["methods"][method]["sigma"]
        errors, seconds, failed, pending = {}, [], 0, 0

        # Classify every planned seed, retaining failures and unfinished analysis.
        for seed in config["seeds"]:
            run = (
                ROOT
                / "output"
                / config["campaign"]
                / "burgers"
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
            error, duration = metrics["final_relative_l2"], metrics["training_seconds"]
            if not all(
                math.isfinite(value) and value >= 0 for value in (error, duration)
            ):
                raise ValueError(f"Invalid final error or training time: {run}")
            errors[seed] = error
            seconds.append(duration)

        rows.append(
            {
                "sigma": sigma,
                "errors": errors,
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

    # Freeze a candidate only after every planned run and its analysis completes.
    if all(len(row["errors"]) == planned for row in rows):
        winner = min(
            rows,
            key=lambda row: (statistics.mean(row["errors"].values()), row["sigma"]),
        )
        lines.append(
            f"\nSelected sigma by mean final relative L2: {winner['sigma']:g}."
        )
        if winner["sigma"] > 0 and winner in (rows[1], rows[-1]):
            lines.append(
                "The selected positive sigma is at a search boundary; review before freezing it. No automatic grid expansion."
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
