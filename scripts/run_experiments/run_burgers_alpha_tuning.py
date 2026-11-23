"""Check, run, or summarize one viscosity's sharpness-controller development."""

import hashlib
import json
import math
import statistics

import yaml

from scripts.run_experiments import run_burgers_campaign as campaign
from src.train import ROOT

# Select one viscosity and action here; candidate settings remain in complete YAMLs.
ACTION = "check"
NU_TAG = "0p0031830989"
BASE_SIGMAS = {
    "0p1": "1e-5",
    "0p031415927": "1e-4",
    "0p01": "1e-3",
    "0p0031830989": "1e-4",
    "0p001": "1e-6",
}
METHOD = "pacmann_sharpness_diffusion"


def load_configurations(nu_tag):
    """Return PACMANN, constant diffusion, and the four finite-alpha settings."""
    directory = ROOT / "config/burgers" / f"nu_{nu_tag}"
    prefix = f"burgers_nu_{nu_tag}"
    filenames = [
        f"{prefix}_sigma_0.yaml",
        f"{prefix}_sigma_{BASE_SIGMAS[nu_tag]}.yaml",
        *[f"{prefix}_alpha_{tag}.yaml" for tag in ("0p1", "1", "10", "100")],
    ]
    return [yaml.safe_load((directory / name).read_text()) for name in filenames]


def check_tuning_configs(configs):
    """Reject unmatched scientific controls, base sigma, or gradient normalization."""
    if len(configs) < 3:
        raise ValueError("Expected PACMANN, constant diffusion, and finite alphas")
    baseline, constant = configs[:2]
    if list(baseline["methods"]) != ["pacmann_adam"] or list(constant["methods"]) != [
        "pacmann_diffusion"
    ]:
        raise ValueError("The first two settings must be the archived controls")
    sigma = constant["methods"]["pacmann_diffusion"]["sigma"]
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("The frozen base sigma must be finite and positive")
    seeds = baseline["seeds"]
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(type(seed) is not int or seed < 0 for seed in seeds)
    ):
        raise ValueError("Development seeds must be distinct nonnegative integers")

    # All variations share training/evaluation controls and coordinate-Adam settings.
    campaigns, previous_alpha = set(), 0.0
    for index, config in enumerate(configs):
        if config["campaign"] in campaigns:
            raise ValueError("Candidate campaigns must have distinct names")
        campaigns.add(config["campaign"])
        for field in ("seeds", "problem", "network", "training", "evaluation"):
            if config[field] != baseline[field]:
                raise ValueError(f"Unmatched tuning control: {field}")
        method = ("pacmann_adam", "pacmann_diffusion")[index] if index < 2 else METHOD
        if list(config["methods"]) != [method]:
            raise ValueError("Each setting must select its one expected method")
        sampling = dict(config["methods"][method])
        if index > 0 and sampling.pop("sigma") != sigma:
            raise ValueError("Alpha tuning must retain the frozen base sigma")
        if index > 1:
            alpha, g_base = sampling.pop("alpha"), sampling.pop("g_base")
            if not math.isfinite(alpha) or alpha <= previous_alpha:
                raise ValueError("Finite alphas must be positive and increasing")
            if not math.isclose(g_base, math.pi, rel_tol=1e-12, abs_tol=0):
                raise ValueError("Burgers g_base must equal the initial maximum pi")
            previous_alpha = alpha
        if sampling != baseline["methods"]["pacmann_adam"]:
            raise ValueError("Coordinate-Adam settings must match ordinary PACMANN")


def read_results(configs):
    """Read planned outcomes and verify saved settings, data, and environment."""
    reference = ROOT / configs[0]["problem"]["reference_path"]
    fingerprint = hashlib.sha256(reference.read_bytes()).hexdigest()
    rows: list[dict] = []
    provenance = None
    for index, config in enumerate(configs):
        method = next(iter(config["methods"]))
        alpha = config["methods"][method]["alpha"] if index > 1 else None
        label = ("PACMANN", "constant")[index] if index < 2 else f"alpha={alpha:g}"
        row: dict = {
            "label": label,
            "alpha": alpha,
            "errors": {},
            "residuals": {},
            "failed": [],
            "pending": [],
        }
        nu_tag = f"{config['problem']['viscosity']:.8g}".replace(".", "p")
        for seed in config["seeds"]:
            run = (
                ROOT
                / "output/burgers"
                / f"nu_{nu_tag}"
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
                row["failed"].append(seed)
                continue
            required = (
                "config.yaml",
                "metadata.json",
                "metrics.json",
                "diagnostics.json",
                "diagnostics.png",
            )
            if status != "complete" or not all(
                (run / name).is_file() for name in required
            ):
                row["pending"].append(seed)
                continue

            # Historical campaign directories were renamed without rewriting artifacts.
            saved = yaml.safe_load((run / "config.yaml").read_text())
            saved["campaign"] = config["campaign"]
            if saved != dict(config, selected_method=method, selected_seed=seed):
                raise ValueError(f"Archived configuration differs: {run}")
            metadata = json.loads((run / "metadata.json").read_text())
            if metadata["dataset_sha256"] != fingerprint:
                raise ValueError(f"Reference fingerprint differs: {run}")
            identity = {
                key: metadata[key] for key in ("versions", "dataset_sha256", "device")
            }
            if provenance is None:
                provenance = identity
            elif identity != provenance:
                raise ValueError(f"Environment, dataset, or device differs: {run}")
            metrics = json.loads((run / "metrics.json").read_text())
            error, residual = (
                metrics["final_relative_l2"],
                metrics["final_residual_rms"],
            )
            if not all(
                math.isfinite(value) and value >= 0 for value in (error, residual)
            ):
                raise ValueError(f"Non-finite or negative final metric: {run}")
            row["errors"][seed], row["residuals"][seed] = error, residual
        rows.append(row)
    return rows


def summarize_tuning(rows, seeds):
    """Print all outcomes and return the solution-error-selected strategy, if complete."""
    print("Setting | Seed: solution L2 (%), residual RMS | Failed | Pending")
    for row in rows:
        values = "; ".join(
            f"{seed}: {100 * row['errors'][seed]:.6g}, {row['residuals'][seed]:.6g}"
            if seed in row["errors"]
            else f"{seed}: unavailable"
            for seed in seeds
        )
        print(f"{row['label']} | {values} | {row['failed']} | {row['pending']}")
    if any(len(row["errors"]) != len(seeds) for row in rows):
        print("No selection: all planned candidates and controls must complete.")
        return None

    # Paired differences retain every seed; residuals do not select another alpha.
    print(
        "Setting | Mean L2 (%) ± SD | Mean L2 change vs constant (pp) / lower pairs | Mean residual ± SD | Mean residual change vs constant / lower pairs"
    )
    constant = rows[1]
    for row in rows:
        errors = [row["errors"][seed] for seed in seeds]
        residuals = [row["residuals"][seed] for seed in seeds]
        error_changes = [
            row["errors"][seed] - constant["errors"][seed] for seed in seeds
        ]
        residual_changes = [
            row["residuals"][seed] - constant["residuals"][seed] for seed in seeds
        ]
        error_sd = statistics.stdev(errors) if len(seeds) > 1 else 0
        residual_sd = statistics.stdev(residuals) if len(seeds) > 1 else 0
        print(
            f"{row['label']} | {100 * statistics.mean(errors):.6g} ± {100 * error_sd:.6g} | "
            f"{100 * statistics.mean(error_changes):+.6g} / {sum(value < 0 for value in error_changes)}/{len(seeds)} | "
            f"{statistics.mean(residuals):.6g} ± {residual_sd:.6g} | "
            f"{statistics.mean(residual_changes):+.6g} / {sum(value < 0 for value in residual_changes)}/{len(seeds)}"
        )
    winner = min(
        rows[1:],
        key=lambda row: (
            statistics.mean(row["errors"].values()),
            -math.inf if row["alpha"] is None else -row["alpha"],
        ),
    )
    print(
        f"Selected strategy: {winner['label']} (constant represents alpha → infinity)."
    )
    if winner is rows[2] or winner is rows[-1]:
        print(
            "Finite boundary winner: report the tested range; no automatic expansion."
        )
    return winner


def main():
    """Check readiness by default, or explicitly run/summarize the selected viscosity."""
    if ACTION not in ("check", "run", "summarize"):
        raise ValueError("ACTION must be 'check', 'run', or 'summarize'")
    configs = load_configurations(NU_TAG)
    check_tuning_configs(configs)
    rows = read_results(configs)
    if ACTION in ("check", "run"):
        if any(len(row["errors"]) != len(configs[0]["seeds"]) for row in rows[:2]):
            raise ValueError("Both archived controls must be complete before execution")
        for config in configs[2:]:
            campaign.check_campaign(config, METHOD)
        print(
            f"Ready: nu_{NU_TAG}, {len(configs) - 2} alphas, "
            f"{len(configs[0]['seeds'])} paired seeds, archived controls verified."
        )
    if ACTION == "run":
        for config in configs[2:]:
            campaign.run_campaign(config, METHOD)
        rows = read_results(configs)
    if ACTION != "check":
        summarize_tuning(rows, configs[0]["seeds"])


if __name__ == "__main__":
    main()
