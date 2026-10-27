"""Check spatial scoring density and resolve the prepared Burgers tuning settings."""

import json
import shutil

import torch
import yaml

from scripts.post_processing.analyze_burgers import (
    analyze_run,
    predict_batches,
    residual_rms,
    solution_errors,
)
from scripts.run_experiments import run_burgers_campaign as campaign
from scripts.run_experiments.run_burgers_sigma_tuning import (
    CONFIG_PATHS,
    check_tuning_configs,
)
from src.artifacts import write_json
from src.burgers import build_model, load_reference
from src.train import ROOT

from deepxde import config as dde_config  # isort: skip

RUN_DIR = ROOT / "output/burgers/nu_0p001/sigma_0/pacmann_adam/seed_3000"
SPATIAL_POINTS = (1024, 4096, 8192, 16384)
TOLERANCE_PERCENT = 1.0


def compare_grids(model, config, fingerprint):
    """Return fixed-model scores and the first coarse grid passing refinement."""
    problem, settings = config["problem"], config["evaluation"]
    rows = []
    selected = None
    for nx in SPATIAL_POINTS:
        # Change spatial sampling only; retain native solution and residual times.
        coordinates, reference, identity = load_reference(ROOT, problem, nx)
        if identity != fingerprint:
            raise ValueError("Reference differs from the baseline dataset fingerprint")
        prediction = predict_batches(model, coordinates, settings["batch_size"])
        row = {
            "spatial_points": nx,
            "solution_relative_l2": solution_errors(reference, prediction)[
                "final_relative_l2"
            ],
            "residual_rms": residual_rms(
                model, problem, nx, settings["residual_nt"], settings["batch_size"]
            ),
        }
        # Compare each metric with the finer-grid denominator, including exact zeros.
        if rows:
            changes = []
            for metric in ("solution_relative_l2", "residual_rms"):
                coarse, fine = rows[-1][metric], row[metric]
                change = (
                    100 * abs(coarse - fine) / fine
                    if fine
                    else (0.0 if coarse == 0 else None)
                )
                row[f"{metric}_change_percent"] = change
                changes.append(change)
            if all(
                change is not None and change < TOLERANCE_PERCENT for change in changes
            ):
                selected = rows[-1]["spatial_points"]
        rows.append(row)
        print(row, flush=True)
        if selected is not None:
            break
    return {
        "dataset_sha256": fingerprint,
        "solution_nt": len(coordinates) // nx,
        "residual_nt": settings["residual_nt"],
        "tolerance_percent": TOLERANCE_PERCENT,
        "selected_spatial_points": selected,
        "scores": rows,
    }


def apply_selected_grid(run, config_paths, spatial_points):
    """Resolve scoring settings, preserving the baseline's original evaluation."""
    if spatial_points is None:
        raise ValueError("No evaluation grid passed the refinement check")
    configs = [yaml.safe_load(path.read_text()) for path in config_paths]
    check_tuning_configs(configs)
    saved = yaml.safe_load((run / "config.yaml").read_text())
    if saved != dict(configs[0], selected_method="pacmann_adam", selected_seed=3000):
        raise ValueError("Baseline configuration differs from the prepared sweep")
    settings = dict(
        saved["evaluation"], solution_nx=spatial_points, residual_nx=spatial_points
    )
    if settings == saved["evaluation"]:
        return

    # Resolve density before other runs exist; their archived settings must stay fixed.
    for index, config in enumerate(configs):
        seeds = (
            [seed for seed in config["seeds"] if seed != 3000]
            if index == 0
            else config["seeds"]
        )
        campaign.check_campaign(
            dict(config, seeds=seeds), next(iter(config["methods"]))
        )

    # Preserve the original record before regenerating only derived evaluation files.
    initial = run / "initial_evaluation"
    initial.mkdir(exist_ok=False)
    for name in (
        "config.yaml",
        "predictions.npz",
        "metrics.json",
        "diagnostics.json",
        "diagnostics.png",
    ):
        if (run / name).exists():
            shutil.move(run / name, initial / name)
    saved["evaluation"] = settings
    (run / "config.yaml").write_text(yaml.safe_dump(saved, sort_keys=False))
    for path, config in zip(config_paths, configs, strict=True):
        config["evaluation"] = dict(settings)
        path.write_text(yaml.safe_dump(config, sort_keys=False))
    analyze_run(run)


def main():
    """Restore the baseline, save the bounded grid check, and resolve scoring density."""
    config = yaml.safe_load((RUN_DIR / "config.yaml").read_text())
    metadata = json.loads((RUN_DIR / "metadata.json").read_text())
    if json.loads((RUN_DIR / "status.json").read_text())["status"] != "complete":
        raise ValueError("The baseline must finish training before the grid check")

    # Restore fixed weights without running an optimizer step.
    dde_config.set_default_float("float64")
    model, _ = build_model(config["problem"], config["network"])
    model.compile("adam", lr=config["training"]["learning_rate"])
    model.net.load_state_dict(
        torch.load(
            RUN_DIR / "models/final.pt",
            weights_only=True,
            map_location=torch.get_default_device(),
        )
    )
    report = compare_grids(model, config, metadata["dataset_sha256"])
    report["training_revision"] = metadata["revision"]
    write_json(RUN_DIR.parent.parent / "evaluation_grid_check.json", report)
    apply_selected_grid(RUN_DIR, CONFIG_PATHS, report["selected_spatial_points"])
    print(f"Selected spatial evaluation points: {report['selected_spatial_points']}")


if __name__ == "__main__":
    main()
