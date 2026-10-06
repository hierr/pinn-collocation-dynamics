"""Run one configured Burgers method and seed in a fresh artifact directory."""

import argparse
import contextlib
import importlib.metadata
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml
from deepxde.optimizers import config as optimizer_config

from src.burgers import build_model, load_reference
from src.sampling import PacmannAdam, interior_points

ROOT = Path(__file__).resolve().parents[1]


class Tee:
    """Copy training output to both the terminal and the run log."""

    def __init__(self, terminal, log):
        """Store the two output streams."""
        self.terminal, self.log = terminal, log

    def write(self, text):
        """Write text to both streams."""
        self.terminal.write(text)
        return self.log.write(text)

    def flush(self):
        """Flush both output streams."""
        self.terminal.flush()
        self.log.flush()


def write_json(path: Path, payload):
    """Write a human-readable JSON artifact, rejecting non-finite values."""
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def synchronize():
    """Synchronize CUDA before measuring elapsed training time."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def create_run(config: dict, method: str, seed_index: int, root: Path = ROOT):
    """Validate selectors and reserve a fresh run directory without overwriting."""
    campaign = config["campaign"]
    if (
        not isinstance(campaign, str)
        or campaign in ("", ".", "..")
        or "/" in campaign
        or "\\" in campaign
    ):
        raise ValueError("Campaign must be one ordinary path component")
    if (
        method not in ("fixed_hammersley", "pacmann_adam")
        or method not in config["methods"]
    ):
        raise ValueError("Unknown sampling method")
    if not 0 <= seed_index < len(config["seeds"]):
        raise ValueError("Seed index is outside the configured list")
    seed = config["seeds"][seed_index]
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("Seeds must be nonnegative integers")
    run = root / "output" / campaign / "burgers" / method / f"seed_{seed}"
    run.mkdir(parents=True, exist_ok=False)
    return run, seed


def snapshot(model, run: Path, name: str, count: int):
    """Save evaluation weights and distinct PDE/interior/constraint coordinates."""
    torch.save(model.net.state_dict(), run / "models" / f"{name}.pt")
    np.savez_compressed(
        run / "points" / f"{name}.npz",
        interior=interior_points(model.data, count),
        pde=model.data.train_x_all.copy(),
        constraints=model.data.train_x_bc.copy(),
    )


def run_training(config: dict, method: str, seed_index: int):
    """Train and evaluate one seed, preserving all artifacts if execution fails."""
    # Verify the immutable evaluation dataset before creating output artifacts.
    coordinates, values, fingerprint = load_reference(ROOT)
    run, seed = create_run(config, method, seed_index)
    resolved = dict(config, selected_method=method, selected_seed=seed)
    (run / "config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))
    for folder in ("models", "points"):
        (run / folder).mkdir()
    versions = {
        name: importlib.metadata.version(name)
        for name in ("deepxde", "torch", "numpy", "scipy", "matplotlib", "PyYAML")
    }
    metadata = {
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            )
        ),
        "versions": versions,
        "dataset_sha256": fingerprint,
        "upstream_commit": "ecf06a6cf277b8beb6f8a99c88baa2535738fd7f",
        "device": str(torch.get_default_device()),
        "seed": seed,
        "snapshot_contract": "Evaluation weights only; not exact resume",
    }
    write_json(run / "metadata.json", metadata)
    write_json(run / "status.json", {"status": "running"})
    with (
        (run / "run.log").open("w") as log,
        contextlib.redirect_stdout(Tee(sys.stdout, log)),
        contextlib.redirect_stderr(Tee(sys.stderr, log)),
    ):
        try:
            execute(resolved, method, seed, coordinates, values, run)
        except Exception as error:
            traceback.print_exc()
            write_json(run / "status.json", {"status": "failed", "error": str(error)})
            raise
    write_json(run / "status.json", {"status": "complete"})
    return run


def execute(config, method, seed, coordinates, values, run):
    """Execute optimizer phases, save history, and check final-weight reload."""
    model, operator = build_model(config, seed, coordinates, values)
    settings = config["training"]
    count = config["problem"]["interior_points"]
    sampler = (
        PacmannAdam(config["methods"][method], count, operator)
        if method == "pacmann_adam"
        else None
    )
    constraints = model.data.train_x_bc.copy()
    initial = model.data.train_x_all.copy()
    np.savez_compressed(
        run / "points" / "initial.npz",
        interior=interior_points(model.data, count),
        pde=initial,
        constraints=constraints,
    )
    stages = []
    total_seconds = 0.0
    optimizer_config.set_LBFGS_options(**settings["lbfgs"])
    # DeepXDE initializes these derived limits only at import time.
    options = optimizer_config.LBFGS_options
    options["iter_per_step"] = min(1000, options["maxiter"])
    options["fun_per_step"] = (
        options["maxfun"] * options["iter_per_step"] // options["maxiter"]
    )
    # Recompile each stage, resetting network optimizer state as upstream does.
    for phase in range(1, settings["phases"] + 1):
        for optimizer in ("adam", "L-BFGS"):
            kwargs = {"lr": settings["learning_rate"]} if optimizer == "adam" else {}
            synchronize()
            start = time.perf_counter()
            model.compile(
                optimizer, metrics=["mean squared error", "l2 relative error"], **kwargs
            )
            before = model.data.train_x_all.copy()
            events_before = sampler.events if sampler else 0
            train_kwargs = (
                {
                    "iterations": settings["adam_steps"],
                    "callbacks": [sampler] if sampler else [],
                }
                if optimizer == "adam"
                else {}
            )
            model.train(display_every=settings["display_every"], **train_kwargs)
            synchronize()
            elapsed = time.perf_counter() - start
            total_seconds += elapsed
            # Verify population invariants after each optimizer endpoint.
            if not np.array_equal(model.data.train_x_bc, constraints):
                raise RuntimeError("Initial/boundary constraint coordinates changed")
            if optimizer == "L-BFGS" and not np.array_equal(
                before, model.data.train_x_all
            ):
                raise RuntimeError("Physics points moved during L-BFGS")
            if sampler is None and not np.array_equal(initial, model.data.train_x_all):
                raise RuntimeError("Fixed Hammersley points moved")
            if not np.isfinite(model.train_state.loss_train).all():
                raise FloatingPointError("Non-finite training loss")
            name = f"phase_{phase:02d}_{optimizer.lower()}"
            snapshot(model, run, name, count)
            state = next(iter(model.opt.state_dict()["state"].values()), {})
            stages.append(
                {
                    "phase": phase,
                    "optimizer": optimizer,
                    "seconds": elapsed,
                    "global_step": int(model.train_state.step),
                    "optimizer_iterations": int(
                        state.get("n_iter", settings["adam_steps"])
                    ),
                    "function_evaluations": int(
                        state.get("func_evals", settings["adam_steps"])
                    ),
                    "movement_events": (sampler.events if sampler else 0)
                    - events_before,
                    "pde_points": len(model.data.train_x_all),
                }
            )
            write_json(run / "training.json", {"stages": stages})
    snapshot(model, run, "final", count)
    predictions = model.predict(coordinates)
    if not np.isfinite(predictions).all():
        raise FloatingPointError("Non-finite final predictions")
    np.savez_compressed(
        run / "predictions.npz",
        coordinates=coordinates,
        reference=values,
        prediction=predictions,
    )
    history = model.losshistory
    loss = np.sum(np.array(history.loss_train), axis=1)
    metrics = np.array(history.metrics_test)
    best = int(np.argmin(loss))
    np.savez_compressed(
        run / "history.npz",
        steps=np.array(history.steps),
        loss_train=np.array(history.loss_train),
        loss_test=np.array(history.loss_test),
        metrics_test=metrics,
    )
    # Rebuild independently and verify evaluation-only weight portability.
    restored, _ = build_model(config, seed, coordinates, values)
    restored.compile("adam", lr=settings["learning_rate"])
    restored.net.load_state_dict(torch.load(run / "models/final.pt", weights_only=True))
    reloaded = restored.predict(coordinates)
    if not np.allclose(predictions, reloaded, rtol=1e-10, atol=1e-12):
        raise RuntimeError("Reloaded predictions differ from final predictions")
    expected_events = (
        settings["phases"]
        * settings["adam_steps"]
        // config["methods"][method]["period"]
        if sampler
        else 0
    )
    if sampler and sampler.events != expected_events:
        raise RuntimeError("Unexpected PACMANN update count")
    write_json(
        run / "metrics.json",
        {
            "final_mse": float(np.mean((predictions - values) ** 2)),
            "final_relative_l2": float(
                np.linalg.norm(predictions - values) / np.linalg.norm(values)
            ),
            "upstream_min_loss_step": int(history.steps[best]),
            "upstream_mse_at_min_loss": float(metrics[best, 0]),
            "upstream_l2_at_min_loss": float(metrics[best, 1]),
            "upstream_min_observed_mse": float(metrics[:, 0].min()),
            "upstream_min_observed_l2": float(metrics[:, 1].min()),
            "movement_events": sampler.events if sampler else 0,
            "reload_max_abs_difference": float(np.max(np.abs(predictions - reloaded))),
            "training_seconds": total_seconds,
        },
    )


def main():
    """Parse execution selectors and run the selected configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--method", required=True, choices=("fixed_hammersley", "pacmann_adam")
    )
    parser.add_argument("--seed-index", required=True, type=int)
    args = parser.parse_args()
    with args.config.open() as stream:
        config = yaml.safe_load(stream)
    print("Artifacts:", run_training(config, args.method, args.seed_index))


if __name__ == "__main__":
    main()
