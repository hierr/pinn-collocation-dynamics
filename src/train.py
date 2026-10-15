"""Train one configured Burgers method and seed in a fresh artifact directory."""

import time
from pathlib import Path

import numpy as np
import torch
from deepxde import config as dde_config
from deepxde.optimizers import config as optimizer_config

from src.artifacts import (
    capture_run_log,
    prepare_run,
    save_points,
    save_snapshot,
    write_json,
)
from src.burgers import build_model, load_reference
from src.sampling import FixedHammersley, PacmannAdam, PacmannDiffusion

ROOT = Path(__file__).resolve().parents[1]


def run_training(config: dict, method: str, seed: int):
    """Train one configured seed, retaining checkpoints and artifacts on failure."""
    problem_settings = config["problem"]
    network_settings = config["network"]
    training_settings = config["training"]
    evaluation_settings = config["evaluation"]
    # Verify reference data and evaluation sizes before reserving the run directory.
    for key in ("residual_nx", "residual_nt", "batch_size"):
        if type(evaluation_settings[key]) is not int or evaluation_settings[key] <= 0:
            raise ValueError(f"Evaluation {key} must be a positive integer")
    _, _, dataset_sha256 = load_reference(
        ROOT, problem_settings, evaluation_settings["solution_nx"]
    )
    run_dir = prepare_run(config, method, seed, dataset_sha256, ROOT)
    sampling_settings = config["methods"][method]
    interior_count = problem_settings["interior_points"]

    with capture_run_log(run_dir):
        # Initialize reproducibility before constructing points and network weights.
        dde_config.set_random_seed(seed)
        dde_config.set_default_float("float64")
        model, residual_gradient = build_model(
            problem_settings=problem_settings,
            network_settings=network_settings,
        )

        # Select the collocation method independently of the network optimizer.
        if method == "fixed_hammersley":
            sampler = FixedHammersley()
        elif method == "pacmann_adam":
            sampler = PacmannAdam(
                interior_points=interior_count,
                residual_gradient=residual_gradient,
                period=sampling_settings["period"],
                coordinate_steps=sampling_settings["steps"],
                step_size=sampling_settings["step_size"],
                betas=sampling_settings["betas"],
                epsilon=sampling_settings["epsilon"],
            )
        else:
            sampler = PacmannDiffusion(
                interior_points=interior_count,
                residual_gradient=residual_gradient,
                period=sampling_settings["period"],
                coordinate_steps=sampling_settings["steps"],
                step_size=sampling_settings["step_size"],
                betas=sampling_settings["betas"],
                epsilon=sampling_settings["epsilon"],
                sigma=sampling_settings["sigma"],
                domain_widths=[
                    problem_settings["x_bounds"][1] - problem_settings["x_bounds"][0],
                    problem_settings["t_bounds"][1] - problem_settings["t_bounds"][0],
                ],
                seed=seed,
            )
        constraints = model.data.train_x_bc.copy()
        initial_pde_points = model.data.train_x_all.copy()
        save_points(model.data, run_dir, "initial", interior_count)

        # Preserve upstream L-BFGS limits and DeepXDE's optimizer-call batching.
        optimizer_config.set_LBFGS_options(**training_settings["lbfgs"])
        options = optimizer_config.LBFGS_options
        # DeepXDE initializes these derived limits only at import time.
        options["iter_per_step"] = min(1000, options["maxiter"])
        options["fun_per_step"] = (
            options["maxfun"] * options["iter_per_step"] // options["maxiter"]
        )
        stages = []

        for phase in range(1, training_settings["phases"] + 1):
            # Optimize network weights with Adam and the sampling callback.
            events_before = sampler.events
            synchronize()
            start = time.perf_counter()
            model.compile("adam", lr=training_settings["learning_rate"])
            model.train(
                iterations=training_settings["adam_steps"],
                callbacks=[sampler],
                display_every=training_settings["display_every"],
            )
            synchronize()
            adam_seconds = time.perf_counter() - start

            # Verify constraints and fixed sampling at the Adam endpoint.
            if not np.array_equal(model.data.train_x_bc, constraints):
                raise RuntimeError("Initial/boundary constraint coordinates changed")
            if method == "fixed_hammersley" and not np.array_equal(
                initial_pde_points, model.data.train_x_all
            ):
                raise RuntimeError("Fixed Hammersley points moved")
            if not np.isfinite(model.train_state.loss_train).all():
                raise FloatingPointError("Non-finite training loss")
            save_snapshot(model, run_dir, f"phase_{phase:02d}_adam", interior_count)
            stages.append(
                {
                    "phase": phase,
                    "optimizer": "adam",
                    "seconds": adam_seconds,
                    "global_step": int(model.train_state.step),
                    "optimizer_iterations": training_settings["adam_steps"],
                    "movement_events": sampler.events - events_before,
                    "pde_points": len(model.data.train_x_all),
                }
            )
            write_json(run_dir / "training.json", {"stages": stages})

            # Recompile for L-BFGS, resetting optimizer state and fixing PDE points.
            pde_before_lbfgs = model.data.train_x_all.copy()
            events_before = sampler.events
            synchronize()
            start = time.perf_counter()
            model.compile("L-BFGS")
            model.train(display_every=training_settings["display_every"])
            synchronize()
            lbfgs_seconds = time.perf_counter() - start

            # Verify fixed populations at the L-BFGS endpoint.
            if not np.array_equal(model.data.train_x_bc, constraints):
                raise RuntimeError("Initial/boundary constraint coordinates changed")
            if not np.array_equal(pde_before_lbfgs, model.data.train_x_all):
                raise RuntimeError("Physics points moved during L-BFGS")
            if not np.isfinite(model.train_state.loss_train).all():
                raise FloatingPointError("Non-finite training loss")
            save_snapshot(model, run_dir, f"phase_{phase:02d}_l-bfgs", interior_count)
            lbfgs_state = next(iter(model.opt.state_dict()["state"].values()))
            stages.append(
                {
                    "phase": phase,
                    "optimizer": "L-BFGS",
                    "seconds": lbfgs_seconds,
                    "global_step": int(model.train_state.step),
                    "optimizer_iterations": int(lbfgs_state["n_iter"]),
                    "movement_events": sampler.events - events_before,
                    "pde_points": len(model.data.train_x_all),
                    "function_evaluations": int(lbfgs_state["func_evals"]),
                }
            )
            write_json(run_dir / "training.json", {"stages": stages})

        # Save the final model and points for independent post-processing.
        save_snapshot(model, run_dir, "final", interior_count)

        # Retain optimizer loss curves for post-processing.
        history = model.losshistory
        np.savez_compressed(
            run_dir / "history.npz",
            steps=np.array(history.steps),
            loss_train=np.array(history.loss_train),
            loss_test=np.array(history.loss_test),
        )

        if method in ("pacmann_adam", "pacmann_diffusion"):
            expected_events = (
                training_settings["phases"]
                * training_settings["adam_steps"]
                // sampling_settings["period"]
            )
            if sampler.events != expected_events:
                raise RuntimeError("Unexpected PACMANN update count")

    return run_dir


def synchronize():
    """Synchronize CUDA before measuring elapsed training time."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
