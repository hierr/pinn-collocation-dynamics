"""Run Burgers method batches sequentially, with concurrent seeds in each batch."""

import multiprocessing
import os

import torch
import yaml

from scripts.post_processing.analyze_burgers import analyze_run
from src.train import ROOT, run_training

# Select the batch order here; the full scientific protocol remains in YAML.
CONFIG_PATH = ROOT / "config/burgers_nu_0p0031830989_10seed.yaml"
METHODS = ["pacmann_diffusion", "fixed_hammersley", "random_resampling", "rad"]


def check_campaign(config: dict, method: str):
    """Check the protocol and selected method's destinations without creating files."""
    campaign = config["campaign"]
    seeds = config["seeds"]
    methods = config["methods"]

    # Check every execution selector before starting a potentially long campaign.
    if (
        not isinstance(campaign, str)
        or campaign in ("", ".", "..")
        or "/" in campaign
        or "\\" in campaign
    ):
        raise ValueError("Campaign must be one ordinary path component")
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("Seeds must be a nonempty list of nonnegative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must not contain duplicates")
    if (
        not methods
        or method not in methods
        or any(
            configured_method
            not in (
                "fixed_hammersley",
                "random_resampling",
                "rad",
                "pacmann_adam",
                "pacmann_diffusion",
            )
            for configured_method in methods
        )
    ):
        raise ValueError("Methods must select supported Burgers sampling methods")

    # Other methods can already be complete; reject conflicts only for this launch.
    for seed in seeds:
        run_dir = ROOT / "output" / "burgers" / campaign / method / f"seed_{seed}"
        if run_dir.exists():
            raise FileExistsError(f"Run already exists: {run_dir}")


def run_seed(config: dict, method: str, seed: int):
    """Train and evaluate one seed in its own process and artifact directory."""
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    print(f"Training {config['campaign']}: {method}, seed {seed}", flush=True)
    run_dir = run_training(config, method, seed)
    print(f"Analyzing {run_dir}", flush=True)
    analyze_run(run_dir)
    print(f"Completed {method}, seed {seed}: {run_dir}", flush=True)


def run_campaign(config: dict, method: str):
    """Launch one process per seed and wait for every worker before reporting failures."""
    check_campaign(config, method)

    # Fresh processes inherit these limits before importing numerical libraries.
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = "1"
    context = multiprocessing.get_context("spawn")
    processes = []

    # Start the whole method batch, then allow every started worker to finish.
    try:
        for seed in config["seeds"]:
            process = context.Process(target=run_seed, args=(config, method, seed))
            process.start()
            processes.append((seed, process))
    finally:
        for _, process in processes:
            process.join()
    failed = [seed for seed, process in processes if process.exitcode != 0]
    if failed:
        raise RuntimeError(
            f"{method} failed for seeds {failed}; inspect retained artifacts"
        )


def main():
    """Check all selected destinations, then run method batches in order."""
    with CONFIG_PATH.open() as stream:
        config = yaml.safe_load(stream)

    # Reject conflicts in later batches before starting the first one.
    for method in METHODS:
        check_campaign(config, method)

    # Each batch finishes training and evaluation before the next one starts.
    for method in METHODS:
        run_campaign(config, method)


if __name__ == "__main__":
    main()
