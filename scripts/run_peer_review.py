"""Train and evaluate one short Burgers demo for the peer review."""

import os

# Limit numerical threads before importing the scientific modules.
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"

import torch
import yaml

from scripts.post_processing.analyze_burgers import analyze_run
from src.train import ROOT, run_training

CONFIG_PATH = ROOT / "config/peer_review.yaml"
METHOD = "pacmann_diffusion"


def main():
    """Read the demo configuration, train one method and seed, then evaluate it."""
    config = yaml.safe_load(CONFIG_PATH.read_text())
    run = run_training(config, METHOD, config["seeds"][0])
    analyze_run(run)
    print(f"Review demonstration saved in {run.relative_to(ROOT)}")


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    main()
