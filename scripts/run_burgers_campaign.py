"""Train and analyze each Burgers seed and method in the selected YAML."""

import yaml

from scripts.analyze_burgers import analyze_run
from src.train import ROOT, run_training

# Select the campaign here; scientific settings remain in the YAML file.
CONFIG_PATH = ROOT / "config/burgers_pilot.yaml"


def check_campaign(config: dict):
    """Check selectors and all run destinations without creating artifacts."""
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
    if not methods or any(
        method not in ("fixed_hammersley", "pacmann_adam", "pacmann_diffusion")
        for method in methods
    ):
        raise ValueError("Methods must select supported Burgers sampling methods")

    # Allow existing campaigns, but reject any conflict before training begins.
    for seed in seeds:
        for method in methods:
            run_dir = ROOT / "output" / campaign / "burgers" / method / f"seed_{seed}"
            if run_dir.exists():
                raise FileExistsError(f"Run already exists: {run_dir}")


def run_campaign(config: dict):
    """Run configured pairs sequentially, stopping on failure without overwriting."""
    check_campaign(config)

    # Train each pair in YAML order and generate its diagnostics immediately.
    for seed in config["seeds"]:
        for method in config["methods"]:
            print(f"Training {config['campaign']}: {method}, seed {seed}", flush=True)
            run_dir = run_training(config, method, seed)
            print(f"Analyzing {run_dir}", flush=True)
            analyze_run(run_dir)
            print(f"Completed {method}, seed {seed}: {run_dir}", flush=True)


def main():
    """Load the configuration selected above and execute its campaign."""
    with CONFIG_PATH.open() as stream:
        config = yaml.safe_load(stream)
    run_campaign(config)


if __name__ == "__main__":
    main()
