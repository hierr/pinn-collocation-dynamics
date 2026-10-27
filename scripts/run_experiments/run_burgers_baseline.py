"""Run one planned Burgers baseline seed before checking evaluation-grid adequacy."""

import yaml

from scripts.run_experiments import run_burgers_campaign as campaign
from src.train import ROOT

CONFIG_PATH = ROOT / "config/burgers/nu_0p001/burgers_nu_0p001_sigma_0.yaml"
METHOD = "pacmann_adam"
SEED = 3000


def main():
    """Train and evaluate only the selected seed, preserving the full campaign plan."""
    config = yaml.safe_load(CONFIG_PATH.read_text())
    # Check only this destination while retaining all planned seeds in the run config.
    campaign.run_campaign(config, METHOD, seeds=[SEED])


if __name__ == "__main__":
    main()
