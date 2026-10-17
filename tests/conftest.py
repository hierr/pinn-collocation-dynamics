"""Select the project backend before test modules import DeepXDE."""

import numpy as np
import pytest
import yaml

import src  # noqa: F401
from src.train import ROOT


@pytest.fixture
def config(tmp_path):
    """Return small controlled settings and an analytic native-grid test reference."""
    result = yaml.safe_load((ROOT / "config/burgers_template.yaml").read_text())
    result["seeds"] = [1000]
    # Keep ordinary fixtures small; sampler-specific tests select the other methods.
    result["methods"] = {
        name: result["methods"][name] for name in ("fixed_hammersley", "pacmann_adam")
    }
    result["problem"]["interior_points"] = 16
    result["evaluation"] = {
        "solution_nx": 17,
        "residual_nx": 16,
        "residual_nt": 8,
        "batch_size": 32,
    }
    x = np.cos(np.pi * np.arange(17) / 16)
    times = np.array([0.0, 0.495, 0.99, 1.0])
    path = tmp_path / "reference.npz"
    np.savez(
        path,
        x=x,
        t=times,
        U=-np.sin(np.pi * x)[None, :] * np.exp(-times[:, None]),
        nu=result["problem"]["viscosity"],
    )
    result["problem"]["reference_path"] = str(path)
    return result


@pytest.fixture
def all_methods_config(config):
    """Select all five methods with small populations and frequent sampling events."""
    config["methods"] = yaml.safe_load(
        (ROOT / "config/burgers_template.yaml").read_text()
    )["methods"]
    for settings in config["methods"].values():
        if "period" in settings:
            settings["period"] = 2
        if "steps" in settings:
            settings["steps"] = 1
    config["methods"]["rad"].update(candidate_points=64, batch_size=13)
    return config
