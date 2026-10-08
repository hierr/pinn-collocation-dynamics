"""Select the project backend before test modules import DeepXDE."""

import pytest
import yaml

import src  # noqa: F401
from src.train import ROOT


@pytest.fixture
def config():
    """Return pilot settings with a small interior population for focused tests."""
    result = yaml.safe_load((ROOT / "config/burgers_pilot.yaml").read_text())
    result["problem"]["interior_points"] = 16
    return result
