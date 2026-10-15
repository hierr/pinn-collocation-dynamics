"""Select the project backend before test modules import DeepXDE."""

import pytest
import yaml

import src  # noqa: F401
from src.train import ROOT


@pytest.fixture
def config():
    """Return template settings with a small interior population for tests."""
    result = yaml.safe_load((ROOT / "config/burgers_template.yaml").read_text())
    result["seeds"] = [1000]
    result["problem"]["interior_points"] = 16
    return result
