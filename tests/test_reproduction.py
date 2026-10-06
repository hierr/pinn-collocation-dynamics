"""Verify data ordering, residuals, and equivalence to the imported sampler."""

import ast
import copy
from types import SimpleNamespace
from typing import Any

import deepxde as dde
import numpy as np
import pytest
import torch
import yaml

import src  # noqa: F401
from src.burgers import build_model, load_reference, residual
from src.sampling import PacmannAdam
from src.train import ROOT, create_run


@pytest.fixture
def config():
    """Return a small point population with the reproduction's constraints."""
    result = yaml.safe_load((ROOT / "config/burgers_validation.yaml").read_text())
    result["problem"]["interior_points"] = 16
    return result


def test_reference_ordering_and_checksum(tmp_path):
    """Check flattening against the original grid and reject altered data."""
    x, y, fingerprint = load_reference(ROOT)
    with np.load(ROOT / "data/burgers/Burgers.npz") as original:
        np.testing.assert_array_equal(x[:256, 0], original["x"].ravel())
        np.testing.assert_array_equal(y[:256, 0], original["usol"][:, 0])
        np.testing.assert_array_equal(x[:256, 1], np.zeros(256))
    assert len(fingerprint) == 64
    (tmp_path / "references/pacmann/burgers").mkdir(parents=True)
    (tmp_path / "data/burgers").mkdir(parents=True)
    (tmp_path / "references/pacmann/burgers/manifest.json").write_bytes(
        (ROOT / "references/pacmann/burgers/manifest.json").read_bytes()
    )
    (tmp_path / "data/burgers/Burgers.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        load_reference(tmp_path)


def test_residual_derivatives():
    """Check nonlinear transport, temporal, and viscous derivatives analytically."""
    x = torch.tensor([[0.2, 0.3], [-0.4, 0.8]], dtype=torch.float64, requires_grad=True)
    y = x[:, :1] ** 2 + x[:, 1:2]
    actual = residual(x, y, 0.1)
    expected = 1 + y * (2 * x[:, :1]) - 0.2
    torch.testing.assert_close(actual, expected)
    dde.grad.clear()


def test_sampler_matches_imported_callback(config):
    """Compare our callback directly with upstream over two events and L-BFGS."""
    x, y, _ = load_reference(ROOT)
    model, operator = build_model(config, 1000, x, y)
    constraints = model.data.train_x_bc.copy()
    assert len(model.data.train_x_all) == 256
    original_data = copy.deepcopy(model.data)
    settings = dict(config["methods"]["pacmann_adam"], period=1)
    # Compile only the nested upstream class, without executing its training function.
    tree = ast.parse(
        (ROOT / "references/pacmann/burgers/PINN_Burgers_function.py").read_text()
    )
    function = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "PINN_Burgers"
    )
    callback = next(
        n for n in function.body if isinstance(n, ast.ClassDef) and n.name == "Adam"
    )
    namespace: dict[str, Any] = {
        "np": np,
        "dde": dde,
        "data": original_data,
        "geom": dde.geometry.Interval(-1, 1),
        "timedomain": dde.geometry.TimeDomain(0, 0.99),
        "geomtime": original_data.geom,
        "residual_gradient_operator": operator,
    }
    exec(  # noqa: S102 - execute only the pinned reference callback for parity
        compile(
            ast.Module(body=[callback], type_ignores=[]), "upstream-callback", "exec"
        ),
        namespace,
    )
    reference = namespace["Adam"](15, 16, 1e-5, period=1)
    predict = lambda points, operator=None: 0.2 * points + 0.1
    reference.model = SimpleNamespace(data=original_data, predict=predict)
    ours = PacmannAdam(settings, 16, operator)
    ours.model = SimpleNamespace(data=model.data, predict=predict, opt_name="adam")
    for _ in range(2):
        reference.on_epoch_end()
        ours.on_epoch_end()
        np.testing.assert_array_equal(model.data.train_x_all, original_data.train_x_all)
        np.testing.assert_array_equal(model.data.train_x_bc, constraints)
        assert len(model.data.train_x_all) == 16
    previous = model.data.train_x_all.copy()
    ours.model.opt_name = "L-BFGS"
    ours.on_epoch_end()
    assert ours.events == 2
    np.testing.assert_array_equal(model.data.train_x_all, previous)


def test_fixed_points_and_run_reservation(config, tmp_path):
    """Verify seeded fixed sampling and non-overwriting output paths."""
    x, y, _ = load_reference(ROOT)
    first, _ = build_model(config, 1000, x, y)
    second, _ = build_model(config, 1000, x, y)
    np.testing.assert_array_equal(first.data.train_x_all, second.data.train_x_all)
    path, seed = create_run(config, "fixed_hammersley", 0, tmp_path)
    assert seed == 1000 and path.exists()
    with pytest.raises(FileExistsError):
        create_run(config, "fixed_hammersley", 0, tmp_path)
    with pytest.raises(ValueError):
        create_run(config, "fixed_hammersley", -1, tmp_path)
