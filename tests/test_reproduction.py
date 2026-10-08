"""Verify data ordering, residuals, and equivalence to the imported sampler."""

import ast
import copy
from types import SimpleNamespace
from typing import Any

import deepxde as dde
import numpy as np
import pytest
import torch
from deepxde import config as dde_config

import src  # noqa: F401
from src.burgers import build_model, load_reference, residual
from src.sampling import FixedHammersley, PacmannAdam
from src.train import ROOT


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


@pytest.mark.parametrize("step_size", [1e-5, 2.0])
def test_sampler_matches_imported_callback(config, step_size):
    """Compare our callback directly with upstream over two events and L-BFGS."""
    x, y, _ = load_reference(ROOT)
    dde_config.set_random_seed(1000)
    dde_config.set_default_float("float64")
    model, residual_gradient = build_model(config["problem"], config["network"], x, y)
    constraints = model.data.train_x_bc.copy()
    assert len(model.data.train_x_all) == 256
    original_data = copy.deepcopy(model.data)
    settings = config["methods"]["pacmann_adam"]
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
        "residual_gradient_operator": residual_gradient,
    }
    exec(  # noqa: S102 - execute only the pinned reference callback for parity
        compile(
            ast.Module(body=[callback], type_ignores=[]), "upstream-callback", "exec"
        ),
        namespace,
    )
    reference = namespace["Adam"](15, 16, step_size, period=1)
    predict = lambda points, operator=None: 0.2 * points + 0.1
    reference.model = SimpleNamespace(data=original_data, predict=predict)
    ours = PacmannAdam(
        interior_points=16,
        residual_gradient=residual_gradient,
        period=1,
        coordinate_steps=settings["steps"],
        step_size=step_size,
        betas=settings["betas"],
        epsilon=settings["epsilon"],
    )
    ours.model = SimpleNamespace(data=model.data, predict=predict, opt_name="adam")
    for _ in range(2):
        # Give both callbacks the same random draws when points escape the domain.
        random_state = np.random.get_state()
        reference.on_epoch_end()
        np.random.set_state(random_state)
        ours.on_epoch_end()
        np.testing.assert_array_equal(model.data.train_x_all, original_data.train_x_all)
        np.testing.assert_array_equal(model.data.train_x_bc, constraints)
        assert len(model.data.train_x_all) == 16
    previous = model.data.train_x_all.copy()
    ours.model.opt_name = "L-BFGS"
    ours.on_epoch_end()
    assert ours.events == 2
    np.testing.assert_array_equal(model.data.train_x_all, previous)


def test_fixed_hammersley_preserves_seeded_points(config):
    """Verify seeded construction and the fixed callback preserve point populations."""
    x, y, _ = load_reference(ROOT)
    dde_config.set_random_seed(1000)
    dde_config.set_default_float("float64")
    first, _ = build_model(config["problem"], config["network"], x, y)
    dde_config.set_random_seed(1000)
    second, _ = build_model(config["problem"], config["network"], x, y)
    sampler = FixedHammersley()
    sampler.set_model(first)
    sampler.on_epoch_end()
    np.testing.assert_array_equal(first.data.train_x_all, second.data.train_x_all)
    np.testing.assert_array_equal(first.data.train_x_bc, second.data.train_x_bc)
    assert sampler.events == 0


def test_nonfinite_coordinate_gradient_is_rejected():
    """Reject an invalid gradient before it corrupts the physics population."""
    data = SimpleNamespace(
        train_x_all=np.array([[0.2, 0.3]]), train_x_bc=np.empty((0, 2))
    )
    sampler = PacmannAdam(
        interior_points=1,
        residual_gradient=None,
        period=1,
        coordinate_steps=1,
        step_size=1e-5,
        betas=(0.9, 0.999),
        epsilon=1e-7,
    )
    sampler.model = SimpleNamespace(
        data=data,
        opt_name="adam",
        predict=lambda points, operator: np.full_like(points, np.nan),
    )
    before = data.train_x_all.copy()
    with pytest.raises(FloatingPointError, match="coordinate gradient"):
        sampler.on_epoch_end()
    np.testing.assert_array_equal(data.train_x_all, before)
