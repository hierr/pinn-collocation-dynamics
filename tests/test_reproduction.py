"""Verify data ordering, residuals, and equivalence to the imported sampler."""

import ast
import copy
import hashlib
import json
from pathlib import Path
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


def test_reference_ordering_and_checksum(config):
    """Check uniform ordering, interpolated initial values, and the data fingerprint."""
    coordinates, values, fingerprint = load_reference(
        ROOT, config["problem"], config["evaluation"]["solution_nx"]
    )
    np.testing.assert_array_equal(coordinates[:17, 0], np.linspace(-1, 1, 17))
    np.testing.assert_allclose(
        values[:17, 0], -np.sin(np.pi * coordinates[:17, 0]), atol=1e-10
    )
    np.testing.assert_array_equal(coordinates[:17, 1], np.zeros(17))
    assert coordinates.shape == (51, 2)
    assert (
        fingerprint
        == hashlib.sha256(
            Path(config["problem"]["reference_path"]).read_bytes()
        ).hexdigest()
    )


def test_original_reference_retains_upstream_checksum():
    """Verify the retained canonical PACMANN dataset against its immutable manifest."""
    manifest = json.loads(
        (ROOT / "references/pacmann/burgers/manifest.json").read_text()
    )
    item = next(
        item for item in manifest["files"] if item["local_path"].endswith(".npz")
    )
    assert (
        hashlib.sha256(
            (ROOT / "data/burgers_dataset/PACMANN_Burgers.npz").read_bytes()
        ).hexdigest()
        == item["sha256"]
    )


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
    dde_config.set_random_seed(1000)
    dde_config.set_default_float("float64")
    model, residual_gradient = build_model(config["problem"], config["network"])
    constraints = model.data.train_x_bc.copy()
    assert len(model.data.train_x_all) == 16
    original_data = copy.deepcopy(model.data)
    # Recreate the released callback's 480-row prefix only in this parity fixture.
    original_data.train_x = np.vstack((constraints, original_data.train_x))
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
    dde_config.set_random_seed(1000)
    dde_config.set_default_float("float64")
    first, _ = build_model(config["problem"], config["network"])
    dde_config.set_random_seed(1000)
    second, _ = build_model(config["problem"], config["network"])
    sampler = FixedHammersley()
    sampler.set_model(first)
    sampler.on_epoch_end()
    np.testing.assert_array_equal(first.data.train_x_all, second.data.train_x_all)
    np.testing.assert_array_equal(first.data.train_x_bc, second.data.train_x_bc)
    assert len(first.data.train_x_all) == config["problem"]["interior_points"]
    constraints = first.data.train_x_bc
    np.testing.assert_array_equal(first.data.train_x[: len(constraints)], constraints)
    np.testing.assert_array_equal(first.data.test_x[: len(constraints)], constraints)
    for name, value in first.net.state_dict().items():
        torch.testing.assert_close(value, second.net.state_dict()[name], rtol=0, atol=0)
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
