"""Check upstream resampling parity, RAD probabilities, and fixed constraints."""

import ast
import copy
from types import SimpleNamespace
from typing import Any

import deepxde as dde
import numpy as np
import pytest
from deepxde import config as dde_config

from src.burgers import build_model
from src.sampling import RAD, RandomResampling
from src.train import ROOT


@pytest.fixture
def model(config):
    """Build a small deterministic Burgers model with matched physics points."""
    dde_config.set_random_seed(1000)
    dde_config.set_default_float("float64")
    config["network"]["layers"] = [2, 8, 1]
    result, _ = build_model(config["problem"], config["network"])
    result.opt_name = "adam"
    return result


@pytest.mark.parametrize("method", ["random_resampling", "rad"])
def test_matches_upstream_callback(model, method):
    """Compare two replacement events with the immutable upstream callbacks."""
    data = copy.deepcopy(model.data)
    # Extract only the relevant nested callback, without running upstream training.
    tree = ast.parse(
        (ROOT / "references/pacmann/burgers/PINN_Burgers_function.py").read_text()
    )
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "PINN_Burgers"
    )
    name = "RAD" if method == "rad" else "Random_resampling"
    callback = next(
        node
        for node in function.body
        if isinstance(node, ast.ClassDef) and node.name == name
    )
    namespace: dict[str, Any] = {
        "np": np,
        "dde": dde,
        "data": data,
        "geomtime": data.geom,
        "pde": data.pde,
    }
    exec(  # noqa: S102 - execute only the pinned reference callback for parity
        compile(
            ast.Module(body=[callback], type_ignores=[]), "upstream-callback", "exec"
        ),
        namespace,
    )
    reference = namespace[name](N_col_points=16, period=2)
    # Use a known nonconstant score so this tests weighted selection, not only RNG.
    predict = lambda points, operator: points[:, :1] + 2 * points[:, 1:2] - 0.3
    reference.model = SimpleNamespace(data=data, predict=predict)
    if method == "rad":
        ours = RAD(
            interior_points=16,
            period=2,
            candidate_points=100000,
            k=1,
            c=1,
            batch_size=4096,
        )
    else:
        ours = RandomResampling(interior_points=16, period=2)
    model.predict = predict
    ours.set_model(model)
    constraints = model.data.train_x_bc.copy()

    for _ in range(4):
        random_state = np.random.get_state()
        reference.on_epoch_end()
        np.random.set_state(random_state)
        ours.on_epoch_end()
        np.testing.assert_array_equal(model.data.train_x_all, data.train_x_all)
        np.testing.assert_array_equal(model.data.train_x_bc, constraints)
    assert ours.events == 2


@pytest.mark.parametrize("method", ["random_resampling", "rad"])
def test_cadence_constraints_and_domain_across_phases(model, method):
    """Carry Adam cadence across phases and leave the population fixed in L-BFGS."""
    sampler = (
        RAD(interior_points=16, period=2, candidate_points=64, k=1, c=1, batch_size=13)
        if method == "rad"
        else RandomResampling(interior_points=16, period=2)
    )
    sampler.set_model(model)
    constraints = model.data.train_x_bc.copy()
    previous = model.data.train_x_all.copy()
    for phase in range(2):
        model.opt_name = "adam"
        sampler.on_train_begin()
        for step in range(3):
            sampler.on_epoch_end()
            points = model.data.train_x_all.copy()
            if (phase * 3 + step + 1) % 2:
                np.testing.assert_array_equal(points, previous)
            else:
                assert not np.array_equal(points, previous)
            assert points.shape == (16, 2)
            assert model.data.geom.geometry.inside(points[:, 0]).all()
            assert model.data.geom.timedomain.inside(points[:, 1]).all()
            np.testing.assert_array_equal(model.data.train_x_bc, constraints)
            previous = points
        model.opt_name = "L-BFGS"
        sampler.on_epoch_end()
        np.testing.assert_array_equal(model.data.train_x_all, previous)
    assert sampler.adam_steps == 6
    assert sampler.events == 3


@pytest.mark.parametrize("k,c", [(1, 1), (2, 0)])
def test_rad_global_weights_and_selection_without_replacement(model, monkeypatch, k, c):
    """Verify the declared probability rule across batches and selected indices."""
    candidates = np.column_stack((np.linspace(-0.9, 0.9, 64), np.full(64, 0.5)))
    monkeypatch.setattr(model.data.geom, "random_points", lambda n, random: candidates)
    monkeypatch.setattr(model, "predict", lambda points, operator: points[:, :1])
    scores = np.abs(candidates[:, 0]) ** k
    expected = scores / scores.mean() + c
    expected /= expected.sum()

    def select(a, size, replace, p):
        """Inspect probabilities and return known distinct candidate indices."""
        assert a == 64 and size == 16 and replace is False
        np.testing.assert_allclose(p, expected, rtol=1e-14)
        return np.arange(16)

    monkeypatch.setattr(np.random, "choice", select)
    sampler = RAD(
        interior_points=16, period=1, candidate_points=64, k=k, c=c, batch_size=13
    )
    sampler.set_model(model)
    sampler.on_epoch_end()
    np.testing.assert_array_equal(model.data.train_x_all, candidates[:16])


def test_rad_batched_scoring_matches_unbatched(model):
    """Check real PDE derivatives and seeded selection across prediction batches."""
    selected = []
    for batch_size in (13, 73):
        dde_config.set_random_seed(2000)
        sampler = RAD(
            interior_points=16,
            period=1,
            candidate_points=73,
            k=1,
            c=1,
            batch_size=batch_size,
        )
        sampler.set_model(model)
        sampler.on_epoch_end()
        selected.append(model.data.train_x_all.copy())
    np.testing.assert_array_equal(*selected)


def test_rad_zero_scores_use_uniform_probabilities(model, monkeypatch):
    """Use uniform selection when the residual vanishes on all candidates."""
    monkeypatch.setattr(
        model, "predict", lambda points, operator: np.zeros((len(points), 1))
    )

    def select(a, size, replace, p):
        """Check the zero-field fallback without relying on chance frequencies."""
        np.testing.assert_allclose(p, np.full(a, 1 / a))
        return np.arange(size)

    monkeypatch.setattr(np.random, "choice", select)
    sampler = RAD(
        interior_points=16, period=1, candidate_points=64, k=1, c=1, batch_size=13
    )
    sampler.set_model(model)
    sampler.on_epoch_end()
    assert sampler.events == 1


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_rad_nonfinite_residual_preserves_points(model, monkeypatch, value):
    """Fail before replacement when candidate residual evaluation is invalid."""
    before = model.data.train_x_all.copy()
    monkeypatch.setattr(
        model, "predict", lambda points, operator: np.full((len(points), 1), value)
    )
    sampler = RAD(
        interior_points=16, period=1, candidate_points=64, k=1, c=1, batch_size=13
    )
    sampler.set_model(model)
    with pytest.raises(FloatingPointError, match="candidate residual"):
        sampler.on_epoch_end()
    np.testing.assert_array_equal(model.data.train_x_all, before)
    assert sampler.events == 0


def test_rad_nonfinite_probabilities_preserve_points(model, monkeypatch):
    """Reject overflow in the weighting rule before modifying the population."""
    before = model.data.train_x_all.copy()
    monkeypatch.setattr(
        model, "predict", lambda points, operator: np.full((len(points), 1), 1e200)
    )
    sampler = RAD(
        interior_points=16, period=1, candidate_points=64, k=2, c=1, batch_size=13
    )
    sampler.set_model(model)
    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(FloatingPointError, match="sampling probabilities"),
    ):
        sampler.on_epoch_end()
    np.testing.assert_array_equal(model.data.train_x_all, before)


@pytest.mark.parametrize(
    "field,value",
    [
        ("period", 0),
        ("candidate_points", 15),
        ("batch_size", 0),
        ("k", -1),
        ("c", np.nan),
    ],
)
def test_rad_rejects_invalid_sampling_settings(field, value):
    """Reject settings that invalidate cadence, point budgets, or probabilities."""
    settings = {
        "interior_points": 16,
        "period": 2,
        "candidate_points": 64,
        "k": 1,
        "c": 1,
        "batch_size": 13,
    }
    settings[field] = value
    with pytest.raises(ValueError):
        RAD(**settings)
