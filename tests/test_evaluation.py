"""Check spatial interpolation and independent final errors against analytic values."""

from pathlib import Path
from types import SimpleNamespace

import deepxde as dde
import numpy as np
import pytest
import torch

from scripts.post_processing.analyze_burgers import (
    evaluate_final,
    predict_batches,
    residual_rms,
    solution_errors,
)
from src.burgers import build_model, load_reference, residual


def analytic_model(solution, viscosity):
    """Return an analytic predictor using the real autograd PDE operator."""

    def predict(coordinates, operator=None):
        """Evaluate the manufactured solution or differentiate it through PyTorch."""
        inputs = torch.tensor(coordinates, dtype=torch.float64, requires_grad=True)
        outputs = solution(inputs)
        if operator is not None:
            outputs = operator(inputs, outputs)
        values = outputs.detach().cpu().numpy()
        dde.grad.clear()
        return values

    return SimpleNamespace(
        predict=predict,
        data=SimpleNamespace(pde=lambda x, y: residual(x, y, viscosity)),
    )


@pytest.mark.parametrize("count", [16, 17])
def test_reference_polynomial_interpolation(config, count):
    """Recover polynomial snapshots exactly on a different uniform spatial grid."""
    x = np.cos(np.pi * np.arange(count) / (count - 1))
    times = np.array([0.0, 0.495, 0.99, 1.0])
    solution = np.array([-np.sin(np.pi * x), x**2, x**4, x**3])
    np.savez(
        config["problem"]["reference_path"],
        x=x,
        t=times,
        U=solution,
        nu=config["problem"]["viscosity"],
    )
    coordinates, values, _ = load_reference(Path.cwd(), config["problem"], 19)
    uniform_x = np.linspace(-1, 1, 19)
    np.testing.assert_array_equal(
        coordinates[:, 0].reshape(3, 19), np.tile(uniform_x, (3, 1))
    )
    np.testing.assert_allclose(values.reshape(3, 19)[1], uniform_x**2, atol=1e-14)
    np.testing.assert_allclose(values.reshape(3, 19)[2], uniform_x**4, atol=1e-14)


def test_ordinary_solution_errors():
    """Give equally weighted solution MSE and relative L2 for known errors."""
    result = solution_errors(np.array([[3.0], [4.0]]), np.array([[3.0], [5.0]]))
    assert result == {"final_mse": 0.5, "final_relative_l2": 0.2}
    with pytest.raises(ValueError, match="zero reference"):
        solution_errors(np.zeros((2, 1)), np.ones((2, 1)))


def test_reference_uniform_grid_and_crop(config):
    """Preserve exact times in time-major order and exclude the outside snapshot."""
    coordinates, values, _ = load_reference(
        Path.cwd(), config["problem"], config["evaluation"]["solution_nx"]
    )
    assert coordinates.shape == (51, 2) and values.shape == (51, 1)
    np.testing.assert_array_equal(coordinates[:, 0], np.tile(np.linspace(-1, 1, 17), 3))
    np.testing.assert_array_equal(coordinates[:, 1], np.repeat([0, 0.495, 0.99], 17))
    expected = -np.sin(np.pi * coordinates[:, :1]) * np.exp(-coordinates[:, 1:2])
    np.testing.assert_allclose(values, expected, atol=1e-10)


@pytest.mark.parametrize(
    "change",
    [
        "shape",
        "finite",
        "viscosity",
        "nodes",
        "times",
        "nonuniform_times",
        "domain",
        "endpoints",
        "initial",
    ],
)
def test_reference_mismatch_rejected(config, change):
    """Reject data that would silently alter the PDE, domain, or evaluation grid."""
    path = Path(config["problem"]["reference_path"])
    with np.load(path) as source:
        arrays = dict(source)
    if change == "shape":
        arrays["U"] = arrays["U"].T
    elif change == "finite":
        arrays["U"][0, 0] = np.nan
    elif change == "viscosity":
        arrays["nu"] = 0.1
    elif change == "nodes":
        arrays["x"][3] += 0.01
    elif change == "times":
        arrays["t"][1] = arrays["t"][0]
    elif change == "nonuniform_times":
        arrays["t"][1] += 0.01
    elif change == "domain":
        config["problem"]["x_bounds"] = [-2.0, 2.0]
    elif change == "endpoints":
        config["problem"]["t_bounds"][1] = 0.9
    else:
        arrays["U"][0] += 0.01
    np.savez(path, **arrays)
    with pytest.raises(ValueError):
        load_reference(
            Path.cwd(), config["problem"], config["evaluation"]["solution_nx"]
        )


def test_residual_rms_and_batched_interior_points(config):
    """Integrate a manufactured Burgers residual using autograd on midpoint cells."""
    nu = config["problem"]["viscosity"]
    model = analytic_model(lambda x: x[:, :1] ** 2 + x[:, 1:2], nu)
    actual = residual_rms(model, config["problem"], 128, 128, 127)
    expected = np.sqrt((1 - 2 * nu) ** 2 + 4 / 7 + 4 * 0.99 / 5 + 4 * 0.99**2 / 9)
    np.testing.assert_allclose(actual, expected, rtol=1e-4)

    # Midpoints stay strictly inside the domain and do not depend on collocation points.
    visited = []
    original_predict = model.predict

    def record(coordinates, operator=None):
        """Record each batch before applying the analytic predictor."""
        visited.append(coordinates.copy())
        return original_predict(coordinates, operator)

    model.predict = record
    residual_rms(model, config["problem"], 5, 3, 4)
    points = np.concatenate(visited)
    assert len(points) == 15 and max(map(len, visited)) == 4
    assert ((points[:, 0] > -1) & (points[:, 0] < 1)).all()
    assert ((points[:, 1] > 0) & (points[:, 1] < 0.99)).all()


def test_zero_residual_does_not_imply_initial_condition_accuracy(config):
    """A zero Burgers solution has zero residual but violates the sine initial data."""
    coordinates, values, _ = load_reference(
        Path.cwd(), config["problem"], config["evaluation"]["solution_nx"]
    )
    model = analytic_model(lambda x: 0 * x[:, :1] ** 2 + 0 * x[:, 1:2], 0.01 / np.pi)
    prediction, metrics = evaluate_final(
        model, coordinates, values, config["problem"], config["evaluation"]
    )
    np.testing.assert_array_equal(prediction, np.zeros_like(values))
    assert metrics["final_relative_l2"] == 1
    assert metrics["final_residual_rms"] == metrics["boundary_rmse"] == 0
    np.testing.assert_allclose(metrics["initial_rmse"], np.sqrt(8 / 17), rtol=1e-12)


def test_constraint_rmse_uses_equal_grid_weights(config):
    """Check boundary-time and initial-space averages with a known predictor."""
    coordinates, values, _ = load_reference(
        Path.cwd(), config["problem"], config["evaluation"]["solution_nx"]
    )
    model = analytic_model(
        lambda x: x[:, :1] ** 2 + x[:, 1:2], config["problem"]["viscosity"]
    )
    _, metrics = evaluate_final(
        model, coordinates, values, config["problem"], config["evaluation"]
    )
    times = np.array([0, 0.495, 0.99])
    x = np.linspace(-1, 1, 17)
    np.testing.assert_allclose(
        metrics["boundary_rmse"], np.sqrt(np.mean((1 + times) ** 2))
    )
    np.testing.assert_allclose(
        metrics["initial_rmse"], np.sqrt(np.mean((x**2 + np.sin(np.pi * x)) ** 2))
    )


@pytest.mark.parametrize("invalid_nx", [1, 2.5])
def test_invalid_solution_resolution(config, invalid_nx):
    """Require an integer spatial resolution with both endpoints."""
    with pytest.raises(ValueError, match="at least two"):
        load_reference(Path.cwd(), config["problem"], invalid_nx)


def test_invalid_evaluation_boundaries(config):
    """Reject an invalid interior point count and non-finite predictions."""
    config["problem"]["interior_points"] = 0
    with pytest.raises(ValueError, match="point count"):
        build_model(config["problem"], config["network"])
    model = analytic_model(lambda x: x[:, :1] ** 2 + x[:, 1:2], 0.01 / np.pi)
    model.predict = lambda coordinates, operator=None: np.full(
        (len(coordinates), 1), np.nan
    )
    with pytest.raises(FloatingPointError):
        predict_batches(model, np.zeros((1, 2)), 1)
