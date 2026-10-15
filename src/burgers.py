"""Construct the Burgers problem and load its independent reference data.

Adapted from PACMANN, copyright 2025 C. Visser, A. Heinlein, B. Giovanardi.
Apache-2.0 license: references/pacmann/burgers/LICENSE.
"""

import hashlib
from pathlib import Path

import deepxde as dde
import numpy as np
from deepxde.nn.pytorch import FNN
from scipy.interpolate import BarycentricInterpolator


def load_reference(root: Path, problem_settings: dict, solution_nx: int):
    """Return uniform-grid coordinates, interpolated values, and data fingerprint."""
    if type(solution_nx) is not int or solution_nx < 2:
        raise ValueError("Solution grid requires at least two spatial points")
    path = root / problem_settings["reference_path"]
    fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path, allow_pickle=False) as data:
        x, times, solution = data["x"].ravel(), data["t"].ravel(), data["U"]
        viscosity = float(data["nu"])
    if solution.shape != (len(times), len(x)) or not all(
        np.isfinite(array).all() for array in (x, times, solution)
    ):
        raise ValueError("Reference shape or values do not match its coordinates")
    if (
        not np.isfinite(viscosity)
        or viscosity <= 0
        or not np.isclose(viscosity, problem_settings["viscosity"], rtol=1e-12, atol=0)
    ):
        raise ValueError("Reference viscosity differs from the configured PDE")

    # Validate native values while sorting spatial nodes and selecting exact times.
    order = np.argsort(x)
    x, solution = x[order], solution[:, order]
    xmin, xmax = problem_settings["x_bounds"]
    tmin, tmax = problem_settings["t_bounds"]
    if len(x) < 2 or len(times) < 2 or xmax <= xmin or tmax <= tmin:
        raise ValueError("Reference and PDE domain must have positive extent")
    expected = (xmin + xmax) / 2 - (xmax - xmin) / 2 * np.cos(
        np.pi * np.arange(len(x)) / (len(x) - 1)
    )
    if not np.allclose(x, expected, rtol=0, atol=1e-12):
        raise ValueError("Reference must use Chebyshev-Lobatto nodes on the PDE domain")
    if np.any(np.diff(times) <= 0):
        raise ValueError("Reference times must be strictly increasing")
    selected = (times >= tmin) & (times <= tmax)
    times, solution = times[selected], solution[selected]
    if len(times) < 2 or not np.allclose(
        times[[0, -1]], [tmin, tmax], rtol=0, atol=1e-12
    ):
        raise ValueError("Reference must include the configured time endpoints")
    if not np.allclose(np.diff(times), times[1] - times[0], rtol=1e-10, atol=1e-12):
        raise ValueError("Reference evaluation times must be uniformly spaced")
    if not np.allclose(solution[0], -np.sin(np.pi * x), rtol=1e-10, atol=1e-12):
        raise ValueError("Reference initial condition differs from the configured PDE")

    # Interpolate in space using analytical Chebyshev-Lobatto barycentric coefficients.
    coefficients = (-1.0) ** np.arange(len(x))
    coefficients[[0, -1]] *= 0.5
    uniform_x = np.linspace(xmin, xmax, solution_nx)
    solution = BarycentricInterpolator(x, solution, axis=1, wi=coefficients)(uniform_x)
    if not np.isfinite(solution).all():
        raise FloatingPointError("Non-finite interpolated reference")
    xx, tt = np.meshgrid(uniform_x, times)
    coordinates = np.column_stack((xx.ravel(), tt.ravel()))
    return coordinates, solution.reshape(-1, 1), fingerprint


def residual(x, y, viscosity: float):
    """Return the viscous Burgers momentum residual."""
    ux = dde.grad.jacobian(y, x, i=0, j=0)
    ut = dde.grad.jacobian(y, x, i=0, j=1)
    uxx = dde.grad.hessian(y, x, i=0, j=0)
    return ut + y * ux - viscosity * uxx


def build_model(
    problem_settings: dict,
    network_settings: dict,
):
    """Build the Burgers model and its squared-residual coordinate gradient."""
    if (
        type(problem_settings["interior_points"]) is not int
        or problem_settings["interior_points"] <= 0
    ):
        raise ValueError("Interior PDE point count must be a positive integer")
    domain = dde.geometry.GeometryXTime(
        dde.geometry.Interval(*problem_settings["x_bounds"]),
        dde.geometry.TimeDomain(*problem_settings["t_bounds"]),
    )

    def pde(x, y):
        """Return the configured Burgers residual."""
        return residual(x, y, problem_settings["viscosity"])

    def residual_gradient(x, y):
        """Return the gradient of squared residual with respect to coordinates."""
        return dde.grad.jacobian(pde(x, y) ** 2, x, i=0)

    # Keep the upstream soft initial and boundary constraint losses.
    bc = dde.icbc.DirichletBC(domain, lambda x: 0, lambda _, boundary: boundary)
    ic = dde.icbc.IC(
        domain, lambda x: -np.sin(np.pi * x[:, :1]), lambda _, initial: initial
    )
    data = dde.data.TimePDE(
        domain,
        pde,
        [bc, ic],
        num_domain=problem_settings["interior_points"],
        num_boundary=problem_settings["boundary_points"],
        num_initial=problem_settings["initial_points"],
        num_test=10000,
        train_distribution="Hammersley",
    )
    # Keep constraints cached while matching the interior PDE population from step zero.
    data.replace_with_anchors(
        data.train_points()[-problem_settings["interior_points"] :].copy()
    )
    net = FNN(network_settings["layers"], "tanh", "Glorot normal")
    model = dde.Model(data, net)
    return model, residual_gradient
