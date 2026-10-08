"""Construct the Burgers problem and load its independent reference data.

Adapted from PACMANN, copyright 2025 C. Visser, A. Heinlein, B. Giovanardi.
Apache-2.0 license: references/pacmann/burgers/LICENSE.
"""

import hashlib
import json
from pathlib import Path

import deepxde as dde
import numpy as np
from deepxde.nn.pytorch import FNN


def load_reference(root: Path):
    """Return reference coordinates, values, and the verified dataset fingerprint."""
    manifest = json.loads(
        (root / "references/pacmann/burgers/manifest.json").read_text()
    )
    item = next(f for f in manifest["files"] if f["local_path"].endswith(".npz"))
    path = root / item["local_path"]
    fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
    if fingerprint != item["sha256"]:
        raise ValueError("Reference dataset checksum mismatch")
    with np.load(path, allow_pickle=False) as data:
        xx, tt = np.meshgrid(data["x"], data["t"])
        coordinates = np.column_stack((xx.ravel(), tt.ravel()))
        values = data["usol"].T.reshape(-1, 1)
    return coordinates, values, fingerprint


def residual(x, y, viscosity: float):
    """Return the viscous Burgers momentum residual."""
    ux = dde.grad.jacobian(y, x, i=0, j=0)
    ut = dde.grad.jacobian(y, x, i=0, j=1)
    uxx = dde.grad.hessian(y, x, i=0, j=0)
    return ut + y * ux - viscosity * uxx


def build_model(
    problem_settings: dict,
    network_settings: dict,
    reference_coordinates,
    reference_values,
):
    """Build the Burgers model and its squared-residual coordinate gradient."""
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
    net = FNN(network_settings["layers"], "tanh", "Glorot normal")
    model = dde.Model(data, net)
    data.test_x, data.test_y = reference_coordinates, reference_values
    return model, residual_gradient
