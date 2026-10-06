"""Implement the released PACMANN coordinate-Adam movement.

Adapted from PACMANN, copyright 2025 C. Visser, A. Heinlein, B. Giovanardi.
Apache-2.0 license: references/pacmann/burgers/LICENSE.
"""

import deepxde as dde
import numpy as np


def move_adam(points, gradient, settings: dict):
    """Return points after coordinate ascent with freshly initialized moments."""
    moved = points.copy()
    first, second = np.zeros_like(moved), np.zeros_like(moved)
    beta1, beta2 = settings["betas"]
    # Match upstream NumPy updates, including epsilon outside the square root.
    for step in range(settings["steps"]):
        grad = gradient(moved)
        if not np.isfinite(grad).all():
            raise FloatingPointError("Non-finite coordinate gradient")
        first = beta1 * first + (1 - beta1) * grad
        second = beta2 * second + (1 - beta2) * grad**2
        corrected_first = first / (1 - beta1 ** (step + 1))
        corrected_second = second / (1 - beta2 ** (step + 1))
        moved += (
            settings["step_size"]
            * corrected_first
            / (np.sqrt(corrected_second) + settings["epsilon"])
        )
    return moved


def interior_points(data, count: int):
    """Return the initial interior tail or the replacement physics population."""
    return data.train_x_all[-count:].copy()


class PacmannAdam(dde.callbacks.Callback):
    """Move interior physics points at the configured Adam-only cadence."""

    def __init__(self, settings: dict, count: int, operator):
        """Initialize movement settings and the persistent cadence counter."""
        super().__init__()
        self.settings, self.count, self.operator = settings, count, operator
        self.adam_steps = 0
        self.events = 0

    def on_epoch_end(self):
        """Apply coordinate ascent and then uniformly replace escaped points."""
        model = self.model
        if model is None:
            raise RuntimeError("Callback is not attached to a model")
        if model.opt_name != "adam":
            return
        self.adam_steps += 1
        if self.adam_steps % self.settings["period"]:
            return
        data = model.data
        points = interior_points(data, self.count)
        constraints = data.train_x_bc.copy()
        moved = move_adam(
            points,
            lambda x: model.predict(x, operator=self.operator),
            self.settings,
        )
        # Replace only after all coordinate steps, preserving upstream timing.
        inside = np.logical_and(
            data.geom.geometry.inside(moved[:, 0]),
            data.geom.timedomain.inside(moved[:, 1]),
        )
        moved[~inside] = data.geom.random_points(int((~inside).sum()), random="pseudo")
        data.replace_with_anchors(moved)
        if not np.array_equal(data.train_x_bc, constraints):
            raise RuntimeError("Constraint coordinates changed during movement")
        self.events += 1
