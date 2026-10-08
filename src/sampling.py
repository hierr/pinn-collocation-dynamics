"""Implement the released PACMANN coordinate-Adam movement.

Adapted from PACMANN, copyright 2025 C. Visser, A. Heinlein, B. Giovanardi.
Apache-2.0 license: references/pacmann/burgers/LICENSE.
"""

import deepxde as dde
import numpy as np


def interior_points(data, count: int):
    """Return the initial interior tail or the replacement physics population."""
    return data.train_x_all[-count:].copy()


class FixedHammersley(dde.callbacks.Callback):
    """Keep the model's initial Hammersley population fixed throughout training."""

    def __init__(self):
        """Initialize the fixed method with zero movement events."""
        super().__init__()
        self.events = 0


class PacmannAdam(dde.callbacks.Callback):
    """Move interior physics points at the configured Adam-only cadence."""

    def __init__(
        self,
        *,
        interior_points: int,
        residual_gradient,
        period: int,
        coordinate_steps: int,
        step_size: float,
        betas,
        epsilon: float,
    ):
        """Initialize movement settings and the persistent cadence counter."""
        super().__init__()
        self.interior_count = interior_points
        self.residual_gradient = residual_gradient
        self.period = period
        self.coordinate_steps = coordinate_steps
        self.step_size = step_size
        self.beta1, self.beta2 = betas
        self.epsilon = epsilon
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
        if self.adam_steps % self.period:
            return
        data = model.data
        points = interior_points(data, self.interior_count)
        constraints = data.train_x_bc.copy()

        # Ascend squared residual with fresh coordinate-Adam moments for this event.
        first, second = np.zeros_like(points), np.zeros_like(points)
        for step in range(self.coordinate_steps):
            residual_gradient = model.predict(points, operator=self.residual_gradient)
            if not np.isfinite(residual_gradient).all():
                raise FloatingPointError("Non-finite coordinate gradient")
            first = self.beta1 * first + (1 - self.beta1) * residual_gradient
            second = self.beta2 * second + (1 - self.beta2) * residual_gradient**2
            corrected_first = first / (1 - self.beta1 ** (step + 1))
            corrected_second = second / (1 - self.beta2 ** (step + 1))
            points += (
                self.step_size
                * corrected_first
                / (np.sqrt(corrected_second) + self.epsilon)
            )

        # Replace only after all coordinate steps, preserving upstream timing.
        inside = np.logical_and(
            data.geom.geometry.inside(points[:, 0]),
            data.geom.timedomain.inside(points[:, 1]),
        )
        points[~inside] = data.geom.random_points(int((~inside).sum()), random="pseudo")
        data.replace_with_anchors(points)
        if not np.array_equal(data.train_x_bc, constraints):
            raise RuntimeError("Constraint coordinates changed during movement")
        self.events += 1
