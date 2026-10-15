"""Implement fixed, resampled, residual-adaptive, and PACMANN collocation.

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


class RandomResampling(dde.callbacks.Callback):
    """Replace interior physics points uniformly at the Adam-only cadence."""

    def __init__(self, *, interior_points: int, period: int):
        """Initialize the population size and persistent resampling counter."""
        super().__init__()
        if type(period) is not int or period <= 0:
            raise ValueError("Resampling period must be a positive integer")
        self.interior_count = interior_points
        self.period = period
        self.adam_steps = 0
        self.events = 0

    def on_epoch_end(self):
        """Replace the interior population after each configured Adam interval."""
        model = self.model
        if model is None:
            raise RuntimeError("Callback is not attached to a model")
        if model.opt_name != "adam":
            return
        self.adam_steps += 1
        if self.adam_steps % self.period:
            return

        # Match upstream's fresh pseudorandom uniform interior population.
        data = model.data
        constraints = data.train_x_bc.copy()
        points = data.geom.random_points(self.interior_count, random="pseudo")
        data.replace_with_anchors(points)
        if not np.array_equal(data.train_x_bc, constraints):
            raise RuntimeError("Constraint coordinates changed during resampling")
        self.events += 1


class RAD(dde.callbacks.Callback):
    """Select a replacement physics population using residual-based probabilities."""

    def __init__(
        self,
        *,
        interior_points: int,
        period: int,
        candidate_points: int,
        k: float,
        c: float,
        batch_size: int,
    ):
        """Initialize candidate scoring, weighted selection, and the Adam counter."""
        super().__init__()
        if type(period) is not int or period <= 0:
            raise ValueError("Resampling period must be a positive integer")
        if type(candidate_points) is not int or candidate_points < interior_points:
            raise ValueError("RAD candidate count must cover the interior population")
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("RAD batch size must be a positive integer")
        if not np.isfinite([k, c]).all() or k < 0 or c < 0:
            raise ValueError("RAD k and c must be finite and nonnegative")
        self.interior_count = interior_points
        self.period = period
        self.candidate_points = candidate_points
        self.k, self.c = k, c
        self.batch_size = batch_size
        self.adam_steps = 0
        self.events = 0

    def on_epoch_end(self):
        """Score fresh candidates and replace points by weighted sampling."""
        model = self.model
        if model is None:
            raise RuntimeError("Callback is not attached to a model")
        if model.opt_name != "adam":
            return
        self.adam_steps += 1
        if self.adam_steps % self.period:
            return
        data = model.data
        constraints = data.train_x_bc.copy()

        # Score all candidates in bounded batches before normalizing globally.
        candidates = data.geom.random_points(self.candidate_points, random="pseudo")
        residuals = (
            np.concatenate(
                [
                    model.predict(
                        candidates[start : start + self.batch_size], operator=data.pde
                    )
                    for start in range(0, self.candidate_points, self.batch_size)
                ]
            )
            .astype(np.float64)
            .ravel()
        )
        if not np.isfinite(residuals).all():
            raise FloatingPointError("Non-finite RAD candidate residual")
        scores = np.abs(residuals) ** self.k
        mean_score = scores.mean()
        # A zero residual field has no preferred region; avoid upstream's 0/0.
        weights = (
            np.ones_like(scores) if mean_score == 0 else scores / mean_score + self.c
        )
        probabilities = weights / weights.sum()
        if not np.isfinite(probabilities).all() or (probabilities < 0).any():
            raise FloatingPointError("Invalid RAD sampling probabilities")

        # Keep the point budget fixed and sample candidate indices without replacement.
        indices = np.random.choice(
            len(candidates), self.interior_count, replace=False, p=probabilities
        )
        data.replace_with_anchors(candidates[indices])
        if not np.array_equal(data.train_x_bc, constraints):
            raise RuntimeError("Constraint coordinates changed during resampling")
        self.events += 1


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


class PacmannDiffusion(dde.callbacks.Callback):
    """Add domain-normalized Gaussian noise to PACMANN coordinate ascent."""

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
        sigma: float,
        domain_widths,
        seed: int,
    ):
        """Initialize PACMANN settings and an independent persistent noise stream."""
        super().__init__()
        widths = np.asarray(domain_widths, dtype=float)
        if not np.isfinite(sigma) or sigma < 0:
            raise ValueError("Sigma must be finite and nonnegative")
        if widths.shape != (2,) or not np.isfinite(widths).all() or (widths <= 0).any():
            raise ValueError("Domain widths must be two positive finite values")
        if type(coordinate_steps) is not int or coordinate_steps <= 0:
            raise ValueError("Coordinate steps must be a positive integer")
        self.interior_count = interior_points
        self.residual_gradient = residual_gradient
        self.period = period
        self.coordinate_steps = coordinate_steps
        self.step_size = step_size
        self.beta1, self.beta2 = betas
        self.epsilon = epsilon
        self.sigma = sigma
        self.noise_scale = sigma * widths / np.sqrt(coordinate_steps)
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
        self.adam_steps = 0
        self.events = 0

    def on_epoch_end(self):
        """Apply noisy coordinate ascent, then uniformly replace escaped points."""
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

        # Keep fresh coordinate-Adam moments for each movement event.
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

            # Sigma scales the accumulated noise component over K coordinate steps.
            if self.sigma > 0:
                points += self.noise_scale * self.rng.standard_normal(points.shape)

        # Preserve upstream replacement timing; never project or move constraints.
        inside = np.logical_and(
            data.geom.geometry.inside(points[:, 0]),
            data.geom.timedomain.inside(points[:, 1]),
        )
        points[~inside] = data.geom.random_points(int((~inside).sum()), random="pseudo")
        data.replace_with_anchors(points)
        if not np.array_equal(data.train_x_bc, constraints):
            raise RuntimeError("Constraint coordinates changed during movement")
        self.events += 1
