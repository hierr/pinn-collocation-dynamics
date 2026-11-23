"""Verify local noise attenuation, coordinate dynamics, and saved-run integration."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from deepxde import config as dde_config

from scripts.post_processing.analyze_burgers import analyze_run
from src import train
from src.burgers import build_model
from src.sampling import PacmannAdam, PacmannDiffusion, PacmannSharpnessDiffusion


def make_sampler(**overrides):
    """Return a small controller with an explicit gradient scale and noise stream."""
    settings = {
        "interior_points": 4,
        "residual_gradient": None,
        "period": 1,
        "coordinate_steps": 3,
        "step_size": 1e-5,
        "betas": (0.9, 0.999),
        "epsilon": 1e-7,
        "sigma": 1e-3,
        "alpha": 2.0,
        "g_base": np.pi,
        "domain_widths": (2.0, 0.99),
        "seed": 2000,
    }
    settings.update(overrides)
    return PacmannSharpnessDiffusion(**settings)


def attach_sampler(sampler, config, predict):
    """Attach known derivatives to real DeepXDE physics-point bookkeeping."""
    dde_config.set_random_seed(2000)
    dde_config.set_default_float("float64")
    model, _ = build_model(config["problem"], config["network"])
    model.data.replace_with_anchors(np.tile([0.0, 0.5], (4, 1)))
    sampler.model = SimpleNamespace(data=model.data, predict=predict, opt_name="adam")
    return model.data


@pytest.mark.parametrize("steps", [1, 4, 15])
def test_seeded_attenuation_and_accumulated_noise(config, steps):
    """Recover the exact noise at zero, half-variance, and very large gradients."""
    sampler = make_sampler(coordinate_steps=steps)
    relative = np.array([0.0, 1.0, -1.0, 1e6])
    gradients = np.column_stack((np.zeros((4, 2)), relative * 2 * np.pi))
    data = attach_sampler(sampler, config, lambda points, operator: gradients)
    constraints, expected = data.train_x_bc.copy(), data.train_x_all.copy()
    amplitudes = np.array([1, np.sqrt(0.5), np.sqrt(0.5), 1 / np.sqrt(1 + 1e12)])
    rng = np.random.default_rng(np.random.SeedSequence([2000, 1]))
    global_before = np.random.get_state(legacy=False)
    for _ in range(2):
        for _ in range(steps):
            expected += (
                1e-3
                * np.array([2.0, 0.99])
                / np.sqrt(steps)
                * amplitudes[:, None]
                * rng.standard_normal(expected.shape)
            )
        sampler.on_epoch_end()
        np.testing.assert_allclose(data.train_x_all, expected, rtol=0, atol=1e-16)
    np.testing.assert_array_equal(data.train_x_bc, constraints)
    np.testing.assert_equal(np.random.get_state(legacy=False), global_before)
    assert sampler.rng.bit_generator.state == rng.bit_generator.state
    assert sampler.noise_point_count == 8 * steps
    assert sampler.noise_multiplier_sum / sampler.noise_point_count == pytest.approx(
        amplitudes.mean()
    )
    assert sampler.half_variance_count / sampler.noise_point_count == 0.75


def test_zero_gradient_matches_constant_diffusion(config):
    """Use identical Gaussian draws when the predicted spatial gradient is zero."""
    controller = make_sampler()
    data = attach_sampler(controller, config, lambda points, operator: np.zeros((4, 3)))
    constant = PacmannDiffusion(
        interior_points=4,
        residual_gradient=None,
        period=1,
        coordinate_steps=3,
        step_size=1e-5,
        betas=(0.9, 0.999),
        epsilon=1e-7,
        sigma=1e-3,
        domain_widths=(2.0, 0.99),
        seed=2000,
    )
    constant.model = SimpleNamespace(
        data=copy.deepcopy(data),
        opt_name="adam",
        predict=lambda points, operator: np.zeros_like(points),
    )
    for _ in range(2):
        controller.on_epoch_end()
        constant.on_epoch_end()
        np.testing.assert_array_equal(data.train_x_all, constant.model.data.train_x_all)
    assert controller.rng.bit_generator.state == constant.rng.bit_generator.state
    assert controller.half_variance_count == 0


@pytest.mark.parametrize("step_size", [1e-5, 2.0])
def test_zero_sigma_matches_pacmann_with_replacement(config, step_size):
    """Preserve PACMANN drift, escaped-point draws, and the unused noise stream."""
    controller = make_sampler(sigma=0, step_size=step_size)
    data = attach_sampler(controller, config, lambda points, operator: np.ones((4, 3)))
    pacmann = PacmannAdam(
        interior_points=4,
        residual_gradient=None,
        period=1,
        coordinate_steps=3,
        step_size=step_size,
        betas=(0.9, 0.999),
        epsilon=1e-7,
    )
    pacmann.model = SimpleNamespace(
        data=copy.deepcopy(data),
        opt_name="adam",
        predict=lambda points, operator: np.ones_like(points),
    )
    noise_before = copy.deepcopy(controller.rng.bit_generator.state)
    for _ in range(2):
        before = np.random.get_state(legacy=False)
        pacmann.on_epoch_end()
        expected_rng = np.random.get_state(legacy=False)
        np.random.set_state(before)
        controller.on_epoch_end()
        np.testing.assert_equal(np.random.get_state(legacy=False), expected_rng)
        np.testing.assert_array_equal(data.train_x_all, pacmann.model.data.train_x_all)
    assert controller.rng.bit_generator.state == noise_before


def test_gate_uses_current_pre_update_coordinates(config):
    """Reevaluate the sensor after each noisy move, before the next drift step."""
    sampler = make_sampler(step_size=0.02)
    visited = []

    def gradients(points, operator):
        """Return a spatially varying sensor and constant coordinate ascent."""
        visited.append(points.copy())
        return np.column_stack((np.ones_like(points), 100 * points[:, 0]))

    data = attach_sampler(sampler, config, gradients)
    expected = data.train_x_all.copy()
    rng = np.random.default_rng(np.random.SeedSequence([2000, 1]))
    for _ in range(3):
        multiplier = 1 / np.sqrt(1 + (100 * expected[:, :1] / (2 * np.pi)) ** 2)
        expected += 0.02 / (1 + 1e-7)
        expected += (
            1e-3
            * np.array([2.0, 0.99])
            / np.sqrt(3)
            * multiplier
            * rng.standard_normal(expected.shape)
        )
    sampler.on_epoch_end()
    assert len(visited) == 3
    assert not np.array_equal(visited[0], visited[1])
    np.testing.assert_allclose(data.train_x_all, expected, rtol=0, atol=1e-15)


def test_manufactured_solution_derivatives(config, monkeypatch):
    """Check du/dx and both residual-score derivatives for u=x³+2t."""
    dde_config.set_default_float("float64")
    model, score_gradient = build_model(config["problem"], config["network"])
    model.opt_name = "adam"
    monkeypatch.setattr(model.net, "forward", lambda x: x[:, :1] ** 3 + 2 * x[:, 1:2])
    points = np.array([[0.25, 0.5], [-0.5, 0.4]])
    model.data.replace_with_anchors(points)
    sampler = make_sampler(
        interior_points=2, residual_gradient=score_gradient, sigma=0, coordinate_steps=1
    )
    sampler.model = model
    original_predict, recorded = model.predict, []

    def predict(coordinates, operator):
        """Record the combined operator's actual automatic derivatives."""
        values = original_predict(coordinates, operator=operator)
        recorded.append(values)
        return values

    monkeypatch.setattr(model, "predict", predict)
    sampler.on_epoch_end()
    x, t = points.T
    nu = config["problem"]["viscosity"]
    residual = 2 + (x**3 + 2 * t) * 3 * x**2 - 6 * nu * x
    expected = np.column_stack(
        (
            2 * residual * (15 * x**4 + 12 * t * x - 6 * nu),
            12 * residual * x**2,
            3 * x**2,
        )
    )
    np.testing.assert_allclose(recorded[0], expected, rtol=1e-12, atol=1e-12)


def test_cadence_and_escaped_points(config, monkeypatch):
    """Keep L-BFGS inactive and replace escaped points only after all substeps."""
    sampler = make_sampler(period=2, coordinate_steps=2, step_size=2)
    visited, replacements = [], []

    def gradients(points, operator):
        """Record pre-update coordinates with a zero spatial sensor."""
        visited.append(points.copy())
        return np.column_stack((np.ones_like(points), np.zeros(len(points))))

    data = attach_sampler(sampler, config, gradients)
    constraints, before = data.train_x_bc.copy(), data.train_x_all.copy()

    def replacement(count, random):
        """Replace escaped points after both coordinate evaluations."""
        replacements.append((count, random, len(visited)))
        return before.copy()

    monkeypatch.setattr(data.geom, "random_points", replacement)
    sampler.on_epoch_end()
    assert sampler.events == 0 and sampler.noise_point_count == 0
    sampler.on_epoch_end()
    assert replacements == [(4, "pseudo", 2)]
    assert (visited[1][:, 0] > 1).all()
    np.testing.assert_array_equal(data.train_x_bc, constraints)
    rng_before = copy.deepcopy(sampler.rng.bit_generator.state)
    sampler.model.opt_name = "L-BFGS"
    sampler.on_epoch_end()
    assert sampler.events == 1 and sampler.adam_steps == 2
    assert sampler.rng.bit_generator.state == rng_before


@pytest.mark.parametrize(
    "settings",
    [
        {"alpha": 0},
        {"alpha": np.inf},
        {"g_base": -1},
        {"g_base": np.nan},
        {"sigma": -1},
        {"domain_widths": (0, 1)},
        {"coordinate_steps": 0},
        {"period": 0},
    ],
)
def test_invalid_settings(settings):
    """Reject parameters that cannot define the intended local noise covariance."""
    with pytest.raises(ValueError):
        make_sampler(**settings)


@pytest.mark.parametrize("gradients", [np.ones((4, 2)), np.full((4, 3), np.nan)])
def test_invalid_derivatives(config, gradients):
    """Reject wrong derivative shapes and non-finite spatial or score gradients."""
    sampler = make_sampler()
    with pytest.raises(RuntimeError, match="attached"):
        sampler.on_epoch_end()
    attach_sampler(sampler, config, lambda points, operator: gradients)
    with pytest.raises((ValueError, FloatingPointError), match="gradients"):
        sampler.on_epoch_end()


def test_tiny_training_parity_and_post_processing(config, tmp_path, monkeypatch):
    """Restore controller weights and retain stage diagnostics without changing controls."""
    monkeypatch.setattr(train, "ROOT", tmp_path)
    monkeypatch.setattr(
        "src.artifacts.subprocess.check_output",
        lambda *args, **kwargs: "test-revision\n",
    )
    config["network"]["layers"] = [2, 8, 1]
    config["problem"].update(boundary_points=4, initial_points=4)
    config["training"].update(phases=2, adam_steps=3, display_every=1)
    config["training"]["lbfgs"].update(maxiter=2, maxfun=3)
    config["methods"]["pacmann_adam"].update(period=2, steps=1)
    settings = dict(
        config["methods"]["pacmann_adam"], sigma=0.0, alpha=1.0, g_base=np.pi
    )
    config["methods"]["pacmann_sharpness_diffusion"] = settings
    ordinary = train.run_training(config, "pacmann_adam", 1000)
    zero = train.run_training(config, "pacmann_sharpness_diffusion", 1000)
    analyze_run(ordinary)
    analyze_run(zero)
    original_weights = torch.load(ordinary / "models/final.pt", weights_only=True)
    zero_weights = torch.load(zero / "models/final.pt", weights_only=True)
    for name in original_weights:
        torch.testing.assert_close(
            original_weights[name], zero_weights[name], rtol=0, atol=0
        )
    with (
        np.load(ordinary / "predictions.npz") as expected,
        np.load(zero / "predictions.npz") as actual,
    ):
        np.testing.assert_array_equal(actual["prediction"], expected["prediction"])

    # Positive-noise integration covers stages both with and without sampling events.
    settings["sigma"] = 1e-3
    for period in (2, 100):
        config["campaign"] = f"controller_validation_{period}"
        settings["period"] = period
        run = train.run_training(config, "pacmann_sharpness_diffusion", 1000)
        analyze_run(run)
        saved = json.loads((run / "training.json").read_text())["stages"]
        for stage in saved:
            if stage["optimizer"] == "adam" and stage["movement_events"]:
                assert 0 < stage["mean_noise_multiplier"] <= 1
                assert 0 <= stage["half_variance_fraction"] <= 1
            else:
                assert "mean_noise_multiplier" not in stage
                assert "half_variance_fraction" not in stage
        metrics = json.loads((run / "metrics.json").read_text())
        assert metrics["movement_events"] == (3 if period == 2 else 0)
        assert (run / "diagnostics.png").stat().st_size > 0
