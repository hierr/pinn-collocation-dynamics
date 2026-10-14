"""Verify diffusion numerics, reproducibility, and training integration."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from deepxde import config as dde_config

from scripts.post_processing.analyze_burgers import analyze_run
from src import train
from src.burgers import build_model, load_reference
from src.sampling import PacmannAdam, PacmannDiffusion


def make_sampler(**overrides):
    """Return a small diffusion callback with explicit numerical settings."""
    settings: dict = {
        "interior_points": 4,
        "residual_gradient": None,
        "period": 1,
        "coordinate_steps": 3,
        "step_size": 1e-5,
        "betas": (0.9, 0.999),
        "epsilon": 1e-7,
        "sigma": 1e-3,
        "domain_widths": (2.0, 0.99),
        "seed": 2000,
    }
    settings.update(overrides)
    return PacmannDiffusion(**settings)


def attach_sampler(sampler, config, predict):
    """Attach the callback to real DeepXDE point bookkeeping and a known gradient."""
    coordinates, values, _ = load_reference(train.ROOT)
    dde_config.set_random_seed(2000)
    dde_config.set_default_float("float64")
    model, _ = build_model(config["problem"], config["network"], coordinates, values)
    sampler.model = SimpleNamespace(data=model.data, predict=predict, opt_name="adam")
    return model.data


@pytest.mark.parametrize("step_size", [1e-5, 2.0])
def test_zero_sigma_matches_pacmann_and_preserves_rng(config, step_size):
    """Match multiple events and escaped-point draws without consuming noise."""
    diffusion = make_sampler(sigma=0, step_size=step_size, interior_points=16)
    data = attach_sampler(
        diffusion, config, lambda points, operator: 0.2 * points + 0.1
    )
    original = copy.deepcopy(data)
    pacmann = PacmannAdam(
        interior_points=16,
        residual_gradient=None,
        period=1,
        coordinate_steps=3,
        step_size=step_size,
        betas=(0.9, 0.999),
        epsilon=1e-7,
    )
    pacmann.model = SimpleNamespace(
        data=original, predict=diffusion.model.predict, opt_name="adam"
    )
    local_before = copy.deepcopy(diffusion.rng.bit_generator.state)
    for _ in range(3):
        before = np.random.get_state(legacy=False)
        pacmann.on_epoch_end()
        expected_rng = np.random.get_state(legacy=False)
        np.random.set_state(before)
        diffusion.on_epoch_end()
        actual_rng = np.random.get_state(legacy=False)
        np.testing.assert_equal(actual_rng, expected_rng)
        np.testing.assert_array_equal(data.train_x_all, original.train_x_all)
        np.testing.assert_array_equal(data.train_x_bc, original.train_x_bc)
    assert diffusion.events == pacmann.events == 3
    assert diffusion.rng.bit_generator.state == local_before


@pytest.mark.parametrize("steps", [1, 4, 15])
def test_noise_matches_seeded_formula_and_persists(config, steps):
    """Recover exact noise increments and their K/domain scaling over two events."""
    sampler = make_sampler(coordinate_steps=steps)
    data = attach_sampler(
        sampler, config, lambda points, operator: np.zeros_like(points)
    )
    # Keep points away from boundaries so replacement cannot mask the noise formula.
    data.replace_with_anchors(np.tile([0.0, 0.5], (4, 1)))
    constraints = data.train_x_bc.copy()
    expected = data.train_x_all.copy()
    rng = np.random.default_rng(np.random.SeedSequence([2000, 1]))
    global_before = np.random.get_state(legacy=False)
    for _ in range(2):
        for _ in range(steps):
            expected += (
                1e-3
                * np.array([2.0, 0.99])
                / np.sqrt(steps)
                * rng.standard_normal(expected.shape)
            )
        sampler.on_epoch_end()
        np.testing.assert_allclose(data.train_x_all, expected, rtol=0, atol=1e-16)
        np.testing.assert_array_equal(data.train_x_bc, constraints)
    global_after = np.random.get_state(legacy=False)
    np.testing.assert_equal(global_before, global_after)
    assert sampler.rng.bit_generator.state == rng.bit_generator.state


def test_cadence_reproducibility_and_lbfgs_inactivity(config):
    """Repeat a seeded noisy event and avoid noise draws between events or in L-BFGS."""
    first, second = make_sampler(period=2), make_sampler(period=2)
    data = attach_sampler(first, config, lambda points, operator: np.zeros_like(points))
    data.replace_with_anchors(np.tile([0.0, 0.5], (4, 1)))
    second.model = SimpleNamespace(
        data=copy.deepcopy(data), predict=first.model.predict, opt_name="adam"
    )
    before, rng_before = (
        data.train_x_all.copy(),
        copy.deepcopy(first.rng.bit_generator.state),
    )
    first.on_epoch_end()
    np.testing.assert_array_equal(data.train_x_all, before)
    assert first.rng.bit_generator.state == rng_before
    first.on_epoch_end()
    second.on_epoch_end()
    second.on_epoch_end()
    np.testing.assert_array_equal(data.train_x_all, second.model.data.train_x_all)
    assert not np.array_equal(data.train_x_all, before)
    moved, rng_moved = (
        data.train_x_all.copy(),
        copy.deepcopy(first.rng.bit_generator.state),
    )
    first.model.opt_name = "L-BFGS"
    first.on_epoch_end()
    np.testing.assert_array_equal(data.train_x_all, moved)
    assert first.rng.bit_generator.state == rng_moved
    assert first.adam_steps == 2 and first.events == 1


def test_escaped_points_replaced_after_all_steps(config, monkeypatch):
    """Evaluate both gradients before uniformly replacing escaped physics points."""
    sampler = make_sampler(sigma=0.1, coordinate_steps=2, step_size=2.0)
    visited, replacements = [], []

    def gradient(points, operator):
        """Record the population evaluated at each coordinate step."""
        visited.append(points.copy())
        return np.ones_like(points)

    data = attach_sampler(sampler, config, gradient)
    constraints = data.train_x_bc.copy()

    def replacement(count, random):
        """Provide known interior coordinates only after both gradient evaluations."""
        replacements.append((count, random, len(visited)))
        return np.tile([0.0, 0.5], (count, 1))

    monkeypatch.setattr(data.geom, "random_points", replacement)
    sampler.on_epoch_end()
    assert replacements == [(4, "pseudo", 2)]
    assert (visited[1][:, 0] > 1).any()
    np.testing.assert_array_equal(data.train_x_all, np.tile([0.0, 0.5], (4, 1)))
    np.testing.assert_array_equal(data.train_x_bc, constraints)


@pytest.mark.parametrize(
    "settings",
    [
        {"sigma": -1},
        {"sigma": np.nan},
        {"sigma": np.inf},
        {"domain_widths": [0, 1]},
        {"domain_widths": [1, np.inf]},
        {"domain_widths": [1]},
        {"coordinate_steps": 0},
        {"coordinate_steps": 1.5},
    ],
)
def test_invalid_diffusion_settings(settings):
    """Reject settings that cannot define a finite normalized noise scale."""
    with pytest.raises(ValueError):
        make_sampler(**settings)


def test_missing_model_and_nonfinite_gradient():
    """Reject an unattached callback and invalid coordinate gradients."""
    sampler = make_sampler()
    with pytest.raises(RuntimeError, match="attached"):
        sampler.on_epoch_end()
    sampler.model = SimpleNamespace(
        opt_name="adam",
        data=SimpleNamespace(train_x_all=np.zeros((4, 2)), train_x_bc=np.zeros((1, 2))),
        predict=lambda points, operator: np.full_like(points, np.nan),
    )
    with pytest.raises(FloatingPointError, match="coordinate gradient"):
        sampler.on_epoch_end()


def test_tiny_training_zero_sigma_parity_and_positive_diffusion(
    config, tmp_path, monkeypatch
):
    """Verify final weights/history parity and real noisy training plus diagnostics."""
    reference = load_reference(train.ROOT)
    monkeypatch.setattr(train, "ROOT", tmp_path)
    monkeypatch.setattr(train, "load_reference", lambda root: reference)
    monkeypatch.setattr(
        "src.artifacts.subprocess.check_output",
        lambda *args, **kwargs: "test-revision\n",
    )
    config["network"]["layers"] = [2, 8, 1]
    config["training"].update(phases=2, adam_steps=3, display_every=1)
    config["training"]["lbfgs"].update(maxiter=2, maxfun=3)
    config["methods"]["pacmann_adam"].update(period=2, steps=1)
    config["methods"]["pacmann_diffusion"] = dict(
        config["methods"]["pacmann_adam"], sigma=0.0
    )
    original = train.run_training(config, "pacmann_adam", 1000)
    zero = train.run_training(config, "pacmann_diffusion", 1000)
    for name in ("predictions.npz", "history.npz", "points/final.npz"):
        with np.load(original / name) as expected, np.load(zero / name) as actual:
            for key in expected.files:
                np.testing.assert_array_equal(actual[key], expected[key])
    expected_weights = torch.load(original / "models/final.pt", weights_only=True)
    actual_weights = torch.load(zero / "models/final.pt", weights_only=True)
    for key in expected_weights:
        torch.testing.assert_close(
            actual_weights[key], expected_weights[key], rtol=0, atol=0
        )
    config["campaign"] = "positive_diffusion_validation"
    config["methods"]["pacmann_diffusion"]["sigma"] = 1e-3
    positive = train.run_training(config, "pacmann_diffusion", 1000)
    analyze_run(positive)
    assert json.loads((positive / "status.json").read_text())["status"] == "complete"
    assert json.loads((positive / "metrics.json").read_text())["movement_events"] == 3
    assert (positive / "diagnostics.png").stat().st_size > 0
