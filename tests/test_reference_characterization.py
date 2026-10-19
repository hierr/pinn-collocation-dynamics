"""Check reference gradients against analytical derivatives and native snapshots."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts.post_processing import characterize_burgers_references as characterization


@pytest.mark.parametrize("degree", [32, 33])
def test_initial_condition_derivative(degree):
    """Recover the sine derivative for even and odd polynomial degrees."""
    x = np.cos(np.pi * np.arange(degree + 1) / degree)
    scales = np.array([1.0, 0.3])[:, None]
    solution = -np.sin(np.pi * x)[None, :] * scales
    expected = -np.pi * np.cos(np.pi * x)[None, :] * scales
    actual = characterization.spatial_derivative(x, solution)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-11)


def test_derivative_uses_physical_domain_width():
    """Differentiate polynomials on [2, 5] without assuming an interval of length 2."""
    x = 3.5 + 1.5 * np.cos(np.pi * np.arange(17) / 16)
    solution = np.vstack((x**3, 2 * x + 7))
    expected = np.vstack((3 * x**2, np.full_like(x, 2)))
    actual = characterization.spatial_derivative(x, solution)
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)


def test_non_lobatto_nodes_are_rejected():
    """Reject coordinates for which the spectral differentiation would be wrong."""
    x = np.linspace(1, -1, 17)
    with pytest.raises(ValueError, match="Chebyshev-Lobatto"):
        characterization.spatial_derivative(x, np.array([x**2]))


def test_native_time_crop_and_spatial_order(config):
    """Preserve exact evaluation times and handle a reference stored in ascending x."""
    path = Path(config["problem"]["reference_path"])
    with np.load(path) as source:
        arrays = dict(source)
    arrays["x"] = arrays["x"][::-1]
    arrays["U"] = arrays["U"][:, ::-1]
    np.savez(path, **arrays)
    original = path.read_bytes()

    x, times, solution, gradients, amplitudes = characterization.characterize_reference(
        config["problem"]
    )
    assert np.all(np.diff(x) < 0)
    assert solution.shape == (3, len(x))
    np.testing.assert_array_equal(times, [0.0, 0.495, 0.99])
    np.testing.assert_allclose(gradients, np.pi * np.exp(-times), atol=1e-8)
    expected_amplitude = np.max(np.abs(np.sin(np.pi * arrays["x"]))) * np.exp(-times)
    np.testing.assert_allclose(amplitudes, expected_amplitude, atol=1e-14)
    assert path.read_bytes() == original


def test_reference_viscosity_must_match(config):
    """Prevent assigning descriptors from one viscosity to a different regime."""
    config["problem"]["viscosity"] *= 2
    with pytest.raises(ValueError, match="viscosity"):
        characterization.characterize_reference(config["problem"])


def test_documentation_figure_and_summary(config, tmp_path, monkeypatch, capsys):
    """Produce both figures and the numerical summary without changing source data."""
    config_path = tmp_path / "reference.yaml"
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(characterization, "ROOT", tmp_path)
    monkeypatch.setattr(characterization, "CONFIG_PATHS", [config_path])
    figure = tmp_path / "regimes.png"
    monkeypatch.setattr(characterization, "OUTPUT_PATH", figure)
    slices = tmp_path / "slices.png"
    monkeypatch.setattr(characterization, "SLICES_OUTPUT_PATH", slices)
    source = Path(config["problem"]["reference_path"])
    original = source.read_bytes()

    characterization.main()

    assert "nu,peak_gradient,time_at_peak,amplitude_at_peak" in capsys.readouterr().out
    assert figure.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert slices.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert source.read_bytes() == original
