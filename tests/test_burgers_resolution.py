"""Check common-grid values, spectral derivatives, and resolution-comparison inputs."""

import json

import numpy as np
import pytest

from scripts.dataset_validation import burgers_compare_resolutions as comparison


def test_polynomial_values_and_spatial_derivatives():
    """Recover a quadratic and its correctly signed derivative between saved nodes."""
    x = np.cos(np.pi * np.arange(17) / 16)
    amplitudes = np.array([1.0, 1.5])[:, None]
    u = amplitudes * (1 - x**2)
    coordinates = np.linspace(-1, 1, 101)

    values, derivatives = comparison.sample_reference(x, u, coordinates)

    np.testing.assert_allclose(values, amplitudes * (1 - coordinates**2), atol=1e-13)
    np.testing.assert_allclose(derivatives, -2 * amplitudes * coordinates, atol=1e-12)


def write_reference(path, degree, scale=1.0, **settings):
    """Save a polynomial case with controlled settings for the comparison tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.cos(np.pi * np.arange(degree + 1) / degree)
    t = np.linspace(0, 1, 5)
    config = dict(nu=0.001, N=degree, number_of_spatial_nodes=degree + 1, **settings)
    np.savez(
        path,
        x=x,
        t=t,
        U=scale * (1 + t[:, None]) * (1 - x**2),
        nu=0.001,
        config_json=json.dumps(config),
    )


def test_comparison_reports_known_relative_disagreement(tmp_path, monkeypatch):
    """Check normalization and time selection using a known multiplicative difference."""
    monkeypatch.setattr(comparison, "ROOT", tmp_path)
    monkeypatch.setattr(comparison, "DEGREES", (8, 16))
    monkeypatch.setattr(comparison, "UNIFORM_POINTS", 129)
    for degree, scale in ((8, 1.001), (16, 1.0)):
        write_reference(
            tmp_path / f"data/burgers_dataset/burgers_nu_0p001_N{degree}.npz",
            degree,
            scale,
        )

    comparison.main()

    output = tmp_path / "output/burgers_reference_validation/nu_0p001_n8_n16"
    summary = json.loads((output / "summary.json").read_text())
    assert summary["time_interval"] == [0.0, 0.75]
    assert summary["snapshots"] == 4
    for field in (
        "shared_nodes_relative_l2",
        "uniform_relative_l2",
        "uniform_gradient_relative_l2",
        "maximum_peak_gradient_relative_change",
    ):
        assert summary[field] == pytest.approx(0.001, abs=1e-12)
    assert (output / "front_profiles.png").is_file()
    assert (output / "resolution_over_time.png").is_file()
    series = np.loadtxt(output / "time_series.csv", delimiter=",", skiprows=1)
    np.testing.assert_allclose(series[:, 1], 0.001, atol=1e-12)


def test_mismatched_temporal_settings_are_rejected(tmp_path):
    """Prevent attributing a temporal-solver change to spatial refinement."""
    paths = [tmp_path / "coarse.npz", tmp_path / "fine.npz"]
    write_reference(paths[0], 8, max_step=0.005)
    write_reference(paths[1], 16, max_step=0.01)
    with pytest.raises(ValueError, match="times and non-spatial solver settings"):
        comparison.load_references(paths)
