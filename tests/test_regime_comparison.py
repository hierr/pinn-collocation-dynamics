"""Check final-error aggregation and same-seed method contrasts."""

import numpy as np
import pytest

from scripts.analysis import burgers_regime_comparison as comparison
from scripts.analysis.burgers_regime_comparison import (
    paired_statistics,
    prediction_errors,
)


def test_physical_time_and_complete_grid_errors():
    """Keep spatial error at each physical time distinct from complete-grid L2."""
    x, times = np.array([-1.0, 0.0, 1.0]), np.array([0.0, 0.99])
    xx, tt = np.meshgrid(x, times)
    reference = np.array([[1.0, 2.0, 2.0], [2.0, 0.0, 1.0]])
    error = np.array([[1.0, -1.0, 0.0], [0.0, 2.0, -1.0]])
    arrays = {
        "coordinates": np.column_stack((xx.ravel(), tt.ravel())),
        "reference": reference.reshape(-1, 1),
        "prediction": (reference + error).reshape(-1, 1),
    }
    actual_times, relative, rmse, overall = prediction_errors(arrays)
    np.testing.assert_array_equal(actual_times, times)
    np.testing.assert_allclose(relative, [np.sqrt(2 / 9), 1.0])
    np.testing.assert_allclose(rmse, np.sqrt([2 / 3, 5 / 3]))
    assert overall == pytest.approx(np.sqrt(7 / 14))
    assert overall != pytest.approx(np.mean(relative))

    arrays["coordinates"] = arrays["coordinates"][::-1]
    with pytest.raises(ValueError, match="time-major"):
        prediction_errors(arrays)


def test_pairing_retains_ties_and_large_errors():
    """Retain a worse mean even when other seeds improve or tie."""
    summary = paired_statistics([0.5, 4.0, 3.0], [1.0, 2.0, 3.0])
    assert summary["mean_difference"] == pytest.approx(0.5)
    assert summary["sample_sd_difference"] == pytest.approx(np.sqrt(1.75))
    assert (summary["improved"], summary["tied"], summary["worse"]) == (1, 1, 1)
    assert summary["median_ratio"] == 1.0


@pytest.mark.parametrize(
    "first,second", [([1, 2], [1]), ([0, 1], [1, 1]), ([np.nan, 1], [1, 1])]
)
def test_undefined_pairing_or_ratios_are_rejected(first, second):
    """Reject mismatched seeds and values that cannot support finite error ratios."""
    with pytest.raises(ValueError):
        paired_statistics(first, second)


def test_selected_figures_preserve_case_values_and_pairing(tmp_path, monkeypatch):
    """Keep omitted display cases from shifting data, sigma labels, or seed ratios."""
    monkeypatch.setattr(comparison, "ROOT", tmp_path)
    figures = {}

    def capture(figure, path, **kwargs):
        """Retain the plotted data for scientific checks without writing images."""
        figures[path.name] = figure

    monkeypatch.setattr(comparison.plt.Figure, "savefig", capture)
    errors, temporal = {}, {}
    times = np.array([0.0, 0.5, 0.99])
    for case in range(5):
        for index, method in enumerate(comparison.METHODS):
            values = 10 * case + (index + 1) ** 2 + np.linspace(0.1, 1, 10)
            for metric, _ in comparison.METRICS:
                errors[case, method, metric] = values
            temporal[case, method] = (times, np.repeat(values[:, None], 3, axis=1))
    selected = (0, 2, 3, 4)
    sigmas = [1e-5, 1e-4, 1e-3, 1e-4, 1e-6]

    comparison.plot_comparisons(
        errors, temporal, sigmas, selected, tmp_path / "figures"
    )

    final_axis = figures["final_errors.png"].axes[3]
    np.testing.assert_allclose(
        final_axis.collections[0].get_offsets()[:, 1],
        errors[4, "fixed_hammersley", "solution_l2_percent"],
    )
    ratio_axis = figures["diffusion_ratios.png"].axes[0]
    last_scatter = ratio_axis.collections[6].get_offsets()
    assert np.mean(last_scatter[:, 0]) == pytest.approx(3)
    np.testing.assert_allclose(
        last_scatter[:, 1],
        errors[4, "pacmann_diffusion", "solution_l2_percent"]
        / errors[4, "pacmann_adam", "solution_l2_percent"],
    )
    assert "σ = 1e-06" in ratio_axis.get_xticklabels()[-1].get_text()
    temporal_axis = figures["solution_error_over_time.png"].axes[3]
    np.testing.assert_allclose(
        temporal_axis.lines[0].get_ydata(),
        temporal[4, "fixed_hammersley"][1].mean(axis=0),
    )
