"""Check final-error aggregation and same-seed method contrasts."""

import numpy as np
import pytest

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
