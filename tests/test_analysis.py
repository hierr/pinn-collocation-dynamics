"""Check grid reconstruction and numerical diagnostics independently of training."""

import numpy as np
import pytest

from scripts.analyze_burgers import calculate_diagnostics, load_arrays, reconstruct_grid


def test_grid_and_known_errors():
    """Recover a small grid and errors with known analytic values."""
    predictions = {
        "coordinates": np.array([[-1, 0], [1, 0], [-1, 0.5], [1, 0.5]]),
        "reference": np.array([[3], [4], [0], [2]]),
        "prediction": np.array([[3], [5], [0], [3]]),
    }
    x, times, reference, predicted = reconstruct_grid(predictions)
    assert reference.shape == (2, 2)
    result = calculate_diagnostics(x, times, reference, predicted)
    assert result["maximum_absolute_error"] == 1
    assert result["maximum_error_coordinate"] == [1, 0]
    np.testing.assert_allclose(result["per_time_relative_l2"], [0.2, 0.5])
    assert result["times"] == [0, 0.5]


@pytest.mark.parametrize("indices", [[1, 0, 2, 3], [0, 1, 2]])
def test_invalid_grid_order_or_missing_coordinate(indices):
    """Reject grids that would silently misalign reshaped predictions."""
    predictions = {
        "coordinates": np.array([[-1, 0], [1, 0], [-1, 1], [1, 1]])[indices],
        "reference": np.ones((len(indices), 1)),
        "prediction": np.ones((len(indices), 1)),
    }
    with pytest.raises(ValueError, match="complete time-major grid"):
        reconstruct_grid(predictions)


def test_nonfinite_artifact_and_zero_reference(tmp_path):
    """Reject inputs that make an error plot or relative error misleading."""
    path = tmp_path / "invalid.npz"
    np.savez(path, prediction=np.array([np.nan]))
    with pytest.raises(ValueError, match="prediction contains non-finite"):
        load_arrays(path)
    with pytest.raises(ValueError, match="zero reference norm"):
        calculate_diagnostics(
            np.array([0]), np.array([0]), np.zeros((1, 1)), np.ones((1, 1))
        )
