"""Check grid reconstruction and numerical diagnostics independently of training."""

import json

import numpy as np
import pytest
import torch
import yaml
from deepxde import config as dde_config

from scripts.post_processing import analyze_burgers as analysis
from scripts.post_processing.analyze_burgers import (
    analyze_run,
    calculate_diagnostics,
    load_arrays,
    reconstruct_grid,
)
from src import train
from src.burgers import build_model, load_reference


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


def test_ordinary_time_slice_errors():
    """Count each spatial point equally in a localized prediction error."""
    result = calculate_diagnostics(
        np.array([-1.0, 0.0, 1.0]),
        np.array([0.0]),
        np.ones((1, 3)),
        np.array([[1.0, 2.0, 1.0]]),
    )
    np.testing.assert_allclose(result["per_time_relative_l2"], [1 / np.sqrt(3)])


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


def test_postprocessing_matches_training_model_and_preserves_artifacts(
    config, tmp_path, monkeypatch
):
    """Restore the final model, match in-memory scores, and reuse saved evaluation."""
    monkeypatch.setattr(train, "ROOT", tmp_path)
    monkeypatch.setattr(
        "src.artifacts.subprocess.check_output",
        lambda *args, **kwargs: "test-revision\n",
    )
    config["network"]["layers"] = [2, 8, 1]
    config["training"].update(phases=1, adam_steps=2, display_every=1)
    config["training"]["lbfgs"].update(maxiter=2, maxfun=3)
    models = []
    build_model = train.build_model

    def record_model(**settings):
        """Keep the trained model for an independent checkpoint-reload comparison."""
        model, gradient = build_model(**settings)
        models.append(model)
        return model, gradient

    monkeypatch.setattr(train, "build_model", record_model)
    run = train.run_training(config, "fixed_hammersley", 1000)
    original_files = {
        path: path.read_bytes() for path in run.rglob("*") if path.is_file()
    }
    coordinates, reference, _ = load_reference(
        tmp_path, config["problem"], config["evaluation"]["solution_nx"]
    )
    expected_prediction, expected_metrics = analysis.evaluate_final(
        models[0], coordinates, reference, config["problem"], config["evaluation"]
    )

    # Final evaluation is generated from the checkpoint, without training again.
    analyze_run(run)
    with np.load(run / "predictions.npz") as saved:
        assert set(saved.files) == {"coordinates", "reference", "prediction"}
        np.testing.assert_array_equal(saved["coordinates"], coordinates)
        np.testing.assert_allclose(
            saved["prediction"], expected_prediction, rtol=1e-10, atol=1e-12
        )
    actual = json.loads((run / "metrics.json").read_text())
    stages = json.loads((run / "training.json").read_text())["stages"]
    assert set(actual) == set(expected_metrics) | {
        "movement_events",
        "training_seconds",
    }
    for name, expected in expected_metrics.items():
        np.testing.assert_allclose(actual[name], expected, rtol=1e-10, atol=1e-12)
    assert actual["training_seconds"] == sum(stage["seconds"] for stage in stages)
    assert actual["movement_events"] == 0
    assert all(path.read_bytes() == content for path, content in original_files.items())

    # Recover a missing summary without replacing the existing predictions.
    saved_prediction_bytes = (run / "predictions.npz").read_bytes()
    (run / "metrics.json").unlink()
    analyze_run(run)
    assert (run / "predictions.npz").read_bytes() == saved_prediction_bytes
    assert json.loads((run / "metrics.json").read_text()) == actual
    saved_metric_bytes = (run / "metrics.json").read_bytes()

    # Recover missing predictions without replacing the consistent metric summary.
    (run / "predictions.npz").unlink()
    analyze_run(run)
    assert (run / "metrics.json").read_bytes() == saved_metric_bytes
    with np.load(run / "predictions.npz") as saved:
        np.testing.assert_allclose(
            saved["prediction"], expected_prediction, rtol=1e-10, atol=1e-12
        )
    saved_prediction_bytes = (run / "predictions.npz").read_bytes()

    # A completed evaluation only needs its saved arrays to regenerate plots.
    monkeypatch.setattr(
        analysis, "build_model", lambda *args: pytest.fail("Unexpected model reload")
    )
    analyze_run(run)
    assert (run / "predictions.npz").read_bytes() == saved_prediction_bytes
    assert (run / "metrics.json").read_bytes() == saved_metric_bytes


@pytest.mark.parametrize(
    "changed_field",
    ["coordinates", "reference", "prediction", "final_relative_l2", "missing_metric"],
)
def test_recovery_rejects_inconsistent_existing_evaluation(
    config, tmp_path, changed_field
):
    """Reject mismatched evaluation files before writing any recovery artifacts."""
    config["network"]["layers"] = [2, 8, 1]
    dde_config.set_default_float("float64")
    coordinates, reference, fingerprint = load_reference(
        tmp_path, config["problem"], config["evaluation"]["solution_nx"]
    )
    model, _ = build_model(config["problem"], config["network"])
    model.compile("adam", lr=config["training"]["learning_rate"])
    prediction, metrics = analysis.evaluate_final(
        model, coordinates, reference, config["problem"], config["evaluation"]
    )
    run = tmp_path / "partial_evaluation"
    (run / "models").mkdir(parents=True)
    torch.save(model.net.state_dict(), run / "models/final.pt")
    (run / "config.yaml").write_text(yaml.safe_dump(config))
    (run / "metadata.json").write_text(json.dumps({"dataset_sha256": fingerprint}))
    (run / "training.json").write_text(
        json.dumps({"stages": [{"seconds": 1.0, "movement_events": 0}]})
    )

    # Leave one counterpart inconsistent with the archived model and settings.
    if changed_field in ("coordinates", "reference", "prediction"):
        arrays = {
            "coordinates": coordinates,
            "reference": reference,
            "prediction": prediction,
        }
        arrays[changed_field] = arrays[changed_field] + 1
        np.savez(run / "predictions.npz", **arrays)
    else:
        metrics.update(training_seconds=1.0, movement_events=0)
        if changed_field == "missing_metric":
            del metrics["final_relative_l2"]
        else:
            metrics[changed_field] += 1
        (run / "metrics.json").write_text(json.dumps(metrics))
    original_files = {
        path: path.read_bytes() for path in run.rglob("*") if path.is_file()
    }

    with pytest.raises(
        ValueError, match="Existing .* differs from checkpoint evaluation"
    ):
        analyze_run(run)
    assert {
        path: path.read_bytes() for path in run.rglob("*") if path.is_file()
    } == original_files


def test_postprocessing_rejects_changed_reference(config, tmp_path):
    """Reject a substituted reference before loading weights or writing evaluation."""
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.yaml").write_text(yaml.safe_dump(config))
    (run / "metadata.json").write_text(json.dumps({"dataset_sha256": "different"}))
    with pytest.raises(ValueError, match="fingerprint"):
        analyze_run(run)
    assert not (run / "predictions.npz").exists()
    assert not (run / "metrics.json").exists()
