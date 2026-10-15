"""Check run artifacts, execution selectors, and optimizer-stage behavior."""

import json

import numpy as np
import pytest
import yaml

from src import train
from src.artifacts import prepare_run
from src.burgers import load_reference


@pytest.fixture
def run_root(config, tmp_path, monkeypatch):
    """Use temporary output and the small independent test reference."""
    reference = load_reference(
        train.ROOT, config["problem"], config["evaluation"]["solution_nx"]
    )
    monkeypatch.setattr(train, "ROOT", tmp_path)
    monkeypatch.setattr(
        train, "load_reference", lambda root, problem, solution_nx: reference
    )
    monkeypatch.setattr(
        "src.artifacts.subprocess.check_output",
        lambda *args, **kwargs: "test-revision\n",
    )
    return tmp_path


@pytest.mark.parametrize("method", ["fixed_hammersley", "pacmann_adam"])
def test_training_preserves_artifacts_and_stage_behavior(config, run_root, method):
    """Exercise both stages, cross-phase sampling cadence, and evaluation snapshots."""
    config["network"]["layers"] = [2, 8, 1]
    config["training"].update(phases=2, adam_steps=3, display_every=1)
    config["training"]["lbfgs"].update(maxiter=2, maxfun=3)
    config["methods"]["pacmann_adam"].update(period=2, steps=1)
    run_dir = train.run_training(config, method, 1000)

    # Verify the run's public artifact contract independently of helper layout.
    assert json.loads((run_dir / "status.json").read_text()) == {"status": "complete"}
    saved_config = yaml.safe_load((run_dir / "config.yaml").read_text())
    assert saved_config == dict(config, selected_method=method, selected_seed=1000)
    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert set(metadata) == {"revision", "versions", "dataset_sha256", "device"}
    assert "Training model" in (run_dir / "run.log").read_text()
    assert not (run_dir / "metrics.json").exists()
    assert not (run_dir / "predictions.npz").exists()
    with np.load(run_dir / "history.npz") as history:
        assert set(history.files) == {
            "steps",
            "loss_train",
            "loss_test",
        }

    stages = json.loads((run_dir / "training.json").read_text())["stages"]
    assert [stage["optimizer"] for stage in stages] == [
        "adam",
        "L-BFGS",
        "adam",
        "L-BFGS",
    ]
    expected_events = [1, 0, 2, 0] if method == "pacmann_adam" else [0, 0, 0, 0]
    assert [stage["movement_events"] for stage in stages] == expected_events
    for stage in stages:
        expected_keys = {
            "phase",
            "optimizer",
            "seconds",
            "global_step",
            "optimizer_iterations",
            "movement_events",
            "pde_points",
        }
        if stage["optimizer"] == "L-BFGS":
            expected_keys.add("function_evaluations")
        else:
            assert stage["optimizer_iterations"] == 3
        assert set(stage) == expected_keys

    # Check coordinate retention across optimizer boundaries.
    with np.load(run_dir / "points/initial.npz") as initial:
        constraints = initial["constraints"].copy()
        initial_pde = initial["pde"].copy()
    for phase in (1, 2):
        with (
            np.load(run_dir / f"points/phase_{phase:02d}_adam.npz") as adam,
            np.load(run_dir / f"points/phase_{phase:02d}_l-bfgs.npz") as lbfgs,
        ):
            for points in (adam, lbfgs):
                assert set(points.files) == {"interior", "pde", "constraints"}
                np.testing.assert_array_equal(points["constraints"], constraints)
                if method == "fixed_hammersley":
                    np.testing.assert_array_equal(points["pde"], initial_pde)
                else:
                    assert len(points["pde"]) == 16
            np.testing.assert_array_equal(adam["pde"], lbfgs["pde"])
    expected_names = {
        "phase_01_adam",
        "phase_01_l-bfgs",
        "phase_02_adam",
        "phase_02_l-bfgs",
        "final",
    }
    assert {path.stem for path in (run_dir / "models").iterdir()} == expected_names
    assert {path.stem for path in (run_dir / "points").iterdir()} == expected_names | {
        "initial"
    }
    with pytest.raises(FileExistsError):
        train.run_training(config, method, 1000)


def test_failed_run_retains_configuration_and_traceback(config, run_root, monkeypatch):
    """Keep failed-run artifacts and reject a retry that would overwrite them."""

    def fail_to_build(**kwargs):
        """Simulate a failure after run preparation."""
        raise RuntimeError("model construction failed")

    monkeypatch.setattr(train, "build_model", fail_to_build)
    with pytest.raises(RuntimeError, match="model construction failed"):
        train.run_training(config, "pacmann_adam", 1000)
    run_dir = (
        run_root / "output" / config["campaign"] / "burgers/pacmann_adam/seed_1000"
    )
    assert (run_dir / "config.yaml").exists()
    assert json.loads((run_dir / "status.json").read_text()) == {
        "status": "failed",
        "error": "model construction failed",
    }
    log = (run_dir / "run.log").read_text()
    assert "Traceback" in log and "model construction failed" in log
    with pytest.raises(FileExistsError):
        train.run_training(config, "pacmann_adam", 1000)


@pytest.mark.parametrize("selector", ["campaign", "method", "seed"])
def test_invalid_selectors_create_no_output(config, tmp_path, selector):
    """Reject invalid external selectors before reserving output directories."""
    method, seed = "fixed_hammersley", 1000
    if selector == "campaign":
        config["campaign"] = "../outside"
    elif selector == "method":
        method = "unknown"
    else:
        seed = 1001
    with pytest.raises(ValueError):
        prepare_run(config, method, seed, "unused", tmp_path)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "field, value",
    [("residual_nx", 0), ("residual_nt", -1), ("batch_size", 0), ("batch_size", 1.5)],
)
def test_invalid_evaluation_sizes_rejected_before_run(
    config, monkeypatch, field, value
):
    """Reject invalid evaluation sizes before creating artifacts or training."""
    config["evaluation"][field] = value
    monkeypatch.setattr(
        train,
        "prepare_run",
        lambda *args: pytest.fail(
            "Invalid evaluation settings reached run preparation"
        ),
    )
    with pytest.raises(ValueError, match=field):
        train.run_training(config, "fixed_hammersley", 1000)
