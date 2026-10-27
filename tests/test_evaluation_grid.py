"""Check bounded grid selection and preservation during baseline rescoring."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from scripts.post_processing import check_burgers_evaluation_grid as grid
from scripts.run_experiments import run_burgers_sigma_tuning as tuning


def test_fixed_model_scores_keep_time_sampling(config, monkeypatch):
    """Use the real scoring functions with a predictor of known solution/residual error."""
    monkeypatch.setattr(grid, "SPATIAL_POINTS", (17, 33))
    visits = []

    def predict(coordinates, operator=None):
        """Return zero solution predictions or a constant PDE residual."""
        visits.append((coordinates.copy(), operator))
        return np.full((len(coordinates), 1), 0.0 if operator is None else 0.25)

    model = SimpleNamespace(predict=predict, data=SimpleNamespace(pde=object()))
    fingerprint = hashlib.sha256(
        Path(config["problem"]["reference_path"]).read_bytes()
    ).hexdigest()
    report = grid.compare_grids(model, config, fingerprint)

    assert report["selected_spatial_points"] == 17
    assert report["solution_nt"] == 3 and report["residual_nt"] == 8
    assert [row["solution_relative_l2"] for row in report["scores"]] == [1.0, 1.0]
    assert [row["residual_rms"] for row in report["scores"]] == [0.25, 0.25]
    residual_times = np.unique(
        np.concatenate(
            [points[:, 1] for points, operator in visits if operator is not None]
        )
    )
    np.testing.assert_allclose(residual_times, (np.arange(8) + 0.5) * 0.99 / 8)


@pytest.mark.parametrize(
    "solution, residual, selected, count",
    [
        ([1.005, 1, 1, 1], [1.005, 1, 1, 1], 1024, 2),
        ([101, 100, 100, 100], [1, 1, 1, 1], 4096, 3),
        ([2, 1.005, 1, 1], [2, 1.005, 1, 1], 4096, 3),
        ([4, 2, 1.005, 1], [4, 2, 1.005, 1], 8192, 4),
        ([4, 3, 2, 1], [4, 3, 2, 1], None, 4),
        ([1, 1, 1, 1], [2, 1.005, 1, 1], 4096, 3),
        ([0, 0, 0, 0], [0, 0, 0, 0], 1024, 2),
        ([1, 0, 0, 0], [1, 0, 0, 0], 4096, 3),
    ],
)
def test_grid_selection_boundaries(
    config, monkeypatch, solution, residual, selected, count
):
    """Select the first verified coarse grid, requiring both metrics to pass."""
    nx_values = list(grid.SPATIAL_POINTS)
    monkeypatch.setattr(
        grid,
        "load_reference",
        lambda root, problem, nx: (np.zeros((nx * 3, 2)), np.ones((nx * 3, 1)), "data"),
    )
    monkeypatch.setattr(
        grid,
        "predict_batches",
        lambda model, coordinates, batch_size: np.zeros_like(coordinates[:, :1]),
    )
    monkeypatch.setattr(
        grid,
        "solution_errors",
        lambda reference, prediction: {
            "final_relative_l2": solution[nx_values.index(len(reference) // 3)]
        },
    )
    monkeypatch.setattr(
        grid,
        "residual_rms",
        lambda model, problem, nx, nt, batch_size: residual[nx_values.index(nx)],
    )

    report = grid.compare_grids(None, config, "data")

    assert report["selected_spatial_points"] == selected
    assert len(report["scores"]) == count
    json.dumps(report, allow_nan=False)


def test_changed_reference_is_rejected(config, monkeypatch):
    """Prevent comparing a baseline checkpoint against a different reference file."""
    monkeypatch.setattr(grid, "load_reference", lambda *args: (None, None, "changed"))
    with pytest.raises(ValueError, match="fingerprint"):
        grid.compare_grids(None, config, "original")


@pytest.fixture
def baseline_record(tmp_path, monkeypatch):
    """Create original baseline records and five disposable full configurations."""
    configs = tuning.load_configurations()
    paths = []
    for index, config in enumerate(configs):
        path = tmp_path / f"config_{index}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        paths.append(path)
    run = tmp_path / "output/burgers/nu_0p001/sigma_0/pacmann_adam/seed_3000"
    run.mkdir(parents=True)
    (run / "config.yaml").write_text(
        yaml.safe_dump(
            dict(configs[0], selected_method="pacmann_adam", selected_seed=3000),
            sort_keys=False,
        )
    )
    (run / "models").mkdir()
    (run / "models/final.pt").write_bytes(b"fixed weights")
    (run / "metadata.json").write_text(
        json.dumps({"revision": "training-source", "dataset_sha256": "data"})
    )
    (run / "status.json").write_text(json.dumps({"status": "complete"}))
    (run / "training.json").write_text('{"stages": []}\n')
    for name in (
        "predictions.npz",
        "metrics.json",
        "diagnostics.json",
        "diagnostics.png",
    ):
        (run / name).write_bytes(f"original {name}".encode())
    monkeypatch.setattr(grid.campaign, "ROOT", tmp_path)
    return run, paths


def test_rescoring_preserves_original_records(baseline_record, monkeypatch):
    """Resolve all scoring configurations without changing training data or weights."""
    run, paths = baseline_record
    originals = {
        name: (run / name).read_bytes()
        for name in (
            "config.yaml",
            "predictions.npz",
            "metrics.json",
            "diagnostics.json",
            "diagnostics.png",
        )
    }
    training = {
        name: (run / name).read_bytes()
        for name in ("metadata.json", "training.json", "models/final.pt")
    }
    calls = []
    monkeypatch.setattr(grid, "analyze_run", lambda path: calls.append(path))

    grid.apply_selected_grid(run, paths, 4096)

    for name, content in originals.items():
        assert (run / "initial_evaluation" / name).read_bytes() == content
    for name, content in training.items():
        assert (run / name).read_bytes() == content
    saved = yaml.safe_load((run / "config.yaml").read_text())
    original = yaml.safe_load(originals["config.yaml"])
    settings = saved.pop("evaluation")
    original.pop("evaluation")
    assert saved == original
    assert settings == {
        "solution_nx": 4096,
        "residual_nx": 4096,
        "residual_nt": 200,
        "batch_size": 4096,
    }
    assert all(
        yaml.safe_load(path.read_text())["evaluation"] == settings for path in paths
    )
    grid.apply_selected_grid(run, paths, 4096)
    assert calls == [run]


def test_unchanged_grid_keeps_files_untouched(baseline_record, monkeypatch):
    """Keep the completed baseline exactly as it is when the original grid passes."""
    run, paths = baseline_record
    before = {path: path.read_bytes() for path in [*run.glob("*.*"), *paths]}
    monkeypatch.setattr(
        grid, "analyze_run", lambda run: pytest.fail("Unexpected rescore")
    )
    grid.apply_selected_grid(run, paths, 1024)
    assert not (run / "initial_evaluation").exists()
    assert all(path.read_bytes() == content for path, content in before.items())


def test_unresolved_grid_changes_nothing(baseline_record):
    """Stop before configuration changes when no tested refinement passes."""
    run, paths = baseline_record
    with pytest.raises(ValueError, match="No evaluation grid"):
        grid.apply_selected_grid(run, paths, None)
    assert not (run / "initial_evaluation").exists()


def test_later_runs_block_grid_changes(baseline_record, monkeypatch):
    """Protect already archived runs from a retroactive campaign density change."""
    run, paths = baseline_record
    (run.parent / "seed_3001").mkdir()
    monkeypatch.setattr(
        grid, "analyze_run", lambda run: pytest.fail("Unexpected rescore")
    )
    with pytest.raises(FileExistsError):
        grid.apply_selected_grid(run, paths, 4096)
    assert not (run / "initial_evaluation").exists()


def test_mismatched_baseline_blocks_grid_changes(baseline_record):
    """Reject a checkpoint whose archived scientific settings differ from the sweep."""
    run, paths = baseline_record
    saved = yaml.safe_load((run / "config.yaml").read_text())
    saved["problem"]["viscosity"] = 0.1
    (run / "config.yaml").write_text(yaml.safe_dump(saved))
    with pytest.raises(ValueError, match="configuration differs"):
        grid.apply_selected_grid(run, paths, 4096)
    assert not (run / "initial_evaluation").exists()


def test_main_reloads_without_training(baseline_record, monkeypatch):
    """Read fixed weights and record the training revision before resolving density."""
    run, paths = baseline_record
    events = []
    model = SimpleNamespace(
        compile=lambda *args, **kwargs: events.append("compile"),
        net=SimpleNamespace(load_state_dict=lambda state: events.append(state)),
    )
    monkeypatch.setattr(grid, "RUN_DIR", run)
    monkeypatch.setattr(grid, "CONFIG_PATHS", paths)
    monkeypatch.setattr(grid, "build_model", lambda *args: (model, None))
    monkeypatch.setattr(grid.torch, "load", lambda *args, **kwargs: "weights")
    monkeypatch.setattr(
        grid, "compare_grids", lambda *args: {"selected_spatial_points": 1024}
    )
    grid.main()
    report = json.loads((run.parent.parent / "evaluation_grid_check.json").read_text())
    assert report["training_revision"] == "training-source"
    assert events == ["compile", "weights"]


def test_main_rejects_incomplete_baseline(baseline_record, monkeypatch):
    """Never score a checkpoint as final before its training has completed."""
    run, _ = baseline_record
    (run / "status.json").write_text('{"status": "running"}')
    monkeypatch.setattr(grid, "RUN_DIR", run)
    with pytest.raises(ValueError, match="finish training"):
        grid.main()
