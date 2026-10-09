"""Check campaign selection, execution order, failures, and generated diagnostics."""

import json

import pytest
import yaml

from scripts import run_burgers_campaign as runner
from src import train
from src.burgers import load_reference


@pytest.fixture
def calls(tmp_path, monkeypatch):
    """Record campaign calls without running training or creating artifacts."""
    recorded = []
    monkeypatch.setattr(runner, "ROOT", tmp_path)

    def record_training(config, method, seed):
        """Return a recognizable run path and record the selected pair."""
        recorded.append(("train", method, seed))
        return tmp_path / method / f"seed_{seed}"

    monkeypatch.setattr(runner, "run_training", record_training)
    monkeypatch.setattr(
        runner, "analyze_run", lambda path: recorded.append(("analyze", path))
    )
    return recorded


def test_configuration_loading_and_execution_order(
    config, tmp_path, monkeypatch, calls
):
    """Use the selected YAML, its seed values, and its method insertion order."""
    config["seeds"] = [2001, 1000]
    config["methods"] = {
        "pacmann_adam": config["methods"]["pacmann_adam"],
        "fixed_hammersley": {},
    }
    config_path = tmp_path / "campaign.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    monkeypatch.setattr(runner, "CONFIG_PATH", config_path)
    runner.main()
    assert calls == [
        ("train", "pacmann_adam", 2001),
        ("analyze", tmp_path / "pacmann_adam/seed_2001"),
        ("train", "fixed_hammersley", 2001),
        ("analyze", tmp_path / "fixed_hammersley/seed_2001"),
        ("train", "pacmann_adam", 1000),
        ("analyze", tmp_path / "pacmann_adam/seed_1000"),
        ("train", "fixed_hammersley", 1000),
        ("analyze", tmp_path / "fixed_hammersley/seed_1000"),
    ]


def test_existing_campaign_with_unused_destinations(config, tmp_path, calls):
    """Allow other runs in the same campaign without changing their artifacts."""
    previous = (
        tmp_path / "output" / config["campaign"] / "burgers/pacmann_adam/seed_999"
    )
    previous.mkdir(parents=True)
    marker = previous / "run.log"
    marker.write_text("previous run\n")
    runner.run_campaign(config)
    assert len(calls) == 4
    assert marker.read_text() == "previous run\n"


def test_later_destination_conflict_prevents_all_training(config, tmp_path, calls):
    """Reject a conflict in the second planned run before executing the first."""
    existing = (
        tmp_path / "output" / config["campaign"] / "burgers/pacmann_adam/seed_1000"
    )
    existing.mkdir(parents=True)
    marker = existing / "run.log"
    marker.write_text("keep this run\n")
    with pytest.raises(FileExistsError, match="Run already exists"):
        runner.run_campaign(config)
    assert calls == []
    assert marker.read_text() == "keep this run\n"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("campaign", "../outside"),
        ("campaign", ""),
        ("campaign", None),
        ("seeds", []),
        ("seeds", [1000, 1000]),
        ("seeds", [1000, -1]),
        ("seeds", [1000, "1001"]),
        ("seeds", [True]),
        ("methods", {}),
        ("methods", {"fixed_hammersley": {}, "unknown": {}}),
    ],
)
def test_invalid_selection_prevents_all_training(config, tmp_path, calls, field, value):
    """Reject invalid or repeated selectors without creating output."""
    config[field] = value
    with pytest.raises(ValueError):
        runner.run_campaign(config)
    assert calls == []
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("failure_stage", ["train", "analyze"])
def test_failure_stops_campaign_and_preserves_artifacts(
    config, tmp_path, monkeypatch, failure_stage
):
    """Preserve the first run and failed pair, and never start later seeds."""
    config["seeds"] = [1000, 1001]
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    started = []

    def simulate_training(config, method, seed):
        """Retain a training artifact before a possible failure."""
        started.append((method, seed))
        path = tmp_path / method / f"seed_{seed}"
        path.mkdir(parents=True)
        (path / "run.log").write_text("retained training output\n")
        if method == "pacmann_adam" and failure_stage == "train":
            raise RuntimeError("training failed")
        return path

    def simulate_analysis(path):
        """Write first-run diagnostics and fail on the second pair if requested."""
        if path.parent.name == "pacmann_adam" and failure_stage == "analyze":
            raise RuntimeError("analysis failed")
        (path / "diagnostics.json").write_text("{}\n")

    monkeypatch.setattr(runner, "run_training", simulate_training)
    monkeypatch.setattr(runner, "analyze_run", simulate_analysis)
    with pytest.raises(RuntimeError, match="failed"):
        runner.run_campaign(config)
    assert started == [("fixed_hammersley", 1000), ("pacmann_adam", 1000)]
    assert (tmp_path / "fixed_hammersley/seed_1000/diagnostics.json").exists()
    assert (tmp_path / "pacmann_adam/seed_1000/run.log").exists()


def test_small_campaign_generates_training_and_diagnostics(
    config, tmp_path, monkeypatch
):
    """Exercise both real methods and their analysis in disposable output."""
    reference = load_reference(train.ROOT)
    monkeypatch.setattr(train, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(train, "load_reference", lambda root: reference)
    monkeypatch.setattr(
        "src.artifacts.subprocess.check_output",
        lambda *args, **kwargs: "test-revision\n",
    )
    config["network"]["layers"] = [2, 8, 1]
    config["training"].update(phases=1, adam_steps=2, display_every=1)
    config["training"]["lbfgs"].update(maxiter=2, maxfun=3)
    config["methods"]["pacmann_adam"].update(period=2, steps=1)
    runner.run_campaign(config)
    for method in config["methods"]:
        run = (
            tmp_path / "output" / config["campaign"] / "burgers" / method / "seed_1000"
        )
        assert json.loads((run / "status.json").read_text()) == {"status": "complete"}
        metrics = json.loads((run / "metrics.json").read_text())
        assert metrics["movement_events"] == (1 if method == "pacmann_adam" else 0)
        diagnostics = json.loads((run / "diagnostics.json").read_text())
        assert diagnostics["maximum_absolute_error"] >= 0
        assert (run / "models/final.pt").is_file()
        assert (run / "predictions.npz").is_file()
        assert (run / "diagnostics.png").stat().st_size > 0
