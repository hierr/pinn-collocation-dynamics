"""Check sequential method batches, concurrent seeds, and saved-run evaluation."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from scripts.run_experiments import run_burgers_campaign as runner
from src import train
from src.burgers import load_reference


@pytest.fixture
def calls(tmp_path, monkeypatch):
    """Record launch order and execute mocked workers when their processes join."""
    recorded = []
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner.torch, "set_num_threads", lambda count: None)
    monkeypatch.setattr(runner.torch, "set_num_interop_threads", lambda count: None)

    def record_training(config, method, seed):
        """Return a recognizable path and record the selected run."""
        recorded.append(("train", method, seed))
        return tmp_path / method / f"seed_{seed}"

    class Process:
        """Model independent child exit codes without starting expensive processes."""

        def __init__(self, target, args):
            """Retain the worker and its selected configuration, method, and seed."""
            self.target, self.args = target, args
            self.exitcode = None

        def start(self):
            """Record that this worker was started before the parent joins any worker."""
            recorded.append(("start", self.args[1], self.args[2]))

        def join(self):
            """Finish the worker independently, preserving a failed exit code."""
            recorded.append(("join", self.args[1], self.args[2]))
            try:
                self.target(*self.args)
            except RuntimeError:
                self.exitcode = 1
            else:
                self.exitcode = 0

    def context(name):
        """Require the CUDA-compatible spawn start method."""
        assert name == "spawn"
        return SimpleNamespace(Process=Process)

    monkeypatch.setattr(runner.multiprocessing, "get_context", context)
    monkeypatch.setattr(runner, "run_training", record_training)
    monkeypatch.setattr(
        runner, "analyze_run", lambda path: recorded.append(("analyze", path))
    )
    return recorded


@pytest.mark.parametrize(
    "methods", [["pacmann_adam"], ["pacmann_adam", "fixed_hammersley"]]
)
def test_configuration_loading_and_execution_order(
    config, tmp_path, monkeypatch, calls, methods
):
    """Finish each selected method's concurrent seed batch before starting the next."""
    config["seeds"] = [2001, 1000]
    config_path = tmp_path / "campaign.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    monkeypatch.setattr(runner, "CONFIG_PATH", config_path)
    monkeypatch.setattr(runner, "METHODS", methods)
    runner.main()
    expected = []
    for method in methods:
        expected.extend(
            [
                ("start", method, 2001),
                ("start", method, 1000),
                ("join", method, 2001),
                ("train", method, 2001),
                ("analyze", tmp_path / method / "seed_2001"),
                ("join", method, 1000),
                ("train", method, 1000),
                ("analyze", tmp_path / method / "seed_1000"),
            ]
        )
    assert calls == expected


def test_later_method_conflict_prevents_entire_sequence(
    config, tmp_path, monkeypatch, calls
):
    """Reject an occupied later batch before starting the first method."""
    config_path = tmp_path / "campaign.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    monkeypatch.setattr(runner, "CONFIG_PATH", config_path)
    monkeypatch.setattr(runner, "METHODS", ["pacmann_adam", "fixed_hammersley"])
    existing = (
        tmp_path
        / "output/burgers/nu_0p0031830989"
        / config["campaign"]
        / "fixed_hammersley/seed_1000"
    )
    existing.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="Run already exists"):
        runner.main()
    assert calls == []


def test_failed_method_prevents_next_batch(config, tmp_path, monkeypatch, calls):
    """Finish healthy workers in a failed batch without launching another method."""
    config["seeds"] = [1000, 1001]
    config_path = tmp_path / "campaign.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    monkeypatch.setattr(runner, "CONFIG_PATH", config_path)
    monkeypatch.setattr(runner, "METHODS", ["pacmann_adam", "fixed_hammersley"])

    def fail_first_seed(config, method, seed):
        """Simulate one failed seed while allowing its batch peer to finish."""
        if seed == 1000:
            raise RuntimeError("training failed")
        return tmp_path / method / f"seed_{seed}"

    monkeypatch.setattr(runner, "run_training", fail_first_seed)
    with pytest.raises(RuntimeError, match=r"failed for seeds \[1000\]"):
        runner.main()
    assert ("analyze", tmp_path / "pacmann_adam/seed_1001") in calls
    assert not any(event[:2] == ("start", "fixed_hammersley") for event in calls)


def test_other_methods_and_unused_destinations_are_allowed(config, tmp_path, calls):
    """Keep other methods and earlier seeds untouched when selecting a new batch."""
    previous = (
        tmp_path
        / "output/burgers/nu_0p0031830989"
        / config["campaign"]
        / "pacmann_adam/seed_1000"
    )
    previous.mkdir(parents=True)
    marker = previous / "run.log"
    marker.write_text("previous run\n")
    runner.run_campaign(config, "fixed_hammersley")
    assert ("train", "fixed_hammersley", 1000) in calls
    assert marker.read_text() == "previous run\n"


def test_later_destination_conflict_prevents_all_workers(config, tmp_path, calls):
    """Reject a conflict at the final selected seed before starting any worker."""
    config["seeds"] = [1000, 1001]
    existing = (
        tmp_path
        / "output/burgers/nu_0p0031830989"
        / config["campaign"]
        / "pacmann_adam/seed_1001"
    )
    existing.mkdir(parents=True)
    marker = existing / "run.log"
    marker.write_text("keep this run\n")
    with pytest.raises(FileExistsError, match="Run already exists"):
        runner.run_campaign(config, "pacmann_adam")
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
def test_invalid_selection_prevents_all_workers(config, tmp_path, calls, field, value):
    """Reject invalid protocol selectors without starting a worker or writing output."""
    config[field] = value
    with pytest.raises(ValueError):
        runner.run_campaign(config, "fixed_hammersley")
    assert calls == []
    assert not (tmp_path / "output").exists()


def test_unconfigured_method_is_rejected(config, calls):
    """Reject a supported method that is absent from the selected protocol."""
    with pytest.raises(ValueError):
        runner.run_campaign(config, "rad")
    assert calls == []


def test_start_failure_waits_for_already_started_worker(config, monkeypatch, calls):
    """Join a started worker if launching the next process fails."""
    config["seeds"] = [1000, 1001]
    context = runner.multiprocessing.get_context("spawn")
    process_type = context.Process

    def process(target, args):
        """Fail startup for the second seed while preserving the first process."""
        result = process_type(target=target, args=args)
        if args[2] == 1001:

            def fail_to_start():
                """Simulate a process-creation failure before this worker starts."""
                raise OSError("cannot start process")

            monkeypatch.setattr(result, "start", fail_to_start)
        return result

    monkeypatch.setattr(
        runner.multiprocessing,
        "get_context",
        lambda name: SimpleNamespace(Process=process),
    )
    with pytest.raises(OSError, match="cannot start process"):
        runner.run_campaign(config, "pacmann_adam")
    assert ("join", "pacmann_adam", 1000) in calls
    assert ("train", "pacmann_adam", 1000) in calls
    assert not any(event[-1] == 1001 for event in calls if event[0] != "analyze")


@pytest.mark.parametrize("failure_stage", ["train", "analyze"])
def test_failure_preserves_artifacts_and_finishes_other_workers(
    config, tmp_path, monkeypatch, calls, failure_stage
):
    """Report a failed seed after allowing the other launched seed to complete."""
    config["seeds"] = [1000, 1001]

    def simulate_training(config, method, seed):
        """Retain a log before failing one seed or returning its directory."""
        path = tmp_path / method / f"seed_{seed}"
        path.mkdir(parents=True)
        (path / "run.log").write_text("retained training output\n")
        if seed == 1000 and failure_stage == "train":
            raise RuntimeError("training failed")
        return path

    def simulate_analysis(path):
        """Write diagnostics except for the deliberately failed analysis."""
        if path.name == "seed_1000" and failure_stage == "analyze":
            raise RuntimeError("analysis failed")
        (path / "diagnostics.json").write_text("{}\n")

    monkeypatch.setattr(runner, "run_training", simulate_training)
    monkeypatch.setattr(runner, "analyze_run", simulate_analysis)
    with pytest.raises(RuntimeError, match=r"failed for seeds \[1000\]"):
        runner.run_campaign(config, "pacmann_adam")
    assert calls[:2] == [
        ("start", "pacmann_adam", 1000),
        ("start", "pacmann_adam", 1001),
    ]
    assert ("join", "pacmann_adam", 1001) in calls
    assert (tmp_path / "pacmann_adam/seed_1000/run.log").exists()
    assert (tmp_path / "pacmann_adam/seed_1001/diagnostics.json").exists()


def record_spawned_seed(config, method, seed):
    """Write a CPU-only marker from a real spawned child, optionally failing one."""
    record = {
        "pid": os.getpid(),
        "method": method,
        "threads": {
            name: os.environ[name]
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
    }
    (Path(config["spawn_output"]) / f"seed_{seed}.json").write_text(json.dumps(record))
    if config["fail_child"] and seed == 1000:
        raise RuntimeError("deliberate child failure")


@pytest.mark.parametrize("fail_child", [False, True])
def test_real_spawn_isolation_and_failure_collection(
    config, tmp_path, monkeypatch, fail_child
):
    """Exercise real process isolation and independent exits without running training."""
    config.update(seeds=[1000, 1001], spawn_output=str(tmp_path), fail_child=fail_child)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "run_seed", record_spawned_seed)
    if fail_child:
        with pytest.raises(RuntimeError, match=r"failed for seeds \[1000\]"):
            runner.run_campaign(config, "fixed_hammersley")
    else:
        runner.run_campaign(config, "fixed_hammersley")
    records = [
        json.loads((tmp_path / f"seed_{seed}.json").read_text())
        for seed in config["seeds"]
    ]
    assert len({record["pid"] for record in records}) == 2
    for record in records:
        assert record["pid"] != os.getpid()
        assert record["method"] == "fixed_hammersley"
        assert set(record["threads"].values()) == {"1"}


def test_workers_generate_training_and_diagnostics(
    all_methods_config, tmp_path, monkeypatch
):
    """Exercise the worker for all methods with matched initial weights and points."""
    config = all_methods_config
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
    # These direct worker calls share the test process; production workers each set this once.
    monkeypatch.setattr(runner.torch, "set_num_interop_threads", lambda count: None)
    initial_weights = []
    build_model = train.build_model

    def capture_initial_weights(**kwargs):
        """Capture construction-time weights before each method starts training."""
        model, gradient = build_model(**kwargs)
        initial_weights.append(
            {
                name: value.detach().clone()
                for name, value in model.net.state_dict().items()
            }
        )
        return model, gradient

    monkeypatch.setattr(train, "build_model", capture_initial_weights)
    config["network"]["layers"] = [2, 8, 1]
    config["training"].update(phases=1, adam_steps=2, display_every=1)
    config["training"]["lbfgs"].update(maxiter=2, maxfun=3)
    initial_points = []
    for method in config["methods"]:
        runner.run_seed(config, method, 1000)
        run = (
            tmp_path
            / "output/burgers/nu_0p0031830989"
            / config["campaign"]
            / method
            / "seed_1000"
        )
        assert json.loads((run / "status.json").read_text()) == {"status": "complete"}
        metrics = json.loads((run / "metrics.json").read_text())
        assert metrics["movement_events"] == (0 if method == "fixed_hammersley" else 1)
        assert (run / "predictions.npz").is_file()
        assert (run / "diagnostics.png").is_file()
        with np.load(run / "points/initial.npz") as points:
            initial_points.append({name: points[name].copy() for name in points.files})
    for weights, points in zip(initial_weights[1:], initial_points[1:]):
        for name, value in weights.items():
            torch.testing.assert_close(value, initial_weights[0][name], rtol=0, atol=0)
        for name, value in points.items():
            np.testing.assert_array_equal(value, initial_points[0][name])
