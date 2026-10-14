"""Verify matched sweep controls, execution safeguards, and artifact summaries."""

import json

import pytest
import yaml

from scripts.run_experiments import run_burgers_campaign as campaign
from scripts.run_experiments import run_burgers_sigma_tuning as tuning
from src import train
from src.burgers import load_reference


@pytest.fixture
def sweep(tmp_path, monkeypatch):
    """Use the actual tuning YAMLs and disposable output destinations."""
    configs = tuning.load_configurations()
    monkeypatch.setattr(tuning, "ROOT", tmp_path)
    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    return configs


def save_completed(configs, root, errors=None):
    """Create minimal saved-run fixtures with known seed errors and provenance."""
    if errors is None:
        errors = [
            [0.1, 0.2, 0.3],
            [0.08, 0.18, 0.28],
            [0.05, 0.15, 0.25],
            [0.05, 0.15, 0.25],
            [0.2, 0.3, 0.4],
        ]
    paths = []
    for config, values in zip(configs, errors, strict=True):
        method = next(iter(config["methods"]))
        for seed, error in zip(config["seeds"], values, strict=True):
            run = (
                root
                / "output"
                / config["campaign"]
                / "burgers"
                / method
                / f"seed_{seed}"
            )
            run.mkdir(parents=True)
            (run / "config.yaml").write_text(
                yaml.safe_dump(dict(config, selected_method=method, selected_seed=seed))
            )
            (run / "status.json").write_text(json.dumps({"status": "complete"}))
            (run / "metrics.json").write_text(
                json.dumps({"final_relative_l2": error, "training_seconds": 10.0})
            )
            (run / "metadata.json").write_text(
                json.dumps(
                    {
                        "revision": "test",
                        "versions": {"torch": "test"},
                        "dataset_sha256": "reference",
                        "device": "cpu",
                    }
                )
            )
            (run / "diagnostics.json").write_text("{}")
            (run / "diagnostics.png").write_bytes(b"test figure")
            paths.append(run)
    return paths


def test_prepared_grid_matches_pilot(sweep):
    """Keep all full-budget controls and three tuning seeds consistent."""
    tuning.check_tuning_configs(sweep)
    pilot = yaml.safe_load(
        (tuning.CONFIG_PATHS[0].parent / "burgers_pilot.yaml").read_text()
    )
    assert sweep[0]["seeds"] == [2000, 2001, 2002]
    for config in sweep:
        for field in ("problem", "network", "training"):
            assert config[field] == pilot[field]
    assert [
        config["methods"]["pacmann_diffusion"]["sigma"] for config in sweep[1:]
    ] == [1e-5, 1e-4, 1e-3, 1e-2]


def test_preflight_entire_sweep_before_execution(sweep, tmp_path, monkeypatch):
    """Find a final-configuration conflict before launching any training."""
    last = sweep[-1]
    existing = (
        tmp_path / "output" / last["campaign"] / "burgers/pacmann_diffusion/seed_2002"
    )
    existing.mkdir(parents=True)
    calls = []
    monkeypatch.setattr(campaign, "run_campaign", lambda config: calls.append(config))
    with pytest.raises(FileExistsError):
        tuning.run_tuning(sweep)
    assert calls == []
    assert not (tmp_path / "output" / sweep[0]["campaign"]).exists()


def test_sweep_order(sweep, monkeypatch):
    """Execute baseline then increasing sigma in configured order."""
    calls = []
    monkeypatch.setattr(
        campaign, "run_campaign", lambda config: calls.append(config["campaign"])
    )
    tuning.run_tuning(sweep)
    assert calls == [config["campaign"] for config in sweep]


def test_sweep_stops_on_failure(sweep, tmp_path, monkeypatch):
    """Retain launched campaign directories and never start later configurations."""
    calls = []

    def execute(config):
        """Fail on the second campaign after recording the execution order."""
        calls.append(config["campaign"])
        if len(calls) == 2:
            raise RuntimeError("analysis failed")

    monkeypatch.setattr(campaign, "run_campaign", execute)
    with pytest.raises(RuntimeError, match="analysis failed"):
        tuning.run_tuning(sweep)
    assert calls == [config["campaign"] for config in sweep[:2]]
    assert not (tmp_path / "output" / sweep[2]["campaign"]).exists()


@pytest.mark.parametrize(
    "change",
    [
        "seed",
        "problem",
        "budget",
        "sampling",
        "method",
        "campaign",
        "sigma",
        "empty",
        "repeated_seeds",
    ],
)
def test_unmatched_controls_rejected(sweep, monkeypatch, change):
    """Reject unmatched settings or overlapping configurations before training."""
    if change == "seed":
        sweep[1]["seeds"] = [1000]
    elif change == "problem":
        sweep[1]["problem"]["viscosity"] *= 2
    elif change == "budget":
        sweep[1]["training"]["adam_steps"] = 1
    elif change == "sampling":
        sweep[1]["methods"]["pacmann_diffusion"]["period"] = 1
    elif change == "method":
        sweep[1]["methods"] = {"fixed_hammersley": {}}
    elif change == "campaign":
        sweep[1]["campaign"] = sweep[0]["campaign"]
    elif change == "sigma":
        sweep[1]["methods"]["pacmann_diffusion"]["sigma"] = float("nan")
    elif change == "empty":
        sweep.clear()
    else:
        sweep[0]["seeds"] = [2000, 2000]
    calls = []
    monkeypatch.setattr(campaign, "run_campaign", lambda config: calls.append(config))
    with pytest.raises(ValueError):
        tuning.run_tuning(sweep)
    assert calls == []


def test_summary_known_statistics_pairing_and_tie(sweep, tmp_path):
    """Compute sample SD, paired differences, timing, and smaller-sigma tie break."""
    save_completed(sweep, tmp_path)
    result = tuning.summarize_tuning(sweep)
    assert "Selected sigma by mean final relative L2: 0.0001." in result
    assert (
        "| 0.0001 | 3 | 3 | 0 | 0 | 2000: 5; 2001: 15; 2002: 25 | 15 | 10 | -5 | 3/3 | 30.0 |"
        in result
    )
    assert "search boundary" not in result


@pytest.mark.parametrize("best", [0, 1, 4])
def test_zero_and_search_boundary_winners(sweep, tmp_path, best):
    """Allow zero to win and flag either positive boundary for review."""
    errors = [[0.2, 0.3, 0.4] for _ in sweep]
    errors[best] = [0.01, 0.02, 0.03]
    save_completed(sweep, tmp_path, errors)
    result = tuning.summarize_tuning(sweep)
    sigma = 0 if best == 0 else sweep[best]["methods"]["pacmann_diffusion"]["sigma"]
    assert f"Selected sigma by mean final relative L2: {sigma:g}." in result
    assert ("search boundary" in result) == (best != 0)


@pytest.mark.parametrize("state", ["missing", "failed", "running", "analysis_missing"])
def test_incomplete_sweep_never_selects_sigma(sweep, tmp_path, state):
    """Keep seed counts visible and avoid means over only surviving seeds."""
    paths = save_completed(sweep, tmp_path)
    if state == "missing":
        (paths[-1] / "status.json").unlink()
    elif state == "analysis_missing":
        (paths[-1] / "diagnostics.png").unlink()
    else:
        (paths[-1] / "status.json").write_text(json.dumps({"status": state}))
    result = tuning.summarize_tuning(sweep)
    assert "No sigma selected:" in result
    assert "Selected sigma by" not in result
    count = "1 | 0" if state == "failed" else "0 | 1"
    assert f"| 0.01 | 3 | 2 | {count} |" in result
    assert "2002: unavailable | — | — | — |" in result


@pytest.mark.parametrize("field", ["config", "metadata", "metric"])
def test_stale_or_invalid_artifacts_rejected(sweep, tmp_path, field):
    """Prevent ranking results from changed settings, provenance, or invalid errors."""
    paths = save_completed(sweep, tmp_path)
    run = paths[-1]
    if field == "config":
        path = run / "config.yaml"
        saved = yaml.safe_load(path.read_text())
        saved["training"]["phases"] += 1
        path.write_text(yaml.safe_dump(saved))
    elif field == "metadata":
        path = run / "metadata.json"
        saved = json.loads(path.read_text())
        saved["revision"] = "another-checkpoint"
        path.write_text(json.dumps(saved))
    else:
        path = run / "metrics.json"
        saved = json.loads(path.read_text())
        saved["final_relative_l2"] = -1
        path.write_text(json.dumps(saved))
    with pytest.raises(ValueError):
        tuning.summarize_tuning(sweep)


def test_summary_action_without_writing_or_training(
    sweep, tmp_path, monkeypatch, capsys
):
    """Summarize without launching training or creating output directories."""
    calls = []
    monkeypatch.setattr(tuning, "load_configurations", lambda: sweep)
    monkeypatch.setattr(tuning, "run_tuning", lambda configs: calls.append(configs))
    monkeypatch.setattr(tuning, "ACTION", "summarize")
    tuning.main()
    assert "No sigma selected:" in capsys.readouterr().out
    assert calls == [] and not (tmp_path / "output").exists()


def test_run_action_and_invalid_action(sweep, monkeypatch, capsys):
    """Dispatch explicit execution before summarization and reject unknown actions."""
    calls = []
    monkeypatch.setattr(tuning, "load_configurations", lambda: sweep)
    monkeypatch.setattr(tuning, "run_tuning", lambda configs: calls.append("run"))
    monkeypatch.setattr(tuning, "summarize_tuning", lambda configs: "summary")
    monkeypatch.setattr(tuning, "ACTION", "run")
    tuning.main()
    assert calls == ["run"] and capsys.readouterr().out == "summary\n"
    monkeypatch.setattr(tuning, "ACTION", "invalid")
    with pytest.raises(ValueError, match="ACTION"):
        tuning.main()


def test_tiny_sweep_trains_analyzes_and_summarizes(sweep, tmp_path, monkeypatch):
    """Exercise all five settings and three seeds with a disposable tiny budget."""
    reference = load_reference(train.ROOT)
    monkeypatch.setattr(train, "ROOT", tmp_path)
    monkeypatch.setattr(train, "load_reference", lambda root: reference)
    monkeypatch.setattr(
        "src.artifacts.subprocess.check_output",
        lambda *args, **kwargs: "test-revision\n",
    )
    for config in sweep:
        config["problem"]["interior_points"] = 16
        config["network"]["layers"] = [2, 8, 1]
        config["training"].update(phases=1, adam_steps=2, display_every=1)
        config["training"]["lbfgs"].update(maxiter=2, maxfun=3)
        method = next(iter(config["methods"]))
        config["methods"][method].update(period=2, steps=1)
    tuning.run_tuning(sweep)
    result = tuning.summarize_tuning(sweep)
    assert "Selected sigma by mean final relative L2:" in result
    assert "No sigma selected" not in result
    for config in sweep:
        directory = tmp_path / "output" / config["campaign"]
        method = next(iter(config["methods"]))
        for seed in config["seeds"]:
            run = directory / "burgers" / method / f"seed_{seed}"
            metrics = json.loads((run / "metrics.json").read_text())
            assert metrics["movement_events"] == 1
            assert (run / "diagnostics.png").stat().st_size > 0
