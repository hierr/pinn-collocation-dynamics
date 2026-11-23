"""Check the controller sweep, archived controls, and solution-error selection."""

import hashlib
import json
import math

import pytest
import yaml

from scripts.run_experiments import run_burgers_alpha_tuning as tuning
from scripts.run_experiments import run_burgers_campaign as campaign


@pytest.fixture
def sweep(tmp_path, monkeypatch):
    """Use real scientific settings with disposable references and destinations."""
    configs = tuning.load_configurations("0p0031830989")
    reference = tmp_path / "reference.npz"
    reference.write_bytes(b"reference bytes used only for provenance testing")
    for config in configs:
        config["problem"]["reference_path"] = str(reference)
    monkeypatch.setattr(tuning, "ROOT", tmp_path)
    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    monkeypatch.setattr(tuning, "load_configurations", lambda tag: configs)
    return configs


def save_completed(configs, root, errors=None):
    """Write known paired metrics with matching scientific provenance."""
    paths = []
    for index, config in enumerate(configs):
        values = errors[index] if errors is not None else [0.1, 0.2, 0.3]
        method = next(iter(config["methods"]))
        nu_tag = f"{config['problem']['viscosity']:.8g}".replace(".", "p")
        fingerprint = hashlib.sha256(
            (root / config["problem"]["reference_path"]).read_bytes()
        ).hexdigest()
        for seed, error in zip(config["seeds"], values, strict=True):
            run = (
                root
                / "output/burgers"
                / f"nu_{nu_tag}"
                / config["campaign"]
                / method
                / f"seed_{seed}"
            )
            run.mkdir(parents=True)
            (run / "config.yaml").write_text(
                yaml.safe_dump(dict(config, selected_method=method, selected_seed=seed))
            )
            (run / "status.json").write_text(json.dumps({"status": "complete"}))
            (run / "metadata.json").write_text(
                json.dumps(
                    {
                        "revision": f"source-{config['campaign']}",
                        "versions": {"torch": "test"},
                        "dataset_sha256": fingerprint,
                        "device": "cpu",
                    }
                )
            )
            (run / "metrics.json").write_text(
                json.dumps(
                    {"final_relative_l2": error, "final_residual_rms": error / 10}
                )
            )
            (run / "diagnostics.json").write_text("{}")
            (run / "diagnostics.png").write_bytes(b"test figure")
            paths.append(run)
    return paths


def test_all_five_configured_grids_and_archived_controls(tmp_path, monkeypatch):
    """Prepare sixty runs with the frozen base sigma and matched full budgets."""
    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    assert len(tuning.BASE_SIGMAS) == 5
    count = 0
    for nu_tag, sigma_tag in tuning.BASE_SIGMAS.items():
        configs = tuning.load_configurations(nu_tag)
        tuning.check_tuning_configs(configs)
        assert configs[0]["seeds"] == [3000, 3001, 3002]
        assert [
            config["methods"][tuning.METHOD]["alpha"] for config in configs[2:]
        ] == [0.1, 1, 10, 100]
        for config in configs[2:]:
            settings = config["methods"][tuning.METHOD]
            assert settings["sigma"] == float(sigma_tag)
            assert settings["g_base"] == math.pi
            assert config["training"]["phases"] == 5
            assert config["training"]["adam_steps"] == 7000
            campaign.check_campaign(config, tuning.METHOD)
            count += len(config["seeds"])
        rows = tuning.read_results(configs[:2])
        assert all(len(row["errors"]) == 3 for row in rows)
    assert count == 60


@pytest.mark.parametrize(
    "change",
    ["sigma", "alpha", "g_base", "training", "seeds", "method", "cadence", "campaign"],
)
def test_mismatched_controls_are_rejected(sweep, change):
    """Reject changes that would confound the intended one-parameter comparison."""
    candidate = sweep[-1]
    sampling = candidate["methods"][tuning.METHOD]
    if change == "sigma":
        sampling["sigma"] *= 10
    elif change == "alpha":
        sampling["alpha"] = math.inf
    elif change == "g_base":
        sampling["g_base"] = 1.0
    elif change == "training":
        candidate["training"]["adam_steps"] += 1
    elif change == "seeds":
        candidate["seeds"] = [4000, 4001, 4002]
    elif change == "method":
        candidate["methods"] = {"pacmann_diffusion": sampling}
    elif change == "cadence":
        sampling["period"] *= 2
    else:
        candidate["campaign"] = sweep[0]["campaign"]
    with pytest.raises(ValueError):
        tuning.check_tuning_configs(sweep)


def test_selection_includes_constant_and_ties_favor_weaker_attenuation(
    sweep, tmp_path, capsys
):
    """Allow constant diffusion to win and break finite ties with larger alpha."""
    paths = save_completed(sweep, tmp_path)
    rows = tuning.read_results(sweep)
    assert tuning.summarize_tuning(rows, sweep[0]["seeds"])["label"] == "constant"
    for run in paths[6:]:
        metrics = json.loads((run / "metrics.json").read_text())
        metrics["final_relative_l2"] /= 2
        (run / "metrics.json").write_text(json.dumps(metrics))
    rows = tuning.read_results(sweep)
    assert tuning.summarize_tuning(rows, sweep[0]["seeds"])["alpha"] == 100
    assert "no automatic expansion" in capsys.readouterr().out


def test_mean_selection_retains_bad_seeds_and_ignores_residual_ranking(sweep, tmp_path):
    """Keep a large unfavorable outcome and select by full-seed solution error."""
    errors = [
        [0.01] * 3,
        [0.2] * 3,
        [0.001, 0.001, 100],
        [0.15] * 3,
        [0.1] * 3,
        [0.18] * 3,
    ]
    paths = save_completed(sweep, tmp_path, errors)
    # The solution winner deliberately has a much worse residual than every rival.
    for run in paths[12:15]:
        metrics = json.loads((run / "metrics.json").read_text())
        metrics["final_residual_rms"] = 10.0
        (run / "metrics.json").write_text(json.dumps(metrics))
    rows = tuning.read_results(sweep)
    assert rows[2]["errors"][3002] == 100
    assert tuning.summarize_tuning(rows, sweep[0]["seeds"])["alpha"] == 10


@pytest.mark.parametrize("state", ["failed", "running", "missing_diagnostics"])
def test_incomplete_or_failed_seed_blocks_selection(sweep, tmp_path, state):
    """Report the planned seed instead of selecting from a favorable subset."""
    run = save_completed(sweep, tmp_path)[-1]
    if state == "missing_diagnostics":
        (run / "diagnostics.png").unlink()
    else:
        (run / "status.json").write_text(json.dumps({"status": state}))
    rows = tuning.read_results(sweep)
    assert 3002 in rows[-1]["failed" if state == "failed" else "pending"]
    assert tuning.summarize_tuning(rows, sweep[0]["seeds"]) is None


@pytest.mark.parametrize(
    "changed", ["configuration", "dataset_sha256", "versions", "device", "metric"]
)
def test_saved_science_and_provenance_are_checked(sweep, tmp_path, changed):
    """Accept source-revision differences while rejecting scientifically stale data."""
    run = save_completed(sweep, tmp_path)[-1]
    if changed == "configuration":
        path = run / "config.yaml"
        value = yaml.safe_load(path.read_text())
        value["training"]["learning_rate"] *= 2
        path.write_text(yaml.safe_dump(value))
    elif changed == "metric":
        path = run / "metrics.json"
        value = json.loads(path.read_text())
        value["final_relative_l2"] = math.inf
        path.write_text(json.dumps(value))
    else:
        path = run / "metadata.json"
        value = json.loads(path.read_text())
        value[changed] = (
            {"torch": "different"} if changed == "versions" else "different"
        )
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        tuning.read_results(sweep)


@pytest.mark.parametrize("action", ["check", "summarize"])
def test_read_only_actions_never_launch_or_write(sweep, tmp_path, monkeypatch, action):
    """Keep default readiness and reporting independent from scientific execution."""
    save_completed(sweep[:2], tmp_path)
    monkeypatch.setattr(tuning, "ACTION", action)
    monkeypatch.setattr(
        campaign, "run_campaign", lambda *args: pytest.fail("Unexpected training")
    )
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    tuning.main()
    after = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert after == before


def test_check_all_destinations_before_any_launch(sweep, tmp_path, monkeypatch):
    """Find a conflict in the last planned seed before starting the first alpha."""
    save_completed(sweep[:2], tmp_path)
    occupied = (
        tmp_path
        / "output/burgers/nu_0p0031830989/alpha_100"
        / tuning.METHOD
        / "seed_3002"
    )
    occupied.mkdir(parents=True)
    monkeypatch.setattr(tuning, "ACTION", "run")
    monkeypatch.setattr(
        campaign, "run_campaign", lambda *args: pytest.fail("Unexpected training")
    )
    with pytest.raises(FileExistsError):
        tuning.main()


def test_run_selects_only_finite_candidates_and_requires_complete_controls(
    sweep, tmp_path, monkeypatch
):
    """Reuse both controls and execute the four new method batches in order."""
    calls = []
    monkeypatch.setattr(tuning, "ACTION", "run")
    monkeypatch.setattr(
        campaign, "run_campaign", lambda config, method: calls.append((config, method))
    )
    with pytest.raises(ValueError, match="controls must be complete"):
        tuning.main()
    assert not calls
    save_completed(sweep[:2], tmp_path)
    tuning.main()
    assert calls == [(config, tuning.METHOD) for config in sweep[2:]]
