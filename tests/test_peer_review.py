"""Check the review demo and completeness of the retained evidence."""

import hashlib
import json

import numpy as np
import pytest
import yaml

from scripts import run_peer_review as runner
from scripts.post_processing import analyze_burgers as analysis
from src import train


def test_review_demo_trains_evaluates_and_rejects_overwrite(
    all_methods_config, tmp_path, monkeypatch
):
    """Exercise the demo with a tiny budget and preserve a completed destination."""
    config = all_methods_config
    config["network"]["layers"] = [2, 8, 1]
    config["training"].update(phases=1, adam_steps=2, display_every=1)
    config["training"]["lbfgs"].update(maxiter=2, maxfun=3)
    config_path = tmp_path / "demo.yaml"
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(runner, "CONFIG_PATH", config_path)
    monkeypatch.setattr(runner, "METHOD", "pacmann_diffusion")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(train, "ROOT", tmp_path)
    monkeypatch.setattr(analysis, "ROOT", tmp_path)
    monkeypatch.setattr(
        "src.artifacts.subprocess.check_output", lambda *args, **kwargs: "test-revision"
    )

    runner.main()
    run = next(tmp_path.glob("output/burgers/*/*/pacmann_diffusion/seed_*"))
    metrics = json.loads((run / "metrics.json").read_text())
    assert np.isfinite(list(metrics.values())).all()
    assert metrics["movement_events"] == 1
    assert (run / "diagnostics.png").is_file()
    assert yaml.safe_load((run / "config.yaml").read_text())["selected_seed"] == 1000

    before = {path: path.read_bytes() for path in run.rglob("*") if path.is_file()}
    with pytest.raises(FileExistsError):
        runner.main()
    assert all(path.read_bytes() == content for path, content in before.items())


def test_compact_evidence_retains_every_evaluation_seed():
    """Keep every planned method/seed and its original PDE and evaluation settings."""
    protocols = sorted(train.ROOT.glob("config/*_10seed.yaml"))
    assert len(protocols) == 5
    count = 0
    for path in protocols:
        config = yaml.safe_load(path.read_text())
        campaign = train.ROOT / "output/burgers" / config["campaign"]
        if not campaign.exists():
            campaign = train.ROOT / "output/burgers/nu_0p001" / config["campaign"]
        fingerprint = hashlib.sha256(
            (train.ROOT / config["problem"]["reference_path"]).read_bytes()
        ).hexdigest()
        assert config["seeds"] == list(range(4000, 4010))
        assert len(config["methods"]) == 5
        for method in config["methods"]:
            for seed in config["seeds"]:
                run = campaign / method / f"seed_{seed}"
                archived = yaml.safe_load((run / "config.yaml").read_text())
                assert archived["selected_seed"] == seed
                assert archived["selected_method"] == method
                for field in (
                    "problem",
                    "network",
                    "training",
                    "evaluation",
                    "methods",
                ):
                    assert archived[field] == config[field]
                metadata = json.loads((run / "metadata.json").read_text())
                assert metadata["dataset_sha256"] == fingerprint
                assert (
                    json.loads((run / "status.json").read_text())["status"]
                    == "complete"
                )
                metrics = json.loads((run / "metrics.json").read_text())
                assert np.isfinite(list(metrics.values())).all()
                count += 1
    assert count == 250
