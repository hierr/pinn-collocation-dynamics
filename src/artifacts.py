"""Save run provenance, logs, evaluation weights, and training coordinates."""

import contextlib
import importlib.metadata
import json
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml

from src.sampling import interior_points


def write_json(path: Path, payload):
    """Write readable JSON, rejecting non-finite values."""
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def prepare_run(config: dict, method: str, seed: int, dataset_sha256: str, root: Path):
    """Reserve a fresh run directory and save its configuration and provenance."""
    # Validate execution selectors before creating any artifacts.
    campaign = config["campaign"]
    if (
        not isinstance(campaign, str)
        or campaign in ("", ".", "..")
        or "/" in campaign
        or "\\" in campaign
    ):
        raise ValueError("Campaign must be one ordinary path component")
    if (
        method
        not in (
            "fixed_hammersley",
            "random_resampling",
            "rad",
            "pacmann_adam",
            "pacmann_diffusion",
        )
        or method not in config["methods"]
    ):
        raise ValueError("Unknown sampling method")
    if not isinstance(seed, int) or seed < 0 or seed not in config["seeds"]:
        raise ValueError("Seed must be a configured nonnegative integer")

    # Never reuse a previous run, including an incomplete or failed run.
    run_dir = root / "output" / "burgers" / campaign / method / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    for folder in ("models", "points"):
        (run_dir / folder).mkdir()
    resolved_config = dict(config, selected_method=method, selected_seed=seed)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(resolved_config, sort_keys=False)
    )
    metadata = {
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("deepxde", "torch", "numpy", "scipy", "matplotlib", "PyYAML")
        },
        "dataset_sha256": dataset_sha256,
        "device": str(torch.get_default_device()),
    }
    write_json(run_dir / "metadata.json", metadata)
    return run_dir


@contextlib.contextmanager
def capture_run_log(run_dir: Path):
    """Capture terminal output and record run completion or failure."""
    write_json(run_dir / "status.json", {"status": "running"})
    with (
        (run_dir / "run.log").open("w") as log,
        contextlib.redirect_stdout(Tee(sys.stdout, log)),
        contextlib.redirect_stderr(Tee(sys.stderr, log)),
    ):
        try:
            yield
        except Exception as error:
            traceback.print_exc()
            write_json(
                run_dir / "status.json", {"status": "failed", "error": str(error)}
            )
            raise
    write_json(run_dir / "status.json", {"status": "complete"})


def save_points(data, run_dir: Path, name: str, interior_count: int):
    """Save distinct interior, complete PDE, and constraint populations."""
    np.savez_compressed(
        run_dir / "points" / f"{name}.npz",
        interior=interior_points(data, interior_count),
        pde=data.train_x_all.copy(),
        constraints=data.train_x_bc.copy(),
    )


def save_snapshot(model, run_dir: Path, name: str, interior_count: int):
    """Save evaluation weights and their corresponding training coordinates."""
    torch.save(model.net.state_dict(), run_dir / "models" / f"{name}.pt")
    save_points(model.data, run_dir, name, interior_count)


class Tee:
    """Copy training output to both the terminal and the run log."""

    def __init__(self, terminal, log):
        """Store the two output streams."""
        self.terminal, self.log = terminal, log

    def write(self, text):
        """Write text to both streams."""
        self.terminal.write(text)
        return self.log.write(text)

    def flush(self):
        """Flush both output streams."""
        self.terminal.flush()
        self.log.flush()
