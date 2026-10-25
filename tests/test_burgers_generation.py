"""Check reference preservation and the generator's saved numerical settings."""

import itertools
import json
import re

import numpy as np
import pytest

from scripts.dataset_generation import burgers_spectral as solver
from scripts.dataset_generation import generate_burgers_nu_0p001 as generation


def test_existing_reference_is_preserved(tmp_path, monkeypatch):
    """Reject an occupied destination before invoking the expensive solver."""
    monkeypatch.setattr(generation, "DATA_DIR", tmp_path)
    destination = tmp_path / f"burgers_nu_0p001_N{generation.RESOLUTION}.npz"
    destination.write_bytes(b"existing reference")
    monkeypatch.setattr(
        generation,
        "save_solution_for_pinn",
        lambda **kwargs: pytest.fail("Existing references must not reach the solver"),
    )

    with pytest.raises(FileExistsError, match="Reference already exists"):
        generation.main()
    assert destination.read_bytes() == b"existing reference"


def test_small_reference_records_the_requested_problem(tmp_path, monkeypatch):
    """Exercise the real writer on a small grid and short pre-front time interval."""
    monkeypatch.setattr(generation, "DATA_DIR", tmp_path)
    monkeypatch.setattr(generation, "RESOLUTION", 16)
    monkeypatch.setattr(generation, "FINAL_TIME", 0.01)
    monkeypatch.setattr(generation, "SNAPSHOTS", 3)

    generation.main()

    with np.load(tmp_path / "burgers_nu_0p001_N16.npz", allow_pickle=False) as saved:
        settings = json.loads(str(saved["config_json"]))
        assert float(saved["nu"]) == settings["nu"] == 0.001
        assert settings["N"] == 16
        assert saved["U"].shape == (3, 17)
        assert np.isfinite(saved["U"]).all()
        np.testing.assert_array_equal(saved["t"], [0.0, 0.005, 0.01])
        np.testing.assert_allclose(
            saved["U"][0], -np.sin(np.pi * saved["x"]), atol=1e-14
        )
        np.testing.assert_array_equal(saved["U"][:, [0, -1]], np.zeros((3, 2)))


def test_progress_logging_preserves_solution(monkeypatch, capsys):
    """Throttle progress reports and preserve identical numerical solution arrays."""
    monkeypatch.setattr(solver, "PROGRESS_INTERVAL_SECONDS", np.inf)
    baseline = solver.solve_burgers(nu=0.001, N=16, T=0.01, n_snap=3, max_step=0.005)
    capsys.readouterr()

    # Advance a fake clock so logging can be tested without waiting a minute.
    monkeypatch.setattr(solver, "perf_counter", itertools.count(0, 30).__next__)
    monkeypatch.setattr(solver, "PROGRESS_INTERVAL_SECONDS", 60.0)
    logged = solver.solve_burgers(nu=0.001, N=16, T=0.01, n_snap=3, max_step=0.005)
    output = capsys.readouterr().out

    for expected, actual in zip(baseline, logged):
        np.testing.assert_array_equal(actual, expected)
    assert "Building Chebyshev operators" in output
    assert "reported RHS times include trial stages" in output
    assert "elapsed=60.0s, RHS trial t=" in output
    assert "RHS calls=2\n" in output
    assert "elapsed=30.0s, RHS trial t=" not in output
    counts = re.search(r"RHS calls=(\d+), nfev=(\d+), njev=(\d+), nlu=(\d+)", output)
    assert counts is not None
    rhs_calls, nfev, njev, nlu = map(int, counts.groups())
    assert rhs_calls > nfev  # The manual counter also includes Jacobian probes.
    assert njev > 0 and nlu > 0
