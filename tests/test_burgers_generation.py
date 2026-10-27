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
        assert settings["jacobian"] == "analytic"
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
    assert rhs_calls == nfev  # Analytical Jacobians require no extra RHS probes.
    assert njev > 0 and nlu > 0


@pytest.mark.parametrize("degree", [16, 32])
def test_analytical_jacobian_matches_directional_finite_differences(
    monkeypatch, degree
):
    """Check the production Jacobian on smooth and irregular interior states."""
    original_solve = solver.solve_ivp

    def verify_and_solve(fun, span, initial, **settings):
        rng = np.random.default_rng(42)
        states = [initial, rng.uniform(-0.9, 0.9, size=initial.shape)]
        for state in states:
            direction = rng.normal(size=initial.shape)
            direction /= np.linalg.norm(direction)
            epsilon = 1e-6
            finite_difference = (
                fun(0, state + epsilon * direction)
                - fun(0, state - epsilon * direction)
            ) / (2 * epsilon)
            jacobian = settings["jac"](0, state)
            assert jacobian.shape == (degree - 1, degree - 1)
            derivative = jacobian @ direction
            relative_difference = np.linalg.norm(
                derivative - finite_difference
            ) / np.linalg.norm(finite_difference)
            assert relative_difference < 1e-7
        return original_solve(fun, span, initial, **settings)

    monkeypatch.setattr(solver, "solve_ivp", verify_and_solve)
    solver.solve_burgers(nu=0.001, N=degree, T=0.01, n_snap=3, max_step=0.005)


def test_analytical_and_numerical_jacobians_give_matching_solutions(monkeypatch):
    """Compare short integrations while changing only Radau's Jacobian source."""
    analytic = solver.solve_burgers(
        nu=0.001, N=64, T=0.05, n_snap=11, rtol=1e-11, atol=1e-13, max_step=0.005
    )
    original_solve = solver.solve_ivp

    def solve_with_finite_differences(fun, span, initial, **settings):
        settings["jac"] = None
        return original_solve(fun, span, initial, **settings)

    monkeypatch.setattr(solver, "solve_ivp", solve_with_finite_differences)
    numerical = solver.solve_burgers(
        nu=0.001, N=64, T=0.05, n_snap=11, rtol=1e-11, atol=1e-13, max_step=0.005
    )
    np.testing.assert_array_equal(analytic[0], numerical[0])
    np.testing.assert_array_equal(analytic[1], numerical[1])
    np.testing.assert_allclose(analytic[2], numerical[2], rtol=0, atol=1e-9)
