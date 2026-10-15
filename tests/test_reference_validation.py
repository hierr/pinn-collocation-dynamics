"""Check common-grid interpolation and saved-weight evaluation with analytic data."""

import csv

import numpy as np
import torch
import yaml

from scripts.dataset_validation import (
    burguers_compare_pinn_solutions as validation,
)
from scripts.dataset_validation import burguers_compare_solvers


def test_reference_check_on_polynomial_solutions(tmp_path, monkeypatch):
    """Recover identical polynomial references across grids and restore saved models."""
    data = tmp_path / "data"
    runs = tmp_path / "runs"
    output = tmp_path / "review"
    data.mkdir()
    monkeypatch.setattr(validation, "DATA", data)
    monkeypatch.setattr(validation, "RUNS", runs)
    monkeypatch.setattr(validation, "OUTPUT", output)
    torch.set_default_device("cpu")
    x = np.linspace(-1, 1, 5)
    times = np.array([0.0, 0.1, 0.2])
    solution = (1 + times[:, None]) * (1 - x[None, :] ** 2)
    np.savez(data / "PACMANN_Burgers.npz", x=x, t=times, usol=solution.T)

    # A quadratic polynomial is exactly recoverable by Chebyshev interpolation.
    snapshots = np.array([0.0, 0.05, 0.1, 0.15, 0.2])
    for resolution in (512, 1024):
        nodes = np.cos(np.pi * np.arange(resolution + 1) / resolution)
        values = (1 + snapshots[:, None]) * (1 - nodes[None, :] ** 2)
        np.savez(
            data / f"burgers_nu_0p0031830989_N{resolution}.npz",
            x=nodes,
            t=snapshots,
            U=values,
            nu=np.float64(0.01 / np.pi),
        )

    # Save independently evaluated weights and predictions as the input artifacts.
    xx, tt = np.meshgrid(x, times)
    coordinates = np.column_stack((xx.ravel(), tt.ravel()))
    for seed in validation.SEEDS:
        run = runs / f"seed_{seed}"
        (run / "models").mkdir(parents=True)
        config = {
            "problem": {"viscosity": 0.01 / np.pi},
            "network": {"layers": [2, 4, 1]},
        }
        (run / "config.yaml").write_text(yaml.safe_dump(config))
        torch.manual_seed(seed)
        network = validation.FNN([2, 4, 1], "tanh", "Glorot normal").double()
        torch.save(network.state_dict(), run / "models/final.pt")
        with torch.no_grad():
            prediction = network(torch.from_numpy(coordinates)).numpy()
        np.savez(
            run / "predictions.npz",
            coordinates=coordinates,
            reference=solution.reshape(-1, 1),
            prediction=prediction,
        )

    validation.main()
    with (output / "model_errors.csv").open() as handle:
        models = list(csv.DictReader(handle))
    assert len(models) == 9
    for seed in validation.SEEDS:
        common = [
            float(row["relative_l2"]) for row in models if int(row["seed"]) == seed
        ]
        np.testing.assert_allclose(common, common[0], rtol=1e-12)
    assert (output / "comparison.png").is_file()

    # The colleague's comparison must also recover identical polynomial references.
    metrics = burguers_compare_solvers.calculate_overall_errors(
        burguers_compare_solvers.load_my_dataset(
            data / "burgers_nu_0p0031830989_N1024.npz"
        ),
        burguers_compare_solvers.load_paper_dataset(data / "PACMANN_Burgers.npz"),
        tmp_path / "colleague_comparison",
    )
    assert metrics["max_absolute_error"] < 1e-12
