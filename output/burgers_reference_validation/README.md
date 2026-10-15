# Burgers reference validation

Compare generated N512 and N1024 references with PACMANN at viscosity `0.01/pi`,
then check how reference choice affects errors of three existing PACMANN models.
Use PACMANN's unchanged 256 spatial coordinates and 100 times through `0.99`;
interpolate generated Chebyshev solutions in space at exact shared times.

Inputs are the [three reference datasets](../../data/burgers_dataset/) and
`source_models/seed_{2000,2001,2002}/`. Each model directory retains its original
configuration, metadata, final weights, and predictions from
`burgers_sigma_tuning_01_baseline/burgers/pacmann_adam` on `hyperparameter-tuning`.
The metadata records the training revision, dependencies, device, and dataset
fingerprint. This check reloads weights on CPU in float64, verifies predictions
against their archived arrays, and holds predictions fixed across references.
No training occurs.

## Execution

From the repository root, select `RESOLUTION = 512` or `1024` in
[`burguers_compare_solvers.py`](../../scripts/dataset_validation/burguers_compare_solvers.py),
then run:

```bash
MPLBACKEND=Agg venv/bin/python -m scripts.dataset_validation.burguers_compare_solvers
MPLBACKEND=Agg venv/bin/python -m scripts.dataset_validation.burguers_compare_pinn_solutions
```

These commands regenerate derived tables and figures while preserving inputs.
The [N512](n512_reference_grid/README.md), [N1024](n1024_reference_grid/README.md),
and [fixed-model comparison](pacmann_3seed/README.md) directories describe the
artifacts. See [study context](../../docs/research_context.md),
[results and limitations](../../docs/results.md), and
[artifact meanings](../../docs/output_format.md). Agreement at this viscosity
supports the accepted benchmark reference; it does not certify other regimes.
