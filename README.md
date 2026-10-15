# PINN Collocation Dynamics — dataset validation

This branch, `dataset-validation`, retains reference-data generation and
validation for the collocation study. It is a continuing workstream: add future
dataset checks here rather than creating a branch for every validation.

- `dataset-validation`: solvers, validation scripts, inputs, and comparison artifacts.
- `hyperparameter-tuning`: tuning protocols, configurations, and development runs.
- `main`: accepted implementations, frozen evaluation settings, full evaluations,
  and concise accepted conclusions.

Transfer selected code or decisions between branches with targeted commits;
avoid merging all validation or tuning artifacts into `main`. Record the source
commit when carrying a conclusion across. Branches preserve work history;
they do not by themselves establish scientific independence.

## Contents and execution

```text
scripts/dataset_generation/           Collaborator's Burgers reference solver
scripts/dataset_validation/           Solver and fixed-PINN reference comparisons
data/burgers_dataset/                 Original/generated reference datasets
output/burgers_reference_validation/  Comparisons and retained source models
docs/                                Study context, results, and artifact meanings
references/                          Immutable PACMANN source snapshot and provenance
```

Use Python 3.12 and the existing uv-managed `venv`. The locked stack includes
DeepXDE 1.14.0, PyTorch 2.7.1, NumPy 2.3.1, and SciPy 1.16.0. Conda is unnecessary.
For a new checkout:

```bash
uv venv --python 3.12 venv
VIRTUAL_ENV="$PWD/venv" uv sync --active --locked
```

From the repository root, select `RESOLUTION` in the solver-comparison script
(512 or 1024), then run:

```bash
MPLBACKEND=Agg venv/bin/python -m scripts.dataset_validation.burguers_compare_solvers
MPLBACKEND=Agg venv/bin/python -m scripts.dataset_validation.burguers_compare_pinn_solutions
```

These regenerate derived comparisons; they do not train models or regenerate
reference datasets. The fixed-model check runs on CPU in float64. Dataset and
model inputs are tracked here. The collaborator's generator is
`scripts/dataset_generation/burgers_spectral.py`; review its settings before generating
new reference data, and preserve existing datasets.

See the [comparison protocol](output/burgers_reference_validation/README.md),
[results](docs/results.md), [study context](docs/research_context.md),
[artifact dictionary](docs/output_format.md), and [upstream provenance](references/upstream.md).

```bash
venv/bin/python -m pytest --cov=scripts.dataset_validation --cov-report=term-missing
venv/bin/ruff check scripts/dataset_validation/burguers_compare_pinn_solutions.py tests
venv/bin/ty check scripts/dataset_validation tests
```

The collaborator's solver retains its original formatting and numerical logic.
