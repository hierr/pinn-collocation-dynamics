# PINN Collocation Dynamics

Residual-guided collocation and stochastic diffusion for PINNs: evaluating
performance across smooth and sharp-feature PDE regimes.

This research investigates adaptive collocation methods for physics-informed
neural networks, with a possible publication as the longer-term objective.

## Research questions

- When does residual-gradient adaptive collocation improve on low-discrepancy
  sampling, and how does that benefit change across smooth and sharp regimes?
- Does adding stochastic diffusion to PACMANN help or hurt under those conditions?

Sharp solution gradients and gradients of the squared PDE residual are distinct.
The questions above are hypotheses to investigate, not established claims.

## First milestone

Reproduce the upstream Burgers benchmark with PACMANN and its Hammersley
baseline, then add diffusion as a controlled change. DeepXDE with the PyTorch
backend is the intended implementation. The pinned upstream Burgers scripts,
requirements, license, and file manifest are imported under
`references/pacmann/burgers/`. The ignored reference dataset can be restored using
[source provenance instructions](references/upstream.md#restore-the-dataset).
The minimal trainer and locked environment are implemented. Short engineering
checks are separate from a completed scientific reproduction.

## Structure

```text
src/                    Burgers definition, sampler callback, and training entrypoint
config/                 Full reproduction and short validation YAML files
experimental/           Research context, methodology, and experiment records
references/             Source provenance and upstream implementation pin
```

Use `data/` for reference data and
`output/<campaign>/<problem>/<method>/<seed>/` for generated run artifacts.
Both are ignored by Git. Before a campaign, document artifact storage and
retrieval; ignored local files alone are not a reproducibility archive.
Each run must archive its resolved configuration and must not overwrite an
existing run directory. Scientific settings belong in configurations.

Add functionality only when the next experiment requires it. Begin with phase
endpoints, final models, and selected collocation snapshots; exact resume and
campaign automation are deferred.

See [source provenance](references/upstream.md) and
[working agreements](AGENTS.md).

## Setup and execution

Use Python 3.12 and uv; conda is not required for this wheel-based stack.
The lock pins the released DeepXDE 1.14.0 environment with PyTorch 2.7.1/cu128.

```bash
uv venv --python 3.12 venv
VIRTUAL_ENV="$PWD/venv" uv sync --active --locked
venv/bin/python -m src.train --config config/burgers_validation.yaml --method fixed_hammersley --seed-index 0
venv/bin/python -m src.train --config config/burgers_validation.yaml --method pacmann_adam --seed-index 0
```

The validation runs use two short optimizer phases. Existing directories are
rejected; use a new campaign name for a new attempt. DeepXDE selects CUDA when
available, otherwise CPU; the actual device is recorded in metadata.
`config/burgers.yaml` holds the full five-phase, ten-seed reproduction settings.
Full reproduction training and diffusion experiments remain future work.

Each run saves its configuration, provenance, status, log, optimizer timings,
loss/metric history, predictions, phase/final weights, and collocation snapshots.
Weights support evaluation, not exact training resume. Final weights are reloaded
into a fresh model and checked against saved predictions. Both final errors and
upstream history-selected errors are explicitly named.

```bash
venv/bin/python -m pytest --cov=src --cov-report=term-missing
venv/bin/ruff check src tests
venv/bin/ty check
uv lock --check
uv pip check --python venv/bin/python
```

Tests cover the reference checksum/order, analytic PDE residual, coordinate-Adam
parity with the imported upstream callback, fixed constraints, and run isolation.
See [methodology](experimental/methodology.md) for the environment decision.
