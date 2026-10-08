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
docs/                   Research context, methodology, experiment records, and output dictionary
references/             Source provenance and upstream implementation pin
```

Use `data/` for reference data and
`output/<campaign>/<problem>/<method>/<seed>/` for generated run artifacts.
Reference datasets are ignored with restoration instructions. Scientific run
directories are retained directly in Git; setup-validation/compatibility outputs
remain disposable and ignored. Compressed archives are not required.
Each run must archive its resolved configuration and must not overwrite an
existing run directory. Scientific settings belong in configurations.

See the [output file dictionary](docs/output_format.md) for file contents, fields,
array layouts, and snapshot meanings. Each scientific campaign has a README
linking its artifacts to the experiment protocol and results; see the
[Burgers pilot](output/burgers_pilot_01/README.md).

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
venv/bin/python -m src.train --config config/burgers_pilot.yaml --method fixed_hammersley --seed-index 0
venv/bin/python -m src.train --config config/burgers_pilot.yaml --method pacmann_adam --seed-index 0
```

These commands select the full-budget, one-seed pilot: five phases of 7,000 Adam
steps followed by L-BFGS with a configured 3,000-iteration limit per phase.
The saved pilot directories already exist, so rerunning these commands is rejected.
Use a configuration with a new campaign name for a new approved run. For disposable
engineering checks, copy the pilot settings into a temporary YAML, reduce the
training budget, and use a campaign name containing `validation` so its output
is ignored. DeepXDE selects CUDA when available, otherwise CPU; the actual device
is recorded in metadata. The ten-seed reproduction and diffusion experiments
remain future work.

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
See [methodology](docs/methodology.md) for the environment decision.

## Analyze a saved run

```bash
venv/bin/python scripts/analyze_burgers.py --run-dir output/burgers_pilot_01/burgers/pacmann_adam/seed_1000
```

This regenerates only `diagnostics.json` and `diagnostics.png` from saved
predictions, history, points, and configuration. Diagnostics contain maximum
absolute error, its coordinates, evaluation times, and per-time relative L2.
Training metrics remain in `metrics.json`; analysis does not certify the training
protocol. Grid dimensions are inferred from saved coordinates. No training or
reference dataset download is needed.
