# PINN Collocation Dynamics — hyperparameter tuning

This persistent `hyperparameter-tuning` branch retains tuning code,
configurations, artifacts, and decisions. Dataset validation lives on
`dataset-validation`; accepted implementations and full frozen-setting
evaluations live on `main`. Transfer selected changes and conclusions between
branches, citing the source commit, rather than merging entire workstreams.

## Retained development campaign

The [original-reference sigma sweep](output/burgers_sigma_tuning_01_baseline/README.md)
retains all 15 runs, archived settings, and the shared comparison design.
[Results](docs/results.md) summarize the historical selection of `1e-4` on the
original PACMANN reference. This is development evidence, not a completed
frozen-setting evaluation or a selection for future generated datasets.

The next study will tune separately at each Burgers viscosity using validated
generated references, then freeze settings before ten-seed evaluation. Its
parameter range, tuning budget, and decision rules still require approval.

## Setup and execution

Use Python 3.12 and the existing uv-managed `venv`; conda is unnecessary. The
locked stack includes DeepXDE 1.14.0 and PyTorch 2.7.1 with CUDA 12.8 wheels.
The backend is PyTorch with float64; saved metadata records the training device,
source revision, dataset fingerprint, and dependency versions.

```bash
uv venv --python 3.12 venv
VIRTUAL_ENV="$PWD/venv" uv sync --active --locked
```

Select `ACTION = "summarize"` in
`scripts/run_experiments/run_burgers_sigma_tuning.py` to read the retained sweep:

```bash
venv/bin/python -m scripts.run_experiments.run_burgers_sigma_tuning
```

Select the YAMLs through `CONFIG_PATHS` in that script. `ACTION = "run"` trains
and analyzes configured runs; all selected destinations are checked before
training and existing runs cannot be overwritten. Obtain approval and commit
the scientific protocol before starting a new campaign. Campaign READMEs are
maintained manually alongside the outputs.

The reusable runner is `scripts/run_experiments/run_burgers_campaign.py`; select
its YAML through `CONFIG_PATH`. `config/burgers_template.yaml` preserves upstream
settings but deliberately has no seeds: it is a starting template, not an
approved campaign. Sampling classes remain in `src/sampling.py`.

```bash
venv/bin/python -m pytest --cov=src --cov=scripts.run_experiments.run_burgers_campaign --cov=scripts.run_experiments.run_burgers_sigma_tuning --cov-report=term-missing
venv/bin/ruff check src scripts tests
venv/bin/ty check src tests scripts/run_experiments
```

See the [artifact dictionary](docs/output_format.md),
[research context](docs/research_context.md), and [source provenance](references/upstream.md).
