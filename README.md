# PINN Collocation Dynamics

This study investigates whether residual-gradient adaptive collocation helps
PINNs more as solutions develop sharp features, and whether stochastic diffusion
improves PACMANN under those conditions. Both benefits remain open questions.

The [research context](docs/research_context.md) records the study design and
accepted decisions; [results](docs/results.md) consolidate accepted comparisons.

## Research workstreams

| Persistent branch | Purpose |
| --- | --- |
| `dataset-validation` | Reference generation, validation scripts, and evidence |
| `hyperparameter-tuning` | Development configurations, tuning campaigns, and selections |
| `main` | Accepted code and conclusions, frozen evaluation settings, and full evaluations |

Transfer selected changes with the supporting commit rather than merging all
side-branch artifacts into `main`. Each branch preserves its own continuing
history; a new branch per campaign is unnecessary.

The generated N1024 Burgers reference at viscosity `0.01/pi` has been
[validated against PACMANN](docs/results.md#generated-burgers-reference-validation).
The next study will validate a viscosity range, tune diffusion separately at
each viscosity, then run ten-seed comparisons with frozen settings. No full
ten-seed evaluation has been completed. Historical pilot and tuning runs remain
in Git history, with the sigma sweep retained on `hyperparameter-tuning`.

## Structure

```text
src/                    Burgers problem, sampling classes, training, and artifacts
scripts/run_experiments/ Reusable campaign runner
scripts/post_processing/ Saved-run diagnostics
scripts/dataset_generation/ Accepted collaborator solver
config/                 Scientific YAML settings
docs/                   Research context, results, and artifact dictionary
references/             Immutable PACMANN source snapshot and provenance
data/                   Original and generated reference datasets
output/                 Full evaluation campaigns, added when executed
```

Scientific run directories are retained directly in Git. Each archives its
resolved configuration, provenance, models, predictions, point snapshots, logs,
and metrics. Never overwrite an existing run or original dataset. Campaign
READMEs belong alongside retained runs; see the [artifact dictionary](docs/output_format.md).

## Setup and execution

Use Python 3.12 and uv with the local `venv`; conda is unnecessary for this
wheel-based stack. `pyproject.toml` and `uv.lock` pin DeepXDE 1.14.0 and PyTorch
2.7.1 with CUDA 12.8 wheels. The backend is PyTorch with float64. The original
reference host was an NVIDIA GeForce RTX 5050 Laptop GPU; each run records its
actual device and dependency versions.

```bash
uv venv --python 3.12 venv
VIRTUAL_ENV="$PWD/venv" uv sync --active --locked
```

Select a YAML through `CONFIG_PATH` in
`scripts/run_experiments/run_burgers_campaign.py`, then run from the repository root:

```bash
venv/bin/python -m scripts.run_experiments.run_burgers_campaign
```

The default `config/burgers_template.yaml` preserves upstream settings and has
an empty seed list, so execution is rejected before writing artifacts. Specify
and approve a campaign protocol before selecting seeds and running it. The
runner trains each seed–method pair in YAML order and generates diagnostics;
it stops on failure and never overwrites, skips, or resumes an existing run.
Weights support later evaluation, not exact training resume.

Currently implemented methods are fixed Hammersley, PACMANN-Adam, and PACMANN
with diffusion. The trainer still loads the original reference at
`data/burgers/Burgers.npz`; generated-data selection, upstream random resampling,
and RAD remain implementation work before the proposed full comparison.
Validation scripts and artifacts are available by switching to
`dataset-validation`. Datasets are tracked; preserve their
[source provenance](references/upstream.md).

```bash
venv/bin/python -m pytest --cov=src --cov=scripts.run_experiments.run_burgers_campaign --cov-report=term-missing
venv/bin/ruff check src scripts/run_experiments scripts/post_processing tests
venv/bin/ty check src tests scripts/run_experiments
```

For saved-run diagnostics, use
`venv/bin/python scripts/post_processing/analyze_burgers.py --run-dir <run-directory>`.
This regenerates diagnostics from completed artifacts without training.
