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
The controlled N1024 development sweep selected `sigma=1e-4` at this viscosity;
see the [development result](docs/results.md#n1024-diffusion-development-selection).
The next evaluation will compare five methods on ten new paired seeds with
frozen settings, before extending to a validated viscosity range and tuning
separately in each regime. No full ten-seed evaluation has been completed.
Development campaigns remain on `hyperparameter-tuning`.

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
2.7.1 with CUDA 12.8 wheels. The backend is PyTorch with float64. The N1024
development sweep used `hier-wsl`, its existing `.venv`, Python 3.12.12, and an
NVIDIA GeForce RTX 5070 (driver 617.42, device `cuda:0`). Each run records its
device and dependency versions. Use the existing remote environment for scientific
campaigns and verify CUDA before execution; local tests use the existing `venv`.

```bash
uv venv --python 3.12 venv
VIRTUAL_ENV="$PWD/venv" uv sync --active --locked
```

Select a YAML through `CONFIG_PATH` in
`scripts/run_experiments/run_burgers_campaign.py`, then run from the repository root:

```bash
venv/bin/python -m scripts.run_experiments.run_burgers_campaign
```

The default `config/burgers_template.yaml` demonstrates all five methods with
matched physics populations and the upstream network/optimizer schedule. It has
an empty seed list, so execution is rejected before writing artifacts. Specify
and approve a campaign protocol before selecting seeds and running it. The
runner trains each seed–method pair in YAML order and generates diagnostics;
it stops on failure and never overwrites, skips, or resumes an existing run.
Weights support later evaluation, not exact training resume.

Implemented methods are fixed Hammersley, upstream random resampling, RAD,
PACMANN-Adam, and PACMANN with diffusion. YAML's `problem.reference_path` selects
the generated reference. All methods start from matched Hammersley interior
points with fixed initial/boundary constraints. Resampling and adaptation occur
only during Adam. The template uses upstream RAD settings; they are not a
separately optimized RAD configuration. Actual optimizer work and callback cost
are recorded rather than assuming equal computation across methods.
Validation scripts and artifacts are available by switching to
`dataset-validation`. Datasets are tracked; preserve their
[source provenance](references/upstream.md).

```bash
venv/bin/python -m pytest --cov=src --cov=scripts.run_experiments.run_burgers_campaign --cov=scripts.post_processing.analyze_burgers --cov-report=term-missing
venv/bin/ruff check src scripts/run_experiments scripts/post_processing scripts/plot_burgers_initial_points.py tests
venv/bin/ty check src tests scripts/run_experiments scripts/post_processing scripts/plot_burgers_initial_points.py
```

For saved-run diagnostics, use
`venv/bin/python -m scripts.post_processing.analyze_burgers --run-dir <run-directory>`.
Post-processing restores final weights when evaluation files are missing, then
computes ordinary solution MSE/relative L2, independent residual RMS, and boundary
and initial RMSE. Existing predictions and metrics are retained when regenerating
plots. Evaluation settings are archived in each run; see the
[evaluation policy](docs/research_context.md#accepted-controls-and-evaluation-policy).

To regenerate the [initial Hammersley figure](docs/burgers_initial_collocation.png)
from the template and reference, without training or tuning artifacts:

```bash
venv/bin/python -m scripts.plot_burgers_initial_points
```
