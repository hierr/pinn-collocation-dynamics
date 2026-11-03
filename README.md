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
| `analysis` | Exploratory post-processing and reproducible comparisons awaiting synthesis |
| `main` | Accepted implementations and analyses, conclusions, frozen settings, and full evaluations |

Transfer selected changes with the supporting commit rather than merging all
side-branch artifacts into `main`. Each branch preserves its own continuing
history; a new branch per campaign is unnecessary.
Develop analyses on `analysis`; transfer scripts that support accepted results
or retained figures to `scripts/analysis/` on `main`. Unadopted explorations
remain on their workstream branch.

The generated N1024 Burgers reference at viscosity `0.01/pi` has been
[validated against PACMANN](docs/results.md#generated-burgers-reference-validation).
The controlled N1024 development sweep selected `sigma=1e-4` at this viscosity;
see the [development result](docs/results.md#n1024-diffusion-development-selection).
The [first frozen evaluation](output/burgers/burgers_nu_0p0031830989_10seed/README.md)
completed all 50 runs across five methods and ten paired seeds. See the
[evaluation results](docs/results.md#frozen-n1024-ten-seed-evaluation).
The comparison extends across viscosity with separate tuning in each regime.
The [pi/100 campaign](output/burgers/burgers_nu_0p031415927_10seed/README.md)
completed all 50 runs using the restored N1024 reference and
development-selected `sigma=1e-4`; see its
[evaluation results](docs/results.md#frozen-evaluation-at-pi100).
Development campaigns remain on `hyperparameter-tuning`.

The sharper [nu = 0.001 campaign](output/burgers/nu_0p001/burgers_nu_0p001_10seed/README.md)
completed all 50 runs using the accepted N4096 reference and separately
selected **sigma = 1e-6**, from tuning checkpoint `56ea078`. It retains the
shared training budgets and evaluation grids. All methods have large solution
errors at this viscosity; random resampling has the lowest mean solution and
residual errors. See the [evaluation results](docs/results.md#frozen-evaluation-at-nu-0001).

## Structure

```text
src/                    Burgers problem, sampling classes, training, and artifacts
scripts/run_experiments/ Reusable campaign runner
scripts/post_processing/ Saved-run diagnostics
scripts/analysis/       Accepted comparisons, reference characterization, and diagnoses
scripts/illustrations/  Explanatory setup and method visuals
scripts/dataset_generation/ Accepted collaborator solver
config/                 Scientific YAML settings
docs/                   Research context, results, and artifact dictionary
docs/figures/           Scientific figures shared by documents and presentations
references/             Immutable PACMANN source snapshot and provenance
data/                   Original and generated reference datasets
output/                 Retained full evaluation campaigns
```

Scientific run directories are retained directly in Git. Each archives its
resolved configuration, provenance, models, predictions, point snapshots, logs,
and metrics. Never overwrite an existing run or original dataset. Campaign
READMEs belong alongside retained runs; see the [artifact dictionary](docs/output_format.md).

## Setup and execution

Use Python 3.12 and uv with the local `venv`; conda is unnecessary for this
wheel-based stack. `pyproject.toml` and `uv.lock` pin DeepXDE 1.14.0 and PyTorch
2.7.1 with CUDA 12.8 wheels. The backend is PyTorch with float64. Reported PINN
development and evaluation campaigns used an NVIDIA GeForce RTX 5070; the
development environment used Python 3.12.12 and driver 617.42. Each run records
its device and dependency versions. The lockfile specifies the software stack;
campaign READMEs specify concurrency and timing conditions. Commands below assume
the repository root and use the project environment `venv`.

```bash
uv venv --python 3.12 venv
VIRTUAL_ENV="$PWD/venv" uv sync --active --locked
```

Select the YAML through `CONFIG_PATH` and an ordered list through `METHODS` in
`scripts/run_experiments/run_burgers_campaign.py`, then run from the repository root:

```bash
venv/bin/python -m scripts.run_experiments.run_burgers_campaign
```

For each selected method, the runner starts all configured seeds in independent
spawned processes and waits for their training and evaluation to finish before
starting the next method. A failed batch stops
the sequence after its other workers finish. Set `METHODS` to a single-entry
list when a pause for review is needed after that method.
All five methods remain in the frozen YAML and its
archived copies; each run records its actual `selected_method` and `selected_seed`.
The runner currently selects the completed `nu = 0.001` configuration. Its 50
destinations are occupied. Earlier campaigns remain at their archived paths;
new runs use `output/burgers/nu_<viscosity>/<campaign>/`.
For a new repetition, copy the
campaign YAML with a new `campaign` identifier, keep its scientific settings,
and select that copy and the desired methods in the runner.
For future campaigns,
`config/burgers_template.yaml` demonstrates all five methods with matched physics
populations and the upstream network/optimizer schedule. The template has an
empty seed list and cannot run unchanged. The runner validates all destinations
for the selected methods before starting, and never overwrites, skips, retries,
or resumes an existing run. Each worker trains and generates its own diagnostics.
Weights support later evaluation, not exact training resume.

Parallel workers use one CPU thread for PyTorch intra-op/inter-op execution and
set `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `OPENBLAS_NUM_THREADS` to `1` before
the child imports numerical libraries. This limits CPU oversubscription while
all seeds share the same GPU. Timing comparisons use this same concurrency policy
for every method.

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
venv/bin/python -m pytest --cov=src \
  --cov=scripts.run_experiments.run_burgers_campaign \
  --cov=scripts.post_processing.analyze_burgers \
  --cov=scripts.analysis --cov=scripts.illustrations --cov-report=term-missing
venv/bin/ruff check src scripts/run_experiments scripts/post_processing scripts/analysis scripts/illustrations tests
venv/bin/ty check src tests scripts/run_experiments scripts/post_processing scripts/analysis scripts/illustrations
```

For saved-run diagnostics, use
`venv/bin/python -m scripts.post_processing.analyze_burgers --run-dir <run-directory>`.
Post-processing restores final weights when evaluation files are missing, then
computes ordinary solution MSE/relative L2, independent residual RMS, and boundary
and initial RMSE. Existing predictions and metrics are retained when regenerating
plots. Evaluation settings are archived in each run; see the
[evaluation policy](docs/research_context.md#accepted-controls-and-evaluation-policy).

To regenerate the [initial Hammersley figure](docs/figures/burgers_initial_collocation.png)
from the template and reference, without training or tuning artifacts:

```bash
venv/bin/python -m scripts.illustrations.plot_burgers_initial_points
```

To compare the five final collocation populations over the same reference
solution, using the first evaluation seed (4000) at `nu=0.01/pi`:

```bash
venv/bin/python -m scripts.analysis.plot_burgers_final_collocation
```

The script reads saved points and reference values without training or inference
and writes `docs/figures/burgers_final_collocation.png`.

To reproduce the accepted five-regime comparison from saved final artifacts:

```bash
venv/bin/python -m scripts.analysis.burgers_regime_comparison
```

The script reads all 250 runs from the fixed campaign snapshot `4562077` in Git.
It writes four complete figures under `docs/figures/burgers_regime_comparison/`,
four-viscosity slide selections under
`docs/presentation/figures/burgers_regime_comparison/`, and regenerable CSV tables
under `output/burgers/regime_comparison/`. The slide selection currently omits
`pi/100`; the full analysis retains it. Figures are tracked; derived tables are
ignored. No training or new inference is required. See the
[comparison and interpretation](docs/results.md#comparison-across-five-burgers-regimes).
