# PINN Collocation Dynamics

Residual-guided collocation and stochastic diffusion for PINNs: evaluating
performance across smooth and sharp-feature PDE regimes.

This research investigates adaptive collocation methods for physics-informed
neural networks, with a possible publication as the longer-term objective.

The [research context](docs/research_context.md) records the study stages and
scientific decisions and study design principles. Campaign READMEs explain
campaign-specific reasoning and execution, and
[results](docs/results.md) consolidate comparisons and brief interpretation.

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
src/                    Burgers definition, sampling callbacks, and training functions
scripts/                Campaign execution and saved-run diagnostics
config/                 Scientific campaign settings in YAML
docs/                   Research context, results, and output dictionary
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
linking its artifacts to the study phase and results; see the
[Burgers sigma-tuning campaign](output/burgers_sigma_tuning_01_baseline/README.md).
Earlier readiness artifacts remain available but are excluded from scientific
documents and comparison tables.

Add functionality only when the next experiment requires it. Begin with phase
endpoints, final models, and selected collocation snapshots. The Burgers campaign
script runs configured seed–method pairs sequentially; exact resume is deferred.

See [source provenance](references/upstream.md) and
[working agreements](AGENTS.md).

## Setup and execution

Use Python 3.12 and uv with the local `venv`; conda is unnecessary for this
wheel-based stack. `pyproject.toml` and `uv.lock` are authoritative for dependency
pins, including DeepXDE 1.14.0 and PyTorch 2.7.1 with CUDA 12.8 wheels. Retain this
released-code stack for the initial reproduction, using the PyTorch backend and
float64. The reference host is an NVIDIA GeForce RTX 5050 Laptop GPU with driver
595.99.02; each run records its actual device and dependency versions. Setup
checks are disposable; document final environment choices here.

```bash
uv venv --python 3.12 venv
VIRTUAL_ENV="$PWD/venv" uv sync --active --locked
```

Restore the reference dataset using the [data instructions](references/upstream.md#restore-the-dataset).
Select the YAML by editing `CONFIG_PATH` at the top of
[`scripts/run_burgers_campaign.py`](scripts/run_burgers_campaign.py), then run from
the repository root:

```bash
venv/bin/python -m scripts.run_burgers_campaign
```

The script takes no CLI arguments. The YAML specifies the campaign name, seeds,
methods, and scientific settings. It runs seeds in their listed order and methods
in their YAML order, generating diagnostics after each training run. For an
individual run, select one seed and one method in the YAML. Obtain approval for
the scientific protocol and commit the source and configuration before material runs.

The default selects the full-budget, one-seed pilot: five phases of 7,000 Adam
steps followed by L-BFGS with a configured 3,000-iteration limit per phase, for
both methods. Its saved run directories already exist, so execution is rejected
before training. Existing campaign directories are allowed when all selected run
destinations are unused. The runner never overwrites, skips, retries, or resumes
an existing run. It stops on the first training or analysis failure and retains
artifacts; an analysis failure leaves the completed training artifacts available
for independent diagnostic regeneration.

For disposable engineering checks, copy the pilot settings into a temporary YAML,
reduce the training budget, select that file in the script, and use a campaign
name containing `validation` so its output is ignored. DeepXDE selects CUDA when
available, otherwise CPU; the actual device is recorded in metadata. The ten-seed
reproduction and diffusion experiments remain future work.

Each run saves its configuration, provenance, status, log, optimizer timings,
loss/metric history, predictions, phase/final weights, and collocation snapshots.
Weights support evaluation, not exact training resume. Final weights are reloaded
into a fresh model and checked against saved predictions. Both final errors and
upstream history-selected errors are explicitly named.

```bash
venv/bin/python -m pytest --cov=src --cov=scripts.run_burgers_campaign --cov=scripts.run_burgers_sigma_tuning --cov-report=term-missing
venv/bin/ruff check src scripts tests
venv/bin/ty check src tests scripts/run_burgers_campaign.py scripts/run_burgers_sigma_tuning.py
uv lock --check
uv pip check --python venv/bin/python
```

Tests cover the reference checksum/order, analytic PDE residual, coordinate-Adam
parity with the imported upstream callback, fixed constraints, run isolation,
campaign execution order, failure handling, and training-to-analysis flow.

## Sigma tuning

The paired sigma sweep is complete and selected `1e-4` for later evaluation on
the original Burgers benchmark. See the
[campaign design](output/burgers_sigma_tuning_01_baseline/README.md#comparison-design),
[results](docs/results.md#original-benchmark-burgers-sigma-tuning), and
[noise definition](docs/output_format.md#configuration-provenance-and-status).

[`scripts/run_burgers_sigma_tuning.py`](scripts/run_burgers_sigma_tuning.py) selects
five self-contained YAMLs through `CONFIG_PATHS` at its top: PACMANN as sigma zero,
then diffusion at `1e-5`, `1e-4`, `1e-3`, and `1e-2`. Each uses seeds 2000–2002.
Its committed `ACTION` is currently `"run"`. To inspect existing outputs, change
it to `"summarize"`, which reads saved artifacts and prints a Markdown comparison
without training or writing files. Then use:

```bash
venv/bin/python -m scripts.run_burgers_sigma_tuning
```

For an approved scientific launch, commit the source and configurations with
`ACTION = "run"` and use the same command to train and analyze all 15 runs. Routine
execution within that approved scope needs no repeated permission. Existing
occupied destinations must be resolved under the
[recovery policy](AGENTS.md#execution-recovery).
The script checks all destinations before starting and stops on failure. It creates short
campaign READMEs linking to the shared design and artifact dictionary, preserving
existing READMEs. The YAML campaign names keep sigma settings in separate output
directories; each seed archives its actual settings.

For independent post-processing, set `ACTION = "summarize"` again. The summary
checks archived configurations and common source, dependency versions, data, and
device. Restore the executed YAMLs if they have changed. It reports individual
errors, planned/completed/failed/pending counts, complete-group means and sample
standard deviations, paired changes, and completed training time. Missing
diagnostics count as pending even when training completed. Regenerate them with
the analysis command below. No sigma is selected until all planned runs and
diagnostics complete successfully; boundary winners require review. Copy the
resulting table and brief interpretation into `docs/results.md`; record the
freeze decision in `docs/research_context.md`. No summary JSON is created.

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
