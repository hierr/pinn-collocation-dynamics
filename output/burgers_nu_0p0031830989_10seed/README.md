# Burgers: frozen-setting ten-seed comparison

**Frozen protocol; all five method batches are complete (50/50 runs).** See the
[consolidated results](../../docs/results.md#frozen-n1024-ten-seed-evaluation).
The [configuration](../../config/burgers_nu_0p0031830989_10seed.yaml) contains
the full five-method study. Run metadata records source revision `9c3f022` for
PACMANN and `e02098d` for the other four methods. Numerical training and
evaluation code is unchanged between these revisions.

## Design and comparisons

The completed comparison evaluated fixed Hammersley, upstream random resampling,
RAD, PACMANN, and PACMANN with diffusion on ten paired seeds (50 runs) at
`nu = 0.01/pi`, using the validated generated N1024 reference. This evaluates
methods at one viscosity; it does not test a sharpness effect or reproduce the
original paper's unequal-population setup.

The primary contrast is diffusion versus PACMANN for final solution relative L2,
using `sigma=1e-4` selected on separate development seeds. Additional contrasts
compare RAD, PACMANN, and PACMANN with diffusion against fixed Hammersley and
random resampling. Assess the same contrasts for independent residual RMS without
selecting another sigma.
Boundary and initial-condition RMSE are constraint diagnostics.

Use the YAML's planned seed list. Match initial weights,
Hammersley points, fixed constraints, PDE point budgets, and optimizer settings
across methods within each seed. Sampling settings are fixed before evaluation:
upstream sampling settings for the baselines and PACMANN, and development-selected
sigma for diffusion. Disclose this tuning asymmetry. Retain the development sweep's
L-BFGS behavior and compare actual iterations, function evaluations, and runtime.

Report every seed's final-model outcomes, means and sample SD, paired absolute
error changes, and counts improved for each declared contrast. Negative paired
changes mean lower error; exact ties count separately. Express solution L2 as
percent and its paired changes as percentage points; residual changes remain in
residual units. Do not retune, select history minima, or replace unfavorable
seeds. Also report the supplementary trimmed means defined in the
[results](../../docs/results.md#consolidated-evaluation-results), while retaining
all seeds in the primary summaries and paired comparisons.
Report a worker failure after all started workers finish; retain every
run's artifacts and review failures before launching another method. Follow the
[recovery policy](../../AGENTS.md#execution-recovery). Do not present a complete
ten-seed mean from a successful subset; report missing/failed outcomes and the
available paired count explicitly.

## Execution

Follow the repository's [setup instructions](../../README.md#setup-and-execution).
In `scripts/run_experiments/run_burgers_campaign.py`, set `CONFIG_PATH` to the
campaign YAML and `METHODS` to the required methods in execution order. A full
repetition includes `pacmann_adam`, `pacmann_diffusion`, `fixed_hammersley`,
`random_resampling`, and `rad`. Use a copy of the YAML with a new `campaign`
identifier to preserve the retained runs, keeping the scientific settings fixed.
Execute from the repository root:

```bash
venv/bin/python -m scripts.run_experiments.run_burgers_campaign
```

Each batch launches all ten seeds concurrently in independent processes. The next
method starts only after every worker in the current batch finishes training and
evaluation. Only one method batch uses the GPU at a time. The reported evaluation
used an NVIDIA GeForce RTX 5070.
Workers use one CPU thread and set OMP/MKL/OpenBLAS thread limits to one, as
described in the [environment instructions](../../README.md#setup-and-execution).

Report mean and sample SD of all ten `training_seconds` values for each method,
describing the timing condition as up to ten concurrent seeds on one GPU. These
recorded times include competition with other workers, including workers doing
final evaluation. Do not combine them with timings from serial tuning runs.

The launcher rejects occupied destinations and stops after a failed batch's
workers finish. It does not skip, retry, or resume runs. Per-run artifacts are
retained here; completed training requires predictions, metrics, and diagnostics
before comparison. Saved weights support evaluation, not exact training resume.

[Study controls](../../docs/research_context.md#accepted-controls-and-evaluation-policy)
· [Development selection](../../docs/results.md#n1024-diffusion-development-selection)
· [Artifact dictionary](../../docs/output_format.md)
· [Environment](../../README.md#setup-and-execution)
· [Optimizer semantics](../../references/upstream.md#controlled-study-choices)
