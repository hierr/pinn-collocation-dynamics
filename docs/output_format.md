# Output file dictionary

This document describes the current Burgers runner's saved artifacts. Scientific
settings belong in archived configurations, and numerical outcomes in run
artifacts and [results](results.md); this dictionary explains how to read the files.

## Directory layout and experiment links

```text
output/<campaign>/
  README.md
  <problem>/<method>/seed_<seed>/
    config.yaml
    metadata.json
    status.json
    run.log
    training.json
    metrics.json
    history.npz
    predictions.npz
    models/phase_<NN>_<optimizer>.pt
    models/final.pt
    points/initial.npz
    points/phase_<NN>_<optimizer>.npz
    points/final.npz
    diagnostics.json
    diagnostics.png
```

Each scientific campaign has a concise, objective `README.md` with its execution
instructions and reasoning not captured by configurations or artifacts. Link to
settings, the relevant [study phase](research_context.md), the comparison in
[`results.md`](results.md), and this dictionary. Keep a shared design in one
README and link from related output directories; do not repeat settings or
create per-seed narratives. Maintain these READMEs as documentation; execution
scripts write the numerical run artifacts.

The trainer writes configuration, provenance, status, logs, optimizer records,
histories, model checkpoints, and training points. Post-processing loads the
final checkpoint and archived settings to write `predictions.npz`, `metrics.json`,
and diagnostics. The campaign runner performs both steps automatically. A failed
or interrupted run can have only a subset of these files.

## Configuration, provenance, and status

| File | Contents and purpose |
| --- | --- |
| `config.yaml` | Configuration copied at run start, including campaign, planned seeds, problem parameters, reference path, point budgets, network layers, optimizer, sampling, and evaluation settings. `selected_method` and `selected_seed` identify this particular run; `seeds` remains the campaign's configured list. |
| `metadata.json` | Code revision, dependency versions, reference dataset identity, and device used by the runner. |
| `status.json` | Execution state. |
| `run.log` | Standard output and errors captured during training, including DeepXDE progress messages and training failure tracebacks. |

`metadata.json` fields:

| Field | Meaning |
| --- | --- |
| `revision` | Git HEAD commit at run start. This identifies the checkout revision; it does not capture uncommitted changes. |
| `versions` | Installed version strings for DeepXDE, PyTorch, NumPy, SciPy, Matplotlib, and PyYAML. |
| `dataset_sha256` | SHA-256 fingerprint of the input reference dataset. Generated data are checked against the configured viscosity, domain, initial condition, and native grid. |
| `device` | PyTorch default device recorded by the runner, for example `cuda:0` or `cpu`. Final hardware choices belong in the repository README; disclose deviations affecting interpretation in the campaign README or results. |

`status.json` contains `status`: `running`, `complete`, or `failed`. A caught
failure also records `error`, the exception message. `complete` means training
and its artifact writes finished; final evaluation and diagnostics
must also exist before a run is ready for comparison. An evaluation failure stops
the campaign without changing the training status or requiring retraining.
Completion does not establish a scientific claim.
An abrupt interruption can leave `running` behind. Upstream attribution and the
pinned source commit are in [source provenance](../references/upstream.md).

These are execution states, not scientific classifications. A `failed` entry
or incomplete files alone cannot distinguish numerical failure from an external
interruption. Apply the [recovery policy](../AGENTS.md#execution-recovery)
before interpreting an attempt or counting scientific failures; no additional
scientific status field is implied.

For `pacmann_diffusion`, the saved method settings include `sigma` alongside the
coordinate-Adam settings. Domain widths are derived from `problem.x_bounds` and
`problem.t_bounds`; the movement-event length is `methods.pacmann_diffusion.steps`.
Each coordinate step adds Gaussian noise with standard deviations
`sigma * domain_widths / sqrt(steps)`. Sigma therefore scales the accumulated
noise component per movement event, before boundary replacement; it is not a
physical PDE diffusion coefficient. Ordinary `pacmann_adam` is the sigma-zero
baseline and has no `sigma` field.

The diffusion callback initializes a persistent local generator using NumPy
`SeedSequence([selected_seed, 1])`. This convention and the archived NumPy version
identify the noise stream without extra metadata fields. Sigma zero draws no
noise. Gaussian draws do not advance the global NumPy stream; uniform escaped-point
replacement still uses upstream sampling and can consume different draws when
positive sigma changes which points escape. Initial points and weights remain
paired across configurations.

For controlled runs, `problem.reference_path` selects a generated NPZ with `x`,
`t`, `U` (time by space), and `nu`. `evaluation.solution_nx` sets the number of
uniform spatial evaluation points, including the boundaries. The reference is
interpolated in space using Chebyshev barycentric interpolation, retaining the
native uniformly spaced times within the exact configured interval.
`evaluation.residual_nx` and `residual_nt` specify the independent uniform interior
midpoint grid; `batch_size` limits prediction and autograd batches. The template
uses 1024 solution points, 1024 by 200 residual points, and batches of 4096.
These settings are archived with every new run.

## Optimizer-stage records: `training.json`

The top-level `stages` list records completed optimizer stages in execution
order. Each phase consists of network Adam followed by L-BFGS; optimizer state
is reset when each stage is compiled.

| Field | Type/unit | Meaning |
| --- | --- | --- |
| `phase` | Integer, starting at 1 | Phase number. |
| `optimizer` | String | `adam` or `L-BFGS`. This is the network optimizer, separate from coordinate Adam used by PACMANN. |
| `seconds` | Seconds | Elapsed stage compilation and training time, with CUDA synchronization. Excludes subsequent snapshot writes and post-processing, including checkpoint loading and final evaluation. |
| `global_step` | Integer | Cumulative DeepXDE network optimizer step at the stage endpoint. Includes previous stages; excludes coordinate movement steps and L-BFGS function evaluations. |
| `optimizer_iterations` | Integer | Stage's network iterations: the configured completed Adam steps or the L-BFGS optimizer's actual `n_iter` counter. |
| `function_evaluations` | Integer; L-BFGS only | Actual L-BFGS `func_evals` counter, including repeated closure evaluations during optimization. Omitted for Adam. |
| `movement_events` | Integer | PACMANN movement events during this stage. One event contains the configured number of coordinate updates; it is not one event per point or per coordinate step. Zero for fixed sampling and L-BFGS. |
| `pde_points` | Integer | Number of PDE-residual coordinates at the endpoint, corresponding to the saved `pde` array. Controlled runs use the configured interior count throughout. |

Actual L-BFGS iterations can exceed the configured limit because DeepXDE makes
optimizer calls in batches. Use the recorded counts when comparing executed
budgets. The file is rewritten after each completed stage, so a failed run may
contain a partial stage list.

## Error and timing summary: `metrics.json`

Let `prediction` be the network solution and `reference` the reference solution
at all saved evaluation coordinates, with `error = prediction - reference`.
MSE is `mean(error**2)`;
relative L2 is `sqrt(sum(error**2) / sum(reference**2))`, counting each grid point
equally. Relative errors are stored as fractions: multiply by 100 for percentages. MSE has squared
solution units; relative L2 is dimensionless. The current Burgers benchmark uses
nondimensional coordinates and solution values.

| Field | Meaning |
| --- | --- |
| `final_mse` | MSE of the final saved model over the complete evaluation grid. |
| `final_relative_l2` | Relative L2 of that final model over the same grid. |
| `final_residual_rms` | RMS of `u_t + u*u_x - viscosity*u_xx`, using PINN autograd derivatives on the independent interior midpoint grid. No reference solution or zero-residual normalization is used. |
| `boundary_rmse` | RMSE against zero at both spatial boundaries, averaged equally over both boundaries and all retained times. |
| `initial_rmse` | RMSE against `-sin(pi*x)` at the configured initial time, averaged equally over the uniform spatial evaluation points. |
| `movement_events` | Total PACMANN events across training; sum of the stage event counts. |
| `training_seconds` | Sum of stage `seconds`, including compilation and training with movement callbacks. Excludes checkpoint loading, final evaluation, and plotting. |

Scores are computed by loading the final checkpoint into a fresh model during
post-processing. Checkpoint/in-memory evaluation agreement is covered by tests.
Training time and movement totals are derived from the saved optimizer-stage
records.

Sigma selection uses mean `final_relative_l2` over development seeds. The same
selected setting is evaluated for residual error, without a separate residual
selection. Pointwise residual arrays are not saved: the archived model and
evaluation settings allow the residual score to be recomputed.

## Recorded convergence: `history.npz`

NPZ files are compressed NumPy archives, readable with `numpy.load` using
`allow_pickle=False`. Here `H` is the number of recorded history rows; floating
arrays use float64.

| Array | Shape/type | Meaning |
| --- | --- | --- |
| `steps` | `(H,)`, integers | Cumulative network optimizer step for each observation. Stage transitions can record the same step more than once. |
| `loss_train` | `(H, 3)` | Training-loss components in order: PDE residual, zero boundary condition, initial condition. |
| `loss_test` | `(H, 3)` | DeepXDE's recorded test-loss components in the same order. Controlled runs retain DeepXDE's test construction with its constraint prefix and approximately 10,000 PDE test points. These are monitoring diagnostics, distinct from the final independent residual metric. |

Recording follows DeepXDE's display cadence and stage endpoints; L-BFGS history
is recorded at optimizer-call boundaries rather than every internal iteration.
Stage records in `training.json` identify optimizer endpoints. Model weights are
saved only at those endpoints; scientific metrics use the final checkpoint.

## Final evaluation data: `predictions.npz`

| Array | Shape | Meaning |
| --- | --- | --- |
| `coordinates` | `(N, 2)` | Evaluation coordinate pairs `[x, t]`. |
| `reference` | `(N, 1)` | Reference solution `u` at each coordinate. |
| `prediction` | `(N, 1)` | Final model's predicted `u` at each coordinate. |

All arrays use float64. With the generated N1024 reference and template settings,
the evaluation grid has 1,024 uniform spatial points and 199 times through
`0.99`, giving 203,776 rows. The source reference has 1,025 Chebyshev nodes;
these are interpolated onto the evaluation grid.
Ordering is time-major: all ascending spatial coordinates
at the first time, then all spatial coordinates at the next time, and so on.
Solution arrays can therefore be reshaped to `(number_of_times, number_of_x)`.

These are evaluation data, generated by post-processing from the final saved
model. Reference solution values are not supervised training labels; training
uses the PDE and analytic boundary and initial conditions. The reference
fingerprint must match training metadata before new evaluation artifacts are
written. Existing predictions and metrics are retained when regenerating plots;
missing evaluation files can be generated from the checkpoint.

## Training coordinates: `points/*.npz`

Each archive contains float64 `[x, t]` arrays. These are training coordinates,
not observations of the reference solution.

| Array | Shape | Meaning |
| --- | --- | --- |
| `interior` | `(I, 2)` | Interior physics population extracted by the sampler; PDE collocation coordinates that PACMANN can move. |
| `pde` | `(P, 2)` | Complete `train_x_all` population on which DeepXDE applies the PDE-residual loss. |
| `constraints` | `(C, 2)` | `train_x_bc` coordinates for soft boundary and initial-condition losses, in boundary-then-initial order. Their target values come from the analytic conditions in the problem definition. |

In controlled runs, `interior` and `pde` contain the same population from
initialization: `I = P = 2500` for the template. `C = 240` comprises 80 boundary
and 160 initial constraints, evaluated separately and kept fixed for every
method. Do not add `interior` and `pde` lengths to calculate the point budget.

Historical runs retain the released setup: initially `P = 2740`, staying 2740
for fixed Hammersley and dropping to 2500 after PACMANN's first event. Their
constraint coordinates also appear in the initial PDE population. Saved arrays
record the populations actually used; these artifacts are not rewritten.

`initial.npz` is saved before network training. `phase_<NN>_adam.npz` and
`phase_<NN>_l-bfgs.npz` are saved at each optimizer endpoint. `final.npz` provides
a common final filename and duplicates the last completed stage's point state
in a successful run. Intermediate movement events are not saved individually.

## Evaluation weights: `models/*.pt`

Each PyTorch file contains only `model.net.state_dict()`. Reconstruct the network
using the saved configuration and load the weights for evaluation. Optimizer
state, random-generator state, and sampler state are absent, so these files do
not support exact training resume.

Names match the phase/optimizer point snapshots. `final.pt` duplicates the last
stage's weights in a successful run, providing a common filename. There is no
initial weight snapshot.

## Derived analysis: `diagnostics.json` and `diagnostics.png`

[`scripts/post_processing/analyze_burgers.py`](../scripts/post_processing/analyze_burgers.py) regenerates these
two files from saved predictions, history, final interior points, and config.
Regenerating diagnostics from completed evaluation artifacts does not require
the reference dataset or model. Generating missing predictions or metrics also
requires the final checkpoint and matching reference dataset; it does not train.

| JSON field | Meaning |
| --- | --- |
| `maximum_absolute_error` | Largest absolute final prediction error over the saved evaluation grid, in solution units. |
| `maximum_error_coordinate` | `[x, t]` where that maximum occurs; the first in saved grid order if tied. |
| `times` | Evaluation times in ascending order. |
| `per_time_relative_l2` | One relative-L2 fraction per entry in `times`, using ordinary spatial norms of the error and reference at that time. Undefined for a zero reference norm. |

The PNG shows recorded training/test losses, final absolute error
over space and time, final interior collocation points, and reference/prediction
slices at the saved times nearest 0.25, 0.50, 0.75, and 0.99. These are derived
diagnostics. Final scores and training-cost totals are in `metrics.json`.
