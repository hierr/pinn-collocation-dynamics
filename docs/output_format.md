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
create per-seed narratives. The sigma-tuning script prepares these READMEs at
launch and preserves existing ones.

The trainer writes each seed directory; the analysis script adds the derived
diagnostics. A failed or interrupted run can have only a subset of these files.

## Configuration, provenance, and status

| File | Contents and purpose |
| --- | --- |
| `config.yaml` | Configuration copied at run start, including campaign, planned seeds, problem parameters and point budgets, network layers, optimizer settings, and sampling settings. `selected_method` and `selected_seed` identify this particular run; `seeds` remains the campaign's configured list. |
| `metadata.json` | Code revision, dependency versions, reference dataset identity, and device used by the runner. |
| `status.json` | Execution state. |
| `run.log` | Standard output and errors captured during training/evaluation, including DeepXDE progress messages and failure tracebacks. |

`metadata.json` fields:

| Field | Meaning |
| --- | --- |
| `revision` | Git HEAD commit at run start. This identifies the checkout revision; it does not capture uncommitted changes. |
| `versions` | Installed version strings for DeepXDE, PyTorch, NumPy, SciPy, Matplotlib, and PyYAML. |
| `dataset_sha256` | SHA-256 fingerprint of the input reference dataset, verified against the imported manifest. It identifies the evaluation data. |
| `device` | PyTorch default device recorded by the runner, for example `cuda:0` or `cpu`. Final hardware choices belong in the repository README; disclose deviations affecting interpretation in the campaign README or results. |

`status.json` contains `status`: `running`, `complete`, or `failed`. A caught
failure also records `error`, the exception message. `complete` means the runner
finished its checks and output writes; it does not establish a scientific claim.
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

## Optimizer-stage records: `training.json`

The top-level `stages` list records completed optimizer stages in execution
order. Each phase consists of network Adam followed by L-BFGS; optimizer state
is reset when each stage is compiled.

| Field | Type/unit | Meaning |
| --- | --- | --- |
| `phase` | Integer, starting at 1 | Phase number. |
| `optimizer` | String | `adam` or `L-BFGS`. This is the network optimizer, separate from coordinate Adam used by PACMANN. |
| `seconds` | Seconds | Elapsed stage compilation and training time, with CUDA synchronization. Excludes subsequent snapshot writes, final evaluation/reload, and post-processing. |
| `global_step` | Integer | Cumulative DeepXDE network optimizer step at the stage endpoint. Includes previous stages; excludes coordinate movement steps and L-BFGS function evaluations. |
| `optimizer_iterations` | Integer | Stage's network iterations: the configured completed Adam steps or the L-BFGS optimizer's actual `n_iter` counter. |
| `function_evaluations` | Integer; L-BFGS only | Actual L-BFGS `func_evals` counter, including repeated closure evaluations during optimization. Omitted for Adam. |
| `movement_events` | Integer | PACMANN movement events during this stage. One event contains the configured number of coordinate updates; it is not one event per point or per coordinate step. Zero for fixed sampling and L-BFGS. |
| `pde_points` | Integer | Number of PDE-residual coordinates at the endpoint, corresponding to the saved `pde` array. Constraints can overlap this population. |

Actual L-BFGS iterations can exceed the configured limit because DeepXDE makes
optimizer calls in batches. Use the recorded counts when comparing executed
budgets. The file is rewritten after each completed stage, so a failed run may
contain a partial stage list.

## Error and timing summary: `metrics.json`

Let `prediction` be the network solution and `reference` the reference solution
at all saved evaluation coordinates. MSE is the mean squared difference;
relative L2 is `norm(prediction - reference) / norm(reference)`. Relative errors
are stored as fractions: multiply by 100 for percentages. MSE has squared
solution units; relative L2 is dimensionless. The current Burgers benchmark uses
nondimensional coordinates and solution values.

| Field | Meaning |
| --- | --- |
| `final_mse` | MSE of the final saved model over the complete evaluation grid. |
| `final_relative_l2` | Relative L2 of that final model over the same grid. |
| `upstream_min_loss_step` | Recorded global step with the smallest sum of training-loss components. The first occurrence is used if sums tie. |
| `upstream_mse_at_min_loss` | Reference-grid MSE at that recorded step. |
| `upstream_l2_at_min_loss` | Reference-grid relative L2 at that recorded step; used for upstream-style reproduction reporting. |
| `upstream_min_observed_mse` | Smallest MSE in the recorded history. |
| `upstream_min_observed_l2` | Smallest relative L2 in the recorded history. Its step may differ from the minimum-MSE or minimum-training-loss step. |
| `movement_events` | Total PACMANN events across training; sum of the stage event counts. |
| `training_seconds` | Sum of stage `seconds`, including compilation and training with movement callbacks, excluding evaluation reload and post-processing. |

The `upstream_` prefix identifies reporting conventions retained from the
released reproduction. Minima refer to recorded observations, not every network
iteration. Minimum observed test errors are descriptive; they do not select
the primary result. The final weights are reloaded into a fresh model and checked
against final predictions during execution; no separate reload metric is stored.

The campaign protocol determines which metric is primary. The current sigma
search selects using `final_relative_l2`, while upstream-style reporting retains
the training-loss-selected errors; these answer different comparisons.

## Recorded convergence: `history.npz`

NPZ files are compressed NumPy archives, readable with `numpy.load` using
`allow_pickle=False`. Here `H` is the number of recorded history rows; floating
arrays use float64.

| Array | Shape/type | Meaning |
| --- | --- | --- |
| `steps` | `(H,)`, integers | Cumulative network optimizer step for each observation. Stage transitions can record the same step more than once. |
| `loss_train` | `(H, 3)` | Training-loss components in order: PDE residual, zero boundary condition, initial condition. Their sum is used for minimum-training-loss selection. |
| `loss_test` | `(H, 3)` | DeepXDE's recorded test-loss components in the same order. These are loss diagnostics, distinct from reference-solution prediction errors in `metrics_test`. |
| `metrics_test` | `(H, 2)` | Column 0: reference-grid MSE. Column 1: reference-grid relative L2, as a fraction. |

Recording follows DeepXDE's display cadence and stage endpoints; L-BFGS history
is recorded at optimizer-call boundaries rather than every internal iteration.
Stage records in `training.json` identify optimizer endpoints. Model weights are
saved only at those endpoints, so a history-selected minimum need not have a
corresponding saved model.

## Final evaluation data: `predictions.npz`

| Array | Shape | Meaning |
| --- | --- | --- |
| `coordinates` | `(N, 2)` | Evaluation coordinate pairs `[x, t]`. |
| `reference` | `(N, 1)` | Reference solution `u` at each coordinate. |
| `prediction` | `(N, 1)` | Final model's predicted `u` at each coordinate. |

All three arrays use float64. The current reference grid has 256 spatial points
and 100 times, giving 25,600 rows. Ordering is time-major: all spatial coordinates
at the first time, then all spatial coordinates at the next time, and so on.
Solution arrays can therefore be reshaped to `(number_of_times, number_of_x)`.

These are evaluation data. Reference solution values are not supervised training
labels in the current reproduction; training uses the PDE and analytic boundary
and initial conditions. Predictions describe the final model, not necessarily
the step selected by minimum training loss.

## Training coordinates: `points/*.npz`

Each archive contains float64 `[x, t]` arrays. These are training coordinates,
not observations of the reference solution.

| Array | Shape | Meaning |
| --- | --- | --- |
| `interior` | `(I, 2)` | Interior physics population extracted by the sampler; PDE collocation coordinates that PACMANN can move. |
| `pde` | `(P, 2)` | Complete `train_x_all` population on which DeepXDE applies the PDE-residual loss. |
| `constraints` | `(C, 2)` | `train_x_bc` coordinates for soft boundary and initial-condition losses, in boundary-then-initial order. Their target values come from the analytic conditions in the problem definition. |

The arrays are not three disjoint point sets. The initial PDE population includes
the interior points and the boundary/initial coordinates. The released PACMANN
callback replaces that PDE population with the moved interior population, while
retaining the separately evaluated constraints. Fixed Hammersley retains the
original PDE population. Constraints remain fixed for both methods.

For the original benchmark settings, `I = 2500` and `C = 240`
(80 boundary and 160 initial).
Initially `P = 2740`; it stays 2740 for Hammersley and becomes 2500 after
PACMANN's first event. After replacement, PACMANN's `interior` and `pde` arrays
are equal. Saved arrays record the populations actually used; do not add their
lengths to calculate a distinct-point budget.

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
initial weight snapshot or separate model for every recorded history minimum.

## Derived analysis: `diagnostics.json` and `diagnostics.png`

[`scripts/analyze_burgers.py`](../scripts/analyze_burgers.py) regenerates these
two files from saved predictions, history, final interior points, and config.
It does not rerun training or require the original dataset file.

| JSON field | Meaning |
| --- | --- |
| `maximum_absolute_error` | Largest absolute final prediction error over the saved evaluation grid, in solution units. |
| `maximum_error_coordinate` | `[x, t]` where that maximum occurs; the first in saved grid order if tied. |
| `times` | Evaluation times in ascending order. |
| `per_time_relative_l2` | One relative-L2 fraction per entry in `times`, normalized by the reference solution norm at that time. Undefined for a zero reference norm; the analysis rejects that case. |

The PNG shows recorded training loss/reference relative L2, final absolute error
over space and time, final interior collocation points, and reference/prediction
slices at the saved times nearest 0.25, 0.50, 0.75, and 0.99. These are derived
diagnostics; the trainer's summary remains in `metrics.json`.
