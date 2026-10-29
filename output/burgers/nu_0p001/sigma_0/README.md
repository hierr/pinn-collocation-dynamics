# Burgers diffusion tuning at ν = 0.001

The 21-run development sweep completed without reported execution failures.
The initial positive winner, `sigma = 1e-5`, triggered one approved lower-boundary
expansion to `1e-7` and `1e-6`. Freeze **sigma = 1e-6**, the positive candidate
with the lowest mean final solution relative L2. It is interior among the six
tested positive values, so no further expansion is required. Sigma zero remains
a comparison baseline; it has lower mean solution and residual error than the
selected coefficient on these development seeds.

## Reference and controls

Use `data/burgers_dataset/burgers_nu_0p001_N4096.npz`, transferred unchanged
from `dataset-validation` checkpoint `41529cb`. Its SHA-256 is
`b628305506f4d3fcfc5052c725bfe595fee716bce21e951165a8ce0af85d89de`.
N4096 denotes Chebyshev degree, giving 4,097 spatial nodes. The solver uses
Radau with an analytical Jacobian, 201 snapshots through `t = 1`,
`rtol = 1e-11`, `atol = 1e-13`, and `max_step = 0.005`.

N2048–N4096 solution relative L2 disagreement is 0.00964855% on 8,193 uniform
points and 199 shared times through `t = 0.99`; spatial-gradient disagreement
is 1.00069%. This supports N4096 as the working reference, without an absolute
error bound or temporal-convergence certification. Resolution artifacts remain
on `dataset-validation`. The reference is used for evaluation only.

The [configurations](../../../../config/burgers/nu_0p001/) retain the
shared network, optimizer schedule, point budgets, and paired seeds 3000–3002.
The initial grid compares ordinary PACMANN with positive sigma values
`1e-5`, `1e-4`, `1e-3`, and `1e-2`: 15 planned runs. Candidates run sequentially,
with three concurrent seeds per candidate and one numerical CPU thread per worker.
The first baseline seed runs alone for the evaluation-grid check; its timing
has different concurrency from the later tuning batches.
The lower expansion adds six runs, giving 21 completed runs in total.

All runs record training revision `ddac4705b14e72c0cf9eeb122affcd438b719110`.
The expansion YAMLs and execution selections were added under the approved
protocol without changing the training, sampling, or evaluation implementation.
All 21 restored final checkpoints reproduce saved predictions and the five
solution, residual, and constraint error metrics.

## Evaluation and selection

The 1,024-point solution grid and 1,024-by-200 residual midpoint grid were
provisional before the baseline check. Train only planned PACMANN baseline seed
3000 first. Using its fixed saved weights and the same N4096 reference, compare
solution relative L2 and residual RMS at 1,024 versus 4,096 spatial evaluation
points. Keep the 199
native solution times through `0.99` and 200 residual midpoint times unchanged.
For each metric, compute `100 * abs(coarse - fine) / abs(fine)`. Retain the
current grid if both changes are below 1%. If either fails, the bounded fallback
compares 4,096 with 8,192, then 8,192 with 16,384
spatial points, retaining the first coarser grid passing both metrics. Stop if
none passes. This screens spatial sampling for one model and does not certify
temporal resolution. Revisit only if sensitivity appears or conclusions depend
on differences of comparable size.

Select the positive sigma with the lowest arithmetic mean final solution
relative L2 over all three seeds; exact ties favor the smaller sigma. Sigma zero
is a comparison baseline only. A positive boundary winner triggers expansion
by one and two decades on that side, with at most two expansion rounds. Freeze
the best tested positive value and disclose any remaining boundary limitation.
Preserve numerical failures for review. Report residual error at the same
solution-selected sigma. Ten-seed evaluation follows separately with frozen settings.

## Baseline and resolved evaluation

PACMANN seed 3000 completed all five Adam/L-BFGS phases in 916.3 training seconds,
with 700 point-movement events.

The [fixed-model grid check](evaluation_grid_check.json) gives:

| Spatial points | Solution relative L2 (%) | PDE residual RMS |
| ---: | ---: | ---: |
| 1,024 | 12.3790 | 0.784389 |
| 4,096 | 12.3783 | 0.784790 |

Relative changes are 0.005708% for solution relative L2 and 0.051008% for residual
RMS, both below the 1% threshold. Retain 1,024 spatial points for both metrics,
199 native solution times, and 200 residual midpoint times. No configuration
change or baseline rescoring was needed; training records and weights are
unchanged. This is one development-seed outcome, not a method comparison.

## Final solution error

All values are final-model relative L2 in percent. Paired changes are candidate
minus the same-seed sigma-zero error in percentage points; negative values favor
diffusion. Aggregates retain all three development seeds and use sample SD.

| Sigma | Seed 3000 | Seed 3001 | Seed 3002 | Mean ± sample SD (%) | Mean paired change (pp) | Improved / 3 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 12.37899 | 42.99267 | 69.90527 | 41.75898 ± 28.78297 | — | — |
| 1e-7 | 47.07456 | 52.67994 | 80.18618 | 59.98023 ± 17.72189 | +18.22125 | 0/3 |
| **1e-6** | **78.27303** | **47.83910** | **6.38810** | **44.16674 ± 36.08290** | **+2.40776** | **1/3** |
| 1e-5 | 50.89774 | 35.96433 | 76.85607 | 54.57271 ± 20.69209 | +12.81373 | 1/3 |
| 1e-4 | 82.27323 | 49.41276 | 81.18083 | 70.95560 ± 18.66465 | +29.19662 | 0/3 |
| 1e-3 | 72.83576 | 41.87048 | 72.11534 | 62.27386 ± 17.67351 | +20.51488 | 1/3 |
| 1e-2 | 55.20920 | 45.75740 | 73.50662 | 58.15774 ± 14.10763 | +16.39876 | 0/3 |

At the selected coefficient, only seed 3002 improves over ordinary PACMANN.
The paired mean change is +2.40776 percentage points with sample SD 64.74006.
Large errors and seed variability remain part of the fixed-budget development
outcomes; successful execution does not imply accurate solutions. These
selection results do not establish held-out performance or a sharpness effect.

## Final PDE residual RMS

Residual RMS uses the independent midpoint grid and is reported at the same
solution-selected sigma; no separate residual optimization was performed.

| Sigma | Seed 3000 | Seed 3001 | Seed 3002 | Mean ± sample SD |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.784389 | 1.923399 | 6.087037 | 2.931608 ± 2.791394 |
| 1e-7 | 2.189146 | 3.069884 | 9.307586 | 4.855539 ± 3.880654 |
| **1e-6** | **4.636818** | **3.045645** | **1.166312** | **2.949592 ± 1.737246** |
| 1e-5 | 2.887658 | 3.202915 | 4.917766 | 3.669446 ± 1.092508 |
| 1e-4 | 9.011616 | 3.757635 | 5.943779 | 6.237677 ± 2.639292 |
| 1e-3 | 4.528236 | 1.905285 | 3.465924 | 3.299815 ± 1.319342 |
| 1e-2 | 1.998411 | 2.293778 | 6.291543 | 3.527911 ± 2.397928 |

At `sigma = 1e-6`, the paired mean residual-RMS change is +0.017983
(sample SD 4.489610), with lower residual only in seed 3002. Neither mean
outcome improves over sigma zero on these development seeds.

## Optimizer work and execution conditions

Training used float64 on one NVIDIA GeForce RTX 5070 (12 GB), with one numerical
CPU thread per worker. All runs completed 35,000 Adam steps and 700 movement
events. L-BFGS completed 15,000 iterations per run except `sigma = 1e-2`, seed
3000, which completed 15,327 under DeepXDE's batched driver. Function-evaluation
counts and stage times remain in `training.json`.

Sigma-zero seed 3000 ran alone (916.3 training seconds); seeds 3001–3002 ran
concurrently (1,779.0 and 1,784.7 seconds). Positive candidates used three
concurrent seeds per batch (2,686.3–2,727.4 seconds per run), with sigma batches
executed sequentially. These different concurrency levels prevent a direct
baseline-versus-diffusion timing comparison. Times include compilation and
training and exclude post-processing. Ten-seed evaluation has not started.

## Reproduction

From the repository root in the [documented environment](../../../../README.md#setup-and-execution):

Run only the first baseline seed:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
venv/bin/python -m scripts.run_experiments.run_burgers_baseline
```

The baseline entry point archives the full three-seed configuration but executes
only seed 3000. Check its fixed weights and resolve the evaluation settings with:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
venv/bin/python -m scripts.post_processing.check_burgers_evaluation_grid
```

The checker writes `evaluation_grid_check.json` in this directory and updates
the listed configurations to the selected density. If density changes, it retains
the baseline's original configuration, predictions, metrics, and diagnostics
in `seed_3000/initial_evaluation/` before regenerating derived evaluation files.
Training records, metadata, and model weights remain unchanged.
The separate `scripts/run_experiments/run_burgers_sigma_tuning.py` retains
`ACTION = "summarize"`; it reads artifacts without training or writing files.
To reproduce the initial search, keep only `sigma_0`, `sigma_1e-5`, `sigma_1e-4`,
`sigma_1e-3`, and `sigma_1e-2` in the runner's `CONFIG_PATHS`. After a successful
grid check, select `ACTION = "run"` and execute:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
venv/bin/python -m scripts.run_experiments.run_burgers_sigma_tuning
```

This runs baseline seeds 3001–3002 and all three seeds for each initial positive
candidate. `BASELINE_SEEDS` selects launch seeds only; archived plans and summary
comparisons retain all three seeds. Keep the preparation revision fixed during
training; approved expansion configurations and execution selections can be
resolved under this protocol and committed with the completed results. Outputs use
`output/burgers/nu_0p001/sigma_*/<method>/seed_<seed>/`.

For the first lower expansion, run only
`config/burgers/nu_0p001/burgers_nu_0p001_sigma_1e-7.yaml` and
`config/burgers/nu_0p001/burgers_nu_0p001_sigma_1e-6.yaml`, in that order.
For each, set `CONFIG_PATH` in `scripts/run_experiments/run_burgers_campaign.py`
to the selected file and `METHODS = ["pacmann_diffusion"]`, then execute:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
venv/bin/python -m scripts.run_experiments.run_burgers_campaign
```

Each batch runs seeds 3000–3002 concurrently. Completed candidates are not rerun;
restore all seven `CONFIG_PATHS` and `ACTION = "summarize"` to compare the
complete search with the same tuning command. Existing artifact directories
cannot be overwritten; reproduction requires a checkout without those run
directories.
See [study context](../../../../docs/research_context.md),
[results](../../../../docs/results.md), and
[artifact meanings](../../../../docs/output_format.md).
