# Burgers: frozen-setting ten-seed comparison

**Frozen protocol; all five method batches are complete (50/50 runs).** See the
[consolidated results](../../../docs/results.md#frozen-n1024-ten-seed-evaluation).
The [configuration](../../../config/burgers_nu_0p0031830989_10seed.yaml) contains
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
[results](../../../docs/results.md#consolidated-evaluation-results), while retaining
all seeds in the primary summaries and paired comparisons.
Report a worker failure after all started workers finish; retain every
run's artifacts and review failures before launching another method. Follow the
[recovery policy](../../../AGENTS.md#execution-recovery). Do not present a complete
ten-seed mean from a successful subset; report missing/failed outcomes and the
available paired count explicitly.

## Execution

Follow the repository's [setup instructions](../../../README.md#setup-and-execution).
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
described in the [environment instructions](../../../README.md#setup-and-execution).

Report mean and sample SD of all ten `training_seconds` values for each method,
describing the timing condition as up to ten concurrent seeds on one GPU. These
recorded times include competition with other workers, including workers doing
final evaluation. Do not combine them with timings from serial tuning runs.

The launcher rejects occupied destinations and stops after a failed batch's
workers finish. It does not skip, retry, or resume runs. Per-run artifacts are
retained here; completed training requires predictions, metrics, and diagnostics
before comparison. Saved weights support evaluation, not exact training resume.

## Exploratory checkpoint diagnostics

Post hoc inspection selected seeds 4000, 4002, 4004, and 4006 across all five
methods, evaluating their ten saved optimizer-stage checkpoints (200 evaluations).
The [checkpoint metrics](diagnostics/stage_metrics.csv) use the archived solution
evaluation coordinates and reference, the configured independent residual grid,
and each checkpoint's saved PDE points. Solution relative L2 is stored as a
fraction. These selected examples describe failure trajectories; they do not
establish causes or change the final-checkpoint evaluation policy.

The transition figures for [PACMANN seed 4002](diagnostics/pacmann_adam_4002_transition.png)
and [diffusion seed 4004](diagnostics/pacmann_diffusion_4004_transition.png) compare
phase 2 L-BFGS, phase 3 Adam, and phase 3 L-BFGS. The top panels show absolute
PDE residual with saved PDE points in black; the bottom panels compare reference
(solid) and predicted (dashed) solutions at `t = 0.5, 0.75, 0.99`. Recreate both
figures from archived checkpoints without training:

```bash
venv/bin/python -m scripts.diagnoses.plot_burgers_nu_0p0031830989_displaced_front
```

## Per-method results

Final-checkpoint outcomes for every planned seed are shown below. Solution L2 is
expressed as percent; the other error metrics are unnormalized. Training time is
in seconds under the concurrency conditions above. Consolidated rows report
mean ± sample SD. Supplementary trimmed means remove one lowest and one highest
value independently for solution L2 and residual RMS; see the
[definition and limitations](../../../docs/results.md#consolidated-evaluation-results).
Cross-method comparisons remain in [results](../../../docs/results.md#frozen-n1024-ten-seed-evaluation).

### [Fixed Hammersley](fixed_hammersley/)

| Seed | Solution MSE | Solution L2 (%) | Residual RMS | Boundary RMSE | Initial RMSE | Training seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4000 | 3.20795e-06 | 0.29105 | 0.0280008 | 0.000188581 | 0.000568541 | 9505.7 |
| 4001 | 1.5728e-05 | 0.64445 | 0.0437026 | 8.5356e-05 | 0.000271495 | 9562.5 |
| 4002 | 6.91911e-06 | 0.42745 | 0.0296447 | 0.000115879 | 0.000270559 | 9354.8 |
| 4003 | 2.04641e-05 | 0.73511 | 0.0323068 | 0.000108367 | 0.000220698 | 9557.2 |
| 4004 | 2.05833e-05 | 0.73725 | 0.0329556 | 7.78325e-05 | 0.000240949 | 9395.7 |
| 4005 | 8.12516e-06 | 0.46320 | 0.0245709 | 0.000110127 | 0.000289011 | 9443.4 |
| 4006 | 0.154927 | 63.96177 | 4.17012 | 0.000327142 | 0.00206593 | 9000.4 |
| 4007 | 9.7826e-07 | 0.16073 | 0.0196895 | 0.000126174 | 0.00037224 | 9572.5 |
| 4008 | 1.92201e-06 | 0.22529 | 0.0307403 | 8.66774e-05 | 0.000236557 | 9468.9 |
| 4009 | 3.52307e-06 | 0.30501 | 0.0328799 | 0.000106927 | 0.000284051 | 9346.9 |
| Mean ± SD | 0.0155009 ± 0.0489895 | 6.79513 ± 20.08737 | 0.444461 ± 1.30908 | 0.000133306 ± 7.48243e-05 | 0.000482003 ± 0.00056573 | 9420.8 ± 169.6 |
| Trimmed mean | — | 0.47860 | 0.0318502 | — | — | — |

Sampling updates: 0 per seed.

### [Random resampling](random_resampling/)

| Seed | Solution MSE | Solution L2 (%) | Residual RMS | Boundary RMSE | Initial RMSE | Training seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4000 | 7.14072e-06 | 0.43424 | 0.0232759 | 0.000124384 | 0.000145046 | 9101.1 |
| 4001 | 1.32556e-07 | 0.05916 | 0.0146084 | 0.00011287 | 0.000202492 | 9119.8 |
| 4002 | 8.30018e-07 | 0.14805 | 0.0119498 | 8.86308e-05 | 0.000305103 | 9147.6 |
| 4003 | 1.97019e-06 | 0.22809 | 0.0200279 | 0.000121474 | 0.000223617 | 9129.6 |
| 4004 | 7.97565e-06 | 0.45892 | 0.0117573 | 0.000214293 | 0.000357311 | 9089.0 |
| 4005 | 1.19249e-06 | 0.17745 | 0.0119512 | 0.000161141 | 0.000294145 | 9140.7 |
| 4006 | 3.14563e-05 | 0.91140 | 0.0267699 | 7.42782e-05 | 0.000171368 | 8752.4 |
| 4007 | 4.07979e-05 | 1.03795 | 0.0287328 | 0.000111822 | 0.000267596 | 9047.1 |
| 4008 | 1.79641e-05 | 0.68875 | 0.0398089 | 8.54762e-05 | 0.000191216 | 8842.9 |
| 4009 | 4.25907e-06 | 0.33536 | 0.0119489 | 0.00012914 | 0.000300202 | 8727.1 |
| Mean ± SD | 1.13719e-05 ± 1.42292e-05 | 0.44794 ± 0.33274 | 0.0200831 ± 0.00952523 | 0.000122351 ± 4.08487e-05 | 0.00024581 ± 6.893e-05 | 9009.7 ± 167.6 |
| Trimmed mean | — | 0.42278 | 0.0186581 | — | — | — |

Sampling updates: 700 per seed.

### [RAD](rad/)

| Seed | Solution MSE | Solution L2 (%) | Residual RMS | Boundary RMSE | Initial RMSE | Training seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4000 | 4.45924e-07 | 0.10851 | 0.00632936 | 0.00013951 | 0.000285945 | 8938.0 |
| 4001 | 1.28637e-07 | 0.05828 | 0.00203292 | 0.00016351 | 0.000282582 | 8911.2 |
| 4002 | 1.16694e-06 | 0.17554 | 0.0192312 | 0.000157578 | 0.00035192 | 8932.1 |
| 4003 | 8.79019e-08 | 0.04818 | 0.00199634 | 0.000112769 | 0.000428856 | 8931.4 |
| 4004 | 1.22076e-07 | 0.05678 | 0.00375718 | 0.00026298 | 0.000329297 | 8936.6 |
| 4005 | 1.2745e-07 | 0.05801 | 0.00352121 | 0.000191178 | 0.000526106 | 8917.2 |
| 4006 | 0.000474156 | 3.53848 | 0.090849 | 0.000119447 | 0.000147056 | 8927.3 |
| 4007 | 2.2739e-07 | 0.07749 | 0.0120701 | 0.000137756 | 0.000217786 | 8919.2 |
| 4008 | 1.12151e-07 | 0.05442 | 0.00222902 | 0.00015422 | 0.000368333 | 8932.0 |
| 4009 | 2.10477e-07 | 0.07455 | 0.00690446 | 0.000186402 | 0.000482698 | 8936.2 |
| Mean ± SD | 4.76785e-05 ± 0.000149849 | 0.42503 ± 1.09462 | 0.0148921 ± 0.027241 | 0.000162535 ± 4.3527e-05 | 0.000342058 ± 0.000116568 | 8928.1 ± 9.2 |
| Trimmed mean | — | 0.08295 | 0.00700943 | — | — | — |

Sampling updates: 700 per seed.

### [PACMANN](pacmann_adam/)

| Seed | Solution MSE | Solution L2 (%) | Residual RMS | Boundary RMSE | Initial RMSE | Training seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4000 | 7.80805e-08 | 0.04541 | 0.00530195 | 0.000167993 | 0.00022868 | 9107.1 |
| 4001 | 7.84028e-08 | 0.04550 | 0.0031081 | 0.000111992 | 0.000369017 | 9065.1 |
| 4002 | 0.114951 | 55.09497 | 2.92553 | 0.000878069 | 0.0137854 | 8965.9 |
| 4003 | 1.03934e-06 | 0.16567 | 0.00519094 | 0.000132565 | 0.000235839 | 8948.0 |
| 4004 | 1.27511e-07 | 0.05803 | 0.00283742 | 0.000170114 | 0.000261211 | 9121.4 |
| 4005 | 1.15184e-07 | 0.05515 | 0.0032305 | 0.000162694 | 0.000363078 | 8954.5 |
| 4006 | 0.152715 | 63.50343 | 2.98031 | 0.00150141 | 0.034423 | 8944.6 |
| 4007 | 5.23628e-08 | 0.03719 | 0.00564723 | 0.000165304 | 0.000273733 | 8977.9 |
| 4008 | 5.27358e-08 | 0.03732 | 0.00362157 | 7.54099e-05 | 0.000312793 | 9080.3 |
| 4009 | 3.11263e-07 | 0.09066 | 0.00452938 | 0.000142517 | 0.000266641 | 9079.1 |
| Mean ± SD | 0.0267667 ± 0.0571265 | 11.91333 ± 25.05309 | 0.593931 ± 1.24336 | 0.000350807 ± 0.000466864 | 0.00505194 ± 0.0111577 | 9024.4 ± 72.0 |
| Trimmed mean | — | 6.94909 | 0.36952 | — | — | — |

Sampling updates: 700 per seed.

### [PACMANN + diffusion](pacmann_diffusion/)

| Seed | Solution MSE | Solution L2 (%) | Residual RMS | Boundary RMSE | Initial RMSE | Training seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4000 | 3.79704e-08 | 0.03166 | 0.00319881 | 0.000108014 | 0.000253213 | 8997.0 |
| 4001 | 8.83845e-08 | 0.04831 | 0.00418438 | 0.000199022 | 0.000340726 | 9099.7 |
| 4002 | 0.18444 | 69.78852 | 4.71596 | 0.000998999 | 0.0612308 | 8999.6 |
| 4003 | 1.05915e-06 | 0.16724 | 0.00428316 | 0.000136612 | 0.000289678 | 9103.8 |
| 4004 | 0.0169002 | 21.12525 | 0.794535 | 0.000326859 | 0.00116677 | 8961.6 |
| 4005 | 1.44869e-06 | 0.19559 | 0.00480385 | 0.000209052 | 0.000249669 | 8984.7 |
| 4006 | 0.0485264 | 35.79689 | 2.25145 | 0.00032623 | 0.00156724 | 8985.8 |
| 4007 | 4.52521e-08 | 0.03457 | 0.00682301 | 0.000169052 | 0.000358389 | 8998.7 |
| 4008 | 1.61956e-07 | 0.06540 | 0.0049314 | 0.000125169 | 0.000318012 | 9068.5 |
| 4009 | 5.65231e-08 | 0.03863 | 0.00373017 | 0.00014924 | 0.000346217 | 9002.7 |
| Mean ± SD | 0.0249869 ± 0.0581331 | 12.72921 ± 23.51803 | 0.77939 ± 1.55871 | 0.000274825 ± 0.00026586 | 0.00661207 ± 0.0191964 | 9020.2 ± 50.8 |
| Trimmed mean | — | 7.18399 | 0.384343 | — | — | — |

Sampling updates: 700 per seed.

[Study controls](../../../docs/research_context.md#accepted-controls-and-evaluation-policy)
· [Development selection](../../../docs/results.md#n1024-diffusion-development-selection)
· [Artifact dictionary](../../../docs/output_format.md)
· [Environment](../../../README.md#setup-and-execution)
· [Optimizer semantics](../../../references/upstream.md#controlled-study-choices)
